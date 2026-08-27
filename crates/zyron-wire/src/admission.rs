//! Query admission on the serving path.
//!
//! A query is priced before it runs, classified by that price, and then either
//! run, held, or refused. What makes this different from a connection limit is
//! that the queue holds a known amount of work rather than a count of unknowns:
//! the planner already knows roughly what a query will cost, so the node can
//! compare its backlog against a latency objective in the same unit instead of
//! guessing from a queue length.
//!
//! A query small enough that the decision costs more than the work skips the
//! whole path. That is not an optimisation detail, it is what keeps a point
//! lookup at point-lookup latency while the node is under pressure.

use std::time::{Duration, Instant};

use zyron_common::ZyronError;
use zyron_planner::PhysicalPlan;
use zyron_pressure::pressure::{AdmitDecision, WorkloadClass};
use zyron_pressure::pressure_control::PressureController;

/// How long a query may sit in the admission queue before it is refused.
///
/// Bounded so a caller waiting on a node that never drains is told so, rather
/// than being held until its own timeout fires with no explanation. The
/// objective for the class is the natural bound: waiting longer than the
/// objective has already failed the objective.
fn max_queue_wait(class: WorkloadClass) -> Duration {
    let slo = class.slo_seconds();
    if slo.is_finite() {
        // Ten objectives, so a transient burst is absorbed and a genuinely
        // stuck class is reported instead of hidden
        Duration::from_secs_f64(slo * 10.0)
    } else {
        Duration::from_secs(300)
    }
}

/// A query that has been admitted and is running.
///
/// Retires itself on drop, so a query that errors, is cancelled, or panics
/// still gives its work back to the class counters. Leaking one would leave
/// the node permanently believing it was busier than it is, and the ceiling
/// would fall for a query that finished long ago.
pub struct AdmittedQuery {
    class: WorkloadClass,
    estimated_work_seconds: f64,
    tenant_id: Option<String>,
    started: Instant,
    retired: bool,
}

impl AdmittedQuery {
    /// Retires the query, reporting what it actually cost so the cost model
    /// can be scored against reality.
    pub fn complete(mut self) {
        self.retire();
    }

    fn retire(&mut self) {
        if self.retired {
            return;
        }
        self.retired = true;
        PressureController::global().complete(
            self.class,
            self.estimated_work_seconds,
            self.started.elapsed().as_secs_f64(),
            self.tenant_id.as_deref(),
        );
    }

    pub fn class(&self) -> WorkloadClass {
        self.class
    }

    pub fn estimated_work_seconds(&self) -> f64 {
        self.estimated_work_seconds
    }
}

impl Drop for AdmittedQuery {
    fn drop(&mut self) {
        self.retire();
    }
}

/// Prices a plan, waits for room if the class is full, and returns the ticket
/// that retires it.
///
/// Refuses rather than waiting forever: a node that cannot drain a class
/// within ten times its objective is not going to serve this query usefully,
/// and saying so is more use to the caller than a silent hold.
pub async fn admit(
    plan: &PhysicalPlan,
    tenant_id: Option<&str>,
    background: bool,
) -> Result<AdmittedQuery, ZyronError> {
    let controller = PressureController::global();
    let price = zyron_planner::work_estimate::price_plan_live(plan);
    let estimated = price.work_seconds;
    // Recorded before the decision, so a shed query still counts as traffic
    // this node was asked to serve. A manifest built only from what was
    // admitted would describe the load the node was able to take rather than
    // the load it was given
    controller.record_shape(price.fingerprint, estimated);

    let (class, first) = controller.admit(estimated, background, tenant_id);
    let ticket = AdmittedQuery {
        class,
        estimated_work_seconds: estimated,
        tenant_id: tenant_id.map(str::to_string),
        started: Instant::now(),
        retired: false,
    };

    match first {
        AdmitDecision::Admit | AdmitDecision::Bypass => return Ok(ticket),
        AdmitDecision::Shed { reason } => {
            // The ticket never became a running query, so it must not retire
            // work the counters never charged
            std::mem::forget(ticket);
            return Err(ZyronError::AdmissionShed(reason));
        }
        AdmitDecision::Delay(_) => {}
    }

    // Queued. The counters are already holding this query's work, so every
    // path out of here either admits it or abandons that charge
    let waited_from = Instant::now();
    let deadline = max_queue_wait(class);
    loop {
        tokio::time::sleep(Duration::from_millis(1)).await;
        let waited = waited_from.elapsed();
        match controller.retry_queued(class, estimated, waited, tenant_id) {
            AdmitDecision::Admit | AdmitDecision::Bypass => return Ok(ticket),
            AdmitDecision::Shed { reason } => {
                controller.abandon_queued(class, estimated);
                std::mem::forget(ticket);
                return Err(ZyronError::AdmissionShed(reason));
            }
            AdmitDecision::Delay(_) => {
                if waited >= deadline {
                    controller.abandon_queued(class, estimated);
                    std::mem::forget(ticket);
                    return Err(ZyronError::AdmissionShed(format!(
                        "waited {:.1}s for {} capacity without being admitted, \
                         node objective is {:.3}s",
                        waited.as_secs_f64(),
                        class.as_str(),
                        class.slo_seconds()
                    )));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_pressure::pressure_control::PressureController;

    /// The controller is process-wide, so two tests both moving the same
    /// class's counters would race on it. They run one at a time instead.
    static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn the_queue_wait_bound_follows_the_class_objective() {
        assert_eq!(
            max_queue_wait(WorkloadClass::Interactive),
            Duration::from_secs_f64(1.0)
        );
        assert_eq!(
            max_queue_wait(WorkloadClass::Bulk),
            Duration::from_secs_f64(300.0)
        );
        // Background has no objective, so it gets a bound rather than none
        assert!(max_queue_wait(WorkloadClass::Background) > Duration::ZERO);
    }

    /// A ticket that is dropped without an explicit completion must still give
    /// the work back, or an errored query would leave the node believing it
    /// was busy forever.
    #[test]
    fn a_dropped_ticket_retires_its_work() {
        let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
        let controller = PressureController::global();
        let class = WorkloadClass::Bulk;
        let before = controller.counters(class).in_flight();
        {
            controller.counters(class).start_direct(1.0);
            let _ticket = AdmittedQuery {
                class,
                estimated_work_seconds: 1.0,
                tenant_id: None,
                started: Instant::now(),
                retired: false,
            };
            assert_eq!(controller.counters(class).in_flight(), before + 1);
        }
        assert_eq!(
            controller.counters(class).in_flight(),
            before,
            "a dropped ticket leaked its work"
        );
    }

    #[test]
    fn completing_twice_only_retires_once() {
        let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
        let controller = PressureController::global();
        let class = WorkloadClass::Bulk;
        controller.counters(class).start_direct(1.0);
        let before = controller.counters(class).in_flight();
        let mut ticket = AdmittedQuery {
            class,
            estimated_work_seconds: 1.0,
            tenant_id: None,
            started: Instant::now(),
            retired: false,
        };
        ticket.retire();
        ticket.retire();
        assert_eq!(controller.counters(class).in_flight(), before - 1);
    }
}
