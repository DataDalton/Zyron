//! Timing hooks that keep the cost model honest.
//!
//! Every operator the planner prices reports what a batch of its rows actually
//! took, so the coefficient the next plan is costed against came from this
//! machine doing this workload rather than from a constant measured somewhere
//! else.
//!
//! Timed once per batch, not once per row. A batch is thousands of rows, so a
//! pair of clock reads spread over it is not measurable, and no sampling is
//! needed here the way it is on the page read path.

use std::time::Instant;

use zyron_pressure::capability::OperatorKind;
use zyron_pressure::pressure_control::PressureController;

/// Times one operator batch and records it on drop.
///
/// Recording in Drop rather than at an explicit call means an operator that
/// returns early, or errors, still reports the time it spent, which is exactly
/// the work the node did and would otherwise vanish from the measurement.
pub struct BatchTimer {
    kind: OperatorKind,
    started: Instant,
    units: u64,
}

impl BatchTimer {
    /// Begins timing a batch of the given operator kind.
    #[inline]
    pub fn start(kind: OperatorKind) -> Self {
        Self {
            kind,
            started: Instant::now(),
            units: 0,
        }
    }

    /// Sets how many rows the batch turned out to hold. A batch that produced
    /// nothing records nothing, because dividing its time by zero rows would
    /// say nothing about what a row costs.
    #[inline]
    pub fn rows(&mut self, units: u64) {
        self.units = units;
    }

    /// Adds to the row count, for an operator that fills a batch in pieces.
    #[inline]
    pub fn add_rows(&mut self, units: u64) {
        self.units = self.units.saturating_add(units);
    }
}

impl Drop for BatchTimer {
    #[inline]
    fn drop(&mut self) {
        if self.units == 0 {
            return;
        }
        PressureController::global().record_operator(self.kind, self.units, self.started.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_with_rows_moves_the_coefficient() {
        let controller = PressureController::global();
        let before = controller.coefficients().samples(OperatorKind::SetOp);
        {
            let mut timer = BatchTimer::start(OperatorKind::SetOp);
            timer.rows(1_000);
        }
        controller.drain_calibration();
        assert!(
            controller.coefficients().samples(OperatorKind::SetOp) >= before + 1_000,
            "the batch was not recorded"
        );
    }

    /// An empty batch says nothing about what a row costs, so it must not be
    /// folded in as though it did.
    #[test]
    fn an_empty_batch_records_nothing() {
        let controller = PressureController::global();
        let before = controller.coefficients().samples(OperatorKind::Window);
        {
            let _timer = BatchTimer::start(OperatorKind::Window);
        }
        controller.drain_calibration();
        assert_eq!(
            controller.coefficients().samples(OperatorKind::Window),
            before
        );
    }

    #[test]
    fn rows_accumulate_across_a_piecewise_fill() {
        let mut timer = BatchTimer::start(OperatorKind::Project);
        timer.add_rows(100);
        timer.add_rows(150);
        assert_eq!(timer.units, 250);
    }
}
