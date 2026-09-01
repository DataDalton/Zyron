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

use zyron_pressure::capability::{CoefficientAccumulator, OperatorKind};
use zyron_pressure::pressure_control::PressureController;

/// Times one operator batch and records it on drop.
///
/// Recording in Drop rather than at an explicit call means an operator that
/// returns early, or errors, still reports the time it spent, which is exactly
/// the work the node did and would otherwise vanish from the measurement.
///
/// The accumulator is held rather than looked up, so a caller measuring
/// against its own can say so. Serving code takes the process controller's
/// through `start`, which is a pointer copy and costs nothing over reaching
/// for the global inside `drop`.
pub struct BatchTimer {
    sink: &'static CoefficientAccumulator,
    kind: OperatorKind,
    started: Instant,
    units: u64,
}

impl BatchTimer {
    /// Begins timing a batch of the given operator kind, reporting to the
    /// process controller.
    #[inline]
    pub fn start(kind: OperatorKind) -> Self {
        Self::start_on(PressureController::global().coefficients(), kind)
    }

    /// Begins timing a batch that reports to `sink` instead of the process
    /// controller, for a caller that has to read its own measurements back
    /// without the rest of the process writing into them.
    #[inline]
    pub fn start_on(sink: &'static CoefficientAccumulator, kind: OperatorKind) -> Self {
        Self {
            sink,
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
        self.sink
            .record_elapsed(self.kind, self.units, self.started.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    /// An accumulator of this test's own.
    ///
    /// The process controller's is written to by every operator in the crate,
    /// including whatever else the test binary is running at the time, so a
    /// test that read a count back from it would be reading other tests'
    /// traffic as well as its own. Each caller gets a separate one, so two
    /// tests naming the same operator kind still cannot see each other.
    macro_rules! private_sink {
        () => {{
            static SINK: OnceLock<CoefficientAccumulator> = OnceLock::new();
            SINK.get_or_init(CoefficientAccumulator::new)
        }};
    }

    #[test]
    fn a_batch_with_rows_moves_the_coefficient() {
        let sink = private_sink!();
        let before = sink.samples(OperatorKind::SetOp);
        {
            let mut timer = BatchTimer::start_on(sink, OperatorKind::SetOp);
            timer.rows(1_000);
        }
        sink.drain();
        assert_eq!(
            sink.samples(OperatorKind::SetOp),
            before + 1_000,
            "the batch was not recorded"
        );
    }

    /// An empty batch says nothing about what a row costs, so it must not be
    /// folded in as though it did.
    #[test]
    fn an_empty_batch_records_nothing() {
        let sink = private_sink!();
        let before = sink.samples(OperatorKind::Window);
        {
            let _timer = BatchTimer::start_on(sink, OperatorKind::Window);
        }
        sink.drain();
        assert_eq!(sink.samples(OperatorKind::Window), before);
    }

    /// The default target is the process controller, so serving code that
    /// calls `start` is measured where the planner reads.
    #[test]
    fn the_default_target_is_the_process_controller() {
        let controller = PressureController::global();
        let timer = BatchTimer::start(OperatorKind::Project);
        assert!(
            std::ptr::eq(timer.sink, controller.coefficients()),
            "start reported somewhere other than the process controller"
        );
    }

    #[test]
    fn rows_accumulate_across_a_piecewise_fill() {
        let mut timer = BatchTimer::start_on(private_sink!(), OperatorKind::Project);
        timer.add_rows(100);
        timer.add_rows(150);
        assert_eq!(timer.units, 250);
    }
}
