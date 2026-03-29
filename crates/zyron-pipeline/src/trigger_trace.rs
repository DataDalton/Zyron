//! Trigger execution tracing for debugging trigger chains.
//!
//! TriggerTracer records a tree of trigger executions within a single
//! transaction, capturing timing, depth, and results. Activated
//! per-session via SET trigger_trace = on.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// A single entry in the trigger execution trace tree.
/// Each entry records one trigger invocation with its timing,
/// result, and any child triggers it caused.
#[derive(Clone, Debug)]
pub struct TriggerTraceEntry {
    pub triggerName: String,
    pub tableName: String,
    pub timing: String,
    pub event: String,
    pub depth: u32,
    pub durationNs: u64,
    pub result: String,
    pub children: Vec<TriggerTraceEntry>,
}

/// Handle returned by beginTrace. Holds the index into the entries
/// vector and the start time for duration measurement.
pub struct TraceHandle {
    pub index: usize,
    pub startTime: Instant,
}

/// Combined state protected by a single mutex.
struct TracerState {
    entries: Vec<TriggerTraceEntry>,
    stack: Vec<usize>,
}

/// Records a tree of trigger executions for debugging.
/// When enabled, each trigger invocation is captured with its
/// timing, depth, and result. Uses a single mutex for all mutable
/// state. When disabled (default), no lock is acquired.
pub struct TriggerTracer {
    enabled: AtomicBool,
    state: parking_lot::Mutex<TracerState>,
}

impl TriggerTracer {
    /// Creates a new tracer in the disabled state.
    pub fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            state: parking_lot::Mutex::new(TracerState {
                entries: Vec::new(),
                stack: Vec::new(),
            }),
        }
    }

    /// Enables or disables trace collection.
    pub fn setEnabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Returns whether tracing is currently enabled.
    pub fn isEnabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Begins tracing a trigger invocation. Returns a TraceHandle
    /// if tracing is enabled, or None if disabled.
    /// The handle must be passed to endTrace when the trigger completes.
    pub fn beginTrace(
        &self,
        triggerName: &str,
        tableName: &str,
        timing: &str,
        event: &str,
        depth: u32,
    ) -> Option<TraceHandle> {
        if !self.isEnabled() {
            return None;
        }

        let entry = TriggerTraceEntry {
            triggerName: triggerName.to_string(),
            tableName: tableName.to_string(),
            timing: timing.to_string(),
            event: event.to_string(),
            depth,
            durationNs: 0,
            result: String::new(),
            children: Vec::new(),
        };

        let mut state = self.state.lock();
        let index = state.entries.len();
        state.entries.push(entry);
        state.stack.push(index);

        Some(TraceHandle {
            index,
            startTime: Instant::now(),
        })
    }

    /// Completes a trace entry with the trigger execution result.
    /// Sets the duration based on the time elapsed since beginTrace
    /// was called.
    pub fn endTrace(&self, handle: TraceHandle, result: &str) {
        let elapsed = handle.startTime.elapsed().as_nanos() as u64;

        let mut state = self.state.lock();
        if handle.index < state.entries.len() {
            state.entries[handle.index].durationNs = elapsed;
            state.entries[handle.index].result = result.to_string();
        }

        // Pop the current entry from the stack.
        if let Some(top) = state.stack.last() {
            if *top == handle.index {
                state.stack.pop();
            }
        }

        // If there is a parent entry on the stack, add this entry
        // as a child of the parent for tree building.
        if let Some(&parentIdx) = state.stack.last() {
            if parentIdx < state.entries.len() && handle.index < state.entries.len() {
                let childClone = state.entries[handle.index].clone();
                state.entries[parentIdx].children.push(childClone);
            }
        }
    }

    /// Returns a clone of all trace entries collected so far.
    pub fn getTrace(&self) -> Vec<TriggerTraceEntry> {
        self.state.lock().entries.clone()
    }

    /// Resets all collected trace entries and clears the stack.
    pub fn clear(&self) {
        let mut state = self.state.lock();
        state.entries.clear();
        state.stack.clear();
    }
}

impl Default for TriggerTracer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_by_default() {
        let tracer = TriggerTracer::new();
        assert!(!tracer.isEnabled());
    }

    #[test]
    fn test_enable_disable() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);
        assert!(tracer.isEnabled());
        tracer.setEnabled(false);
        assert!(!tracer.isEnabled());
    }

    #[test]
    fn test_begin_returns_none_when_disabled() {
        let tracer = TriggerTracer::new();
        let handle = tracer.beginTrace("trig", "orders", "BEFORE", "INSERT", 0);
        assert!(handle.is_none());
    }

    #[test]
    fn test_single_trace_entry() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);

        let handle = tracer.beginTrace("audit_trig", "orders", "AFTER", "INSERT", 0);
        assert!(handle.is_some());
        tracer.endTrace(handle.expect("handle exists"), "ok");

        let entries = tracer.getTrace();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].triggerName, "audit_trig");
        assert_eq!(entries[0].tableName, "orders");
        assert_eq!(entries[0].timing, "AFTER");
        assert_eq!(entries[0].event, "INSERT");
        assert_eq!(entries[0].depth, 0);
        assert_eq!(entries[0].result, "ok");
    }

    #[test]
    fn test_nested_trace_entries() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);

        let h1 = tracer
            .beginTrace("parent_trig", "orders", "AFTER", "INSERT", 0)
            .expect("handle");
        let h2 = tracer
            .beginTrace("child_trig", "audit_log", "AFTER", "INSERT", 1)
            .expect("handle");

        tracer.endTrace(h2, "ok");
        tracer.endTrace(h1, "ok");

        let entries = tracer.getTrace();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].children.len(), 1);
        assert_eq!(entries[0].children[0].triggerName, "child_trig");
    }

    #[test]
    fn test_clear_resets_state() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);

        let h = tracer
            .beginTrace("trig", "t", "BEFORE", "DELETE", 0)
            .expect("handle");
        tracer.endTrace(h, "ok");
        assert_eq!(tracer.getTrace().len(), 1);

        tracer.clear();
        assert!(tracer.getTrace().is_empty());
    }

    #[test]
    fn test_multiple_sequential_traces() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);

        let h1 = tracer
            .beginTrace("t1", "orders", "BEFORE", "INSERT", 0)
            .expect("handle");
        tracer.endTrace(h1, "ok");

        let h2 = tracer
            .beginTrace("t2", "orders", "AFTER", "INSERT", 0)
            .expect("handle");
        tracer.endTrace(h2, "ok");

        let entries = tracer.getTrace();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].triggerName, "t1");
        assert_eq!(entries[1].triggerName, "t2");
    }

    #[test]
    fn test_error_result_recorded() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);

        let h = tracer
            .beginTrace("failing_trig", "t", "BEFORE", "UPDATE", 0)
            .expect("handle");
        tracer.endTrace(h, "error: constraint violation");

        let entries = tracer.getTrace();
        assert_eq!(entries[0].result, "error: constraint violation");
    }

    #[test]
    fn test_depth_recorded() {
        let tracer = TriggerTracer::new();
        tracer.setEnabled(true);

        let h = tracer
            .beginTrace("deep_trig", "t", "AFTER", "INSERT", 5)
            .expect("handle");
        tracer.endTrace(h, "ok");

        assert_eq!(tracer.getTrace()[0].depth, 5);
    }
}
