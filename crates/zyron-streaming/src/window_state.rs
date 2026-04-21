//! Per-window, per-key state store for streaming aggregates.
//!
//! Keys are opaque byte slices. Values are opaque state bytes that the
//! aggregator serializes and deserializes. The store bridges the windowing
//! primitives in crate::window and the aggregate operators in
//! crate::agg_runner.

use crate::window::WindowRange;

// ---------------------------------------------------------------------------
// WindowStateStore
// ---------------------------------------------------------------------------

/// Concurrent per-window, per-key state store.
///
/// Keys are caller-composed byte slices. State slots are opaque Vec<u8> so
/// the aggregator retains full control over its accumulator layout.
pub struct WindowStateStore {
    inner: scc::HashMap<(WindowRange, Vec<u8>), Vec<u8>>,
}

impl WindowStateStore {
    pub fn new() -> Self {
        Self {
            inner: scc::HashMap::new(),
        }
    }

    /// Returns a clone of the state bytes for the given (window, key), or
    /// None if no entry exists.
    pub fn get(&self, window: WindowRange, key: &[u8]) -> Option<Vec<u8>> {
        let composed = (window, key.to_vec());
        let mut out = None;
        self.inner
            .read_sync(&composed, |_, v| out = Some(v.clone()));
        out
    }

    /// Inserts or replaces the state for (window, key).
    pub fn put(&self, window: WindowRange, key: Vec<u8>, state: Vec<u8>) {
        let composed = (window, key);
        let entry = self.inner.entry_sync(composed);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.insert(state);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(state);
            }
        }
    }

    /// Applies `f` to the current state bytes for (window, key), inserting
    /// the returned bytes. The closure receives None if no entry exists.
    pub fn update<F>(&self, window: WindowRange, key: &[u8], f: F)
    where
        F: FnOnce(Option<&[u8]>) -> Vec<u8>,
    {
        let composed = (window, key.to_vec());
        let entry = self.inner.entry_sync(composed);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let next = f(Some(occ.get().as_slice()));
                occ.insert(next);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let next = f(None);
                vac.insert_entry(next);
            }
        }
    }

    /// Drains all entries whose window end_ms is at or before watermark_ms.
    /// Returns (window, key, state) tuples. The aggregator uses this to emit
    /// final results and release memory when a window closes.
    pub fn drain_closed(&self, watermark_ms: i64) -> Vec<(WindowRange, Vec<u8>, Vec<u8>)> {
        let mut keys = Vec::new();
        self.inner.iter_sync(|k, _v| {
            if k.0.end_ms <= watermark_ms {
                keys.push(k.clone());
            }
            true
        });
        let mut drained = Vec::with_capacity(keys.len());
        for k in keys {
            if let Some((composed, state)) = self.inner.remove_sync(&k) {
                drained.push((composed.0, composed.1, state));
            }
        }
        drained
    }

    /// Number of state entries currently held.
    pub fn size(&self) -> usize {
        self.inner.len()
    }
}

impl Default for WindowStateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn w(start: i64, end: i64) -> WindowRange {
        WindowRange::new(start, end)
    }

    #[test]
    fn put_get_roundtrip() {
        let s = WindowStateStore::new();
        s.put(w(0, 100), b"k1".to_vec(), vec![1, 2, 3]);
        assert_eq!(s.get(w(0, 100), b"k1"), Some(vec![1, 2, 3]));
        assert_eq!(s.get(w(0, 100), b"missing"), None);
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn update_applies_function() {
        let s = WindowStateStore::new();
        // First call sees None, initializes to [1].
        s.update(w(0, 100), b"k", |prev| {
            assert!(prev.is_none());
            vec![1]
        });
        // Second call sees [1], appends to [1, 2].
        s.update(w(0, 100), b"k", |prev| {
            let mut buf = prev.unwrap().to_vec();
            buf.push(2);
            buf
        });
        assert_eq!(s.get(w(0, 100), b"k"), Some(vec![1, 2]));
    }

    #[test]
    fn drain_closed_removes_past_windows() {
        let s = WindowStateStore::new();
        s.put(w(0, 100), b"a".to_vec(), vec![1]);
        s.put(w(100, 200), b"b".to_vec(), vec![2]);
        s.put(w(200, 300), b"c".to_vec(), vec![3]);

        // Watermark at 150 closes the first window only.
        let drained = s.drain_closed(150);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, w(0, 100));
        assert_eq!(drained[0].1, b"a".to_vec());
        assert_eq!(drained[0].2, vec![1]);
        assert_eq!(s.size(), 2);

        // Advance to 250 closes the middle window.
        let drained = s.drain_closed(250);
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, w(100, 200));
        assert_eq!(s.size(), 1);
    }
}
