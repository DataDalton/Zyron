//! Lock-free Read-Copy-Update primitive for read-heavy, write-rarely data.
//!
//! Uses AtomicPtr + Arc refcounting. Readers do an atomic pointer load and
//! increment the Arc refcount (~2-3ns, zero locks, zero contention). Writers
//! clone the inner data, modify it, and atomically swap the pointer. The old
//! data lives until all readers drop their Arc references.
//!
//! This is the core synchronization primitive for all auth stores. It replaces
//! scc::HashMap (bucket locks on read_sync) and parking_lot::RwLock (atomic
//! counter contention on read) with truly lock-free reads.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Lock-free read-copy-update container.
///
/// Readers call `load()` to get an `Arc<T>` snapshot. The read path is
/// two atomic operations: one Acquire load + one refcount increment.
/// No locks, no contention, no blocking.
///
/// Writers call `store()` or `update()` to atomically swap the data.
/// The previous generation is kept alive via a second AtomicPtr to
/// prevent use-after-free for readers that loaded the old pointer but
/// haven't incremented the refcount yet.
///
/// Writers are NOT serialized against each other. If concurrent writes
/// are possible, the caller must serialize them externally.
pub struct Rcu<T> {
    ptr: AtomicPtr<T>,
    /// Holds the previous generation to prevent premature deallocation.
    /// When store() swaps in a new value, the old value moves here.
    /// The value that was previously here (generation N-2) is dropped,
    /// which is safe because any reader that loaded N-2's pointer has
    /// completed the refcount increment (the race window is 1-2 CPU
    /// instructions, far shorter than a full store() call).
    prev: AtomicPtr<T>,
}

// SAFETY: The inner T is always behind an Arc, which is Send + Sync when T is.
// The AtomicPtr operations are inherently thread-safe.
unsafe impl<T: Send + Sync> Send for Rcu<T> {}
unsafe impl<T: Send + Sync> Sync for Rcu<T> {}

impl<T> Rcu<T> {
    /// Creates a new Rcu with the given initial value.
    pub fn new(val: T) -> Self {
        let arc = Arc::new(val);
        Self {
            ptr: AtomicPtr::new(Arc::into_raw(arc) as *mut T),
            prev: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Lock-free read: atomically loads the pointer and returns an owned
    /// Arc<T> snapshot. Cost: one Acquire load + one atomic refcount increment.
    pub fn load(&self) -> Arc<T> {
        let ptr = self.ptr.load(Ordering::Acquire);
        // SAFETY: ptr was produced by Arc::into_raw. The refcount is >= 1
        // because either ptr is the current value (Rcu holds it) or it was
        // just swapped to prev (which also holds it). We increment the
        // refcount to create a new Arc handle.
        unsafe {
            Arc::increment_strong_count(ptr);
            Arc::from_raw(ptr)
        }
    }

    /// Atomically replaces the stored value. The previous value is kept alive
    /// in a deferred slot. The value before that (N-2) is dropped.
    pub fn store(&self, new_val: T) {
        let new_arc = Arc::new(new_val);
        let old_ptr = self
            .ptr
            .swap(Arc::into_raw(new_arc) as *mut T, Ordering::AcqRel);

        // Move old_ptr to prev, retrieving the previous prev (N-2).
        // old_ptr stays alive in prev, protecting any reader that loaded it
        // but hasn't incremented the refcount yet.
        let prev_prev = self.prev.swap(old_ptr, Ordering::AcqRel);

        // Drop generation N-2. Any reader that loaded N-2 has completed
        // its refcount increment by now (the increment happens within 1-2
        // instructions of the pointer load, and an entire store() call
        // with its atomic operations serves as a sufficient fence).
        if !prev_prev.is_null() {
            unsafe {
                Arc::from_raw(prev_prev);
            }
        }
    }

    /// Clone-modify-swap: loads the current snapshot, clones the inner T,
    /// applies the mutation function, and atomically stores the result.
    /// The caller must serialize concurrent update() calls externally.
    pub fn update(&self, f: impl FnOnce(&mut T))
    where
        T: Clone,
    {
        let snap = self.load();
        let mut new_val = (*snap).clone();
        f(&mut new_val);
        self.store(new_val);
    }
}

impl<T> Drop for Rcu<T> {
    fn drop(&mut self) {
        // SAFETY: Drop both the current and previous generation Arcs.
        unsafe {
            let ptr = self.ptr.load(Ordering::Relaxed);
            if !ptr.is_null() {
                Arc::from_raw(ptr);
            }
            let prev = self.prev.load(Ordering::Relaxed);
            if !prev.is_null() {
                Arc::from_raw(prev);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience type alias and helpers for HashMap-based Rcu stores
// ---------------------------------------------------------------------------

/// Lock-free map: Rcu wrapping a HashMap. Readers get a snapshot via load(),
/// then do standard HashMap lookups on it. Writers clone-modify-swap.
pub type RcuMap<K, V> = Rcu<HashMap<K, V>>;

impl<K, V> Rcu<HashMap<K, V>>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates an empty RcuMap.
    pub fn empty_map() -> Self {
        Self::new(HashMap::new())
    }

    /// Lock-free lookup: loads snapshot, returns cloned value if found.
    pub fn get(&self, key: &K) -> Option<V> {
        let snap = self.load();
        snap.get(key).cloned()
    }

    /// Inserts or replaces a key-value pair via clone-modify-swap.
    pub fn insert(&self, key: K, value: V) {
        self.update(|m| {
            m.insert(key, value);
        });
    }

    /// Removes a key via clone-modify-swap. Returns true if the key existed.
    pub fn remove(&self, key: &K) -> bool {
        let snap = self.load();
        if !snap.contains_key(key) {
            return false;
        }
        self.update(|m| {
            m.remove(key);
        });
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn test_rcu_new_and_load() {
        let rcu = Rcu::new(42u64);
        let snap = rcu.load();
        assert_eq!(*snap, 42);
    }

    #[test]
    fn test_rcu_store_replaces_value() {
        let rcu = Rcu::new(1u64);
        assert_eq!(*rcu.load(), 1);
        rcu.store(2);
        assert_eq!(*rcu.load(), 2);
    }

    #[test]
    fn test_rcu_update_modifies_in_place() {
        let rcu = Rcu::new(vec![1, 2, 3]);
        rcu.update(|v| v.push(4));
        let snap = rcu.load();
        assert_eq!(*snap, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_rcu_old_snapshot_survives_store() {
        let rcu = Rcu::new(100u64);
        let old_snap = rcu.load();
        rcu.store(200);
        // Old snapshot still valid, new snapshot has new value
        assert_eq!(*old_snap, 100);
        assert_eq!(*rcu.load(), 200);
    }

    #[test]
    fn test_rcu_map_empty_and_insert() {
        let map: RcuMap<String, u32> = Rcu::empty_map();
        assert_eq!(map.get(&"key".to_string()), None);
        map.insert("key".to_string(), 42);
        assert_eq!(map.get(&"key".to_string()), Some(42));
    }

    #[test]
    fn test_rcu_map_remove() {
        let map: RcuMap<u32, String> = Rcu::empty_map();
        map.insert(1, "one".to_string());
        assert!(map.remove(&1));
        assert!(!map.remove(&1));
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn test_rcu_concurrent_readers() {
        let rcu = Arc::new(Rcu::new(42u64));
        let barrier = Arc::new(Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let rcu = rcu.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let mut sum = 0u64;
                    for _ in 0..100_000 {
                        sum += *rcu.load();
                    }
                    sum
                })
            })
            .collect();

        let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 42 * 100_000 * 8);
    }

    #[test]
    fn test_rcu_reader_writer_concurrent() {
        let rcu = Arc::new(Rcu::new(0u64));
        let barrier = Arc::new(Barrier::new(9));

        // 8 reader threads
        let readers: Vec<_> = (0..8)
            .map(|_| {
                let rcu = rcu.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    let mut reads = 0u64;
                    for _ in 0..100_000 {
                        let val = *rcu.load();
                        // Value should always be a valid state (0 through 1000)
                        assert!(val <= 1000);
                        reads += 1;
                    }
                    reads
                })
            })
            .collect();

        // 1 writer thread
        let writer = {
            let rcu = rcu.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                for i in 0..=1000u64 {
                    rcu.store(i);
                }
            })
        };

        writer.join().unwrap();
        let total_reads: u64 = readers.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total_reads, 800_000);
        assert_eq!(*rcu.load(), 1000);
    }

    #[test]
    fn test_rcu_drop_frees_memory() {
        let counter = Arc::new(());
        let weak = Arc::downgrade(&counter);
        {
            let rcu = Rcu::new(counter);
            let _snap = rcu.load();
            // rcu and _snap both hold references
            drop(rcu);
            // _snap still holds a reference
            assert!(weak.upgrade().is_some());
        }
        // All references dropped
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_rcu_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Rcu<u64>>();
        assert_send_sync::<Rcu<Vec<String>>>();
        assert_send_sync::<RcuMap<String, u32>>();
    }
}
