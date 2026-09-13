//! Lock-free read-copy-update primitive for read-heavy, write-rarely data.
//!
//! Readers take a snapshot of the current value as an `Arc<T>`, writers
//! publish a whole new value, and the old value lives until its last reader
//! drops it. The swap and the reclamation are `arc_swap`'s, which parks a
//! reader's claim in a per-thread slot before it dereferences the pointer,
//! so a reader preempted between reading the pointer and taking its
//! reference never sees a value a writer freed underneath it.
//!
//! This is the synchronization primitive of the auth stores. A read is a
//! few atomic operations with no lock and no contention between readers. A
//! write is a clone, the mutation, and one pointer swap

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use arc_swap::ArcSwap;

/// Lock-free read-copy-update container
pub struct Rcu<T> {
    inner: ArcSwap<T>,
}

impl<T> Rcu<T> {
    /// Creates a new Rcu with the given initial value
    pub fn new(val: T) -> Self {
        Self {
            inner: ArcSwap::from_pointee(val),
        }
    }

    /// Lock-free read, an owned `Arc<T>` snapshot of the current value
    pub fn load(&self) -> Arc<T> {
        self.inner.load_full()
    }

    /// Lock-free read that borrows the current value for the guard's
    /// lifetime without taking a reference count on it. The guard holds a
    /// per-thread claim that keeps the value alive, so a lookup that
    /// finishes before returning costs a claim and its release rather than
    /// two reference count changes. Held briefly, never across an await,
    /// since a thread has few claim slots and a read that finds none free
    /// takes the reference count instead
    #[inline]
    pub fn read(&self) -> arc_swap::Guard<Arc<T>> {
        self.inner.load()
    }

    /// Atomically replaces the stored value. A reader holding the previous
    /// snapshot keeps it until it drops it
    pub fn store(&self, new_val: T) {
        self.inner.store(Arc::new(new_val));
    }

    /// Clone-modify-swap. `f` receives a clone of the current value, and the
    /// result is published only when no other writer published in between.
    /// Otherwise the clone is discarded and `f` runs again over the newer
    /// value, so concurrent writers never lose one another's changes. `f`
    /// is therefore repeatable and must not move anything it captures
    pub fn update(&self, mut f: impl FnMut(&mut T))
    where
        T: Clone,
    {
        self.inner.rcu(|current| {
            let mut next = (**current).clone();
            f(&mut next);
            next
        });
    }
}

// ---------------------------------------------------------------------------
// Convenience type alias and helpers for HashMap-based Rcu stores
// ---------------------------------------------------------------------------

/// Lock-free map, an `Rcu` over a `HashMap`. Readers look up in a snapshot,
/// writers clone-modify-swap
pub type RcuMap<K, V> = Rcu<HashMap<K, V>>;

impl<K, V> Rcu<HashMap<K, V>>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates an empty RcuMap
    pub fn empty_map() -> Self {
        Self::new(HashMap::new())
    }

    /// Lock-free lookup, the value cloned out of the current snapshot
    /// without taking a reference on the snapshot itself
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.load().get(key).cloned()
    }

    /// Inserts or replaces a key-value pair via clone-modify-swap
    pub fn insert(&self, key: K, value: V) {
        self.update(|m| {
            m.insert(key.clone(), value.clone());
        });
    }

    /// Removes a key via clone-modify-swap. Returns true if the key existed
    /// in the map the removal was published over
    pub fn remove(&self, key: &K) -> bool {
        if !self.inner.load().contains_key(key) {
            return false;
        }
        let mut removed = false;
        self.update(|m| {
            removed = m.remove(key).is_some();
        });
        removed
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

    /// Several writers updating the same value at once, none of their
    /// changes lost, which the clone-modify-swap of a single writer at a
    /// time never had to prove
    #[test]
    fn test_rcu_concurrent_updates_lose_nothing() {
        let rcu = Arc::new(Rcu::new(Vec::<u64>::new()));
        let barrier = Arc::new(Barrier::new(8));
        let writers: Vec<_> = (0..8u64)
            .map(|w| {
                let rcu = rcu.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    for i in 0..500u64 {
                        rcu.update(|v| v.push(w * 1000 + i));
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        let mut all = (*rcu.load()).clone();
        all.sort_unstable();
        let expected: Vec<u64> = (0..8u64)
            .flat_map(|w| (0..500u64).map(move |i| w * 1000 + i))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(all, expected);
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
