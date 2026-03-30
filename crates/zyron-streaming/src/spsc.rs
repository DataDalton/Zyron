//! Lock-free single-producer, single-consumer (SPSC) ring buffer.
//!
//! Replaces crossbeam bounded channels between streaming operators.
//! Uses a fixed-size array with AtomicU64 write/read positions and
//! bitmask wrapping. No CAS needed since there is exactly one producer
//! and one consumer. Only Relaxed loads + Release/Acquire stores.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Creates a bounded SPSC channel with the given capacity (rounded up to power of 2).
/// Returns (sender, receiver).
pub fn spsc_channel<T: Send>(capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
    let buffer = SpscRingBuffer::new(capacity);
    let arc = Arc::new(buffer);
    (
        SpscSender {
            inner: Arc::clone(&arc),
        },
        SpscReceiver { inner: arc },
    )
}

/// Write half of an SPSC channel.
pub struct SpscSender<T: Send> {
    inner: Arc<SpscRingBuffer<T>>,
}

// Safety: only one sender thread writes to write_pos.
unsafe impl<T: Send> Send for SpscSender<T> {}

impl<T: Send> SpscSender<T> {
    /// Tries to send a value without blocking. Returns Err(value) if the buffer is full.
    #[inline]
    pub fn try_send(&self, value: T) -> Result<(), T> {
        self.inner.try_push(value)
    }

    /// Sends a value, spinning until space is available.
    pub fn send(&self, mut value: T) {
        loop {
            match self.inner.try_push(value) {
                Ok(()) => return,
                Err(v) => {
                    value = v;
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Returns current number of items in the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Returns the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }
}

/// Read half of an SPSC channel.
pub struct SpscReceiver<T: Send> {
    inner: Arc<SpscRingBuffer<T>>,
}

// Safety: only one receiver thread reads from read_pos.
unsafe impl<T: Send> Send for SpscReceiver<T> {}

impl<T: Send> SpscReceiver<T> {
    /// Tries to receive a value without blocking. Returns None if the buffer is empty.
    #[inline]
    pub fn try_recv(&self) -> Option<T> {
        self.inner.try_pop()
    }

    /// Receives a value, spinning until one is available.
    pub fn recv(&self) -> T {
        loop {
            if let Some(value) = self.inner.try_pop() {
                return value;
            }
            std::hint::spin_loop();
        }
    }

    /// Returns current number of items in the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Returns the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }
}

// ---------------------------------------------------------------------------
// Internal ring buffer
// ---------------------------------------------------------------------------

#[repr(C)]
struct SpscRingBuffer<T> {
    /// Slots stored in UnsafeCell for interior mutability.
    slots: Box<[UnsafeCell<Option<T>>]>,
    /// Power-of-2 capacity.
    capacity: usize,
    /// Bitmask for index wrapping (capacity - 1).
    mask: usize,
    // Producer-side (own cache line)
    write_pos: AtomicU64,
    _pad1: [u8; 120], // 8 + 120 = 128 bytes, isolates write_pos on its own cache line
    // Consumer-side (own cache line)
    read_pos: AtomicU64,
    _pad2: [u8; 120], // 8 + 120 = 128 bytes, isolates read_pos on its own cache line
}

// Safety: T is Send, and producer/consumer access disjoint slots.
unsafe impl<T: Send> Sync for SpscRingBuffer<T> {}

impl<T: Send> SpscRingBuffer<T> {
    fn new(min_capacity: usize) -> Self {
        let capacity = min_capacity.next_power_of_two().max(2);
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(UnsafeCell::new(None));
        }

        Self {
            slots: slots.into_boxed_slice(),
            capacity,
            mask: capacity - 1,
            write_pos: AtomicU64::new(0),
            _pad1: [0u8; 120],
            read_pos: AtomicU64::new(0),
            _pad2: [0u8; 120],
        }
    }

    #[inline]
    fn try_push(&self, value: T) -> Result<(), T> {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Acquire);

        // Check if buffer is full.
        if (write - read) as usize >= self.capacity {
            return Err(value);
        }

        let idx = (write as usize) & self.mask;
        // Safety: only the producer writes to this slot, and it is guaranteed
        // not to overlap with the consumer's current read slot.
        unsafe {
            *self.slots[idx].get() = Some(value);
        }

        // Release store ensures the value write is visible before the consumer
        // sees the updated write_pos.
        self.write_pos.store(write + 1, Ordering::Release);
        Ok(())
    }

    #[inline]
    fn try_pop(&self) -> Option<T> {
        let read = self.read_pos.load(Ordering::Relaxed);
        let write = self.write_pos.load(Ordering::Acquire);

        // Check if buffer is empty.
        if read >= write {
            return None;
        }

        let idx = (read as usize) & self.mask;
        // Safety: only the consumer reads from this slot, and it is guaranteed
        // not to overlap with the producer's current write slot.
        let value = unsafe { (*self.slots[idx].get()).take() };

        // Release store ensures the slot read is complete before the producer
        // sees the updated read_pos and potentially overwrites the slot.
        self.read_pos.store(read + 1, Ordering::Release);
        value
    }

    fn len(&self) -> usize {
        let write = self.write_pos.load(Ordering::Relaxed);
        let read = self.read_pos.load(Ordering::Relaxed);
        (write - read) as usize
    }
}

impl<T> Drop for SpscRingBuffer<T> {
    fn drop(&mut self) {
        // Drain remaining items by directly clearing slots.
        let write = *self.write_pos.get_mut();
        let read = *self.read_pos.get_mut();
        for i in read..write {
            let idx = (i as usize) & self.mask;
            // Safety: we have exclusive access in drop.
            unsafe {
                *self.slots[idx].get() = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_send_recv() {
        let (tx, rx) = spsc_channel::<u64>(4);
        tx.send(1);
        tx.send(2);
        tx.send(3);
        assert_eq!(rx.recv(), 1);
        assert_eq!(rx.recv(), 2);
        assert_eq!(rx.recv(), 3);
    }

    #[test]
    fn test_try_send_full() {
        let (tx, rx) = spsc_channel::<u64>(2);
        assert!(tx.try_send(1).is_ok());
        assert!(tx.try_send(2).is_ok());
        // Buffer should be full now (capacity rounded to 2).
        assert!(tx.try_send(3).is_err());
        // After consuming one, should be able to send again.
        rx.recv();
        assert!(tx.try_send(3).is_ok());
    }

    #[test]
    fn test_try_recv_empty() {
        let (_tx, rx) = spsc_channel::<u64>(4);
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn test_len_and_empty() {
        let (tx, rx) = spsc_channel::<u64>(8);
        assert!(tx.is_empty());
        assert_eq!(tx.len(), 0);
        tx.send(1);
        tx.send(2);
        assert_eq!(rx.len(), 2);
        assert!(!rx.is_empty());
        rx.recv();
        assert_eq!(rx.len(), 1);
    }

    #[test]
    fn test_wraparound() {
        let (tx, rx) = spsc_channel::<u64>(4);
        // Fill and drain multiple times to test wraparound.
        for round in 0..10 {
            for i in 0..4 {
                tx.send(round * 4 + i);
            }
            for i in 0..4 {
                assert_eq!(rx.recv(), round * 4 + i);
            }
        }
    }

    #[test]
    fn test_threaded() {
        let (tx, rx) = spsc_channel::<u64>(1024);
        let count = 100_000u64;

        let producer = std::thread::spawn(move || {
            for i in 0..count {
                tx.send(i);
            }
        });

        let consumer = std::thread::spawn(move || {
            let mut sum = 0u64;
            for _ in 0..count {
                sum += rx.recv();
            }
            sum
        });

        producer.join().expect("producer panicked");
        let sum = consumer.join().expect("consumer panicked");
        assert_eq!(sum, count * (count - 1) / 2);
    }

    #[test]
    fn test_capacity_rounding() {
        let (tx, _rx) = spsc_channel::<u64>(3);
        // Capacity should be rounded to 4 (next power of 2).
        assert_eq!(tx.capacity(), 4);
    }
}
