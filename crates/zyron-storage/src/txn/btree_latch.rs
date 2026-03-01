//! B+Tree node latch coupling for concurrent transactional mutations.
//!
//! NodeLatch wraps an AtomicU64 version stamp with a writer bit.
//! Even version = no writer active, odd version = writer active.
//! Readers optimistically load the version, perform their read, then
//! validate the version has not changed. Writers CAS to set the writer
//! bit, perform their mutation, then increment to the next even version.
//!
//! Latch coupling protocol (top-down, deadlock-free):
//! 1. Acquire write latch on parent node
//! 2. Acquire write latch on child node
//! 3. If child is non-full, release parent latch
//! 4. If child is full, hold parent while splitting child, then release parent
//! 5. Always top-down (root first), never hold child while acquiring parent

use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::{Result, ZyronError};

/// Per-node optimistic latch using a version counter with writer bit.
///
/// Version layout:
/// - Bit 0 (LSB): writer flag. 0 = no writer, 1 = writer active.
/// - Bits 1..63: monotonic version counter.
///
/// State transitions:
/// - read_version: loads version, returns Err if writer active (odd)
/// - validate_version: checks version has not changed since read_version
/// - acquire_write: CAS even -> odd (sets writer bit)
/// - release_write: stores old_version + 2 (clears writer bit, bumps version)
pub struct NodeLatch {
    state: AtomicU64,
}

impl NodeLatch {
    /// Creates a new latch with version 0 (unlocked, no writer).
    pub fn new() -> Self {
        Self {
            state: AtomicU64::new(0),
        }
    }

    /// Creates a latch with a specific initial version. Must be even.
    pub fn with_version(version: u64) -> Self {
        debug_assert!(version & 1 == 0, "initial version must be even (no writer)");
        Self {
            state: AtomicU64::new(version),
        }
    }

    /// Loads the current version for an optimistic read.
    ///
    /// Returns the version if no writer is active (even version).
    /// Returns WriteConflict if a writer is active (odd version).
    #[inline]
    pub fn read_version(&self) -> Result<u64> {
        let v = self.state.load(Ordering::Acquire);
        if v & 1 != 0 {
            return Err(ZyronError::VersionConflict);
        }
        Ok(v)
    }

    /// Validates that the version has not changed since read_version.
    ///
    /// Returns true if the version is the same (no writer intervened).
    /// The read is valid only if this returns true.
    #[inline]
    pub fn validate_version(&self, expected: u64) -> bool {
        // Acquire fence to ensure all prior reads are visible before checking.
        self.state.load(Ordering::Acquire) == expected
    }

    /// Acquires the write latch via CAS.
    ///
    /// Sets the writer bit (even -> odd). Returns the old (even) version
    /// on success. Returns WriteConflict if another writer is active or
    /// the version changed.
    #[inline]
    pub fn acquire_write(&self) -> Result<u64> {
        let current = self.state.load(Ordering::Acquire);
        if current & 1 != 0 {
            return Err(ZyronError::VersionConflict);
        }
        // CAS: current (even) -> current | 1 (odd, writer active)
        match self
            .state
            .compare_exchange(current, current | 1, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Ok(current),
            Err(_) => Err(ZyronError::VersionConflict),
        }
    }

    /// Releases the write latch by advancing to the next even version.
    ///
    /// old_version is the value returned by acquire_write.
    /// New version = old_version + 2 (clears writer bit, bumps version).
    #[inline]
    pub fn release_write(&self, old_version: u64) {
        debug_assert!(
            self.state.load(Ordering::Relaxed) & 1 == 1,
            "release_write called without holding write latch"
        );
        // Store the next even version. No CAS needed because we hold the writer bit.
        self.state.store(old_version + 2, Ordering::Release);
    }

    /// Returns the current raw version value (for debugging/testing).
    pub fn current_version(&self) -> u64 {
        self.state.load(Ordering::Relaxed)
    }

    /// Returns true if a writer is currently active.
    #[inline]
    pub fn is_write_locked(&self) -> bool {
        self.state.load(Ordering::Acquire) & 1 != 0
    }
}

impl Default for NodeLatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_latch_unlocked() {
        let latch = NodeLatch::new();
        assert_eq!(latch.current_version(), 0);
        assert!(!latch.is_write_locked());
    }

    #[test]
    fn test_read_version_succeeds_when_unlocked() {
        let latch = NodeLatch::new();
        let v = latch.read_version().unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn test_acquire_and_release_write() {
        let latch = NodeLatch::new();

        let old_v = latch.acquire_write().unwrap();
        assert_eq!(old_v, 0);
        assert!(latch.is_write_locked());

        // read_version should fail while writer active
        assert!(latch.read_version().is_err());

        latch.release_write(old_v);
        assert!(!latch.is_write_locked());
        assert_eq!(latch.current_version(), 2);
    }

    #[test]
    fn test_version_increments() {
        let latch = NodeLatch::new();

        // First write cycle: version 0 -> 2
        let v0 = latch.acquire_write().unwrap();
        latch.release_write(v0);
        assert_eq!(latch.current_version(), 2);

        // Second write cycle: version 2 -> 4
        let v2 = latch.acquire_write().unwrap();
        assert_eq!(v2, 2);
        latch.release_write(v2);
        assert_eq!(latch.current_version(), 4);

        // Third write cycle: version 4 -> 6
        let v4 = latch.acquire_write().unwrap();
        assert_eq!(v4, 4);
        latch.release_write(v4);
        assert_eq!(latch.current_version(), 6);
    }

    #[test]
    fn test_validate_version_unchanged() {
        let latch = NodeLatch::new();
        let v = latch.read_version().unwrap();

        // No writes happened, validation should pass
        assert!(latch.validate_version(v));
    }

    #[test]
    fn test_validate_version_changed() {
        let latch = NodeLatch::new();
        let v = latch.read_version().unwrap();

        // Write cycle changes the version
        let old = latch.acquire_write().unwrap();
        latch.release_write(old);

        // Validation should fail (version changed from 0 to 2)
        assert!(!latch.validate_version(v));
    }

    #[test]
    fn test_double_acquire_fails() {
        let latch = NodeLatch::new();
        let _v = latch.acquire_write().unwrap();

        // Second acquire should fail (writer already active)
        assert!(latch.acquire_write().is_err());
    }

    #[test]
    fn test_with_version() {
        let latch = NodeLatch::with_version(100);
        assert_eq!(latch.current_version(), 100);
        assert!(!latch.is_write_locked());

        let v = latch.read_version().unwrap();
        assert_eq!(v, 100);

        let old = latch.acquire_write().unwrap();
        assert_eq!(old, 100);
        latch.release_write(old);
        assert_eq!(latch.current_version(), 102);
    }

    #[test]
    fn test_optimistic_read_protocol() {
        let latch = NodeLatch::new();

        // Step 1: Load version before read
        let v = latch.read_version().unwrap();

        // Step 2: Perform read (simulated - no actual data here)

        // Step 3: Validate version unchanged
        assert!(latch.validate_version(v));

        // Now a writer intervenes
        let old = latch.acquire_write().unwrap();
        latch.release_write(old);

        // Step 3 (retry): Validation fails, must re-read
        assert!(!latch.validate_version(v));
    }

    #[test]
    fn test_concurrent_read_write() {
        use std::sync::Arc;
        use std::thread;

        let latch = Arc::new(NodeLatch::new());
        let latch_clone = Arc::clone(&latch);

        // Writer thread
        let writer = thread::spawn(move || {
            for _ in 0..100 {
                loop {
                    match latch_clone.acquire_write() {
                        Ok(v) => {
                            // Hold latch briefly
                            std::hint::spin_loop();
                            latch_clone.release_write(v);
                            break;
                        }
                        Err(_) => {
                            // Retry on CAS failure
                            std::hint::spin_loop();
                        }
                    }
                }
            }
        });

        // Reader thread: optimistic read loop
        let latch_reader = Arc::clone(&latch);
        let reader = thread::spawn(move || {
            let mut validated = 0u64;
            let mut retried = 0u64;
            for _ in 0..1000 {
                loop {
                    match latch_reader.read_version() {
                        Ok(v) => {
                            if latch_reader.validate_version(v) {
                                validated += 1;
                                break;
                            } else {
                                retried += 1;
                            }
                        }
                        Err(_) => {
                            retried += 1;
                            std::hint::spin_loop();
                        }
                    }
                }
            }
            (validated, retried)
        });

        writer.join().unwrap();
        let (validated, _retried) = reader.join().unwrap();

        // All reads should eventually validate
        assert_eq!(validated, 1000);
        // Final version should be 200 (100 write cycles, each bumps by 2)
        assert_eq!(latch.current_version(), 200);
    }
}
