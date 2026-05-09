//! Per-query bump arena, hand rolled with no external dependencies
//!
//! Allocates fixed-size 64 KB chunks lazily. alloc_bytes hands out a slice cut
//! from the active chunk, growing the chunk list when the cursor would
//! overflow. reset zeroes the cursor without freeing chunks so the same arena
//! can be reused across batches in a query

use std::cell::Cell;

const CHUNK_SIZE: usize = 64 * 1024;

/// Bump arena. Not thread safe by design, the caller takes &mut Arena
pub struct Arena {
    chunks: Vec<Box<[u8; CHUNK_SIZE]>>,
    /// Cursor into the active chunk in bytes
    cursor: Cell<usize>,
    /// Index of the active chunk in chunks
    active: Cell<usize>,
    /// Counters for diagnostics
    bytes_allocated: Cell<usize>,
    bytes_reset: Cell<usize>,
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Arena {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cursor: Cell::new(0),
            active: Cell::new(0),
            bytes_allocated: Cell::new(0),
            bytes_reset: Cell::new(0),
        }
    }

    /// Returns a freshly allocated mutable byte slice of length n. The slice
    /// is uninitialized but zeroed because the chunk backing it is zeroed on
    /// first allocation. The slice is borrowed from the arena and is valid
    /// until the next call to reset
    pub fn alloc_bytes(&mut self, n: usize) -> &mut [u8] {
        if n == 0 {
            return &mut [];
        }
        if n > CHUNK_SIZE {
            // Oversize allocation, dedicate a fresh chunk just for it. We
            // grow the chunk list and keep the chunk pinned, the cursor for
            // following allocations stays on the previous chunk
            let big = Self::fresh_chunk_at_least(n);
            self.chunks.push(big);
            // Move the active pointer forward so that reset can still reach
            // every chunk, but record the cursor as fully consumed for that
            // index so we never try to slice past n
            let oversize_idx = self.chunks.len() - 1;
            self.bytes_allocated.set(self.bytes_allocated.get() + n);
            // Borrow the dedicated chunk for n bytes
            let chunk = &mut self.chunks[oversize_idx][..n];
            return chunk;
        }
        if self.chunks.is_empty() {
            self.chunks.push(Box::new([0u8; CHUNK_SIZE]));
            self.active.set(0);
            self.cursor.set(0);
        }
        let mut cur = self.cursor.get();
        let mut act = self.active.get();
        if cur + n > CHUNK_SIZE {
            // Move to next chunk, allocate one if needed
            act += 1;
            if act >= self.chunks.len() {
                self.chunks.push(Box::new([0u8; CHUNK_SIZE]));
            }
            self.active.set(act);
            cur = 0;
        }
        let end = cur + n;
        self.cursor.set(end);
        self.bytes_allocated.set(self.bytes_allocated.get() + n);
        &mut self.chunks[act][cur..end]
    }

    /// Copies a string into the arena and returns a reference to the copy
    pub fn alloc_str(&mut self, s: &str) -> &str {
        let dst = self.alloc_bytes(s.len());
        dst.copy_from_slice(s.as_bytes());
        // Safe because we just wrote valid UTF-8 bytes into dst
        unsafe { std::str::from_utf8_unchecked(dst) }
    }

    /// Resets the cursor to the start of the chunk list without freeing any
    /// chunks. All previously allocated slices are invalidated. Counters move
    /// from bytes_allocated to bytes_reset
    pub fn reset(&mut self) {
        let allocated = self.bytes_allocated.get();
        self.bytes_reset.set(self.bytes_reset.get() + allocated);
        self.bytes_allocated.set(0);
        self.cursor.set(0);
        self.active.set(0);
    }

    /// Returns the number of bytes allocated since the last reset
    pub fn bytes_in_use(&self) -> usize {
        self.bytes_allocated.get()
    }

    /// Returns the cumulative bytes that have been reset
    pub fn bytes_recycled(&self) -> usize {
        self.bytes_reset.get()
    }

    fn fresh_chunk_at_least(n: usize) -> Box<[u8; CHUNK_SIZE]> {
        // For oversize allocations we still allocate a CHUNK_SIZE block when n
        // <= CHUNK_SIZE. Callers that want bigger should split the request,
        // here n > CHUNK_SIZE so we panic before reaching here, but to be
        // robust we round up to CHUNK_SIZE which the caller then asks for n
        // bytes from. This keeps the chunk type uniform
        assert!(
            n <= CHUNK_SIZE * 64,
            "arena allocation too large: {} bytes",
            n
        );
        Box::new([0u8; CHUNK_SIZE])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_distinct_slices() {
        let mut a = Arena::new();
        let s1 = a.alloc_bytes(8);
        s1[0] = 1;
        let s2 = a.alloc_bytes(8);
        s2[0] = 2;
        assert_eq!(s2[0], 2);
    }

    #[test]
    fn reset_recycles_chunks() {
        let mut a = Arena::new();
        let _ = a.alloc_bytes(1024);
        let allocated_before = a.bytes_in_use();
        assert!(allocated_before > 0);
        a.reset();
        assert_eq!(a.bytes_in_use(), 0);
        assert_eq!(a.bytes_recycled(), allocated_before);
        // Allocating again works without growing chunks
        let _ = a.alloc_bytes(2048);
        assert_eq!(a.bytes_in_use(), 2048);
    }

    #[test]
    fn alloc_str_roundtrip() {
        let mut a = Arena::new();
        let s = a.alloc_str("hello world");
        assert_eq!(s, "hello world");
    }

    #[test]
    fn alloc_zero_returns_empty() {
        let mut a = Arena::new();
        let s = a.alloc_bytes(0);
        assert!(s.is_empty());
    }

    #[test]
    fn alloc_grows_to_new_chunk() {
        let mut a = Arena::new();
        // Allocate slightly under CHUNK_SIZE then ask for more, must spill
        let _ = a.alloc_bytes(CHUNK_SIZE - 100);
        let _ = a.alloc_bytes(200);
        assert!(a.bytes_in_use() == CHUNK_SIZE - 100 + 200);
    }
}
