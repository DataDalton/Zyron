//! Bounds-checked little endian reader shared by every lake codec.
//!
//! Every decode failure surfaces as ManifestCorrupted carrying the caller
//! supplied context string, so an error names the file it came from

use zyron_common::ZyronError;

pub(crate) fn corrupt(ctx: &str, reason: String) -> ZyronError {
    ZyronError::ManifestCorrupted {
        path: ctx.to_string(),
        reason,
    }
}

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    ctx: &'a str,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], ctx: &'a str) -> Self {
        Self { bytes, pos: 0, ctx }
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    pub(crate) fn corrupt(&self, reason: String) -> ZyronError {
        corrupt(self.ctx, reason)
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], ZyronError> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let s = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(s)
            }
            None => Err(self.corrupt(format!(
                "truncated at offset {}, needed {} more bytes of {}",
                self.pos,
                n,
                self.bytes.len()
            ))),
        }
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], ZyronError> {
        let s = self.take(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(s);
        Ok(a)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, ZyronError> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, ZyronError> {
        Ok(u16::from_le_bytes(self.array::<2>()?))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, ZyronError> {
        Ok(u32::from_le_bytes(self.array::<4>()?))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, ZyronError> {
        Ok(u64::from_le_bytes(self.array::<8>()?))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, ZyronError> {
        Ok(i64::from_le_bytes(self.array::<8>()?))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, ZyronError> {
        Ok(i32::from_le_bytes(self.array::<4>()?))
    }

    pub(crate) fn utf8(&mut self, n: usize, what: &str) -> Result<String, ZyronError> {
        let s = self.take(n)?;
        std::str::from_utf8(s)
            .map(|s| s.to_string())
            .map_err(|_| self.corrupt(format!("{} is not valid UTF-8", what)))
    }

    /// Guards a count field against a corrupt value driving a huge
    /// preallocation, each counted record needs at least `min_size` bytes
    pub(crate) fn check_count(
        &self,
        count: usize,
        min_size: usize,
        what: &str,
    ) -> Result<(), ZyronError> {
        if count > self.remaining() / min_size.max(1) + 1 {
            return Err(self.corrupt(format!("{} count {} exceeds section size", what, count)));
        }
        Ok(())
    }
}
