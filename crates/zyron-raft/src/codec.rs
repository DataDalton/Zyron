//! The one binary encoding, used by the log file and by the wire.
//!
//! A log entry that has been replicated is byte-identical to the entry the
//! leader wrote to its own file, because both go through the functions here.
//! That is what lets a follower checksum what it received against what the
//! leader checksummed, and it is why there is one encoding rather than a disk
//! format and a wire format that drift apart.
//!
//! Everything is little endian and length prefixed. Nothing is self
//! describing: the reader knows the shape it expects from the frame kind or
//! the record type, so a field cannot be silently skipped by a peer running a
//! different build. A short buffer is an error rather than a default value.

use zyron_common::error::{Result, ZyronError};

/// Longest byte string any single field may carry.
///
/// Sixty four megabytes covers a snapshot chunk and a large value while
/// keeping a corrupt length prefix from being taken at face value.
pub const MAX_FIELD_BYTES: usize = 64 * 1024 * 1024;

#[inline]
pub fn put_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

#[inline]
pub fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
pub fn put_bool(buf: &mut Vec<u8>, v: bool) {
    buf.push(u8::from(v));
}

/// Writes a length prefixed byte string
#[inline]
pub fn put_bytes(buf: &mut Vec<u8>, v: &[u8]) {
    put_u32(buf, v.len() as u32);
    buf.extend_from_slice(v);
}

/// Writes a length prefixed UTF-8 string
#[inline]
pub fn put_str(buf: &mut Vec<u8>, v: &str) {
    put_bytes(buf, v.as_bytes());
}

/// Writes an optional node id as a presence byte and the value.
///
/// Zero is not used as the absent marker because a node id is minted from
/// entropy and an operator can legitimately configure any value
#[inline]
pub fn put_opt_u64(buf: &mut Vec<u8>, v: Option<u64>) {
    match v {
        Some(x) => {
            buf.push(1);
            put_u64(buf, x);
        }
        None => buf.push(0),
    }
}

/// A read position over a borrowed buffer.
///
/// Borrowed rather than owned so decoding a batch of log entries out of one
/// received frame copies only the fields that have to be owned
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(ZyronError::EncodingFailed(format!(
                "raft frame truncated: wanted {n} bytes at offset {}, {} left",
                self.pos,
                self.remaining()
            )));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    #[inline]
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    #[inline]
    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    pub fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    /// Borrows a length prefixed byte string.
    ///
    /// The length is checked against what is actually left before the slice is
    /// taken, so a corrupt prefix is an error rather than a panic
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        if len > MAX_FIELD_BYTES {
            return Err(ZyronError::EncodingFailed(format!(
                "raft field length {len} exceeds the {MAX_FIELD_BYTES} byte limit"
            )));
        }
        self.take(len)
    }

    pub fn vec(&mut self) -> Result<Vec<u8>> {
        Ok(self.bytes()?.to_vec())
    }

    pub fn str(&mut self) -> Result<&'a str> {
        let raw = self.bytes()?;
        std::str::from_utf8(raw)
            .map_err(|e| ZyronError::EncodingFailed(format!("raft field is not UTF-8: {e}")))
    }

    pub fn string(&mut self) -> Result<String> {
        Ok(self.str()?.to_string())
    }

    /// Everything not yet read, for a nested decoder that frames itself.
    ///
    /// Log records carry their own length, so a batch of them travels without
    /// a second length prefix per entry and decodes straight out of the
    /// received buffer
    #[inline]
    pub fn rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    /// Skips bytes a nested decoder has already consumed
    #[inline]
    pub fn advance(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }

    pub fn opt_u64(&mut self) -> Result<Option<u64>> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.u64()?)),
            other => Err(ZyronError::EncodingFailed(format!(
                "raft optional field has presence byte {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_primitive() {
        let mut buf = Vec::new();
        put_u8(&mut buf, 7);
        put_u16(&mut buf, 700);
        put_u32(&mut buf, 70_000);
        put_u64(&mut buf, 7_000_000_000);
        put_bool(&mut buf, true);
        put_bytes(&mut buf, b"payload");
        put_str(&mut buf, "node-3");
        put_opt_u64(&mut buf, Some(11));
        put_opt_u64(&mut buf, None);

        let mut c = Cursor::new(&buf);
        assert_eq!(c.u8().expect("u8"), 7);
        assert_eq!(c.u16().expect("u16"), 700);
        assert_eq!(c.u32().expect("u32"), 70_000);
        assert_eq!(c.u64().expect("u64"), 7_000_000_000);
        assert!(c.bool().expect("bool"));
        assert_eq!(c.bytes().expect("bytes"), b"payload");
        assert_eq!(c.str().expect("str"), "node-3");
        assert_eq!(c.opt_u64().expect("some"), Some(11));
        assert_eq!(c.opt_u64().expect("none"), None);
        assert_eq!(c.remaining(), 0);
    }

    #[test]
    fn truncation_is_an_error_not_a_panic() {
        let mut buf = Vec::new();
        put_u64(&mut buf, 5);
        buf.truncate(4);
        let mut c = Cursor::new(&buf);
        assert!(c.u64().is_err());
    }

    #[test]
    fn oversized_length_prefix_is_refused() {
        let mut buf = Vec::new();
        put_u32(&mut buf, u32::MAX);
        let mut c = Cursor::new(&buf);
        assert!(c.bytes().is_err());
    }
}
