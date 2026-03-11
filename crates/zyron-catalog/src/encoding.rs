//! Binary encoding helpers for catalog entry serialization.
//!
//! All catalog entries are stored as heap tuples using a compact binary format:
//! fixed-width fields in little-endian, variable-length strings with u32 length prefix.

use zyron_common::{Result, ZyronError};

/// Writes a u8 to the buffer.
#[inline]
pub fn write_u8(buf: &mut Vec<u8>, val: u8) {
    buf.push(val);
}

/// Reads a u8 from the slice at offset, advancing by 1.
#[inline]
pub fn read_u8(data: &[u8], offset: &mut usize) -> Result<u8> {
    if *offset >= data.len() {
        return Err(ZyronError::CatalogCorrupted(
            "unexpected end of data reading u8".to_string(),
        ));
    }
    let val = data[*offset];
    *offset += 1;
    Ok(val)
}

/// Writes a u16 in little-endian to the buffer.
#[inline]
pub fn write_u16(buf: &mut Vec<u8>, val: u16) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Reads a u16 from the slice at offset, advancing by 2.
#[inline]
pub fn read_u16(data: &[u8], offset: &mut usize) -> Result<u16> {
    if *offset + 2 > data.len() {
        return Err(ZyronError::CatalogCorrupted(
            "unexpected end of data reading u16".to_string(),
        ));
    }
    let val = u16::from_le_bytes([data[*offset], data[*offset + 1]]);
    *offset += 2;
    Ok(val)
}

/// Writes a u32 in little-endian to the buffer.
#[inline]
pub fn write_u32(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Reads a u32 from the slice at offset, advancing by 4.
#[inline]
pub fn read_u32(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        return Err(ZyronError::CatalogCorrupted(
            "unexpected end of data reading u32".to_string(),
        ));
    }
    let val = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(val)
}

/// Writes a u64 in little-endian to the buffer.
#[inline]
pub fn write_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Reads a u64 from the slice at offset, advancing by 8.
#[inline]
pub fn read_u64(data: &[u8], offset: &mut usize) -> Result<u64> {
    if *offset + 8 > data.len() {
        return Err(ZyronError::CatalogCorrupted(
            "unexpected end of data reading u64".to_string(),
        ));
    }
    let val = u64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(val)
}

/// Writes a bool as a single byte (0 = false, 1 = true).
#[inline]
pub fn write_bool(buf: &mut Vec<u8>, val: bool) {
    buf.push(val as u8);
}

/// Reads a bool from the slice at offset, advancing by 1.
#[inline]
pub fn read_bool(data: &[u8], offset: &mut usize) -> Result<bool> {
    let val = read_u8(data, offset)?;
    Ok(val != 0)
}

/// Writes a u32 length-prefixed UTF-8 string.
#[inline]
pub fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

/// Reads a u32 length-prefixed UTF-8 string from the slice at offset.
#[inline]
pub fn read_string(data: &[u8], offset: &mut usize) -> Result<String> {
    let len = read_u32(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(ZyronError::CatalogCorrupted(format!(
            "string length {len} exceeds remaining data at offset {}",
            *offset
        )));
    }
    let s = std::str::from_utf8(&data[*offset..*offset + len]).map_err(|e| {
        ZyronError::CatalogCorrupted(format!("invalid UTF-8 in catalog string: {e}"))
    })?;
    *offset += len;
    Ok(s.to_string())
}

/// Writes an optional string: 0 tag for None, 1 tag + string for Some.
#[inline]
pub fn write_option_string(buf: &mut Vec<u8>, val: &Option<String>) {
    match val {
        None => write_u8(buf, 0),
        Some(s) => {
            write_u8(buf, 1);
            write_string(buf, s);
        }
    }
}

/// Reads an optional string from the slice at offset.
#[inline]
pub fn read_option_string(data: &[u8], offset: &mut usize) -> Result<Option<String>> {
    let tag = read_u8(data, offset)?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(read_string(data, offset)?)),
        _ => Err(ZyronError::CatalogCorrupted(format!(
            "invalid option tag {tag}, expected 0 or 1"
        ))),
    }
}

/// Writes an optional usize: 0 tag for None, 1 tag + u32 value for Some.
#[inline]
pub fn write_option_usize(buf: &mut Vec<u8>, val: &Option<usize>) {
    match val {
        None => write_u8(buf, 0),
        Some(v) => {
            write_u8(buf, 1);
            write_u32(buf, *v as u32);
        }
    }
}

/// Reads an optional usize from the slice at offset.
#[inline]
pub fn read_option_usize(data: &[u8], offset: &mut usize) -> Result<Option<usize>> {
    let tag = read_u8(data, offset)?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(read_u32(data, offset)? as usize)),
        _ => Err(ZyronError::CatalogCorrupted(format!(
            "invalid option tag {tag}, expected 0 or 1"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u8_roundtrip() {
        let mut buf = Vec::new();
        write_u8(&mut buf, 0);
        write_u8(&mut buf, 255);
        let mut off = 0;
        assert_eq!(read_u8(&buf, &mut off).unwrap(), 0);
        assert_eq!(read_u8(&buf, &mut off).unwrap(), 255);
    }

    #[test]
    fn test_u16_roundtrip() {
        let mut buf = Vec::new();
        write_u16(&mut buf, 0);
        write_u16(&mut buf, 12345);
        write_u16(&mut buf, u16::MAX);
        let mut off = 0;
        assert_eq!(read_u16(&buf, &mut off).unwrap(), 0);
        assert_eq!(read_u16(&buf, &mut off).unwrap(), 12345);
        assert_eq!(read_u16(&buf, &mut off).unwrap(), u16::MAX);
    }

    #[test]
    fn test_u32_roundtrip() {
        let mut buf = Vec::new();
        write_u32(&mut buf, 0);
        write_u32(&mut buf, u32::MAX);
        let mut off = 0;
        assert_eq!(read_u32(&buf, &mut off).unwrap(), 0);
        assert_eq!(read_u32(&buf, &mut off).unwrap(), u32::MAX);
    }

    #[test]
    fn test_u64_roundtrip() {
        let mut buf = Vec::new();
        write_u64(&mut buf, 0);
        write_u64(&mut buf, u64::MAX);
        let mut off = 0;
        assert_eq!(read_u64(&buf, &mut off).unwrap(), 0);
        assert_eq!(read_u64(&buf, &mut off).unwrap(), u64::MAX);
    }

    #[test]
    fn test_bool_roundtrip() {
        let mut buf = Vec::new();
        write_bool(&mut buf, true);
        write_bool(&mut buf, false);
        let mut off = 0;
        assert_eq!(read_bool(&buf, &mut off).unwrap(), true);
        assert_eq!(read_bool(&buf, &mut off).unwrap(), false);
    }

    #[test]
    fn test_string_roundtrip() {
        let mut buf = Vec::new();
        write_string(&mut buf, "hello");
        write_string(&mut buf, "");
        write_string(&mut buf, "unicode: \u{1F600}");
        let mut off = 0;
        assert_eq!(read_string(&buf, &mut off).unwrap(), "hello");
        assert_eq!(read_string(&buf, &mut off).unwrap(), "");
        assert_eq!(read_string(&buf, &mut off).unwrap(), "unicode: \u{1F600}");
    }

    #[test]
    fn test_option_string_roundtrip() {
        let mut buf = Vec::new();
        write_option_string(&mut buf, &None);
        write_option_string(&mut buf, &Some("test".to_string()));
        let mut off = 0;
        assert_eq!(read_option_string(&buf, &mut off).unwrap(), None);
        assert_eq!(
            read_option_string(&buf, &mut off).unwrap(),
            Some("test".to_string())
        );
    }

    #[test]
    fn test_option_usize_roundtrip() {
        let mut buf = Vec::new();
        write_option_usize(&mut buf, &None);
        write_option_usize(&mut buf, &Some(255));
        let mut off = 0;
        assert_eq!(read_option_usize(&buf, &mut off).unwrap(), None);
        assert_eq!(read_option_usize(&buf, &mut off).unwrap(), Some(255));
    }

    #[test]
    fn test_truncated_data_errors() {
        let buf = vec![1u8];
        let mut off = 0;
        assert!(read_u16(&buf, &mut off).is_err());
        let mut off = 0;
        assert!(read_u32(&buf, &mut off).is_err());
        let mut off = 0;
        assert!(read_u64(&buf, &mut off).is_err());
    }

    #[test]
    fn test_string_length_overflow_errors() {
        let mut buf = Vec::new();
        write_u32(&mut buf, 9999);
        let mut off = 0;
        assert!(read_string(&buf, &mut off).is_err());
    }
}
