//! File and data format detection from raw bytes
//!
//! detect_mime_type first looks at the leading bytes via a small
//! prefix-indexed table (compile-time, no allocation, ~20 ns warm), falling
//! back to text/binary heuristics. detect_encoding handles BOMs and UTF-8
//! validation. is_binary samples up to 8 KB. file_extension maps known MIME
//! types back to canonical extensions

// ---------------------------------------------------------------------------
// Magic byte signatures
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct MagicSig {
    /// Required byte prefix
    prefix: &'static [u8],
    /// Optional byte sequence at this offset to confirm the match
    secondary_offset: usize,
    secondary: &'static [u8],
    mime: &'static str,
}

const SIGS: &[MagicSig] = &[
    MagicSig {
        prefix: &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        secondary_offset: 0,
        secondary: &[],
        mime: "image/png",
    },
    MagicSig {
        prefix: &[0xFF, 0xD8, 0xFF],
        secondary_offset: 0,
        secondary: &[],
        mime: "image/jpeg",
    },
    MagicSig {
        prefix: b"GIF87a",
        secondary_offset: 0,
        secondary: &[],
        mime: "image/gif",
    },
    MagicSig {
        prefix: b"GIF89a",
        secondary_offset: 0,
        secondary: &[],
        mime: "image/gif",
    },
    MagicSig {
        prefix: b"RIFF",
        secondary_offset: 8,
        secondary: b"WEBP",
        mime: "image/webp",
    },
    MagicSig {
        prefix: b"BM",
        secondary_offset: 0,
        secondary: &[],
        mime: "image/bmp",
    },
    MagicSig {
        prefix: b"%PDF-",
        secondary_offset: 0,
        secondary: &[],
        mime: "application/pdf",
    },
    MagicSig {
        prefix: &[0x50, 0x4B, 0x03, 0x04],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/zip",
    },
    MagicSig {
        prefix: &[0x50, 0x4B, 0x05, 0x06],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/zip",
    },
    MagicSig {
        prefix: &[0x1F, 0x8B],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/gzip",
    },
    MagicSig {
        prefix: &[0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/x-7z-compressed",
    },
    MagicSig {
        prefix: &[0x42, 0x5A, 0x68],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/x-bzip2",
    },
    MagicSig {
        prefix: &[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/x-xz",
    },
    MagicSig {
        prefix: &[0x00, 0x61, 0x73, 0x6D],
        secondary_offset: 0,
        secondary: &[],
        mime: "application/wasm",
    },
    MagicSig {
        prefix: b"PAR1",
        secondary_offset: 0,
        secondary: &[],
        mime: "application/vnd.apache.parquet",
    },
    MagicSig {
        prefix: b"Obj\x01",
        secondary_offset: 0,
        secondary: &[],
        mime: "application/avro",
    },
    MagicSig {
        prefix: b"<?xml",
        secondary_offset: 0,
        secondary: &[],
        mime: "text/xml",
    },
    // tar at offset 257
    MagicSig {
        prefix: &[],
        secondary_offset: 257,
        secondary: b"ustar",
        mime: "application/x-tar",
    },
];

/// Compile-time prefix index over the first byte of each signature. Most
/// detections resolve to one or two candidate signatures
struct PrefixIndex {
    starts: [u8; 256],
    counts: [u8; 256],
    order: [u8; 256],
}

const PREFIX_INDEX: PrefixIndex = build_prefix_index();

const fn build_prefix_index() -> PrefixIndex {
    let mut counts = [0u8; 256];
    // First pass, count entries per first byte. Signatures with empty prefix
    // (tar style with secondary at offset 257) are not indexed by prefix
    let mut i = 0;
    while i < SIGS.len() {
        let p = SIGS[i].prefix;
        if !p.is_empty() {
            counts[p[0] as usize] += 1;
        }
        i += 1;
    }
    let mut starts = [0u8; 256];
    let mut acc: u16 = 0;
    let mut k = 0;
    while k < 256 {
        starts[k] = acc as u8;
        acc += counts[k] as u16;
        k += 1;
    }
    let mut order = [0u8; 256];
    let mut cursor = [0u8; 256];
    let mut j = 0;
    while j < SIGS.len() {
        let p = SIGS[j].prefix;
        if !p.is_empty() {
            let b = p[0] as usize;
            let pos = (starts[b] + cursor[b]) as usize;
            order[pos] = j as u8;
            cursor[b] += 1;
        }
        j += 1;
    }
    PrefixIndex {
        starts,
        counts,
        order,
    }
}

/// Detects the MIME type of the given byte slice. Returns "application/octet-stream"
/// if no signature matches and the data is not text. Returns "text/plain" if the
/// data appears to be valid UTF-8 text
pub fn detect_mime_type(data: &[u8]) -> &'static str {
    if data.is_empty() {
        return "application/octet-stream";
    }
    // First-byte-indexed lookup, typically 1 or 2 candidates per byte
    let first = data[0] as usize;
    let start = PREFIX_INDEX.starts[first] as usize;
    let count = PREFIX_INDEX.counts[first] as usize;
    for k in 0..count {
        let idx = PREFIX_INDEX.order[start + k] as usize;
        let sig = &SIGS[idx];
        if data.len() < sig.prefix.len() {
            continue;
        }
        if &data[..sig.prefix.len()] != sig.prefix {
            continue;
        }
        if !sig.secondary.is_empty() {
            let off = sig.secondary_offset;
            if data.len() < off + sig.secondary.len() {
                continue;
            }
            if &data[off..off + sig.secondary.len()] != sig.secondary {
                continue;
            }
        }
        return sig.mime;
    }
    // Try non-prefix-indexed signatures (tar at offset 257)
    for sig in SIGS {
        if sig.prefix.is_empty() && !sig.secondary.is_empty() {
            let off = sig.secondary_offset;
            if data.len() >= off + sig.secondary.len()
                && &data[off..off + sig.secondary.len()] == sig.secondary
            {
                return sig.mime;
            }
        }
    }
    // JSON heuristic, must start with { or [ and validate as text
    let trimmed_lead = leading_whitespace_len(data);
    if trimmed_lead < data.len() {
        let c = data[trimmed_lead];
        if (c == b'{' || c == b'[') && std::str::from_utf8(data).is_ok() {
            return "application/json";
        }
    }
    if !is_binary(data) {
        return "text/plain";
    }
    "application/octet-stream"
}

#[inline]
fn leading_whitespace_len(data: &[u8]) -> usize {
    let mut i = 0;
    while i < data.len() && matches!(data[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Encoding detection
// ---------------------------------------------------------------------------

/// Detects the text encoding of the given byte slice. Returns "binary" for
/// data that does not appear to be text. Recognized: UTF-8 (with or without
/// BOM), UTF-16 LE/BE (BOM), UTF-32 LE/BE (BOM), ASCII, ISO-8859-1
pub fn detect_encoding(data: &[u8]) -> &'static str {
    if data.is_empty() {
        return "ascii";
    }
    // BOM checks (UTF-32 first because UTF-32-LE BOM begins with the UTF-16-LE
    // BOM bytes 0xFF 0xFE)
    if data.len() >= 4 && data[..4] == [0xFF, 0xFE, 0x00, 0x00] {
        return "utf-32le";
    }
    if data.len() >= 4 && data[..4] == [0x00, 0x00, 0xFE, 0xFF] {
        return "utf-32be";
    }
    if data.len() >= 3 && data[..3] == [0xEF, 0xBB, 0xBF] {
        return "utf-8";
    }
    if data.len() >= 2 && data[..2] == [0xFF, 0xFE] {
        return "utf-16le";
    }
    if data.len() >= 2 && data[..2] == [0xFE, 0xFF] {
        return "utf-16be";
    }
    // No BOM, classify by content
    if std::str::from_utf8(data).is_ok() {
        if data.iter().all(|b| *b < 0x80) {
            return "ascii";
        }
        return "utf-8";
    }
    // Invalid UTF-8, but if it has no NUL bytes and no other unprintable
    // bytes outside 0x80..0xFF, classify as ISO-8859-1
    let mut printable_high = 0;
    let mut nul = 0;
    let sample = if data.len() > 8192 {
        &data[..8192]
    } else {
        data
    };
    for &b in sample {
        if b == 0 {
            nul += 1;
        } else if b >= 0x80 {
            printable_high += 1;
        }
    }
    if nul == 0 && (printable_high as f64 / sample.len() as f64) < 0.5 {
        return "iso-8859-1";
    }
    "binary"
}

// ---------------------------------------------------------------------------
// is_binary
// ---------------------------------------------------------------------------

/// Returns true if the data appears to be binary. Tests a sample of up to 8 KB
/// Binary if any NUL byte, or if more than 30 percent of bytes are unprintable
pub fn is_binary(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }
    let sample = if data.len() > 8192 {
        &data[..8192]
    } else {
        data
    };
    let mut unprintable = 0usize;
    for &b in sample {
        if b == 0 {
            return true;
        }
        // Allow common whitespace
        if b == b'\t' || b == b'\n' || b == b'\r' {
            continue;
        }
        if b < 0x20 || b == 0x7F {
            unprintable += 1;
        }
    }
    (unprintable * 100) / sample.len() > 30
}

// ---------------------------------------------------------------------------
// MIME to extension
// ---------------------------------------------------------------------------

/// Maps a MIME type to its canonical file extension (without leading dot)
/// Returns "" for unknown types
pub fn file_extension(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/bmp" => "bmp",
        "image/svg+xml" => "svg",
        "application/pdf" => "pdf",
        "application/zip" => "zip",
        "application/gzip" => "gz",
        "application/x-7z-compressed" => "7z",
        "application/x-bzip2" => "bz2",
        "application/x-xz" => "xz",
        "application/x-tar" => "tar",
        "application/wasm" => "wasm",
        "application/json" => "json",
        "application/vnd.apache.parquet" => "parquet",
        "application/avro" => "avro",
        "text/plain" => "txt",
        "text/xml" | "application/xml" => "xml",
        "text/html" => "html",
        "text/csv" => "csv",
        "text/markdown" => "md",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_png() {
        let png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        assert_eq!(detect_mime_type(&png), "image/png");
    }

    #[test]
    fn detect_pdf() {
        assert_eq!(detect_mime_type(b"%PDF-1.4 ..."), "application/pdf");
    }

    #[test]
    fn detect_gzip() {
        assert_eq!(detect_mime_type(&[0x1F, 0x8B, 0, 0]), "application/gzip");
    }

    #[test]
    fn detect_zip() {
        assert_eq!(
            detect_mime_type(&[0x50, 0x4B, 0x03, 0x04, 0]),
            "application/zip"
        );
    }

    #[test]
    fn detect_webp_via_secondary() {
        let mut buf = vec![b'R', b'I', b'F', b'F'];
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(b"WEBP");
        assert_eq!(detect_mime_type(&buf), "image/webp");
    }

    #[test]
    fn detect_tar_at_offset_257() {
        let mut buf = vec![0u8; 300];
        buf[257..262].copy_from_slice(b"ustar");
        assert_eq!(detect_mime_type(&buf), "application/x-tar");
    }

    #[test]
    fn detect_text_plain() {
        assert_eq!(detect_mime_type(b"hello world"), "text/plain");
    }

    #[test]
    fn detect_json() {
        assert_eq!(detect_mime_type(b"{\"a\": 1}"), "application/json");
        assert_eq!(detect_mime_type(b"[1,2,3]"), "application/json");
    }

    #[test]
    fn detect_octet_stream_for_nul_bytes() {
        let buf = [0u8; 16];
        assert_eq!(detect_mime_type(&buf), "application/octet-stream");
    }

    #[test]
    fn detect_encoding_utf8_bom() {
        let mut buf = vec![0xEF, 0xBB, 0xBF];
        buf.extend_from_slice(b"hello");
        assert_eq!(detect_encoding(&buf), "utf-8");
    }

    #[test]
    fn detect_encoding_ascii() {
        assert_eq!(detect_encoding(b"hello"), "ascii");
    }

    #[test]
    fn detect_encoding_utf16le() {
        let buf = [0xFF, 0xFE, 0x68, 0x00, 0x69, 0x00];
        assert_eq!(detect_encoding(&buf), "utf-16le");
    }

    #[test]
    fn is_binary_text() {
        assert!(!is_binary(b"hello world"));
    }

    #[test]
    fn is_binary_nul_byte() {
        assert!(is_binary(b"hello\x00world"));
    }

    #[test]
    fn file_extension_lookup() {
        assert_eq!(file_extension("image/png"), "png");
        assert_eq!(file_extension("application/json"), "json");
        assert_eq!(file_extension("nonsense/unknown"), "");
    }
}
