//! Canonical byte encoding for ARRAY values.
//!
//! An array is stored as one variable-length payload, the same way JSON and
//! geometry are, so no storage path needs to know it is an array. The layout
//! keeps element access O(1) rather than requiring a walk: fixed-width
//! elements are addressed by multiplication, and variable-width elements are
//! addressed through an offset table.
//!
//! Layout:
//! ```text
//!   [0]        element type id
//!   [1]        flags, bit 0 set when elements are fixed width
//!   [2..4]     element width when fixed, zero otherwise (u16, little-endian)
//!   [4..8]     element count (u32, little-endian)
//!   [8..8+b]   presence bitmap, b = ceil(count / 8), a set bit means present
//!   fixed:     count * width payload bytes
//!   varlen:    (count + 1) u32 end offsets, then the payload blob
//! ```
//! A null element occupies its slot in the fixed layout and a zero-length
//! range in the variable layout, so element index and payload position stay
//! in step without a second pass.

use crate::types::TypeId;

/// Set when every element occupies the same number of bytes.
const FLAG_FIXED_WIDTH: u8 = 0x01;

/// Bytes before the presence bitmap.
const HEADER_SIZE: usize = 8;

/// Bytes the presence bitmap occupies for a given element count.
#[inline]
fn bitmap_bytes(count: usize) -> usize {
    count.div_ceil(8)
}

/// Builds the canonical encoding from element payloads in order.
///
/// `elements` holds one entry per element: `None` for a null element,
/// `Some(bytes)` for its payload in the element type's own encoding. A fixed
/// width type is written without an offset table; every other type gets one.
pub fn encode(element_type: TypeId, elements: &[Option<&[u8]>]) -> Vec<u8> {
    encode_from(element_type, elements.len(), |i| elements[i])
}

/// Builds the canonical encoding from elements named as ranges into one
/// buffer.
///
/// A caller that has already written every element of a row end to end into a
/// single buffer names them by range rather than gathering a vector of slices
/// first. That vector is one allocation per row, and a row is the unit these
/// callers work in, so it is one allocation per row of the input.
pub fn encode_spans(
    element_type: TypeId,
    payload: &[u8],
    spans: &[Option<(usize, usize)>],
) -> Vec<u8> {
    encode_from(element_type, spans.len(), |i| {
        spans[i].map(|(from, to)| &payload[from..to])
    })
}

/// The encoder both public forms share, reading each element through a
/// closure so neither has to materialize the elements as a slice of slices.
fn encode_from<'a, F>(element_type: TypeId, count: usize, element_at: F) -> Vec<u8>
where
    F: Fn(usize) -> Option<&'a [u8]>,
{
    let bitmap_len = bitmap_bytes(count);
    let fixed = element_type
        .fixed_size()
        .filter(|w| *w > 0 && *w <= u16::MAX as usize);

    let payload_len: usize = (0..count).filter_map(&element_at).map(|b| b.len()).sum();
    let body_len = match fixed {
        Some(width) => count * width,
        None => (count + 1) * 4 + payload_len,
    };
    let mut out = Vec::with_capacity(HEADER_SIZE + bitmap_len + body_len);

    out.push(element_type as u8);
    out.push(if fixed.is_some() { FLAG_FIXED_WIDTH } else { 0 });
    out.extend_from_slice(&(fixed.unwrap_or(0) as u16).to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());

    let bitmap_at = out.len();
    out.resize(bitmap_at + bitmap_len, 0);
    for i in 0..count {
        if element_at(i).is_some() {
            out[bitmap_at + i / 8] |= 1 << (i % 8);
        }
    }

    match fixed {
        Some(width) => {
            for i in 0..count {
                match element_at(i) {
                    // A null element still occupies its slot, so the payload
                    // stays addressable by index alone
                    None => out.resize(out.len() + width, 0),
                    Some(bytes) if bytes.len() >= width => {
                        out.extend_from_slice(&bytes[..width]);
                    }
                    Some(bytes) => {
                        out.extend_from_slice(bytes);
                        out.resize(out.len() + width - bytes.len(), 0);
                    }
                }
            }
        }
        None => {
            let offsets_at = out.len();
            out.resize(offsets_at + (count + 1) * 4, 0);
            let mut end = 0u32;
            out[offsets_at..offsets_at + 4].copy_from_slice(&end.to_le_bytes());
            for i in 0..count {
                if let Some(bytes) = element_at(i) {
                    end += bytes.len() as u32;
                }
                let at = offsets_at + (i + 1) * 4;
                out[at..at + 4].copy_from_slice(&end.to_le_bytes());
            }
            for i in 0..count {
                if let Some(bytes) = element_at(i) {
                    out.extend_from_slice(bytes);
                }
            }
        }
    }
    out
}

/// A borrowed view over an encoded array. Parsing validates the header and
/// the section extents once, so element access after it is unchecked
/// arithmetic rather than repeated bounds work.
#[derive(Debug, Clone, Copy)]
pub struct ArrayView<'a> {
    element_type: TypeId,
    count: usize,
    /// Element width when fixed, zero when the offset table addresses elements
    width: usize,
    bitmap: &'a [u8],
    /// Offset table for the variable-width layout, empty when fixed
    offsets: &'a [u8],
    payload: &'a [u8],
}

impl<'a> ArrayView<'a> {
    /// Parses an encoded array, or None when the bytes are not one.
    pub fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < HEADER_SIZE {
            return None;
        }
        let element_type = TypeId::from_u8(bytes[0])?;
        let fixed = bytes[1] & FLAG_FIXED_WIDTH != 0;
        let width = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let bitmap_len = bitmap_bytes(count);
        let bitmap_end = HEADER_SIZE + bitmap_len;
        if bytes.len() < bitmap_end {
            return None;
        }
        let bitmap = &bytes[HEADER_SIZE..bitmap_end];
        if fixed {
            let payload_end = bitmap_end.checked_add(count.checked_mul(width)?)?;
            if width == 0 || bytes.len() < payload_end {
                return None;
            }
            return Some(Self {
                element_type,
                count,
                width,
                bitmap,
                offsets: &[],
                payload: &bytes[bitmap_end..payload_end],
            });
        }
        let offsets_end = bitmap_end.checked_add(count.checked_add(1)?.checked_mul(4)?)?;
        if bytes.len() < offsets_end {
            return None;
        }
        Some(Self {
            element_type,
            count,
            width: 0,
            bitmap,
            offsets: &bytes[bitmap_end..offsets_end],
            payload: &bytes[offsets_end..],
        })
    }

    pub fn element_type(&self) -> TypeId {
        self.element_type
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The payload of one element by zero-based index. The outer None means
    /// the index is past the end, the inner None means the element is null.
    pub fn get(&self, index: usize) -> Option<Option<&'a [u8]>> {
        if index >= self.count {
            return None;
        }
        if self.bitmap[index / 8] & (1 << (index % 8)) == 0 {
            return Some(None);
        }
        if self.width > 0 {
            let at = index * self.width;
            return Some(Some(&self.payload[at..at + self.width]));
        }
        let read = |slot: usize| -> u32 {
            let at = slot * 4;
            u32::from_le_bytes([
                self.offsets[at],
                self.offsets[at + 1],
                self.offsets[at + 2],
                self.offsets[at + 3],
            ])
        };
        let start = read(index) as usize;
        let end = read(index + 1) as usize;
        if end > self.payload.len() || start > end {
            return Some(None);
        }
        Some(Some(&self.payload[start..end]))
    }

    /// Every element in order.
    pub fn iter(&self) -> impl Iterator<Item = Option<&'a [u8]>> + '_ {
        (0..self.count).map(move |i| self.get(i).unwrap_or(None))
    }

    /// The braced form an array reads and writes as, `{1,2,NULL}`. A text
    /// element is quoted when it holds a character that would otherwise
    /// change how the list parses.
    pub fn render_text(&self) -> String {
        let mut out = String::with_capacity(2 + self.count * 8);
        out.push('{');
        for (i, element) in self.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            match element {
                None => out.push_str("NULL"),
                Some(bytes) => render_element(self.element_type, bytes, &mut out),
            }
        }
        out.push('}');
        out
    }
}

/// Appends one element's text form. Byte-backed types that have no textual
/// spelling are rendered as lowercase hex so the output is still exact.
fn render_element(element_type: TypeId, bytes: &[u8], out: &mut String) {
    use std::fmt::Write as _;
    let int = |n: usize| -> i128 {
        let mut buf = [0u8; 16];
        let take = n.min(bytes.len());
        buf[..take].copy_from_slice(&bytes[..take]);
        let raw = i128::from_le_bytes(buf);
        // Sign-extend from the element's own width
        let shift = 128 - (n * 8) as u32;
        if shift == 0 {
            raw
        } else {
            (raw << shift) >> shift
        }
    };
    let uint = |n: usize| -> u128 {
        let mut buf = [0u8; 16];
        let take = n.min(bytes.len());
        buf[..take].copy_from_slice(&bytes[..take]);
        u128::from_le_bytes(buf)
    };
    match element_type {
        TypeId::Boolean => out.push_str(if bytes.first() == Some(&0) {
            "false"
        } else {
            "true"
        }),
        TypeId::Int8 => {
            let _ = write!(out, "{}", int(1));
        }
        TypeId::Int16 => {
            let _ = write!(out, "{}", int(2));
        }
        TypeId::Int32 | TypeId::Date => {
            let _ = write!(out, "{}", int(4));
        }
        TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
            let _ = write!(out, "{}", int(8));
        }
        TypeId::Int128 | TypeId::Decimal | TypeId::Hlc => {
            let _ = write!(out, "{}", int(16));
        }
        TypeId::UInt8 => {
            let _ = write!(out, "{}", uint(1));
        }
        TypeId::UInt16 => {
            let _ = write!(out, "{}", uint(2));
        }
        TypeId::UInt32 => {
            let _ = write!(out, "{}", uint(4));
        }
        TypeId::UInt64 => {
            let _ = write!(out, "{}", uint(8));
        }
        TypeId::UInt128 => {
            let _ = write!(out, "{}", uint(16));
        }
        TypeId::Float32 => {
            let mut buf = [0u8; 4];
            let take = 4.min(bytes.len());
            buf[..take].copy_from_slice(&bytes[..take]);
            let _ = write!(out, "{}", f32::from_le_bytes(buf));
        }
        TypeId::Float64 => {
            let mut buf = [0u8; 8];
            let take = 8.min(bytes.len());
            buf[..take].copy_from_slice(&bytes[..take]);
            let _ = write!(out, "{}", f64::from_le_bytes(buf));
        }
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            let text = String::from_utf8_lossy(bytes);
            let needs_quotes = text.is_empty()
                || text.eq_ignore_ascii_case("null")
                || text
                    .chars()
                    .any(|c| matches!(c, ',' | '{' | '}' | '"' | '\\') || c.is_whitespace());
            if needs_quotes {
                out.push('"');
                for c in text.chars() {
                    if c == '"' || c == '\\' {
                        out.push('\\');
                    }
                    out.push(c);
                }
                out.push('"');
            } else {
                out.push_str(&text);
            }
        }
        _ => {
            for b in bytes {
                let _ = write!(out, "{b:02x}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_width_elements_round_trip_with_nulls() {
        let a = 7i64.to_le_bytes();
        let b = (-3i64).to_le_bytes();
        let encoded = encode(TypeId::Int64, &[Some(&a), None, Some(&b)]);
        let view = ArrayView::parse(&encoded).expect("parses");
        assert_eq!(view.element_type(), TypeId::Int64);
        assert_eq!(view.len(), 3);
        assert_eq!(view.get(0), Some(Some(&a[..])));
        assert_eq!(view.get(1), Some(None), "the null element keeps its slot");
        assert_eq!(view.get(2), Some(Some(&b[..])));
        assert_eq!(view.get(3), None, "past the end reads as absent");
    }

    #[test]
    fn test_variable_width_elements_round_trip_with_nulls() {
        let encoded = encode(
            TypeId::Text,
            &[Some(b"alpha"), None, Some(b""), Some(b"omega")],
        );
        let view = ArrayView::parse(&encoded).expect("parses");
        assert_eq!(view.element_type(), TypeId::Text);
        assert_eq!(view.len(), 4);
        assert_eq!(view.get(0), Some(Some(&b"alpha"[..])));
        assert_eq!(view.get(1), Some(None));
        assert_eq!(
            view.get(2),
            Some(Some(&b""[..])),
            "an empty element is not a null element"
        );
        assert_eq!(view.get(3), Some(Some(&b"omega"[..])));
    }

    /// The bytes an array occupies on disk, pinned.
    ///
    /// Every array in a heap page, a lake file and a changeset record is in
    /// this layout, and nothing reads it through a version. A round trip
    /// proves the encoder and the reader agree with each other, which they
    /// would go on doing if both moved together; only the bytes themselves
    /// prove that data already written still reads.
    #[test]
    fn the_encoded_layout_is_the_one_already_on_disk() {
        let encoded = encode(TypeId::Int32, &[Some(&1i32.to_le_bytes()), None]);
        assert_eq!(
            encoded,
            vec![
                // element type, then the fixed-width flag
                TypeId::Int32 as u8,
                FLAG_FIXED_WIDTH,
                // element width, little endian u16
                4,
                0,
                // element count, little endian u32
                2,
                0,
                0,
                0,
                // presence bitmap: the first element is set, the second is not
                0b0000_0001,
                // the first element's four bytes, then the null element's
                // slot, which is held open so an index still addresses it
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        );

        let text = encode(TypeId::Text, &[Some(b"ab"), None]);
        assert_eq!(
            text,
            vec![
                TypeId::Text as u8,
                // no fixed-width flag, so an offset table addresses elements
                0,
                // width zero for a variable-width element
                0,
                0,
                2,
                0,
                0,
                0,
                0b0000_0001,
                // offsets, one per element plus a final end, little endian u32
                0,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                // the payload, with nothing written for the null element
                b'a',
                b'b',
            ]
        );
    }

    /// The span form writes the same bytes as the slice form.
    ///
    /// Both are the same encoder reached two ways, and a caller picks between
    /// them on how it happens to hold its elements, never on what it wants
    /// written.
    #[test]
    fn spans_and_slices_encode_alike() {
        let payload = b"alphaomega";
        let spans = [Some((0usize, 5usize)), None, Some((5, 10))];
        assert_eq!(
            encode_spans(TypeId::Text, payload, &spans),
            encode(TypeId::Text, &[Some(b"alpha"), None, Some(b"omega")])
        );
        assert_eq!(
            encode_spans(TypeId::Int64, &[], &[]),
            encode(TypeId::Int64, &[])
        );
    }

    #[test]
    fn test_empty_array_round_trips() {
        let encoded = encode(TypeId::Int32, &[]);
        let view = ArrayView::parse(&encoded).expect("parses");
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.get(0), None);
    }

    #[test]
    fn test_render_text_spells_elements_and_quotes_only_when_needed() {
        let one = 1i32.to_le_bytes();
        let two = 2i32.to_le_bytes();
        let ints = encode(TypeId::Int32, &[Some(&one), None, Some(&two)]);
        assert_eq!(ArrayView::parse(&ints).unwrap().render_text(), "{1,NULL,2}");

        let floats = encode(
            TypeId::Float64,
            &[Some(&1.5f64.to_le_bytes()), Some(&(-0.25f64).to_le_bytes())],
        );
        assert_eq!(
            ArrayView::parse(&floats).unwrap().render_text(),
            "{1.5,-0.25}"
        );

        // A comma, a brace, a quote, whitespace, the empty string and the
        // word null all have to survive a round trip through the braces
        let texts = encode(
            TypeId::Text,
            &[
                Some(b"plain"),
                Some(b"has,comma"),
                Some(b"has\"quote"),
                Some(b""),
                Some(b"null"),
            ],
        );
        assert_eq!(
            ArrayView::parse(&texts).unwrap().render_text(),
            "{plain,\"has,comma\",\"has\\\"quote\",\"\",\"null\"}"
        );

        assert_eq!(
            ArrayView::parse(&encode(TypeId::Int32, &[]))
                .unwrap()
                .render_text(),
            "{}"
        );
    }

    #[test]
    fn test_truncated_bytes_are_refused_rather_than_read_past() {
        let a = 1i32.to_le_bytes();
        let encoded = encode(TypeId::Int32, &[Some(&a), Some(&a)]);
        for cut in 0..encoded.len() {
            // A short blob must fail to parse or answer only within itself,
            // never index outside the slice it was given
            if let Some(view) = ArrayView::parse(&encoded[..cut]) {
                for i in 0..view.len() {
                    let _ = view.get(i);
                }
            }
        }
    }
}
