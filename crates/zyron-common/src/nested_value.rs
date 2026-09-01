//! Canonical byte encodings for STRUCT and MAP values.
//!
//! Both are stored as one variable-length payload, the same way an ARRAY is,
//! so no storage path needs to know which one it holds.
//!
//! A STRUCT declares its fields, so the encoding stores none of their names
//! and addresses them by position. A three-field struct costs an offset table
//! rather than the field names repeated on every row, and reaching one field
//! is two loads instead of a scan over the whole value.
//!
//! A MAP declares only its key and value types, so the keys travel with the
//! value. They are held sorted by their encoded bytes, which makes a lookup a
//! binary search rather than a walk.
//!
//! STRUCT layout:
//! ```text
//!   [0]        format tag, STRUCT_TAG
//!   [1]        flags, reserved, zero
//!   [2..4]     field count (u16, little-endian)
//!   [4..4+b]   presence bitmap, b = ceil(count / 8), a set bit means present
//!              (count + 1) u32 end offsets
//!              payload blob
//! ```
//!
//! MAP layout:
//! ```text
//!   [0]        format tag, MAP_TAG
//!   [1]        flags, reserved, zero
//!   [2..4]     reserved, zero
//!   [4..8]     entry count (u32, little-endian)
//!   [8..8+b]   value presence bitmap, b = ceil(count / 8)
//!              (count + 1) u32 key end offsets
//!              (count + 1) u32 value end offsets
//!              key blob
//!              value blob
//! ```
//!
//! A null field or value occupies a zero-length range, so position and
//! payload stay in step without a second pass. A key is never null, which is
//! why only the values carry a bitmap.

/// Leading byte of an encoded STRUCT.
pub const STRUCT_TAG: u8 = 1;

/// Leading byte of an encoded MAP.
pub const MAP_TAG: u8 = 2;

/// Bytes before the presence bitmap in a STRUCT.
const STRUCT_HEADER_SIZE: usize = 4;

/// Bytes before the presence bitmap in a MAP.
const MAP_HEADER_SIZE: usize = 8;

/// Bytes the presence bitmap occupies for a given count.
#[inline]
fn bitmap_bytes(count: usize) -> usize {
    count.div_ceil(8)
}

/// Reads a little-endian u32 out of an offset table.
#[inline]
fn offset_at(table: &[u8], index: usize) -> usize {
    let at = index * 4;
    u32::from_le_bytes([table[at], table[at + 1], table[at + 2], table[at + 3]]) as usize
}

/// Writes `(count + 1)` end offsets for `items` and returns the payload total.
fn write_offsets(out: &mut Vec<u8>, items: impl Iterator<Item = usize>, count: usize) -> usize {
    let offsets_at = out.len();
    out.resize(offsets_at + (count + 1) * 4, 0);
    let mut end = 0u32;
    out[offsets_at..offsets_at + 4].copy_from_slice(&end.to_le_bytes());
    for (i, len) in items.enumerate() {
        end += len as u32;
        let at = offsets_at + (i + 1) * 4;
        out[at..at + 4].copy_from_slice(&end.to_le_bytes());
    }
    end as usize
}

// ---------------------------------------------------------------------------
// STRUCT
// ---------------------------------------------------------------------------

/// Builds the canonical encoding of a struct from its fields in declared
/// order.
///
/// `fields` holds one entry per declared field: `None` for a null field,
/// `Some(bytes)` for its payload in that field's own encoding. The order is
/// the declaration's, which is what lets a read address a field by position.
pub fn encode_struct(fields: &[Option<&[u8]>]) -> Vec<u8> {
    let count = fields.len();
    let bitmap_len = bitmap_bytes(count);
    let payload_len: usize = fields.iter().flatten().map(|b| b.len()).sum();
    let mut out =
        Vec::with_capacity(STRUCT_HEADER_SIZE + bitmap_len + (count + 1) * 4 + payload_len);

    out.push(STRUCT_TAG);
    out.push(0);
    out.extend_from_slice(&(count as u16).to_le_bytes());

    let bitmap_at = out.len();
    out.resize(bitmap_at + bitmap_len, 0);
    for (i, field) in fields.iter().enumerate() {
        if field.is_some() {
            out[bitmap_at + i / 8] |= 1 << (i % 8);
        }
    }

    write_offsets(
        &mut out,
        fields.iter().map(|f| f.map(|b| b.len()).unwrap_or(0)),
        count,
    );
    for field in fields.iter().flatten() {
        out.extend_from_slice(field);
    }
    out
}

/// A borrowed view over an encoded struct. Parsing validates the header and
/// the section extents once, so field access after it is arithmetic rather
/// than repeated bounds work.
#[derive(Debug, Clone, Copy)]
pub struct StructView<'a> {
    count: usize,
    bitmap: &'a [u8],
    offsets: &'a [u8],
    payload: &'a [u8],
}

impl<'a> StructView<'a> {
    /// Parses an encoded struct, or None when the bytes are not one.
    pub fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < STRUCT_HEADER_SIZE || bytes[0] != STRUCT_TAG {
            return None;
        }
        let count = u16::from_le_bytes([bytes[2], bytes[3]]) as usize;
        let bitmap_end = STRUCT_HEADER_SIZE + bitmap_bytes(count);
        let offsets_end = bitmap_end.checked_add((count + 1).checked_mul(4)?)?;
        if bytes.len() < offsets_end {
            return None;
        }
        let offsets = &bytes[bitmap_end..offsets_end];
        let payload = &bytes[offsets_end..];
        // The last end offset is the payload length, so a truncated value is
        // refused here rather than read past
        if offset_at(offsets, count) > payload.len() {
            return None;
        }
        Some(Self {
            count,
            bitmap: &bytes[STRUCT_HEADER_SIZE..bitmap_end],
            offsets,
            payload,
        })
    }

    /// How many fields the value carries.
    pub fn field_count(&self) -> usize {
        self.count
    }

    /// Whether the field at `index` is null. A field past the end reads null,
    /// which is what a value written before a field was declared holds.
    pub fn is_null(&self, index: usize) -> bool {
        if index >= self.count {
            return true;
        }
        self.bitmap[index / 8] & (1 << (index % 8)) == 0
    }

    /// The bytes of the field at `index`, or None when it is null or past the
    /// end of the value.
    pub fn field(&self, index: usize) -> Option<&'a [u8]> {
        if self.is_null(index) {
            return None;
        }
        let start = offset_at(self.offsets, index);
        let end = offset_at(self.offsets, index + 1);
        self.payload.get(start..end)
    }
}

// ---------------------------------------------------------------------------
// MAP
// ---------------------------------------------------------------------------

/// Builds the canonical encoding of a map.
///
/// `entries` holds `(key, value)` pairs, the value `None` when it is null.
/// They are sorted by key bytes here rather than trusted from the caller, so
/// two writes of the same map produce the same bytes and a lookup can binary
/// search. A duplicate key keeps the last entry given for it, matching what
/// an object literal means everywhere else.
pub fn encode_map(entries: &[(&[u8], Option<&[u8]>)]) -> Vec<u8> {
    let mut ordered: Vec<(&[u8], Option<&[u8]>)> = entries.to_vec();
    ordered.sort_by(|a, b| a.0.cmp(b.0));
    ordered.dedup_by(|later, earlier| {
        if later.0 == earlier.0 {
            // dedup_by keeps the earlier element, so the later value moves
            // onto it before the later one is dropped
            earlier.1 = later.1;
            true
        } else {
            false
        }
    });

    let count = ordered.len();
    let bitmap_len = bitmap_bytes(count);
    let key_len: usize = ordered.iter().map(|(k, _)| k.len()).sum();
    let value_len: usize = ordered
        .iter()
        .filter_map(|(_, v)| *v)
        .map(|v| v.len())
        .sum();
    let mut out =
        Vec::with_capacity(MAP_HEADER_SIZE + bitmap_len + (count + 1) * 8 + key_len + value_len);

    out.push(MAP_TAG);
    out.push(0);
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());

    let bitmap_at = out.len();
    out.resize(bitmap_at + bitmap_len, 0);
    for (i, (_, value)) in ordered.iter().enumerate() {
        if value.is_some() {
            out[bitmap_at + i / 8] |= 1 << (i % 8);
        }
    }

    write_offsets(&mut out, ordered.iter().map(|(k, _)| k.len()), count);
    write_offsets(
        &mut out,
        ordered.iter().map(|(_, v)| v.map(|b| b.len()).unwrap_or(0)),
        count,
    );
    for (key, _) in &ordered {
        out.extend_from_slice(key);
    }
    for value in ordered.iter().filter_map(|(_, v)| *v) {
        out.extend_from_slice(value);
    }
    out
}

/// A borrowed view over an encoded map, with entries in key order.
#[derive(Debug, Clone, Copy)]
pub struct MapView<'a> {
    count: usize,
    bitmap: &'a [u8],
    key_offsets: &'a [u8],
    value_offsets: &'a [u8],
    keys: &'a [u8],
    values: &'a [u8],
}

impl<'a> MapView<'a> {
    /// Parses an encoded map, or None when the bytes are not one.
    pub fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < MAP_HEADER_SIZE || bytes[0] != MAP_TAG {
            return None;
        }
        let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let bitmap_end = MAP_HEADER_SIZE.checked_add(bitmap_bytes(count))?;
        let table_len = (count + 1).checked_mul(4)?;
        let key_offsets_end = bitmap_end.checked_add(table_len)?;
        let value_offsets_end = key_offsets_end.checked_add(table_len)?;
        if bytes.len() < value_offsets_end {
            return None;
        }
        let key_offsets = &bytes[bitmap_end..key_offsets_end];
        let value_offsets = &bytes[key_offsets_end..value_offsets_end];
        let keys_len = offset_at(key_offsets, count);
        let keys_end = value_offsets_end.checked_add(keys_len)?;
        let values_len = offset_at(value_offsets, count);
        let values_end = keys_end.checked_add(values_len)?;
        if bytes.len() < values_end {
            return None;
        }
        Some(Self {
            count,
            bitmap: &bytes[MAP_HEADER_SIZE..bitmap_end],
            key_offsets,
            value_offsets,
            keys: &bytes[value_offsets_end..keys_end],
            values: &bytes[keys_end..values_end],
        })
    }

    /// How many entries the map holds.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the map holds no entries.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The key of the entry at `index`, in key order.
    pub fn key(&self, index: usize) -> Option<&'a [u8]> {
        if index >= self.count {
            return None;
        }
        let start = offset_at(self.key_offsets, index);
        let end = offset_at(self.key_offsets, index + 1);
        self.keys.get(start..end)
    }

    /// The value of the entry at `index`, None when that value is null.
    pub fn value(&self, index: usize) -> Option<&'a [u8]> {
        if index >= self.count || self.bitmap[index / 8] & (1 << (index % 8)) == 0 {
            return None;
        }
        let start = offset_at(self.value_offsets, index);
        let end = offset_at(self.value_offsets, index + 1);
        self.values.get(start..end)
    }

    /// Finds `key`, returning the entry's value. The outer None says the key
    /// is absent, the inner None says it is present with a null value, which
    /// are different answers to a lookup.
    pub fn lookup(&self, key: &[u8]) -> Option<Option<&'a [u8]>> {
        let mut lo = 0usize;
        let mut hi = self.count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let at = self.key(mid)?;
            match at.cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(self.value(mid)),
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_struct_reads_back_every_field_by_position() {
        let name = b"ada".as_slice();
        let age = 36i32.to_le_bytes();
        let encoded = encode_struct(&[Some(name), Some(&age), None]);

        let view = StructView::parse(&encoded).expect("parses");
        assert_eq!(view.field_count(), 3);
        assert_eq!(view.field(0), Some(name));
        assert_eq!(view.field(1), Some(age.as_slice()));
        assert_eq!(view.field(2), None, "a null field reads as absent");
        assert!(view.is_null(2));
        assert!(!view.is_null(0));
    }

    #[test]
    fn a_struct_stores_no_field_names() {
        // The declaration carries the names, so the value carries only the
        // bytes, a bitmap and an offset table
        let encoded = encode_struct(&[Some(b"x".as_slice()), Some(b"y".as_slice())]);
        assert_eq!(
            encoded.len(),
            STRUCT_HEADER_SIZE + 1 + 3 * 4 + 2,
            "a two field struct costs its header, bitmap, offsets and payload"
        );
    }

    #[test]
    fn a_field_past_the_end_reads_null() {
        // A value written before a field was declared has fewer fields than
        // the declaration, and the field it never held is null rather than an
        // error
        let encoded = encode_struct(&[Some(b"only".as_slice())]);
        let view = StructView::parse(&encoded).expect("parses");
        assert_eq!(view.field(0), Some(b"only".as_slice()));
        assert!(view.is_null(1));
        assert_eq!(view.field(1), None);
    }

    #[test]
    fn an_empty_struct_round_trips() {
        let encoded = encode_struct(&[]);
        let view = StructView::parse(&encoded).expect("parses");
        assert_eq!(view.field_count(), 0);
        assert_eq!(view.field(0), None);
    }

    #[test]
    fn a_struct_holds_a_nested_struct_as_one_field() {
        let inner = encode_struct(&[Some(b"leeds".as_slice())]);
        let outer = encode_struct(&[Some(b"ada".as_slice()), Some(&inner)]);

        let view = StructView::parse(&outer).expect("parses");
        let nested = StructView::parse(view.field(1).expect("the nested field")).expect("nested");
        assert_eq!(nested.field(0), Some(b"leeds".as_slice()));
    }

    #[test]
    fn a_map_finds_its_entries_by_key() {
        let entries: Vec<(&[u8], Option<&[u8]>)> = vec![
            (b"zeta", Some(b"26".as_slice())),
            (b"alpha", Some(b"1".as_slice())),
            (b"mid", None),
        ];
        let encoded = encode_map(&entries);
        let view = MapView::parse(&encoded).expect("parses");

        assert_eq!(view.len(), 3);
        assert_eq!(view.lookup(b"alpha"), Some(Some(b"1".as_slice())));
        assert_eq!(view.lookup(b"zeta"), Some(Some(b"26".as_slice())));
        assert_eq!(
            view.lookup(b"mid"),
            Some(None),
            "a present key with a null value is not an absent key"
        );
        assert_eq!(view.lookup(b"missing"), None);
    }

    #[test]
    fn a_map_is_stored_in_key_order_whatever_order_it_arrives_in() {
        let one: Vec<(&[u8], Option<&[u8]>)> = vec![
            (b"c", Some(b"3".as_slice())),
            (b"a", Some(b"1".as_slice())),
            (b"b", Some(b"2".as_slice())),
        ];
        let other: Vec<(&[u8], Option<&[u8]>)> = vec![
            (b"a", Some(b"1".as_slice())),
            (b"b", Some(b"2".as_slice())),
            (b"c", Some(b"3".as_slice())),
        ];
        assert_eq!(
            encode_map(&one),
            encode_map(&other),
            "the same map written twice has to give the same bytes"
        );

        let bytes = encode_map(&one);
        let view = MapView::parse(&bytes).expect("parses");
        assert_eq!(view.key(0), Some(b"a".as_slice()));
        assert_eq!(view.key(1), Some(b"b".as_slice()));
        assert_eq!(view.key(2), Some(b"c".as_slice()));
    }

    #[test]
    fn a_repeated_key_keeps_the_last_value_given() {
        let entries: Vec<(&[u8], Option<&[u8]>)> = vec![
            (b"k", Some(b"first".as_slice())),
            (b"k", Some(b"second".as_slice())),
        ];
        let bytes = encode_map(&entries);
        let view = MapView::parse(&bytes).expect("parses");
        assert_eq!(view.len(), 1);
        assert_eq!(view.lookup(b"k"), Some(Some(b"second".as_slice())));
    }

    #[test]
    fn an_empty_map_round_trips() {
        let bytes = encode_map(&[]);
        let view = MapView::parse(&bytes).expect("parses");
        assert!(view.is_empty());
        assert_eq!(view.lookup(b"anything"), None);
    }

    #[test]
    fn truncated_bytes_are_refused_rather_than_read_past() {
        let encoded = encode_struct(&[Some(b"payload".as_slice())]);
        for cut in 0..encoded.len() {
            // Parsing either refuses or yields a view whose fields stay
            // inside the bytes it was given
            if let Some(view) = StructView::parse(&encoded[..cut]) {
                for i in 0..view.field_count() {
                    let _ = view.field(i);
                }
            }
        }

        let entries: Vec<(&[u8], Option<&[u8]>)> = vec![(b"key", Some(b"value".as_slice()))];
        let encoded = encode_map(&entries);
        for cut in 0..encoded.len() {
            if let Some(view) = MapView::parse(&encoded[..cut]) {
                for i in 0..view.len() {
                    let _ = view.key(i);
                    let _ = view.value(i);
                }
                let _ = view.lookup(b"key");
            }
        }
    }

    #[test]
    fn the_two_encodings_do_not_parse_as_each_other() {
        let s = encode_struct(&[Some(b"x".as_slice())]);
        let m = encode_map(&[(b"x".as_slice(), Some(b"y".as_slice()))]);
        assert!(MapView::parse(&s).is_none());
        assert!(StructView::parse(&m).is_none());
    }
}
