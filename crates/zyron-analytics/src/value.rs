// AnalyticsValue, the runtime value type used by the analytics engine
// Self contained, decoupled from executor's ScalarValue, with conversions
// at the wire boundary done by callers

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use zyron_common::fx_mix;

/// Two independent 64-bit hashes of an AnalyticsValue, packed as
/// (low, high). The two halves use independent fx_mix chains seeded with
/// `seed_low` and `seed_high`. Combined they form a 128-bit fingerprint
/// with negligible collision probability at any practical scale.
///
/// The two chains are advanced in lockstep over a single payload walk so
/// each input byte is touched once and feeds two parallel ALU chains -
/// roughly half the work of calling hash_value_into twice.
#[inline]
pub fn hash_value_128(seed_low: u64, seed_high: u64, v: &AnalyticsValue) -> (u64, u64) {
    let mut h_lo = seed_low;
    let mut h_hi = seed_high;
    match v {
        AnalyticsValue::Null => {
            h_lo = fx_mix(h_lo, 0xA1);
            h_hi = fx_mix(h_hi, 0xA1);
        }
        AnalyticsValue::Bool(b) => {
            let tag = 0xB2u64;
            let val = *b as u64;
            h_lo = fx_mix(fx_mix(h_lo, tag), val);
            h_hi = fx_mix(fx_mix(h_hi, tag), val);
        }
        AnalyticsValue::Int(x) => {
            let tag = 0xC3u64;
            let val = *x as u64;
            h_lo = fx_mix(fx_mix(h_lo, tag), val);
            h_hi = fx_mix(fx_mix(h_hi, tag), val);
        }
        AnalyticsValue::UInt(x) => {
            let tag = 0xD4u64;
            h_lo = fx_mix(fx_mix(h_lo, tag), *x);
            h_hi = fx_mix(fx_mix(h_hi, tag), *x);
        }
        AnalyticsValue::Float(x) => {
            let tag = 0xE5u64;
            // Use the raw bit pattern so +0.0 and -0.0 hash distinctly,
            // matching the total_cmp ordering that puts -0.0 < +0.0.
            // NaN is canonicalised so all NaN payloads collide into one
            // hash bucket (the only case where the bit pattern needs to
            // be normalised, since total_cmp also collapses NaNs).
            let bits = if x.is_nan() {
                f64::NAN.to_bits()
            } else {
                x.to_bits()
            };
            h_lo = fx_mix(fx_mix(h_lo, tag), bits);
            h_hi = fx_mix(fx_mix(h_hi, tag), bits);
        }
        AnalyticsValue::Text(s) => {
            let tag = 0xF6u64;
            h_lo = fx_mix(h_lo, tag);
            h_hi = fx_mix(h_hi, tag);
            for &b in s.as_bytes() {
                let bv = b as u64;
                h_lo = fx_mix(h_lo, bv);
                h_hi = fx_mix(h_hi, bv);
            }
        }
        AnalyticsValue::Timestamp(x) => {
            let tag = 0x07u64;
            let val = *x as u64;
            h_lo = fx_mix(fx_mix(h_lo, tag), val);
            h_hi = fx_mix(fx_mix(h_hi, tag), val);
        }
        AnalyticsValue::Date(x) => {
            let tag = 0x18u64;
            let val = *x as u64;
            h_lo = fx_mix(fx_mix(h_lo, tag), val);
            h_hi = fx_mix(fx_mix(h_hi, tag), val);
        }
    }
    (h_lo, h_hi)
}

/// Streams an AnalyticsValue's payload bytes through the project's
/// canonical fx_mix primitive. The caller seeds with prior state (e.g.
/// a column index or grouping-set ID) so distinct contexts produce
/// independent hashes without re-hashing the key
#[inline]
pub fn hash_value_into(seed: u64, v: &AnalyticsValue) -> u64 {
    let mut h = seed;
    match v {
        AnalyticsValue::Null => h = fx_mix(h, 0xA1),
        AnalyticsValue::Bool(b) => h = fx_mix(fx_mix(h, 0xB2), *b as u64),
        AnalyticsValue::Int(x) => h = fx_mix(fx_mix(h, 0xC3), *x as u64),
        AnalyticsValue::UInt(x) => h = fx_mix(fx_mix(h, 0xD4), *x),
        AnalyticsValue::Float(x) => {
            // Bit pattern preserves the +0.0 / -0.0 distinction that
            // total_cmp orders distinctly. Only NaN is canonicalised so
            // every NaN payload collides into one bucket.
            let bits = if x.is_nan() {
                f64::NAN.to_bits()
            } else {
                x.to_bits()
            };
            h = fx_mix(fx_mix(h, 0xE5), bits);
        }
        AnalyticsValue::Text(s) => {
            h = fx_mix(h, 0xF6);
            for &b in s.as_bytes() {
                h = fx_mix(h, b as u64);
            }
        }
        AnalyticsValue::Timestamp(x) => h = fx_mix(fx_mix(h, 0x07), *x as u64),
        AnalyticsValue::Date(x) => h = fx_mix(fx_mix(h, 0x18), *x as u64),
    }
    h
}

pub const MS_PER_DAY: i64 = 86_400_000;
pub const MS_PER_HOUR: i64 = 3_600_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
    // Timestamp expressed as milliseconds since the Unix epoch UTC
    Timestamp(i64),
    // Date expressed as days since the Unix epoch UTC
    Date(i32),
}

impl AnalyticsValue {
    pub fn is_null(&self) -> bool {
        matches!(self, AnalyticsValue::Null)
    }

    /// Copies `other` into `self`, reusing the existing inner allocation
    /// when both sides are the same variant. The default enum Clone path
    /// does `*self = other.clone()`, which drops the existing String for
    /// the Text variant and heap-allocates a fresh one. This method
    /// dispatches the matching-variant cases through `Clone::clone_from`,
    /// which String overrides to reuse the buffer when the new content
    /// fits. For mismatched variants we fall back to the full clone.
    pub fn assign_from(&mut self, other: &AnalyticsValue) {
        use AnalyticsValue::*;
        match (self, other) {
            (Text(a), Text(b)) => a.clone_from(b),
            (Int(a), Int(b)) => *a = *b,
            (UInt(a), UInt(b)) => *a = *b,
            (Float(a), Float(b)) => *a = *b,
            (Bool(a), Bool(b)) => *a = *b,
            (Timestamp(a), Timestamp(b)) => *a = *b,
            (Date(a), Date(b)) => *a = *b,
            (Null, Null) => {}
            (slot, other) => *slot = other.clone(),
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AnalyticsValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            AnalyticsValue::Int(v) => Some(*v as f64),
            AnalyticsValue::UInt(v) => Some(*v as f64),
            AnalyticsValue::Float(v) => Some(*v),
            AnalyticsValue::Timestamp(v) => Some(*v as f64),
            AnalyticsValue::Date(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            AnalyticsValue::Int(v) => Some(*v),
            AnalyticsValue::UInt(v) => i64::try_from(*v).ok(),
            AnalyticsValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            AnalyticsValue::Timestamp(v) => Some(*v),
            AnalyticsValue::Date(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            AnalyticsValue::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    // Returns timestamp in milliseconds since epoch for any temporal value
    pub fn as_timestamp_ms(&self) -> Option<i64> {
        match self {
            AnalyticsValue::Timestamp(v) => Some(*v),
            AnalyticsValue::Date(v) => Some((*v as i64) * MS_PER_DAY),
            AnalyticsValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    // Discriminant order used to break cross-variant ties. Lower variants
    // sort first, except Null which sorts last (SQL convention).
    #[inline]
    fn variant_rank(&self) -> u8 {
        use AnalyticsValue::*;
        match self {
            Bool(_) => 0,
            Int(_) => 1,
            UInt(_) => 2,
            Float(_) => 3,
            Text(_) => 4,
            Timestamp(_) => 5,
            Date(_) => 6,
            Null => 255,
        }
    }

    // Total ordering used for histograms, percentiles, and grouping keys.
    // Same-variant comparisons use the natural ordering for that type.
    // Cross-variant comparisons fall back to discriminant order, which
    // keeps the result deterministic without allocating debug strings and
    // keeps Hash and Eq consistent (two values compare Equal only when
    // they have the same discriminant and the same payload).
    pub fn total_cmp(&self, other: &AnalyticsValue) -> Ordering {
        use AnalyticsValue::*;
        match (self, other) {
            (Null, Null) => Ordering::Equal,
            (Bool(a), Bool(b)) => a.cmp(b),
            (Int(a), Int(b)) => a.cmp(b),
            (UInt(a), UInt(b)) => a.cmp(b),
            (Float(a), Float(b)) => a.total_cmp(b),
            (Text(a), Text(b)) => a.cmp(b),
            (Timestamp(a), Timestamp(b)) => a.cmp(b),
            (Date(a), Date(b)) => a.cmp(b),
            (a, b) => a.variant_rank().cmp(&b.variant_rank()),
        }
    }
}

impl PartialEq for AnalyticsValue {
    fn eq(&self, other: &Self) -> bool {
        self.total_cmp(other) == Ordering::Equal
    }
}

impl Eq for AnalyticsValue {}

impl PartialOrd for AnalyticsValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.total_cmp(other))
    }
}

impl Ord for AnalyticsValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl std::hash::Hash for AnalyticsValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Discriminant first so disjoint variants never collide
        std::mem::discriminant(self).hash(state);
        match self {
            AnalyticsValue::Null => {}
            AnalyticsValue::Bool(b) => b.hash(state),
            AnalyticsValue::Int(v) => v.hash(state),
            AnalyticsValue::UInt(v) => v.hash(state),
            // Hash f64 by raw bit pattern so +0.0 and -0.0 hash distinctly
            // (matching total_cmp ordering). NaN is the only value
            // canonicalised, since total_cmp collapses all NaN payloads.
            AnalyticsValue::Float(v) => {
                let bits = if v.is_nan() {
                    f64::NAN.to_bits()
                } else {
                    v.to_bits()
                };
                bits.hash(state);
            }
            AnalyticsValue::Text(s) => s.hash(state),
            AnalyticsValue::Timestamp(v) => v.hash(state),
            AnalyticsValue::Date(v) => v.hash(state),
        }
    }
}

// A row is an ordered list of named columns
// Vec<(name, value)> keeps insertion order without pulling in IndexMap
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyticsRow {
    pub columns: Vec<(String, AnalyticsValue)>,
}

impl AnalyticsRow {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            columns: Vec::with_capacity(n),
        }
    }

    pub fn push(&mut self, name: impl Into<String>, value: AnalyticsValue) {
        self.columns.push((name.into(), value));
    }

    pub fn get(&self, name: &str) -> Option<&AnalyticsValue> {
        self.columns.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

// VerifiedKeyMap: a PreHashMap-backed table keyed by a 128-bit hash split
// into a low half (used for bucketing) and a high half (used to verify
// the slot belongs to the caller's intended key, so distinct semantic
// keys never silently merge).
//
// 128-bit collision probability is negligible at any practical scale, so
// the per-access verify cost is one u64 compare regardless of how heavy
// the actual key type is. The original key is still stored alongside the
// value for callers that need it for output rows; it is never read on the
// hot path.
//
// Bucket layout: each bucket holds a SmallChain whose head is inline and
// whose tail is a Vec holding any extra entries that share the bucket
// index but differ in the high half. The tail is only allocated on the
// rare collision-of-low-half-only case.

pub struct SmallChain<K, V> {
    head: (u64, K, V),
    tail: Vec<(u64, K, V)>,
}

impl<K, V> SmallChain<K, V> {
    fn new(hash_high: u64, k: K, v: V) -> Self {
        Self {
            head: (hash_high, k, v),
            tail: Vec::new(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        std::iter::once((&self.head.1, &self.head.2))
            .chain(self.tail.iter().map(|(_, k, v)| (k, v)))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        let (head_k, head_v) = (&self.head.1, &mut self.head.2);
        std::iter::once((head_k, head_v)).chain(self.tail.iter_mut().map(|(_, k, v)| (&*k, v)))
    }

    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        std::iter::once((self.head.1, self.head.2))
            .chain(self.tail.into_iter().map(|(_, k, v)| (k, v)))
    }

    pub fn len(&self) -> usize {
        1 + self.tail.len()
    }
}

pub struct VerifiedKeyMap<K, V> {
    inner: zyron_common::PreHashMap<u64, SmallChain<K, V>>,
}

impl<K, V> VerifiedKeyMap<K, V> {
    pub fn new() -> Self {
        Self {
            inner: zyron_common::PreHashMap::default(),
        }
    }

    /// Total entry count, including chain overflow. Walks the bucket map,
    /// so prefer to call once at finalise rather than per-record.
    pub fn len(&self) -> usize {
        self.inner.values().map(|c| c.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a mutable reference to the value for the (hash_low, hash_high)
    /// key. If no slot exists yet, the key is built via `build_key` and the
    /// value via `make_default`; both closures fire only on the miss path.
    /// If a different key shares hash_low (a 64-bit-only collision), we
    /// scan/append the chain matching on hash_high.
    pub fn entry_or_insert<KeyF, ValF>(
        &mut self,
        hash_low: u64,
        hash_high: u64,
        build_key: KeyF,
        make_default: ValF,
    ) -> &mut V
    where
        KeyF: FnOnce() -> K,
        ValF: FnOnce() -> V,
    {
        use std::collections::hash_map::Entry;
        match self.inner.entry(hash_low) {
            Entry::Vacant(vac) => {
                let chain = SmallChain::new(hash_high, build_key(), make_default());
                &mut vac.insert(chain).head.2
            }
            Entry::Occupied(occ) => {
                let chain = occ.into_mut();
                if chain.head.0 == hash_high {
                    return &mut chain.head.2;
                }
                if let Some(idx) = chain.tail.iter().position(|(h, _, _)| *h == hash_high) {
                    return &mut chain.tail[idx].2;
                }
                chain.tail.push((hash_high, build_key(), make_default()));
                &mut chain.tail.last_mut().unwrap().2
            }
        }
    }

    pub fn get(&self, hash_low: u64, hash_high: u64) -> Option<&V> {
        let chain = self.inner.get(&hash_low)?;
        if chain.head.0 == hash_high {
            return Some(&chain.head.2);
        }
        chain
            .tail
            .iter()
            .find(|(h, _, _)| *h == hash_high)
            .map(|(_, _, v)| v)
    }

    pub fn get_mut(&mut self, hash_low: u64, hash_high: u64) -> Option<&mut V> {
        let chain = self.inner.get_mut(&hash_low)?;
        if chain.head.0 == hash_high {
            return Some(&mut chain.head.2);
        }
        chain
            .tail
            .iter_mut()
            .find(|(h, _, _)| *h == hash_high)
            .map(|(_, _, v)| v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.values().flat_map(|chain| chain.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.inner.values_mut().flat_map(|chain| chain.iter_mut())
    }

    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.inner
            .into_iter()
            .flat_map(|(_, chain)| chain.into_iter())
    }
}

impl<K, V> Default for VerifiedKeyMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}
