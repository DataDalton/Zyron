//! Struct-of-arrays projection of one manifest version for file pruning.
//!
//! A manifest stores each file's statistics as a sorted list of typed
//! values. That form is exact and it is what a delete has to read, but it
//! answers one file at a time through a binary search and an enum match,
//! so a plan against a hundred thousand files pays a hundred thousand of
//! them per predicate term. This projects the same statistics into
//! contiguous u64 arrays keyed by column, where applying one term across
//! every file is a branch-free sweep the compiler vectorizes.
//!
//! The projection is derived state, so it is built once per log version
//! and shared by every plan that reads that version.
//!
//! Soundness is the contract and completeness is reported. A one in the
//! mask means the exact statistics also reject the file. A zero means the
//! sweep has no proof, never that the file matches. Two things cost the
//! sweep its completeness: a `stats_key` is a 64-bit summary, so a long
//! string or a 128-bit integer keeps only its leading bits and equal keys
//! prove nothing, and a value bloom has no vectorized form at all. When
//! the sweep reports itself incomplete the caller runs the exact check
//! over the survivors, which is where the files are already few.

use crate::manifest::ManifestFile;
use crate::predicate::{CompareOp, LakePredicate, LakeValue};

/// Columns projected at most, cluster keys first because those are the
/// ones the layout was chosen to make prunable
const MAX_INDEXED_COLUMNS: usize = 32;

/// Byte ceiling for one version's projection. A wide table with very many
/// files indexes its leading columns and leaves the rest to the exact
/// path rather than holding tens of megabytes per cached version
const MAX_INDEX_BYTES: usize = 16 << 20;

/// Per file, per indexed column: two bounds and four flag bytes
const BYTES_PER_COLUMN_FILE: usize = 2 * 8 + 4;

/// What every recorded bound of one indexed column has in common
#[derive(Debug, Clone, Copy)]
struct KeyMeta {
    /// Value variant every recorded bound uses. A constant of a different
    /// variant is not swept, the exact path decides it
    family: u8,
    /// Every recorded bound survives `stats_key` unchanged, so a
    /// non-strict key comparison carries the same proof the values do
    exact: bool,
    /// Some file carries a value bloom for this column. The sweep cannot
    /// probe one, so an equality term over this column is sound but
    /// prunes less than the exact path and is reported incomplete
    bloomed: bool,
}

/// One log version's file statistics projected for vectorized pruning
pub struct PruneIndex {
    file_count: usize,
    /// Manifest order throughout, so a survivor position addresses the
    /// same entry in `ManifestFile::entries`
    partition_ids: Box<[u64]>,
    size_bytes: Box<[u64]>,
    row_counts: Box<[u64]>,
    /// Indexed column ids, ascending
    key_ids: Box<[u32]>,
    keys: Box<[KeyMeta]>,
    /// `key_min[k * file_count + f]`, and the same addressing throughout
    key_min: Box<[u64]>,
    key_max: Box<[u64]>,
    /// The file carries statistics for this column at all. A file that
    /// does not is undecided rather than rejected, which is what the
    /// exact path answers too
    key_has_stats: Box<[u8]>,
    /// Both bounds known, so a comparison has something to decide against
    key_usable: Box<[u8]>,
    /// Statistics present and no null in this file, which is what an
    /// IS NULL term needs
    key_no_nulls: Box<[u8]>,
    /// Every row null, which rejects every comparison and IS NOT NULL
    key_all_null: Box<[u8]>,
}

/// Reusable working memory for a pruning sweep.
///
/// One sweep needs a byte per file for each nesting level of the
/// predicate. Holding the buffer across sweeps is what makes them
/// allocation free once it has grown to the widest plan a process runs
#[derive(Debug, Default)]
pub struct PruneScratch {
    buf: Vec<u8>,
}

impl PruneScratch {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Bytes currently held, so a caller can assert a warm sweep grew
    /// nothing
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }
}

impl PruneIndex {
    /// Projects one manifest. Cost is one pass over the file entries
    pub fn build(manifest: &ManifestFile) -> Self {
        let file_count = manifest.entries.len();
        let mut partition_ids = Vec::with_capacity(file_count);
        let mut size_bytes = Vec::with_capacity(file_count);
        let mut row_counts = Vec::with_capacity(file_count);
        for entry in &manifest.entries {
            partition_ids.push(entry.partition_id);
            size_bytes.push(entry.size_bytes);
            row_counts.push(entry.row_count);
        }

        let key_ids = select_columns(manifest, file_count);
        let block = key_ids.len() * file_count;
        let mut key_min = vec![0u64; block];
        let mut key_max = vec![0u64; block];
        let mut key_has_stats = vec![0u8; block];
        let mut key_usable = vec![0u8; block];
        let mut key_no_nulls = vec![0u8; block];
        let mut key_all_null = vec![0u8; block];
        let mut keys = vec![
            KeyMeta {
                family: FAMILY_UNSET,
                exact: true,
                bloomed: false,
            };
            key_ids.len()
        ];

        for (k, column_id) in key_ids.iter().enumerate() {
            let base = k * file_count;
            let meta = &mut keys[k];
            for (f, entry) in manifest.entries.iter().enumerate() {
                let Some(stats) = entry.stats_for(*column_id) else {
                    continue;
                };
                let bounds = &stats.bounds;
                if stats.bloom.is_some() {
                    meta.bloomed = true;
                }
                key_has_stats[base + f] = 1;
                key_no_nulls[base + f] = (bounds.null_count == 0) as u8;
                key_all_null[base + f] =
                    (bounds.null_count == entry.row_count && entry.row_count > 0) as u8;
                let (Some(min), Some(max)) = (bounds.min.as_ref(), bounds.max.as_ref()) else {
                    continue;
                };
                let (Some(min_key), Some(max_key)) = (min.stats_key(), max.stats_key()) else {
                    continue;
                };
                // A column whose bounds disagree on variant cannot be
                // swept at all, one key scale does not order two families
                let family = family_tag(min);
                if family != family_tag(max) {
                    meta.family = FAMILY_MIXED;
                    continue;
                }
                match meta.family {
                    FAMILY_UNSET => meta.family = family,
                    existing if existing != family => {
                        meta.family = FAMILY_MIXED;
                        continue;
                    }
                    _ => {}
                }
                meta.exact &= key_is_lossless(min) && key_is_lossless(max);
                key_min[base + f] = min_key;
                key_max[base + f] = max_key;
                key_usable[base + f] = 1;
            }
        }

        Self {
            file_count,
            partition_ids: partition_ids.into_boxed_slice(),
            size_bytes: size_bytes.into_boxed_slice(),
            row_counts: row_counts.into_boxed_slice(),
            key_ids: key_ids.into_boxed_slice(),
            keys: keys.into_boxed_slice(),
            key_min: key_min.into_boxed_slice(),
            key_max: key_max.into_boxed_slice(),
            key_has_stats: key_has_stats.into_boxed_slice(),
            key_usable: key_usable.into_boxed_slice(),
            key_no_nulls: key_no_nulls.into_boxed_slice(),
            key_all_null: key_all_null.into_boxed_slice(),
        }
    }

    pub fn file_count(&self) -> usize {
        self.file_count
    }

    /// Data file identities in manifest order
    pub fn partition_ids(&self) -> &[u64] {
        &self.partition_ids
    }

    /// File sizes in manifest order, the currency skip rate is measured in
    pub fn size_bytes(&self) -> &[u64] {
        &self.size_bytes
    }

    pub fn row_counts(&self) -> &[u64] {
        &self.row_counts
    }

    /// Column ids this index can sweep, ascending
    pub fn indexed_columns(&self) -> &[u32] {
        &self.key_ids
    }

    /// Bytes the projection holds
    pub fn heap_bytes(&self) -> usize {
        self.file_count * 3 * 8 + self.key_ids.len() * self.file_count * BYTES_PER_COLUMN_FILE
    }

    /// Marks every file the statistics prove the predicate cannot match.
    ///
    /// The returned slice holds one byte per file in manifest order: one
    /// is a proof the file can be skipped, zero is the absence of a proof
    /// and never a proof of the opposite. The flag reports whether the
    /// sweep decided every file, which is what lets a caller drop the
    /// exact check rather than run it over the survivors
    pub fn cannot_match<'s>(
        &self,
        predicate: &LakePredicate,
        scratch: &'s mut PruneScratch,
    ) -> (&'s [u8], bool) {
        let slots = predicate.eval_slots();
        let needed = slots * self.file_count;
        if scratch.buf.len() < needed {
            scratch.buf.resize(needed, 0);
        }
        let complete = self.eval(predicate, false, &mut scratch.buf[..needed]);
        (&scratch.buf[..self.file_count], complete)
    }

    /// Writes one predicate's proof mask into `buf[..file_count]`, using
    /// the rest of `buf` for children. Returns whether the answer is exact
    fn eval(&self, predicate: &LakePredicate, negated: bool, buf: &mut [u8]) -> bool {
        let n = self.file_count;
        // Negation is carried rather than applied, matching the exact
        // path, so a Not consumes no slot and allocates nothing
        if let LakePredicate::Not(inner) = predicate {
            return self.eval(inner, !negated, buf);
        }
        let (here, rest) = buf.split_at_mut(n);
        match predicate {
            LakePredicate::Compare {
                column_id,
                op,
                value,
            } => {
                let op = if negated { op.negated() } else { *op };
                self.sweep_compare(*column_id, op, value, here)
            }
            LakePredicate::IsNull { column_id } => {
                if negated {
                    self.sweep_flag(*column_id, FlagKind::AllNull, here)
                } else {
                    self.sweep_flag(*column_id, FlagKind::NoNulls, here)
                }
            }
            LakePredicate::IsNotNull { column_id } => {
                if negated {
                    self.sweep_flag(*column_id, FlagKind::NoNulls, here)
                } else {
                    self.sweep_flag(*column_id, FlagKind::AllNull, here)
                }
            }
            LakePredicate::In { column_id, values } => {
                if values.is_empty() {
                    // IN () selects nothing, NOT IN () selects everything
                    here.fill(!negated as u8);
                    return true;
                }
                let child = &mut rest[..n];
                let mut complete = true;
                if negated {
                    // NOT IN is the conjunction, so one rejected value
                    // rejects the file
                    here.fill(0);
                    for value in values {
                        complete &= self.sweep_compare(*column_id, CompareOp::NotEq, value, child);
                        for f in 0..n {
                            here[f] |= child[f];
                        }
                    }
                } else {
                    // IN is the disjunction, so every value must be rejected
                    here.fill(1);
                    for value in values {
                        complete &= self.sweep_compare(*column_id, CompareOp::Eq, value, child);
                        for f in 0..n {
                            here[f] &= child[f];
                        }
                    }
                }
                complete
            }
            LakePredicate::And(children) | LakePredicate::Or(children) => {
                // A conjunction rejects a file when any arm does, a
                // disjunction only when every arm does, and negation
                // exchanges the two
                let conjunction = matches!(predicate, LakePredicate::And(_)) != negated;
                let mut complete = true;
                here.fill(!conjunction as u8);
                for c in children {
                    // A child sweep needs its own slot plus the slots its
                    // own children need, which is what eval_slots counts
                    let child_buf = &mut rest[..(c.eval_slots() * n)];
                    complete &= self.eval(c, negated, child_buf);
                    if conjunction {
                        for f in 0..n {
                            here[f] |= child_buf[f];
                        }
                    } else {
                        for f in 0..n {
                            here[f] &= child_buf[f];
                        }
                    }
                }
                complete
            }
            LakePredicate::Not(_) => unreachable!("Not is handled before the slot split"),
        }
    }

    /// Null-shaped terms read a precomputed flag column directly
    fn sweep_flag(&self, column_id: u32, kind: FlagKind, out: &mut [u8]) -> bool {
        let Some(k) = self.slot_index(column_id) else {
            out.fill(0);
            return false;
        };
        let base = k * self.file_count;
        let src = match kind {
            FlagKind::HasStats => &self.key_has_stats[base..base + self.file_count],
            FlagKind::NoNulls => &self.key_no_nulls[base..base + self.file_count],
            FlagKind::AllNull => &self.key_all_null[base..base + self.file_count],
        };
        out.copy_from_slice(src);
        true
    }

    /// One comparison across every file.
    ///
    /// The operator and the exactness of the key are resolved once, so the
    /// per-file body is two loads, one compare and two byte operations,
    /// which is the shape that vectorizes
    fn sweep_compare(
        &self,
        column_id: u32,
        op: CompareOp,
        value: &LakeValue,
        out: &mut [u8],
    ) -> bool {
        let n = self.file_count;
        let Some(k) = self.slot_index(column_id) else {
            out.fill(0);
            return false;
        };
        let meta = self.keys[k];
        // A comparison against null is unknown for every row, whatever the
        // operator, so every file carrying statistics for the column is
        // rejected. One that carries none is undecided, which is what the
        // exact path answers too
        if matches!(value, LakeValue::Null) {
            return self.sweep_flag(column_id, FlagKind::HasStats, out);
        }
        let Some((vk, constant_lossless)) = constant_key(meta.family, value) else {
            out.fill(0);
            return false;
        };
        // Equality is the one term a value bloom refines, and no bloom has
        // a vectorized form, so bounds still prune but the answer is not
        // the last word
        let bloom_pending = meta.bloomed && op == CompareOp::Eq;
        let exact = meta.exact && constant_lossless;

        let base = k * n;
        let mins = &self.key_min[base..base + n];
        let maxs = &self.key_max[base..base + n];
        let usable = &self.key_usable[base..base + n];
        let all_null = &self.key_all_null[base..base + n];

        // With a lossless key a key comparison decides exactly what a value
        // comparison would. With a truncating one only a strict inequality
        // carries proof, so those arms weaken to the strict form and the
        // sweep reports itself incomplete
        match (op, exact) {
            (CompareOp::Eq, _) => {
                sweep_bounds(out, mins, maxs, usable, all_null, vk, |min, max, v| {
                    (v < min) | (v > max)
                })
            }
            (CompareOp::NotEq, true) => {
                sweep_bounds(out, mins, maxs, usable, all_null, vk, |min, max, v| {
                    (v == min) & (v == max)
                })
            }
            // Proving every value equal needs an injective key
            (CompareOp::NotEq, false) => out.copy_from_slice(all_null),
            (CompareOp::Lt, true) => {
                sweep_bounds(out, mins, maxs, usable, all_null, vk, |min, _, v| min >= v)
            }
            (CompareOp::Lt, false) | (CompareOp::LtEq, _) => {
                sweep_bounds(out, mins, maxs, usable, all_null, vk, |min, _, v| min > v)
            }
            (CompareOp::Gt, true) => {
                sweep_bounds(out, mins, maxs, usable, all_null, vk, |_, max, v| max <= v)
            }
            (CompareOp::Gt, false) | (CompareOp::GtEq, _) => {
                sweep_bounds(out, mins, maxs, usable, all_null, vk, |_, max, v| max < v)
            }
        }
        exact && !bloom_pending
    }

    fn slot_index(&self, column_id: u32) -> Option<usize> {
        self.key_ids.binary_search(&column_id).ok()
    }
}

thread_local! {
    /// One sweep buffer per thread. Plans are built on worker threads
    /// that run many sweeps, so holding it here is what makes a warm
    /// thread prune without allocating at all
    static THREAD_SCRATCH: std::cell::RefCell<PruneScratch> =
        std::cell::RefCell::new(PruneScratch::new());
}

/// Runs one sweep on the calling thread's reusable buffer and hands the
/// proof mask and its exactness to `f`.
///
/// The mask borrows the buffer, so it is passed to a closure rather than
/// returned, which is what keeps the buffer owned by the thread instead
/// of allocated per plan
pub fn with_sweep<R>(
    index: &PruneIndex,
    predicate: &LakePredicate,
    f: impl FnOnce(&[u8], bool) -> R,
) -> R {
    THREAD_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let (mask, complete) = index.cannot_match(predicate, &mut scratch);
        f(mask, complete)
    })
}

enum FlagKind {
    HasStats,
    NoNulls,
    AllNull,
}

/// One comparison across every file, given the per-file condition.
///
/// Generic over the condition so each of the six operator arms inlines
/// its own copy, leaving a body of two loads, one compare and two byte
/// operations with no branch and no bounds check
#[inline(always)]
fn sweep_bounds(
    out: &mut [u8],
    mins: &[u64],
    maxs: &[u64],
    usable: &[u8],
    all_null: &[u8],
    vk: u64,
    cond: impl Fn(u64, u64, u64) -> bool,
) {
    let n = out.len();
    let mins = &mins[..n];
    let maxs = &maxs[..n];
    let usable = &usable[..n];
    let all_null = &all_null[..n];
    for f in 0..n {
        let hit = cond(mins[f], maxs[f], vk);
        out[f] = (usable[f] & hit as u8) | all_null[f];
    }
}

/// No bound recorded yet, so the column carries no orderable statistics
const FAMILY_UNSET: u8 = 0;
/// Bounds of more than one variant, which one key scale cannot order
const FAMILY_MIXED: u8 = u8::MAX;

fn family_tag(value: &LakeValue) -> u8 {
    match value {
        LakeValue::Null => FAMILY_UNSET,
        LakeValue::Bool(_) => 1,
        LakeValue::Int(_) => 2,
        LakeValue::Int128(_) => 3,
        LakeValue::UInt(_) => 4,
        LakeValue::UInt128(_) => 5,
        LakeValue::Float(_) => 6,
        LakeValue::Str(_) => 7,
        LakeValue::Bytes(_) => 8,
    }
}

/// Whether `stats_key` maps this value injectively, which is what makes a
/// non-strict key comparison a proof
fn key_is_lossless(value: &LakeValue) -> bool {
    match value {
        LakeValue::Null => false,
        LakeValue::Bool(_) | LakeValue::Int(_) | LakeValue::UInt(_) | LakeValue::Float(_) => true,
        LakeValue::Str(s) => s.len() <= 8,
        LakeValue::Bytes(b) => b.len() <= 8,
        // The key keeps the leading 64 bits, so only a value whose low 64
        // are zero survives it
        LakeValue::Int128(v) => (*v as u128) & (u64::MAX as u128) == 0,
        LakeValue::UInt128(v) => *v & (u64::MAX as u128) == 0,
    }
}

/// The constant's key on the column's scale, and whether that key is
/// lossless.
///
/// A constant of a different integer width is rewritten into the column's
/// family when that loses nothing, so an `i64` literal still sweeps a
/// `u32` column. Anything the families cannot reconcile exactly returns
/// None and the term goes to the exact path rather than being guessed.
/// The rewrite builds no value with a heap payload, so a sweep over a
/// string column costs no allocation either
fn constant_key(family: u8, value: &LakeValue) -> Option<(u64, bool)> {
    if family_tag(value) == family {
        return Some((value.stats_key()?, key_is_lossless(value)));
    }
    let as_i128 = match value {
        LakeValue::Int(v) => *v as i128,
        LakeValue::Int128(v) => *v,
        LakeValue::UInt(v) => *v as i128,
        LakeValue::UInt128(v) => i128::try_from(*v).ok()?,
        _ => return None,
    };
    let converted = match family {
        2 => LakeValue::Int(i64::try_from(as_i128).ok()?),
        3 => LakeValue::Int128(as_i128),
        4 => LakeValue::UInt(u64::try_from(as_i128).ok()?),
        5 => LakeValue::UInt128(u128::try_from(as_i128).ok()?),
        _ => return None,
    };
    Some((converted.stats_key()?, key_is_lossless(&converted)))
}

/// Columns worth projecting, ascending.
///
/// Cluster keys come first because the layout was chosen to make those
/// prunable, then every other column carrying statistics. Both caps exist
/// so a wide table with very many files indexes its leading columns rather
/// than holding tens of megabytes per cached version
fn select_columns(manifest: &ManifestFile, file_count: usize) -> Vec<u32> {
    let schema_ids: Vec<u32> = {
        let mut ids: Vec<u32> = manifest.schema.columns.iter().map(|c| c.id).collect();
        ids.sort_unstable();
        ids
    };
    let mut present = vec![false; schema_ids.len()];
    for entry in &manifest.entries {
        for stat in &entry.column_stats {
            if let Ok(i) = schema_ids.binary_search(&stat.column_id) {
                present[i] = true;
            }
        }
    }

    let budget = if file_count == 0 {
        MAX_INDEXED_COLUMNS
    } else {
        (MAX_INDEX_BYTES / (file_count * BYTES_PER_COLUMN_FILE)).max(1)
    };
    let cap = MAX_INDEXED_COLUMNS.min(budget);

    let mut chosen: Vec<u32> = Vec::with_capacity(cap);
    let push = |id: u32, chosen: &mut Vec<u32>| {
        if chosen.len() < cap && !chosen.contains(&id) {
            chosen.push(id);
        }
    };
    for key in &manifest.cluster_spec.keys {
        if schema_ids
            .binary_search(&key.column_id)
            .map(|i| present[i])
            .unwrap_or(false)
        {
            push(key.column_id, &mut chosen);
        }
    }
    for (i, id) in schema_ids.iter().enumerate() {
        if present[i] {
            push(*id, &mut chosen);
        }
    }
    chosen.sort_unstable();
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ClusterSpec, ColumnStatsEntry, PartitionEntry};
    use crate::predicate::{ColumnBounds, PruneDecision};
    use crate::schema::{LakeColumn, LakeSchema};
    use std::collections::BTreeMap;
    use zyron_common::TypeId;

    fn schema(types: &[TypeId]) -> LakeSchema {
        LakeSchema::new(
            1,
            types
                .iter()
                .enumerate()
                .map(|(i, t)| LakeColumn {
                    id: i as u32,
                    name: format!("c{}", i),
                    type_id: *t,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                })
                .collect(),
        )
        .expect("valid schema")
    }

    fn entry(id: u64, stats: Vec<ColumnStatsEntry>, row_count: u64) -> PartitionEntry {
        PartitionEntry {
            partition_id: id,
            size_bytes: 1024,
            row_count,
            added_version: 1,
            cluster_spec_id: 0,
            column_stats: stats,
            delete_predicate_ids: Vec::new(),
        }
    }

    fn bounded(
        column_id: u32,
        min: Option<LakeValue>,
        max: Option<LakeValue>,
        null_count: u64,
        row_count: u64,
    ) -> ColumnStatsEntry {
        ColumnStatsEntry {
            column_id,
            bounds: ColumnBounds {
                min,
                max,
                null_count,
                row_count,
            },
            bloom: None,
            ndv: None,
        }
    }

    fn manifest_of(schema: LakeSchema, entries: Vec<PartitionEntry>) -> ManifestFile {
        ManifestFile {
            snapshot_id: 1,
            parent_snapshot_id: 0,
            timestamp_us: 0,
            schema,
            cluster_spec: ClusterSpec::none(),
            entries,
            delete_predicates: Vec::new(),
            properties: BTreeMap::new(),
            indexes: Vec::new(),
            index_files: Vec::new(),
        }
    }

    /// The exact answer, file by file, so a sweep can be checked against it
    fn scalar_mask(manifest: &ManifestFile, predicate: &LakePredicate) -> Vec<u8> {
        manifest
            .entries
            .iter()
            .map(|e| (manifest.prune_file(predicate, e) == PruneDecision::CannotMatch) as u8)
            .collect()
    }

    fn int_manifest(files: usize) -> ManifestFile {
        let entries = (0..files)
            .map(|i| {
                let lo = (i as i64) * 100;
                entry(
                    i as u64,
                    vec![bounded(
                        0,
                        Some(LakeValue::Int(lo)),
                        Some(LakeValue::Int(lo + 99)),
                        0,
                        100,
                    )],
                    100,
                )
            })
            .collect();
        manifest_of(schema(&[TypeId::Int64]), entries)
    }

    fn cmp(column_id: u32, op: CompareOp, value: LakeValue) -> LakePredicate {
        LakePredicate::Compare {
            column_id,
            op,
            value,
        }
    }

    #[test]
    fn test_sweep_matches_the_exact_reference_on_losslessly_keyed_columns() {
        let manifest = int_manifest(64);
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();

        let cases = [
            cmp(0, CompareOp::Eq, LakeValue::Int(1_050)),
            cmp(0, CompareOp::NotEq, LakeValue::Int(0)),
            cmp(0, CompareOp::Lt, LakeValue::Int(2_000)),
            cmp(0, CompareOp::LtEq, LakeValue::Int(2_000)),
            cmp(0, CompareOp::Gt, LakeValue::Int(4_999)),
            cmp(0, CompareOp::GtEq, LakeValue::Int(5_000)),
            LakePredicate::In {
                column_id: 0,
                values: vec![LakeValue::Int(10), LakeValue::Int(6_300)],
            },
            LakePredicate::And(vec![
                cmp(0, CompareOp::GtEq, LakeValue::Int(1_000)),
                cmp(0, CompareOp::Lt, LakeValue::Int(2_000)),
            ]),
            LakePredicate::Or(vec![
                cmp(0, CompareOp::Lt, LakeValue::Int(200)),
                cmp(0, CompareOp::GtEq, LakeValue::Int(6_000)),
            ]),
            LakePredicate::Not(Box::new(cmp(0, CompareOp::Lt, LakeValue::Int(3_000)))),
            LakePredicate::Not(Box::new(LakePredicate::And(vec![
                cmp(0, CompareOp::GtEq, LakeValue::Int(0)),
                cmp(0, CompareOp::Lt, LakeValue::Int(3_000)),
            ]))),
            LakePredicate::Not(Box::new(LakePredicate::In {
                column_id: 0,
                values: vec![LakeValue::Int(10)],
            })),
            LakePredicate::IsNull { column_id: 0 },
            LakePredicate::IsNotNull { column_id: 0 },
        ];

        for predicate in cases {
            let expected = scalar_mask(&manifest, &predicate);
            let (mask, complete) = index.cannot_match(&predicate, &mut scratch);
            assert!(
                complete,
                "an int column with no bloom decides exactly: {:?}",
                predicate
            );
            assert_eq!(mask, &expected[..], "mask differs for {:?}", predicate);
        }
    }

    #[test]
    fn test_sweep_never_marks_a_file_the_exact_check_keeps() {
        // Strings longer than the key, 128-bit integers and a bloom are
        // the three cases the sweep cannot decide. None of them may
        // produce a mark the exact path disagrees with
        let entries = vec![
            entry(
                0,
                vec![
                    bounded(
                        0,
                        Some(LakeValue::Str("aaaaaaaaAAAA".into())),
                        Some(LakeValue::Str("aaaaaaaaZZZZ".into())),
                        0,
                        10,
                    ),
                    bounded(
                        1,
                        Some(LakeValue::Int128(1 << 70)),
                        Some(LakeValue::Int128((1 << 70) + 5)),
                        0,
                        10,
                    ),
                ],
                10,
            ),
            entry(
                1,
                vec![
                    bounded(
                        0,
                        Some(LakeValue::Str("m".into())),
                        Some(LakeValue::Str("z".into())),
                        0,
                        10,
                    ),
                    bounded(
                        1,
                        Some(LakeValue::Int128(0)),
                        Some(LakeValue::Int128(4)),
                        0,
                        10,
                    ),
                ],
                10,
            ),
        ];
        let manifest = manifest_of(schema(&[TypeId::Varchar, TypeId::Int128]), entries);
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();

        let cases = [
            cmp(0, CompareOp::Eq, LakeValue::Str("aaaaaaaaBBBB".into())),
            cmp(0, CompareOp::Lt, LakeValue::Str("aaaaaaaaAAAA".into())),
            cmp(0, CompareOp::NotEq, LakeValue::Str("m".into())),
            cmp(0, CompareOp::GtEq, LakeValue::Str("b".into())),
            cmp(1, CompareOp::Eq, LakeValue::Int128((1 << 70) + 1)),
            cmp(1, CompareOp::Gt, LakeValue::Int128(1 << 70)),
        ];
        for predicate in cases {
            let expected = scalar_mask(&manifest, &predicate);
            let (mask, _) = index.cannot_match(&predicate, &mut scratch);
            for (f, bit) in mask.iter().enumerate() {
                assert!(
                    *bit == 0 || expected[f] == 1,
                    "sweep marked file {} that the exact check keeps for {:?}",
                    f,
                    predicate
                );
            }
        }
    }

    #[test]
    fn test_a_bloomed_equality_prunes_on_bounds_but_reports_itself_incomplete() {
        let mut stats = bounded(0, Some(LakeValue::Int(0)), Some(LakeValue::Int(99)), 0, 100);
        stats.bloom = Some(vec![0u8; 16]);
        let manifest = manifest_of(
            schema(&[TypeId::Int64]),
            vec![
                entry(0, vec![stats], 100),
                entry(
                    1,
                    vec![bounded(
                        0,
                        Some(LakeValue::Int(500)),
                        Some(LakeValue::Int(599)),
                        0,
                        100,
                    )],
                    100,
                ),
            ],
        );
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();

        let predicate = cmp(0, CompareOp::Eq, LakeValue::Int(42));
        let (mask, complete) = index.cannot_match(&predicate, &mut scratch);
        assert!(
            !complete,
            "a bloom the sweep cannot probe is not the last word"
        );
        assert_eq!(mask[1], 1, "bounds still reject the file out of range");

        // A range term over the same column carries no bloom question
        let ranged = cmp(0, CompareOp::GtEq, LakeValue::Int(500));
        let (mask, complete) = index.cannot_match(&ranged, &mut scratch);
        assert!(complete);
        assert_eq!(mask, &[1, 0][..]);
    }

    #[test]
    fn test_an_unindexed_column_decides_nothing_rather_than_guessing() {
        let manifest = int_manifest(4);
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();

        let predicate = cmp(7, CompareOp::Eq, LakeValue::Int(1));
        let (mask, complete) = index.cannot_match(&predicate, &mut scratch);
        assert!(!complete);
        assert!(mask.iter().all(|b| *b == 0));

        // A conjunction with one undecidable arm keeps whatever the other
        // arm proves, which is still sound
        let mixed = LakePredicate::And(vec![
            predicate,
            cmp(0, CompareOp::GtEq, LakeValue::Int(300)),
        ]);
        let (mask, complete) = index.cannot_match(&mixed, &mut scratch);
        assert!(!complete);
        assert_eq!(mask, &[1, 1, 1, 0][..]);
    }

    #[test]
    fn test_a_constant_of_another_integer_width_still_sweeps() {
        let manifest = int_manifest(4);
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();

        let predicate = cmp(0, CompareOp::GtEq, LakeValue::UInt(300));
        let expected = scalar_mask(&manifest, &predicate);
        let (mask, complete) = index.cannot_match(&predicate, &mut scratch);
        assert!(
            complete,
            "an unsigned literal converts into the signed key exactly"
        );
        assert_eq!(mask, &expected[..]);

        // One that cannot be represented is left to the exact path
        let unrepresentable = cmp(0, CompareOp::GtEq, LakeValue::UInt(u64::MAX));
        let (mask, complete) = index.cannot_match(&unrepresentable, &mut scratch);
        assert!(!complete);
        assert!(mask.iter().all(|b| *b == 0));
    }

    #[test]
    fn test_null_and_missing_bounds_follow_the_exact_rules() {
        let entries = vec![
            // Every row null: no comparison can match
            entry(0, vec![bounded(0, None, None, 100, 100)], 100),
            // Bounds unknown but rows present: nothing is decided
            entry(1, vec![bounded(0, None, None, 0, 100)], 100),
            entry(
                2,
                vec![bounded(
                    0,
                    Some(LakeValue::Int(0)),
                    Some(LakeValue::Int(9)),
                    3,
                    100,
                )],
                100,
            ),
        ];
        let manifest = manifest_of(schema(&[TypeId::Int64]), entries);
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();

        for predicate in [
            cmp(0, CompareOp::Eq, LakeValue::Int(5)),
            cmp(0, CompareOp::Lt, LakeValue::Int(5)),
            cmp(0, CompareOp::Eq, LakeValue::Null),
            LakePredicate::IsNull { column_id: 0 },
            LakePredicate::IsNotNull { column_id: 0 },
        ] {
            let expected = scalar_mask(&manifest, &predicate);
            let (mask, _) = index.cannot_match(&predicate, &mut scratch);
            assert_eq!(mask, &expected[..], "mask differs for {:?}", predicate);
        }
    }

    #[test]
    fn test_projection_stays_inside_its_byte_budget() {
        // Enough columns and files that the caps bind
        let types = vec![TypeId::Int64; 40];
        let entries: Vec<PartitionEntry> = (0..64)
            .map(|i| {
                let stats = (0..40)
                    .map(|c| {
                        bounded(
                            c,
                            Some(LakeValue::Int(i as i64)),
                            Some(LakeValue::Int(i as i64 + 1)),
                            0,
                            10,
                        )
                    })
                    .collect();
                entry(i as u64, stats, 10)
            })
            .collect();
        let manifest = manifest_of(schema(&types), entries);
        let index = PruneIndex::build(&manifest);
        assert_eq!(index.indexed_columns().len(), MAX_INDEXED_COLUMNS);
        assert!(index.heap_bytes() <= MAX_INDEX_BYTES);
        assert!(
            index.indexed_columns().windows(2).all(|w| w[0] < w[1]),
            "indexed columns are ascending so the lookup can binary search"
        );
    }

    #[test]
    fn test_a_warm_scratch_sweeps_without_growing() {
        let manifest = int_manifest(4_096);
        let index = PruneIndex::build(&manifest);
        let mut scratch = PruneScratch::new();
        let predicate = LakePredicate::And(vec![
            cmp(0, CompareOp::GtEq, LakeValue::Int(1_000)),
            LakePredicate::Or(vec![
                cmp(0, CompareOp::Lt, LakeValue::Int(2_000)),
                LakePredicate::In {
                    column_id: 0,
                    values: vec![LakeValue::Int(9_000), LakeValue::Int(9_100)],
                },
            ]),
        ]);

        let _ = index.cannot_match(&predicate, &mut scratch);
        let warm = scratch.capacity();
        for _ in 0..8 {
            let _ = index.cannot_match(&predicate, &mut scratch);
        }
        assert_eq!(scratch.capacity(), warm, "a warm sweep allocates nothing");
    }
}
