// Grouping set expansion for ROLLUP, CUBE, and explicit GROUPING SETS
// Single pass execution: each input row is grouped against every requested
// set in one scan rather than re-running the input for each set

use crate::value::{AnalyticsValue, VerifiedKeyMap, hash_value_into};
use zyron_common::mix_finalize_2round;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupingSetType {
    // Hierarchical subtotals: ROLLUP(a,b,c) -> (a,b,c), (a,b), (a), ()
    Rollup(Vec<String>),
    // Power set: CUBE(a,b) -> (a,b), (a), (b), ()
    Cube(Vec<String>),
    // Explicit list of grouping column tuples
    GroupingSets(Vec<Vec<String>>),
}

impl GroupingSetType {
    // Expand the requested grouping descriptor into the explicit list of
    // grouping sets that the executor must produce. Each set is a list of
    // column names. The empty set represents the grand total row.
    pub fn expand(&self) -> Vec<Vec<String>> {
        match self {
            GroupingSetType::Rollup(cols) => expand_rollup(cols),
            GroupingSetType::Cube(cols) => expand_cube(cols),
            GroupingSetType::GroupingSets(sets) => sets.clone(),
        }
    }

    // The full set of distinct columns referenced across all grouping sets
    pub fn columns(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for set in self.expand() {
            for c in set {
                if !seen.contains(&c) {
                    seen.push(c);
                }
            }
        }
        seen
    }
}

fn expand_rollup(cols: &[String]) -> Vec<Vec<String>> {
    // ROLLUP(a,b,c) yields prefixes from full down to empty
    let mut out = Vec::with_capacity(cols.len() + 1);
    for i in (0..=cols.len()).rev() {
        out.push(cols[..i].to_vec());
    }
    out
}

fn expand_cube(cols: &[String]) -> Vec<Vec<String>> {
    // CUBE(a,b,...) yields every subset ordered from full set down to empty
    let n = cols.len();
    let total = 1usize << n;
    let mut out = Vec::with_capacity(total);
    // Iterate masks in descending order so the full grouping appears first
    // and the grand total last, matching SQL convention
    for mask in (0..total).rev() {
        let mut set = Vec::new();
        for (i, c) in cols.iter().enumerate() {
            if mask & (1 << i) != 0 {
                set.push(c.clone());
            }
        }
        out.push(set);
    }
    out
}

// Free function form for ergonomic external use
pub fn expand_grouping_sets(spec: &GroupingSetType) -> Vec<Vec<String>> {
    spec.expand()
}

// Stable, hashable row key produced by selecting only the grouped columns
// from a row. NULLs are represented as AnalyticsValue::Null.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RowKey {
    pub set_index: u32,
    pub values: Vec<AnalyticsValue>,
}

// GROUPING(col): returns 1 if the column is aggregated away in this row
pub fn grouping_bit(active_columns: &[String], column_name: &str) -> i32 {
    if active_columns.iter().any(|c| c == column_name) {
        0
    } else {
        1
    }
}

// GROUPING_ID(c1, c2, ...): bitmask of grouping bits, c1 is the highest bit
pub fn grouping_id_bits(active_columns: &[String], requested: &[String]) -> i64 {
    let mut bits: i64 = 0;
    for (idx, col) in requested.iter().enumerate() {
        let bit_pos = requested.len() - 1 - idx;
        if grouping_bit(active_columns, col) == 1 {
            bits |= 1i64 << bit_pos;
        }
    }
    bits
}

// Trait describing an aggregator that can ingest one row's worth of input
// and emit a final value. Used by the single-pass runner. Aggregators are
// monomorphised into the runner via a generic parameter, so the per-row
// `ingest` call is a direct (inlinable) call rather than a vtable hop and
// the per-group state lives inline in the bucket - no Box per group.
//
// The `box_clone` method on the trait remains so callers that genuinely
// need a heterogeneous list of aggregators (the rare case) can still hold
// `Box<dyn Aggregator>` values; the runner does not require it.
pub trait Aggregator: Send + Clone {
    fn ingest(&mut self, row: &[AnalyticsValue]);
    fn finalise(&self) -> AnalyticsValue;
    /// Type-erased clone. Optional for callers that hold a Vec of mixed
    /// aggregator types behind a Box; the GroupingSetsRunner does not use
    /// this method directly.
    fn box_clone(&self) -> Box<dyn AggregatorObject>;
}

/// Object-safe view of `Aggregator` for the rare callers that need a
/// heterogeneous list. Splitting it out keeps the main `Aggregator` trait
/// non-object-safe (it has Clone) so the runner can still monomorphise.
pub trait AggregatorObject: Send {
    fn ingest(&mut self, row: &[AnalyticsValue]);
    fn finalise(&self) -> AnalyticsValue;
    fn box_clone(&self) -> Box<dyn AggregatorObject>;
}

impl<T: Aggregator + 'static> AggregatorObject for T {
    fn ingest(&mut self, row: &[AnalyticsValue]) {
        Aggregator::ingest(self, row)
    }
    fn finalise(&self) -> AnalyticsValue {
        Aggregator::finalise(self)
    }
    fn box_clone(&self) -> Box<dyn AggregatorObject> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AggregatorObject> {
    fn clone(&self) -> Self {
        AggregatorObject::box_clone(&**self)
    }
}

// SUM aggregator over a single column index
#[derive(Debug, Clone)]
pub struct SumAgg {
    pub col_index: usize,
    pub total: f64,
    pub count: u64,
}

impl SumAgg {
    pub fn new(col_index: usize) -> Self {
        Self {
            col_index,
            total: 0.0,
            count: 0,
        }
    }
}

impl Aggregator for SumAgg {
    fn ingest(&mut self, row: &[AnalyticsValue]) {
        if let Some(v) = row.get(self.col_index).and_then(|v| v.as_f64()) {
            self.total += v;
            self.count += 1;
        }
    }
    fn finalise(&self) -> AnalyticsValue {
        if self.count == 0 {
            AnalyticsValue::Null
        } else {
            AnalyticsValue::Float(self.total)
        }
    }
    fn box_clone(&self) -> Box<dyn AggregatorObject> {
        Box::new(self.clone())
    }
}

// COUNT(*) aggregator
#[derive(Debug, Clone, Default)]
pub struct CountStarAgg {
    pub n: u64,
}

impl Aggregator for CountStarAgg {
    fn ingest(&mut self, _row: &[AnalyticsValue]) {
        self.n += 1;
    }
    fn finalise(&self) -> AnalyticsValue {
        AnalyticsValue::Int(self.n as i64)
    }
    fn box_clone(&self) -> Box<dyn AggregatorObject> {
        Box::new(self.clone())
    }
}

// AVG aggregator
#[derive(Debug, Clone)]
pub struct AvgAgg {
    pub col_index: usize,
    pub total: f64,
    pub count: u64,
}

impl AvgAgg {
    pub fn new(col_index: usize) -> Self {
        Self {
            col_index,
            total: 0.0,
            count: 0,
        }
    }
}

impl Aggregator for AvgAgg {
    fn ingest(&mut self, row: &[AnalyticsValue]) {
        if let Some(v) = row.get(self.col_index).and_then(|v| v.as_f64()) {
            self.total += v;
            self.count += 1;
        }
    }
    fn finalise(&self) -> AnalyticsValue {
        if self.count == 0 {
            AnalyticsValue::Null
        } else {
            AnalyticsValue::Float(self.total / self.count as f64)
        }
    }
    fn box_clone(&self) -> Box<dyn AggregatorObject> {
        Box::new(self.clone())
    }
}

// MIN aggregator
#[derive(Debug, Clone)]
pub struct MinAgg {
    pub col_index: usize,
    pub current: Option<AnalyticsValue>,
}

impl MinAgg {
    pub fn new(col_index: usize) -> Self {
        Self {
            col_index,
            current: None,
        }
    }
}

impl Aggregator for MinAgg {
    fn ingest(&mut self, row: &[AnalyticsValue]) {
        if let Some(v) = row.get(self.col_index) {
            if v.is_null() {
                return;
            }
            if let Some(slot) = self.current.as_mut() {
                if v.total_cmp(slot) == std::cmp::Ordering::Less {
                    slot.assign_from(v);
                }
            } else {
                self.current = Some(v.clone());
            }
        }
    }
    fn finalise(&self) -> AnalyticsValue {
        self.current.clone().unwrap_or(AnalyticsValue::Null)
    }
    fn box_clone(&self) -> Box<dyn AggregatorObject> {
        Box::new(self.clone())
    }
}

// MAX aggregator
#[derive(Debug, Clone)]
pub struct MaxAgg {
    pub col_index: usize,
    pub current: Option<AnalyticsValue>,
}

impl MaxAgg {
    pub fn new(col_index: usize) -> Self {
        Self {
            col_index,
            current: None,
        }
    }
}

impl Aggregator for MaxAgg {
    fn ingest(&mut self, row: &[AnalyticsValue]) {
        if let Some(v) = row.get(self.col_index) {
            if v.is_null() {
                return;
            }
            if let Some(slot) = self.current.as_mut() {
                if v.total_cmp(slot) == std::cmp::Ordering::Greater {
                    slot.assign_from(v);
                }
            } else {
                self.current = Some(v.clone());
            }
        }
    }
    fn finalise(&self) -> AnalyticsValue {
        self.current.clone().unwrap_or(AnalyticsValue::Null)
    }
    fn box_clone(&self) -> Box<dyn AggregatorObject> {
        Box::new(self.clone())
    }
}

// Single pass expander: input columns are positional, each grouping set is
// a list of column names and is mapped to indices once at construction.
pub struct GroupingSetExpander {
    pub schema: Vec<String>,
    // Each entry is (set_index, column_indices_within_schema)
    pub sets: Vec<(usize, Vec<usize>)>,
}

impl GroupingSetExpander {
    pub fn new(schema: Vec<String>, spec: &GroupingSetType) -> Self {
        Self::from_expanded(schema, spec.expand())
    }

    // Build directly from an already-expanded list of grouping sets so
    // callers that need both the expanded form and the index map (such as
    // GroupingSetsRunner) avoid re-expanding the spec twice.
    pub fn from_expanded(schema: Vec<String>, expanded: Vec<Vec<String>>) -> Self {
        let mut sets = Vec::with_capacity(expanded.len());
        for (i, set_cols) in expanded.into_iter().enumerate() {
            let mut indices = Vec::with_capacity(set_cols.len());
            for col in set_cols {
                if let Some(pos) = schema.iter().position(|c| c == &col) {
                    indices.push(pos);
                }
            }
            sets.push((i, indices));
        }
        Self { schema, sets }
    }

    // Produce the per-set keys for a single input row
    pub fn keys_for_row(&self, row: &[AnalyticsValue]) -> Vec<RowKey> {
        let mut out = Vec::with_capacity(self.sets.len());
        for (set_idx, indices) in &self.sets {
            let mut values = Vec::with_capacity(indices.len());
            for &i in indices {
                values.push(row.get(i).cloned().unwrap_or(AnalyticsValue::Null));
            }
            out.push(RowKey {
                set_index: *set_idx as u32,
                values,
            });
        }
        out
    }
}

// Single pass runner: given a schema, a grouping spec, and an aggregator
// template, produces output rows for every grouping set in one input scan.
//
// Generic over A so the per-row `ingest` is a direct (inlinable) method
// call. Each bucket holds an A inline, no heap Box per group.
pub struct GroupingSetsRunner<A: Aggregator> {
    pub expander: GroupingSetExpander,
    pub spec: GroupingSetType,
    pub expanded_sets: Vec<Vec<String>>,
    state: VerifiedKeyMap<(u32, Vec<AnalyticsValue>), A>,
    template: A,
}

impl<A: Aggregator> GroupingSetsRunner<A> {
    pub fn new(schema: Vec<String>, spec: GroupingSetType, template: A) -> Self {
        // Expand the spec once and share the result between the expander
        // (which needs the column->index map) and the runner (which needs
        // the original column-name lists for output rows).
        let expanded_sets = spec.expand();
        let expander = GroupingSetExpander::from_expanded(schema, expanded_sets.clone());
        Self {
            expander,
            spec,
            expanded_sets,
            state: VerifiedKeyMap::new(),
            template,
        }
    }

    pub fn ingest_row(&mut self, row: &[AnalyticsValue]) {
        // Per row: stream-hash each grouping set's key columns into a
        // 128-bit (low, high) pair and look up the VerifiedKeyMap. The
        // map verifies via the high half on every access; full 128-bit
        // collision probability is negligible at any practical scale.
        // The owned key Vec is only built on the miss path so the hot
        // hit path does no allocation per ingest.
        const SEED_LOW: u64 = 0x9E37_79B9_7F4A_7C15;
        const SEED_HIGH: u64 = 0xBF58_476D_1CE4_E5B9;
        for (set_idx, indices) in &self.expander.sets {
            let mut h_low: u64 = (*set_idx as u64).wrapping_add(SEED_LOW);
            let mut h_high: u64 = (*set_idx as u64).wrapping_add(SEED_HIGH);
            for &i in indices {
                let v = row.get(i).unwrap_or(&AnalyticsValue::Null);
                h_low = hash_value_into(h_low, v);
                h_high = hash_value_into(h_high, v);
            }
            let key_low = mix_finalize_2round(h_low);
            let key_high = mix_finalize_2round(h_high);
            let template = &self.template;
            let set_idx_copy = *set_idx as u32;
            let agg = self.state.entry_or_insert(
                key_low,
                key_high,
                || {
                    let mut owned: Vec<AnalyticsValue> = Vec::with_capacity(indices.len());
                    for &i in indices {
                        owned.push(row.get(i).cloned().unwrap_or(AnalyticsValue::Null));
                    }
                    (set_idx_copy, owned)
                },
                || template.clone(),
            );
            agg.ingest(row);
        }
    }

    // Emit one output row per (grouping set, key tuple). Each output row
    // has the full schema as columns: columns not present in the active
    // grouping set are emitted as NULL, plus the aggregate result and
    // GROUPING_ID for the active set.
    pub fn finalise(self) -> Vec<GroupingSetOutput> {
        let mut out = Vec::with_capacity(self.state.len());
        let expanded = self.expanded_sets;
        for ((set_idx, key_values), agg) in self.state.into_iter() {
            let active_cols = expanded.get(set_idx as usize).cloned().unwrap_or_default();
            out.push(GroupingSetOutput {
                set_index: set_idx,
                active_columns: active_cols,
                key_values,
                aggregate: agg.finalise(),
            });
        }
        // Stable order: by set_index, then by key tuple
        out.sort_by(|a, b| {
            a.set_index
                .cmp(&b.set_index)
                .then_with(|| a.key_values.cmp(&b.key_values))
        });
        out
    }
}

#[derive(Debug, Clone)]
pub struct GroupingSetOutput {
    pub set_index: u32,
    pub active_columns: Vec<String>,
    pub key_values: Vec<AnalyticsValue>,
    pub aggregate: AnalyticsValue,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> String {
        x.to_string()
    }

    #[test]
    fn rollup_expands_prefixes() {
        let spec = GroupingSetType::Rollup(vec![s("a"), s("b"), s("c")]);
        let sets = spec.expand();
        assert_eq!(sets.len(), 4);
        assert_eq!(sets[0], vec![s("a"), s("b"), s("c")]);
        assert_eq!(sets[1], vec![s("a"), s("b")]);
        assert_eq!(sets[2], vec![s("a")]);
        assert_eq!(sets[3], Vec::<String>::new());
    }

    #[test]
    fn cube_expands_power_set() {
        let spec = GroupingSetType::Cube(vec![s("a"), s("b")]);
        let sets = spec.expand();
        assert_eq!(sets.len(), 4);
        // descending mask order produces (a,b), (b), (a), ()
        assert_eq!(sets[0], vec![s("a"), s("b")]);
        assert_eq!(sets[3], Vec::<String>::new());
    }

    #[test]
    fn grouping_bit_and_id() {
        let active = vec![s("region")];
        assert_eq!(grouping_bit(&active, "region"), 0);
        assert_eq!(grouping_bit(&active, "city"), 1);
        let id = grouping_id_bits(&active, &[s("region"), s("city")]);
        // active=region, requested=[region, city] -> bits=01 (city is grouped away)
        assert_eq!(id, 0b01);
    }

    #[test]
    fn single_pass_rollup() {
        let schema = vec![s("region"), s("city"), s("revenue")];
        let spec = GroupingSetType::Rollup(vec![s("region"), s("city")]);
        let mut runner = GroupingSetsRunner::new(schema, spec, SumAgg::new(2));

        let rows = vec![
            vec![
                AnalyticsValue::Text(s("US")),
                AnalyticsValue::Text(s("NYC")),
                AnalyticsValue::Float(1000.0),
            ],
            vec![
                AnalyticsValue::Text(s("US")),
                AnalyticsValue::Text(s("LA")),
                AnalyticsValue::Float(800.0),
            ],
            vec![
                AnalyticsValue::Text(s("EU")),
                AnalyticsValue::Text(s("LON")),
                AnalyticsValue::Float(500.0),
            ],
        ];
        for r in &rows {
            runner.ingest_row(r);
        }
        let out = runner.finalise();
        // 3 groups for full set + 2 region subtotals + 1 grand total
        assert_eq!(out.len(), 3 + 2 + 1);
        let grand = out.iter().find(|o| o.active_columns.is_empty()).unwrap();
        match &grand.aggregate {
            AnalyticsValue::Float(v) => assert!((v - 2300.0).abs() < 1e-9),
            _ => panic!(),
        }
    }
}
