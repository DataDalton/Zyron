//! Cost model for query plan optimization.
//!
//! Estimates plan costs using catalog statistics (histograms, NDV, null fractions)
//! for selectivity estimation and cardinality calculations. The cost model
//! compares alternative physical plans to select the cheapest execution strategy.

use crate::binder::BoundExpr;
use crate::logical::{JoinCondition, LogicalPlan};
use zyron_catalog::{Catalog, ColumnStats, TableStats};
use zyron_parser::ast::JoinType;

/// One network round trip to a peer, in the same units as a page read.
/// Reaching another node costs about as much as reading a hundred local
/// pages, which is what makes a foreign scan worth pushing work into rather
/// than issuing twice.
const FOREIGN_ROUND_TRIP_COST: f64 = 100.0;

/// The floor on how much of its own table a peer reads under a pushed
/// predicate. No filter eliminates every file, and a factor of zero would
/// make an arbitrarily selective foreign scan look free.
const MIN_REMOTE_WORK: f64 = 0.01;

/// Bytes a column contributes to a returned row, averaged over types. Used
/// only to make a narrow projection cost less than a wide one, so the exact
/// figure matters less than the ratio between projections.
const AVG_COLUMN_BYTES: f64 = 16.0;

/// Bytes a row of unknown width is assumed to hold when deciding whether an
/// operator would spill.
///
/// Four columns at the average above. Every alternative the planner compares
/// is measured with the same figure, so it shifts where spilling starts
/// rather than which of two spilling plans looks cheaper.
pub const ASSUMED_ROW_BYTES: f64 = AVG_COLUMN_BYTES * 4.0;

/// Bytes a page holds, for turning spilled bytes into page reads and writes.
const SPILL_PAGE_BYTES: f64 = 8192.0;

/// Rows one input row expands into when nothing recorded a length. Applies
/// to an array column with no statistics and to every document walk, whose
/// shape no statistic describes.
const DEFAULT_EXPANSION_FANOUT: f64 = 4.0;

/// Bytes an encoded array spends before its first element: the type id, the
/// flags, the element width and the element count.
const ARRAY_HEADER_BYTES: f64 = 8.0;

/// The share of left rows an ASOF join matches when neither match column has
/// a histogram. A time series joined to its own quotes matches nearly always,
/// so assuming most rows match is closer than assuming half do.
const DEFAULT_ASOF_MATCH_PROBABILITY: f64 = 0.9;

/// The table and element type behind a column reference, found by walking to
/// the scan that produces it. None when the reference does not resolve to a
/// base table below this plan.
fn scan_column_source(
    plan: &LogicalPlan,
    reference: &crate::binder::ColumnRef,
) -> Option<(zyron_catalog::TableId, zyron_common::TypeId)> {
    if let LogicalPlan::Scan {
        table_id,
        table_idx,
        columns,
        ..
    } = plan
        && *table_idx == reference.table_idx
    {
        let column = columns
            .iter()
            .find(|c| c.column_id == reference.column_id)?;
        return Some((*table_id, column.type_id));
    }
    plan.children()
        .into_iter()
        .find_map(|child| scan_column_source(child, reference))
}

/// How far apart two encoded values are, read as a big-endian magnitude over
/// the leading bytes. Only the ratio between two spans is used, so reading a
/// prefix is enough to order and to size them.
fn byte_span(low: &[u8], high: &[u8]) -> f64 {
    let read = |bytes: &[u8]| -> f64 {
        let mut value = 0.0f64;
        for i in 0..8 {
            value = value * 256.0 + bytes.get(i).copied().unwrap_or(0) as f64;
        }
        value
    };
    (read(high) - read(low)).max(0.0)
}

// ---------------------------------------------------------------------------
// Plan cost
// ---------------------------------------------------------------------------

/// Estimated cost of executing a plan node.
#[derive(Debug, Clone, Copy)]
pub struct PlanCost {
    pub io_cost: f64,
    pub cpu_cost: f64,
    pub row_count: f64,
}

impl PlanCost {
    /// Total cost with IO weighted higher than CPU.
    pub fn total(&self) -> f64 {
        self.io_cost + self.cpu_cost * 0.01
    }

    pub fn zero() -> Self {
        Self {
            io_cost: 0.0,
            cpu_cost: 0.0,
            row_count: 0.0,
        }
    }

    pub fn add(&self, other: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: self.io_cost + other.io_cost,
            cpu_cost: self.cpu_cost + other.cpu_cost,
            row_count: self.row_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Cost component breakdown
// ---------------------------------------------------------------------------

/// Detailed cost breakdown by resource type for hardware-aware plan comparison.
#[derive(Debug, Clone, Copy)]
pub struct CostComponent {
    /// Sequential page reads (base weight: 1.0).
    pub seq_io: f64,
    /// Random page reads (base weight: 4.0).
    pub random_io: f64,
    /// Per-tuple processing cost (base weight: 0.01).
    pub cpu_tuple: f64,
    /// Per-operator evaluation cost (base weight: 0.0025).
    pub cpu_operator: f64,
    /// Per-byte network transfer cost (base weight: 0.0001).
    pub network: f64,
    /// Per-byte working memory pressure (base weight: 0.00001).
    pub memory: f64,
}

impl CostComponent {
    pub fn zero() -> Self {
        Self {
            seq_io: 0.0,
            random_io: 0.0,
            cpu_tuple: 0.0,
            cpu_operator: 0.0,
            network: 0.0,
            memory: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding cost parameters
// ---------------------------------------------------------------------------

/// Cost parameters for columnar-encoded scans. Controls how zone maps,
/// bloom filters, and encoding-specific evaluation affect scan cost.
#[derive(Debug, Clone, Copy)]
pub struct EncodingCostParameters {
    /// Fraction of segments skipped via zone maps or bloom filters (0.0 to 1.0).
    pub skip_rate: f64,
    /// CPU cost to decode one value from the encoded format.
    pub decode_cost_per_value: f64,
    /// Speedup ratio of encoded scan vs unencoded scan (< 1.0 means faster).
    pub encoded_scan_speedup: f64,
}

impl Default for EncodingCostParameters {
    fn default() -> Self {
        Self {
            skip_rate: 0.0,
            decode_cost_per_value: 0.0001,
            encoded_scan_speedup: 0.3,
        }
    }
}

/// Pre-defined encoding decode costs (relative to unencoded = 1.0).
pub mod encoding_costs {
    use super::EncodingCostParameters;

    pub fn fastlanes() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.00015,
            encoded_scan_speedup: 0.15,
        }
    }

    pub fn dictionary() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.0001,
            encoded_scan_speedup: 0.10,
        }
    }

    pub fn rle() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.00005,
            encoded_scan_speedup: 0.05,
        }
    }

    pub fn bitpack() -> EncodingCostParameters {
        EncodingCostParameters {
            skip_rate: 0.0,
            decode_cost_per_value: 0.00008,
            encoded_scan_speedup: 0.08,
        }
    }
}

// ---------------------------------------------------------------------------
// Default selectivity constants
// ---------------------------------------------------------------------------

pub const DEFAULT_EQUALITY_SELECTIVITY: f64 = 0.1;
pub const DEFAULT_RANGE_SELECTIVITY: f64 = 0.33;
pub const DEFAULT_LIKE_SELECTIVITY: f64 = 0.2;
pub const DEFAULT_IN_LIST_SELECTIVITY: f64 = 0.05;
pub const DEFAULT_NULL_SELECTIVITY: f64 = 0.05;

/// Index scan is preferred when selectivity is below this threshold.
pub const INDEX_SCAN_SELECTIVITY_THRESHOLD: f64 = 0.10;

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Cost estimation using catalog statistics.
#[derive(Debug, Clone)]
pub struct CostModel {
    pub seq_page_cost: f64,
    pub random_page_cost: f64,
    pub cpu_tuple_cost: f64,
    pub cpu_index_tuple_cost: f64,
    pub cpu_operator_cost: f64,
    /// Per-byte cost for network transfer (local queries use 0.0001).
    pub network_cost_per_byte: f64,
    /// Per-byte cost for working memory pressure (hash tables, sort buffers).
    pub memory_cost_per_byte: f64,
    /// Per-tuple cost for transferring tuples between parallel workers.
    pub parallel_tuple_cost: f64,
    /// Fixed startup cost for launching parallel workers.
    pub parallel_setup_cost: f64,
    /// Bytes one query's materializing operators may hold before they start
    /// writing to disk. Zero means nothing bounds them, which is the default
    /// and means no plan is costed as spilling.
    pub working_memory_bytes: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            seq_page_cost: 1.0,
            random_page_cost: 4.0,
            cpu_tuple_cost: 0.01,
            cpu_index_tuple_cost: 0.005,
            cpu_operator_cost: 0.0025,
            network_cost_per_byte: 0.0001,
            memory_cost_per_byte: 0.00001,
            parallel_tuple_cost: 0.1,
            parallel_setup_cost: 1000.0,
            working_memory_bytes: 0.0,
        }
    }
}

impl CostModel {
    // -----------------------------------------------------------------------
    // Cost component decomposition
    // -----------------------------------------------------------------------

    /// Breaks a PlanCost into a detailed CostComponent for hardware-aware comparison.
    pub fn decompose(&self, plan_cost: &PlanCost) -> CostComponent {
        CostComponent {
            seq_io: plan_cost.io_cost,
            random_io: 0.0,
            cpu_tuple: plan_cost.row_count * self.cpu_tuple_cost,
            cpu_operator: plan_cost.cpu_cost - plan_cost.row_count * self.cpu_tuple_cost,
            network: 0.0,
            memory: 0.0,
        }
    }

    /// Computes a total cost from components weighted by hardware-specific parameters.
    pub fn weighted_total(&self, components: &CostComponent) -> f64 {
        components.seq_io * self.seq_page_cost
            + components.random_io * self.random_page_cost
            + components.cpu_tuple * self.cpu_tuple_cost
            + components.cpu_operator * self.cpu_operator_cost
            + components.network * self.network_cost_per_byte
            + components.memory * self.memory_cost_per_byte
    }

    // -----------------------------------------------------------------------
    // Encoding-aware scan cost
    // -----------------------------------------------------------------------

    /// Estimates the cost of a columnar-encoded scan using encoding parameters.
    /// Accounts for segment skipping (zone maps, bloom filters) and decode overhead.
    pub fn cost_encoded_scan(
        &self,
        stats: &TableStats,
        params: &EncodingCostParameters,
    ) -> PlanCost {
        let rows = stats.row_count as f64;
        let pages = stats.page_count as f64;

        // IO: sequential scan with skip_rate reducing pages read
        let pages_read = pages * (1.0 - params.skip_rate);
        let io_cost = pages_read * self.seq_page_cost * params.encoded_scan_speedup;

        // CPU: decode cost for non-skipped rows
        let rows_read = rows * (1.0 - params.skip_rate);
        let cpu_cost = rows_read * params.decode_cost_per_value + rows_read * self.cpu_tuple_cost;

        PlanCost {
            io_cost,
            cpu_cost,
            row_count: rows_read,
        }
    }

    // -----------------------------------------------------------------------
    // Parallel cost estimation
    // -----------------------------------------------------------------------

    /// Estimates the cost of a parallel sequential scan split across workers.
    pub fn cost_parallel_scan(&self, stats: &TableStats, num_workers: usize) -> PlanCost {
        let workers = (num_workers as f64).max(1.0);
        let rows = stats.row_count as f64;
        let pages = stats.page_count as f64;

        // IO is divided among workers (each reads a partition of pages)
        let io_cost = (pages / workers) * self.seq_page_cost;

        // CPU divided among workers plus per-tuple coordination overhead
        let cpu_cost = (rows / workers) * self.cpu_tuple_cost
            + rows * self.parallel_tuple_cost
            + self.parallel_setup_cost;

        PlanCost {
            io_cost,
            cpu_cost,
            row_count: rows,
        }
    }

    /// Estimates the cost of a parallel hash join with partitioned build and probe.
    pub fn cost_parallel_hash_join(
        &self,
        left: &PlanCost,
        right: &PlanCost,
        num_workers: usize,
    ) -> PlanCost {
        let workers = (num_workers as f64).max(1.0);

        // Build hash table in parallel (smaller side)
        let (build, probe) = if left.row_count <= right.row_count {
            (left, right)
        } else {
            (right, left)
        };

        // IO from both sides. This join partitions in memory, so both inputs
        // are held whole before any of it starts: past the budget it hands
        // the work to the serial join, which partitions onto disk and gives
        // up the parallelism this cost was granted for. Charged as the serial
        // join is charged, so the plan that fits wins where it should
        let build_bytes = Self::working_bytes(build.row_count);
        let spill = if build_bytes > self.working_memory_bytes && self.working_memory_bytes > 0.0 {
            self.spill_io(build_bytes, 2.0)
                + self.spill_io(Self::working_bytes(probe.row_count), 2.0)
        } else {
            0.0
        };
        let io_cost = left.io_cost + right.io_cost + spill;

        // CPU: build and probe divided by workers, plus coordination overhead
        let build_cpu = (build.row_count / workers) * self.cpu_operator_cost;
        let probe_cpu = (probe.row_count / workers) * self.cpu_operator_cost;
        let coordination = (left.row_count + right.row_count) * self.parallel_tuple_cost;
        let cpu_cost = build_cpu
            + probe_cpu
            + coordination
            + left.cpu_cost
            + right.cpu_cost
            + self.parallel_setup_cost;

        PlanCost {
            io_cost,
            cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    // -----------------------------------------------------------------------
    // Selectivity estimation
    // -----------------------------------------------------------------------

    /// Estimates selectivity of a predicate on a table.
    /// Delegates to CardinalityEstimator for MCV + histogram + NDV based estimation.
    pub fn estimate_selectivity(
        &self,
        predicate: &BoundExpr,
        table_stats: Option<&TableStats>,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        crate::optimizer::cardinality::CardinalityEstimator::estimate_selectivity(
            predicate,
            table_stats,
            column_stats,
        )
    }

    // -----------------------------------------------------------------------
    // Operator cost estimation
    // -----------------------------------------------------------------------

    /// Estimates the cost of a sequential table scan.
    pub fn cost_seq_scan(&self, stats: &TableStats) -> PlanCost {
        PlanCost {
            io_cost: stats.page_count as f64 * self.seq_page_cost,
            cpu_cost: stats.row_count as f64 * self.cpu_tuple_cost,
            row_count: stats.row_count as f64,
        }
    }

    /// Estimates the cost of reading a table on a peer.
    ///
    /// Two things dominate and neither is local IO: the round trip, which is
    /// paid once whatever the query asks for, and the bytes coming back,
    /// which the pushed projection and predicate are what reduce. So the
    /// estimate is a fixed latency term plus a transfer term over the rows
    /// the remote is expected to return.
    ///
    /// The peer's mode decides how much a pushed predicate is worth, which
    /// is the whole reason the mode is tracked. A `lake` peer prunes files
    /// with it and never reads what it skips, so a selective filter cuts the
    /// remote's own work roughly in proportion. A `db` peer walks an index:
    /// a selective filter is cheap there too, but a broad one degrades to a
    /// heap scan on the far side, so the saving flattens out as selectivity
    /// rises. An unknown mode is costed as `db`, the more pessimistic of the
    /// two, because guessing the cheaper one would let a plan commit to a
    /// pushdown the peer cannot honor.
    pub fn cost_foreign_scan(
        &self,
        mode: Option<zyron_common::DeploymentMode>,
        stats: &TableStats,
        selectivity: f64,
        projected_columns: usize,
        pushed: bool,
    ) -> PlanCost {
        let selectivity = selectivity.clamp(0.0, 1.0);
        let rows = (stats.row_count as f64 * selectivity).max(1.0);

        // What the remote saves by applying the filter itself, rather than
        // returning rows for this node to discard. Absent a pushed
        // predicate it saves nothing and reads everything
        let remote_factor = if !pushed {
            1.0
        } else {
            match mode {
                // File pruning: skipped files are never opened, so the
                // remote's work tracks selectivity closely
                Some(zyron_common::DeploymentMode::Lake) => selectivity.max(MIN_REMOTE_WORK),
                // An index walk saves most of the work when the filter is
                // narrow and little when it is broad, so the curve flattens
                // toward a full scan rather than following selectivity down
                _ => selectivity.sqrt().max(MIN_REMOTE_WORK),
            }
        };

        // The peer pays for its own read, and this node waits on it, so the
        // remote's work is part of this plan's cost rather than free
        let remote_io = stats.page_count as f64 * self.seq_page_cost * remote_factor;

        // Only the projected columns cross the wire, which is why a narrow
        // projection against a wide table is worth pushing
        let width = projected_columns.max(1) as f64 * AVG_COLUMN_BYTES;
        let transfer = rows * width * self.network_cost_per_byte;

        PlanCost {
            io_cost: FOREIGN_ROUND_TRIP_COST + remote_io + transfer,
            cpu_cost: rows * self.cpu_tuple_cost,
            row_count: rows,
        }
    }

    /// Estimates the cost of an index scan.
    pub fn cost_index_scan(&self, stats: &TableStats, selectivity: f64) -> PlanCost {
        let rows = (stats.row_count as f64 * selectivity).max(1.0);
        let rows_per_page = if stats.page_count > 0 {
            stats.row_count as f64 / stats.page_count as f64
        } else {
            1.0
        };
        let pages = (rows / rows_per_page).ceil().max(1.0);
        PlanCost {
            io_cost: pages * self.random_page_cost,
            cpu_cost: rows * self.cpu_index_tuple_cost + rows * self.cpu_tuple_cost,
            row_count: rows,
        }
    }

    // -----------------------------------------------------------------------
    // Spilling
    // -----------------------------------------------------------------------

    /// IO an operator pays for the part of its working set that does not fit.
    ///
    /// `passes` is how many times the spilled bytes cross the device: two for
    /// a sort, which writes runs and reads them back once, and four for a
    /// partitioned join, which writes and reads both sides. Only the excess is
    /// charged, so an operator that fits pays nothing and one that is just
    /// over pays a little, which is what keeps the model from treating the
    /// budget as a cliff.
    fn spill_io(&self, working_bytes: f64, passes: f64) -> f64 {
        if self.working_memory_bytes <= 0.0 || working_bytes <= self.working_memory_bytes {
            return 0.0;
        }
        let excess = working_bytes - self.working_memory_bytes;
        (excess / SPILL_PAGE_BYTES) * passes * self.seq_page_cost
    }

    /// Bytes an operator holds for a given number of rows.
    fn working_bytes(rows: f64) -> f64 {
        rows.max(0.0) * ASSUMED_ROW_BYTES
    }

    /// Estimates the cost of a hash join.
    pub fn cost_hash_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        // Build hash table on the smaller side, probe with larger
        let (build, probe) = if left.row_count <= right.row_count {
            (left, right)
        } else {
            (right, left)
        };
        // A build side past the budget partitions both inputs to disk and
        // reads both back, so the excess crosses the device four times. The
        // probe side is charged only when the build side spilled, because a
        // build side that fits means the probe side streams past it and is
        // never written
        let build_bytes = Self::working_bytes(build.row_count);
        let spill = if build_bytes > self.working_memory_bytes && self.working_memory_bytes > 0.0 {
            self.spill_io(build_bytes, 2.0)
                + self.spill_io(Self::working_bytes(probe.row_count), 2.0)
        } else {
            0.0
        };
        PlanCost {
            io_cost: left.io_cost + right.io_cost + spill,
            cpu_cost: build.row_count * self.cpu_operator_cost  // hash build
                + probe.row_count * self.cpu_operator_cost      // hash probe
                + left.cpu_cost + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a nested loop join.
    pub fn cost_nested_loop_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: left.io_cost + left.row_count * right.io_cost,
            cpu_cost: left.row_count * right.row_count * self.cpu_operator_cost
                + left.cpu_cost
                + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a merge join (both sides assumed sorted).
    ///
    /// A merge join over sorted inputs holds only the rows sharing the current
    /// key, so nothing here spills. What it costs is in the sorts the planner
    /// prices separately, and those do.
    pub fn cost_merge_join(&self, left: &PlanCost, right: &PlanCost) -> PlanCost {
        PlanCost {
            io_cost: left.io_cost + right.io_cost,
            cpu_cost: (left.row_count + right.row_count) * self.cpu_operator_cost
                + left.cpu_cost
                + right.cpu_cost,
            row_count: estimate_join_rows(left.row_count, right.row_count),
        }
    }

    /// Estimates the cost of a sort operation.
    ///
    /// Past the budget the sort writes sorted runs and merges them back, so
    /// the excess crosses the device twice. Cheaper per byte than a
    /// partitioned join, which is what makes a sort-based plan worth
    /// considering when a hash plan would not fit.
    pub fn cost_sort(&self, input: &PlanCost) -> PlanCost {
        let n = input.row_count.max(1.0);
        let comparisons = n * n.log2();
        PlanCost {
            io_cost: input.io_cost + self.spill_io(Self::working_bytes(input.row_count), 2.0),
            cpu_cost: input.cpu_cost + comparisons * self.cpu_operator_cost,
            row_count: input.row_count,
        }
    }

    /// Estimates the cost of a hash aggregation.
    pub fn cost_hash_aggregate(&self, input: &PlanCost, group_count: f64) -> PlanCost {
        // The table holds one entry per group, not one per input row, so a
        // grouping that collapses its input costs nothing here however large
        // that input is. Past the budget the groups are partitioned to disk
        // and read back, which is two crossings
        PlanCost {
            io_cost: input.io_cost + self.spill_io(Self::working_bytes(group_count), 2.0),
            cpu_cost: input.cpu_cost + input.row_count * self.cpu_operator_cost,
            row_count: group_count.max(1.0),
        }
    }

    // -----------------------------------------------------------------------
    // Cardinality estimation
    // -----------------------------------------------------------------------

    /// Estimates the output cardinality of a join.
    pub fn estimate_join_cardinality(
        &self,
        left_rows: f64,
        right_rows: f64,
        join_type: &JoinType,
        condition: &JoinCondition,
        _catalog: &Catalog,
    ) -> f64 {
        let base = match condition {
            JoinCondition::Cross => left_rows * right_rows,
            JoinCondition::On(_) => {
                // Equi-join estimate: use the smaller table's cardinality
                // as a rough approximation
                estimate_join_rows(left_rows, right_rows)
            }
            JoinCondition::Using(_) => estimate_join_rows(left_rows, right_rows),
            JoinCondition::Natural => estimate_join_rows(left_rows, right_rows),
        };

        // Outer joins produce at least as many rows as the preserved side
        match join_type {
            JoinType::Left => base.max(left_rows),
            JoinType::Right => base.max(right_rows),
            JoinType::Full => base.max(left_rows.max(right_rows)),
            JoinType::Inner | JoinType::Cross => base,
        }
    }

    /// How many output rows one input row expands into.
    ///
    /// An array column's mean length is derived from the column's recorded
    /// mean encoded width: the encoding is a fixed header, a presence bitmap
    /// and one slot per element, so for a fixed-width element type the count
    /// follows from the width. Without statistics, or for an element type
    /// whose width varies, the estimate is 4.
    pub(crate) fn expansion_fanout(
        &self,
        spec: &crate::logical::ExpandSpec,
        child: &LogicalPlan,
        catalog: &Catalog,
    ) -> f64 {
        use crate::logical::ExpandSpec;
        match spec {
            // A group list is fixed, so the fanout is exact
            ExpandSpec::Unpivot { groups, .. } => groups.len() as f64,
            ExpandSpec::Unnest { arrays, .. } => {
                // Several arrays zip to the longest, so the fanout is the
                // largest of their mean lengths
                arrays
                    .iter()
                    .map(|a| self.mean_array_length(a, child, catalog))
                    .fold(0.0f64, f64::max)
                    .max(1.0)
            }
            // A document walk has no length recorded anywhere, and a
            // recursive walk reaches more than a shallow one
            ExpandSpec::Flatten { recursive, .. } => {
                if *recursive {
                    DEFAULT_EXPANSION_FANOUT * DEFAULT_EXPANSION_FANOUT
                } else {
                    DEFAULT_EXPANSION_FANOUT
                }
            }
        }
    }

    /// The mean number of elements in the array a column holds, from that
    /// column's recorded mean width, or the default when nothing recorded it.
    fn mean_array_length(&self, expr: &BoundExpr, child: &LogicalPlan, catalog: &Catalog) -> f64 {
        let BoundExpr::ColumnRef(reference) = expr else {
            return DEFAULT_EXPANSION_FANOUT;
        };
        let Some((table_id, element_type)) = scan_column_source(child, reference) else {
            return DEFAULT_EXPANSION_FANOUT;
        };
        let Some(recorded) = catalog.get_stats(table_id) else {
            return DEFAULT_EXPANSION_FANOUT;
        };
        let Some(stats) = recorded
            .1
            .iter()
            .find(|c| c.column_id == reference.column_id)
        else {
            return DEFAULT_EXPANSION_FANOUT;
        };
        // Header, then a presence bit per element, then one slot per element
        let Some(width) = element_type.fixed_size().filter(|w| *w > 0) else {
            return DEFAULT_EXPANSION_FANOUT;
        };
        let payload = (stats.avg_width as f64) - ARRAY_HEADER_BYTES;
        if payload <= 0.0 {
            return DEFAULT_EXPANSION_FANOUT;
        }
        // Each element costs its width plus one eighth of a byte of bitmap
        (payload / (width as f64 + 0.125)).max(1.0)
    }

    /// The share of left rows that find a right row, from how far the two
    /// match columns' recorded ranges overlap. Without a histogram on both
    /// sides the estimate is that most left rows match.
    fn match_probability(
        &self,
        match_left: &BoundExpr,
        match_right: &BoundExpr,
        left: &LogicalPlan,
        right: &LogicalPlan,
        catalog: &Catalog,
    ) -> f64 {
        let bounds = |expr: &BoundExpr, plan: &LogicalPlan| -> Option<(Vec<u8>, Vec<u8>)> {
            let BoundExpr::ColumnRef(reference) = expr else {
                return None;
            };
            let (table_id, _) = scan_column_source(plan, reference)?;
            let recorded = catalog.get_stats(table_id)?;
            let stats = recorded
                .1
                .iter()
                .find(|c| c.column_id == reference.column_id)?;
            let histogram = stats.histogram.as_ref()?;
            let low = histogram.bounds.first()?.clone();
            let high = histogram.bounds.last()?.clone();
            Some((low, high))
        };
        let (Some((left_low, left_high)), Some((right_low, right_high))) =
            (bounds(match_left, left), bounds(match_right, right))
        else {
            return DEFAULT_ASOF_MATCH_PROBABILITY;
        };
        // The comparison is over the columns' own encoded bytes, which the
        // histogram stores and which order the same way the values do for
        // every orderable type an ASOF match accepts
        let overlap_low = left_low.clone().max(right_low);
        let overlap_high = left_high.clone().min(right_high);
        if overlap_high < overlap_low {
            // The ranges do not meet, so almost nothing matches
            return 0.01;
        }
        let span = byte_span(&left_low, &left_high);
        if span <= 0.0 {
            return DEFAULT_ASOF_MATCH_PROBABILITY;
        }
        (byte_span(&overlap_low, &overlap_high) / span).clamp(0.01, 1.0)
    }

    /// Estimates the total cost of a logical plan tree.
    pub fn estimate_plan_cost(&self, plan: &LogicalPlan, catalog: &Catalog) -> PlanCost {
        match plan {
            LogicalPlan::Scan { table_id, .. } => {
                if let Some(s) = catalog.get_stats(*table_id) {
                    self.cost_seq_scan(&s.0)
                } else {
                    // No stats: assume 1000 rows, 10 pages
                    PlanCost {
                        io_cost: 10.0 * self.seq_page_cost,
                        cpu_cost: 1000.0 * self.cpu_tuple_cost,
                        row_count: 1000.0,
                    }
                }
            }
            LogicalPlan::Filter { predicate, child } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let selectivity = self.estimate_selectivity(predicate, None, None);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: (child_cost.row_count * selectivity).max(1.0),
                }
            }
            LogicalPlan::Project { child, .. } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: child_cost.row_count,
                }
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let left_cost = self.estimate_plan_cost(left, catalog);
                let right_cost = self.estimate_plan_cost(right, catalog);
                let rows = self.estimate_join_cardinality(
                    left_cost.row_count,
                    right_cost.row_count,
                    join_type,
                    condition,
                    catalog,
                );
                PlanCost {
                    io_cost: left_cost.io_cost + right_cost.io_cost,
                    cpu_cost: left_cost.cpu_cost
                        + right_cost.cpu_cost
                        + rows * self.cpu_operator_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::LateralJoin { left, .. } => {
                // The subquery runs once per left row. Estimate its per-row cost
                // as a small constant fanout since its plan is not costed here.
                let left_cost = self.estimate_plan_cost(left, catalog);
                let per_row = self.cpu_operator_cost * 4.0;
                PlanCost {
                    io_cost: left_cost.io_cost,
                    cpu_cost: left_cost.cpu_cost + left_cost.row_count * per_row,
                    row_count: left_cost.row_count,
                }
            }
            LogicalPlan::ExpandRows {
                child,
                spec,
                outer_input,
                ..
            } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let fanout = self.expansion_fanout(spec, child, catalog);
                // An outer expansion emits a row for an input row that
                // produced none, so it never falls below the input count
                let rows = if *outer_input {
                    (child_cost.row_count * fanout).max(child_cost.row_count)
                } else {
                    (child_cost.row_count * fanout).max(1.0)
                };
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + rows * self.cpu_operator_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::AsofJoin {
                left,
                right,
                match_on,
            } => {
                let crate::logical::AsofMatchOn {
                    match_left,
                    match_right,
                    join_type,
                    ..
                } = match_on.as_ref();
                let left_cost = self.estimate_plan_cost(left, catalog);
                let right_cost = self.estimate_plan_cost(right, catalog);
                // One left row matches at most one right row, so the left
                // cardinality is the ceiling. The LEFT form keeps every left
                // row; the inner form keeps the share that finds a match,
                // which the two match columns' range overlap estimates
                let rows = match join_type {
                    JoinType::Left => left_cost.row_count,
                    _ => {
                        let probability =
                            self.match_probability(match_left, match_right, left, right, catalog);
                        (left_cost.row_count * probability).max(1.0)
                    }
                };
                // Two sorts plus one merge pass over both inputs
                let sort_cost = self.cost_sort(&left_cost).cpu_cost.max(left_cost.cpu_cost)
                    + self
                        .cost_sort(&right_cost)
                        .cpu_cost
                        .max(right_cost.cpu_cost);
                PlanCost {
                    io_cost: left_cost.io_cost + right_cost.io_cost,
                    cpu_cost: sort_cost
                        + (left_cost.row_count + right_cost.row_count) * self.cpu_operator_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::Aggregate {
                group_by, child, ..
            } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let group_count = if group_by.is_empty() {
                    1.0
                } else {
                    // Rough estimate: assume grouping reduces to sqrt(rows)
                    child_cost.row_count.sqrt().max(1.0)
                };
                self.cost_hash_aggregate(&child_cost, group_count)
            }
            LogicalPlan::Sort { child, .. } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                self.cost_sort(&child_cost)
            }
            LogicalPlan::Limit {
                limit,
                offset: _,
                child,
            } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                let rows = if let Some(l) = limit {
                    (*l as f64).min(child_cost.row_count)
                } else {
                    child_cost.row_count
                };
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost,
                    row_count: rows,
                }
            }
            LogicalPlan::Distinct { child } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: child_cost.row_count * 0.8, // Assume 20% duplicates
                }
            }
            LogicalPlan::LockRows { child, .. } => {
                let child_cost = self.estimate_plan_cost(child, catalog);
                // one lock table insert per emitted row
                PlanCost {
                    io_cost: child_cost.io_cost,
                    cpu_cost: child_cost.cpu_cost + child_cost.row_count * self.cpu_operator_cost,
                    row_count: child_cost.row_count,
                }
            }
            LogicalPlan::SetOp { left, right, .. } => {
                let left_cost = self.estimate_plan_cost(left, catalog);
                let right_cost = self.estimate_plan_cost(right, catalog);
                left_cost.add(&right_cost)
            }
            LogicalPlan::Values { rows, .. } => PlanCost {
                io_cost: 0.0,
                cpu_cost: rows.len() as f64 * self.cpu_tuple_cost,
                row_count: rows.len() as f64,
            },
            LogicalPlan::Insert { source, .. } => self.estimate_plan_cost(source, catalog),
            LogicalPlan::ViewTriggerWrite { source, .. } => {
                self.estimate_plan_cost(source, catalog)
            }
            LogicalPlan::Update { child, .. } => self.estimate_plan_cost(child, catalog),
            LogicalPlan::Delete { child, .. } => self.estimate_plan_cost(child, catalog),
            LogicalPlan::GraphAlgorithm { algorithm, .. } => {
                // Mirrors the physical builder's per-algorithm estimates using
                // a nominal graph of V=10_000 nodes and E=100_000 edges.
                let v: f64 = 10_000.0;
                let e: f64 = 100_000.0;
                let (cpu, row_count) = match algorithm.as_str() {
                    "pagerank" => (20.0 * (v + e), v),
                    "shortest_path" => (v + e, v.sqrt()),
                    "bfs" => (v + e, v),
                    "connected_components" => (v + e, v),
                    "community_detection" => (10.0 * (v + e), v),
                    "betweenness_centrality" => (v * (v + e), v),
                    _ => (v + e, v),
                };
                PlanCost {
                    io_cost: v,
                    cpu_cost: cpu,
                    row_count,
                }
            }
            LogicalPlan::AnalyticsTableFunction {
                function_name,
                output_columns,
                positional_args,
                ..
            } => {
                let nominal_rows: f64 = 10_000.0;
                let row_count = match function_name.as_str() {
                    "DATA_PROFILE" | "COLUMN_PROFILE" => output_columns.len() as f64,
                    "CORRELATION_MATRIX" => positional_args.len().pow(2) as f64,
                    _ => nominal_rows,
                };
                PlanCost {
                    io_cost: nominal_rows,
                    cpu_cost: nominal_rows * 4.0,
                    row_count,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Default join cardinality estimate when no detailed stats are available.
/// Uses the geometric mean of the two input sizes.
fn estimate_join_rows(left: f64, right: f64) -> f64 {
    (left * right).sqrt().max(1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::BoundExpr;
    use zyron_catalog::{ColumnId, TableId};
    use zyron_common::TypeId;
    use zyron_parser::ast::{BinaryOperator, LiteralValue};

    fn make_cost_model() -> CostModel {
        CostModel::default()
    }

    fn make_table_stats(rows: u64, pages: u32) -> TableStats {
        TableStats {
            table_id: TableId(1),
            row_count: rows,
            page_count: pages,
            avg_row_size: 64,
            last_analyzed: 0,
        }
    }

    #[test]
    fn test_seq_scan_cost() {
        let model = make_cost_model();
        let stats = make_table_stats(10000, 100);
        let cost = model.cost_seq_scan(&stats);
        assert_eq!(cost.io_cost, 100.0); // 100 pages * 1.0
        assert_eq!(cost.row_count, 10000.0);
    }

    #[test]
    fn test_index_scan_cost_low_selectivity() {
        let model = make_cost_model();
        let stats = make_table_stats(10000, 100);
        let cost = model.cost_index_scan(&stats, 0.01); // 1% selectivity
        assert!(cost.row_count < 200.0);
        // Random IO cost should be relatively small for low selectivity
        assert!(cost.io_cost < stats.page_count as f64 * model.seq_page_cost);
    }

    #[test]
    fn test_seq_scan_cheaper_than_index_for_high_selectivity() {
        let model = make_cost_model();
        let stats = make_table_stats(10000, 100);
        let seq_cost = model.cost_seq_scan(&stats);
        let idx_cost = model.cost_index_scan(&stats, 0.5); // 50% selectivity
        // Sequential should be cheaper for high selectivity due to random IO penalty
        assert!(seq_cost.total() < idx_cost.total());
    }

    #[test]
    fn test_selectivity_and_condition() {
        let model = make_cost_model();
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            op: BinaryOperator::And,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = model.estimate_selectivity(&pred, None, None);
        assert!((sel - 1.0).abs() < 0.001); // true AND true = 1.0
    }

    #[test]
    fn test_selectivity_or_condition() {
        let model = make_cost_model();
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                type_id: TypeId::Boolean,
            }),
            op: BinaryOperator::Or,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = model.estimate_selectivity(&pred, None, None);
        // false OR true = 0 + 1 - 0*1 = 1.0
        assert!((sel - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_equality_selectivity_with_stats() {
        let model = make_cost_model();
        let stats = vec![ColumnStats {
            table_id: TableId(1),
            column_id: ColumnId(0),
            null_fraction: 0.0,
            distinct_count: 100,
            avg_width: 8,
            histogram: None,
            most_common_values: vec![],
            most_common_freqs: vec![],
        }];
        // col0 = 42
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::ColumnRef(crate::binder::ColumnRef {
                table_idx: 0,
                column_id: ColumnId(0),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            })),
            op: BinaryOperator::Eq,
            right: Box::new(BoundExpr::Literal {
                value: LiteralValue::Integer(42),
                type_id: TypeId::Int64,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = model.estimate_selectivity(&pred, None, Some(&stats));
        assert!((sel - 0.01).abs() < 0.001); // 1/100
    }

    #[test]
    fn test_hash_join_cheaper_than_nested_loop() {
        let model = make_cost_model();
        let left = PlanCost {
            io_cost: 100.0,
            cpu_cost: 10000.0,
            row_count: 10000.0,
        };
        let right = PlanCost {
            io_cost: 50.0,
            cpu_cost: 5000.0,
            row_count: 5000.0,
        };
        let hash = model.cost_hash_join(&left, &right);
        let nl = model.cost_nested_loop_join(&left, &right);
        assert!(hash.total() < nl.total());
    }

    #[test]
    fn test_plan_cost_zero() {
        let cost = PlanCost::zero();
        assert_eq!(cost.total(), 0.0);
        assert_eq!(cost.row_count, 0.0);
    }

    #[test]
    fn test_sort_cost_increases_with_rows() {
        let model = make_cost_model();
        let small = PlanCost {
            io_cost: 10.0,
            cpu_cost: 100.0,
            row_count: 100.0,
        };
        let large = PlanCost {
            io_cost: 100.0,
            cpu_cost: 10000.0,
            row_count: 10000.0,
        };
        let small_sort = model.cost_sort(&small);
        let large_sort = model.cost_sort(&large);
        assert!(large_sort.cpu_cost > small_sort.cpu_cost);
    }

    #[test]
    fn test_cost_component_decompose() {
        let model = make_cost_model();
        let cost = PlanCost {
            io_cost: 100.0,
            cpu_cost: 50.0,
            row_count: 1000.0,
        };
        let components = model.decompose(&cost);
        assert_eq!(components.seq_io, 100.0);
        assert!((components.cpu_tuple - 1000.0 * 0.01).abs() < 0.001);
    }

    #[test]
    fn test_weighted_total() {
        let model = make_cost_model();
        let components = CostComponent {
            seq_io: 10.0,
            random_io: 5.0,
            cpu_tuple: 100.0,
            cpu_operator: 50.0,
            network: 0.0,
            memory: 0.0,
        };
        let total = model.weighted_total(&components);
        let expected = 10.0 * 1.0 + 5.0 * 4.0 + 100.0 * 0.01 + 50.0 * 0.0025;
        assert!((total - expected).abs() < 0.001);
    }

    #[test]
    fn test_encoded_scan_cheaper_with_high_skip_rate() {
        let model = make_cost_model();
        let stats = make_table_stats(100_000, 1000);
        let no_skip = model.cost_encoded_scan(
            &stats,
            &EncodingCostParameters {
                skip_rate: 0.0,
                decode_cost_per_value: 0.0001,
                encoded_scan_speedup: 0.3,
            },
        );
        let high_skip = model.cost_encoded_scan(
            &stats,
            &EncodingCostParameters {
                skip_rate: 0.9,
                decode_cost_per_value: 0.0001,
                encoded_scan_speedup: 0.3,
            },
        );
        assert!(high_skip.total() < no_skip.total());
        assert!(high_skip.row_count < no_skip.row_count);
    }

    #[test]
    fn test_parallel_scan_cheaper_for_large_tables() {
        let model = make_cost_model();
        let stats = make_table_stats(1_000_000, 10_000);
        let serial = model.cost_seq_scan(&stats);
        let parallel = model.cost_parallel_scan(&stats, 4);
        // Parallel IO should be roughly 1/4 of serial
        assert!(parallel.io_cost < serial.io_cost);
    }

    #[test]
    fn test_parallel_hash_join_cost() {
        let model = make_cost_model();
        let left = PlanCost {
            io_cost: 100.0,
            cpu_cost: 10000.0,
            row_count: 10000.0,
        };
        let right = PlanCost {
            io_cost: 50.0,
            cpu_cost: 5000.0,
            row_count: 5000.0,
        };
        let serial = model.cost_hash_join(&left, &right);
        let parallel = model.cost_parallel_hash_join(&left, &right, 4);
        // Parallel build+probe CPU per-worker should be lower than serial
        // (total may be higher due to coordination overhead, but wall-clock is less)
        let serial_build_probe = 10000.0 * 0.0025 + 10000.0 * 0.0025;
        let parallel_build_probe = (5000.0 / 4.0) * 0.0025 + (10000.0 / 4.0) * 0.0025;
        assert!(parallel_build_probe < serial_build_probe);
        // IO cost is the same
        assert_eq!(parallel.io_cost, serial.io_cost);
    }

    // -----------------------------------------------------------------------
    // Spilling
    // -----------------------------------------------------------------------

    /// A plan cost with a row count and nothing else, for comparing shapes.
    fn rows(count: f64) -> PlanCost {
        PlanCost {
            io_cost: 0.0,
            cpu_cost: 0.0,
            row_count: count,
        }
    }

    /// A model that lets a query hold the given number of rows.
    fn bounded(row_capacity: f64) -> CostModel {
        CostModel {
            working_memory_bytes: row_capacity * ASSUMED_ROW_BYTES,
            ..CostModel::default()
        }
    }

    /// With no configured limit nothing spills, so no plan pays for it
    /// however large it is. This is the default and the behaviour every plan
    /// had before spilling was costed.
    #[test]
    fn an_unbounded_model_charges_nothing_for_size() {
        let model = make_cost_model();
        assert_eq!(model.working_memory_bytes, 0.0);
        let small = model.cost_hash_join(&rows(10.0), &rows(10.0));
        let huge = model.cost_hash_join(&rows(100_000_000.0), &rows(10.0));
        assert_eq!(small.io_cost, 0.0);
        assert_eq!(huge.io_cost, 0.0, "an unbounded model invented spill IO");
    }

    /// A join whose build side does not fit pays for writing both sides out
    /// and reading them back. One that fits pays nothing.
    #[test]
    fn a_join_past_the_budget_costs_more_than_one_inside_it() {
        let model = bounded(1_000.0);
        let fits = model.cost_hash_join(&rows(900.0), &rows(50_000.0));
        let spills = model.cost_hash_join(&rows(1_100.0), &rows(50_000.0));
        assert_eq!(fits.io_cost, 0.0, "a join inside the budget paid spill IO");
        assert!(
            spills.io_cost > 0.0,
            "a join past the budget paid nothing for spilling"
        );
        assert!(spills.total() > fits.total());
    }

    /// Only the excess is charged, so passing the budget is a slope rather
    /// than a cliff. A cliff would make the planner treat a plan one row over
    /// as equal to one a thousand times over.
    #[test]
    fn the_charge_grows_with_the_excess() {
        let model = bounded(1_000.0);
        let barely = model.cost_hash_join(&rows(1_010.0), &rows(1_010.0));
        let far = model.cost_hash_join(&rows(100_000.0), &rows(100_000.0));
        assert!(barely.io_cost > 0.0);
        assert!(
            far.io_cost > barely.io_cost * 50.0,
            "a join far past the budget cost {} against {} for one barely past",
            far.io_cost,
            barely.io_cost
        );
    }

    /// Sorting the excess is cheaper per byte than partitioning it, because a
    /// sort writes runs and reads them back while a join writes and reads both
    /// of its sides. That difference is what lets a sort-based plan win where
    /// a hash plan would not fit.
    #[test]
    fn a_sort_pays_less_for_the_same_excess_than_a_join() {
        let model = bounded(1_000.0);
        let sort = model.cost_sort(&rows(100_000.0));
        let join = model.cost_hash_join(&rows(100_000.0), &rows(100_000.0));
        assert!(sort.io_cost > 0.0);
        assert!(
            sort.io_cost < join.io_cost,
            "sorting {} cost as much as partitioning {}",
            sort.io_cost,
            join.io_cost
        );
    }

    /// A grouping that collapses its input holds one entry per group, so a
    /// huge input that produces few groups does not spill.
    #[test]
    fn an_aggregate_is_measured_by_its_groups_not_its_input() {
        let model = bounded(1_000.0);
        let collapsing = model.cost_hash_aggregate(&rows(10_000_000.0), 12.0);
        let sprawling = model.cost_hash_aggregate(&rows(10_000_000.0), 5_000_000.0);
        assert_eq!(
            collapsing.io_cost, 0.0,
            "an aggregate with twelve groups was costed as spilling"
        );
        assert!(sprawling.io_cost > 0.0);
    }

    /// The parallel join holds both inputs whole to partition them, so past
    /// the budget it is charged what the serial spilling join is charged
    /// rather than looking free because it has workers.
    #[test]
    fn the_parallel_join_pays_for_spilling_too() {
        let model = bounded(1_000.0);
        let serial = model.cost_hash_join(&rows(100_000.0), &rows(100_000.0));
        let parallel = model.cost_parallel_hash_join(&rows(100_000.0), &rows(100_000.0), 8);
        assert!(parallel.io_cost > 0.0);
        assert_eq!(
            parallel.io_cost, serial.io_cost,
            "the parallel join was costed as if partitioning were free"
        );
    }
}
