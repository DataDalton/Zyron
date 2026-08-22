//! The .zym manifest checkpoint codec.
//!
//! A manifest is the complete table state at one log version: schema,
//! cluster spec, every live data file with its statistics, and every
//! active delete predicate. Readers open the newest checkpoint and replay
//! only the version files after it, so the manifest is the random access
//! point for time travel and the statistics source for pruning.
//!
//! Layout: 64-byte header, schema section, cluster spec section, file
//! manifest section, delete predicate section, 48-byte footer holding the
//! five section offsets, a CRC32 over everything before the checksum, and
//! a trailing magic. All integers little endian.
//!
//! Two deliberate deviations from the original specification. The partition
//! spec section carries the cluster spec, which subsumes a partition scheme.
//! File entries are keyed by partition id, the data file path derives from
//! it, so there is no path string table and no file lookup bloom, entries
//! are sorted and binary searched exactly. Per-column value blooms are
//! carried as opaque bytes and interpreted by the segment bloom reader

use std::collections::BTreeMap;

use zyron_common::ZyronError;

use zyron_storage::columnar::might_contain_serialized;

use crate::cells::value_to_cell;
use crate::codec::{Cursor, corrupt};
use crate::index::{IndexFileEntry, LakeIndexSpec};
use crate::predicate::{
    ColumnBounds, LakePredicate, LakeValue, PruneDecision, StatsSource, decode_value, encode_value,
};
use crate::schema::LakeSchema;

pub const MANIFEST_MAGIC: [u8; 4] = *b"ZYLK";
pub const MANIFEST_FORMAT_VERSION: u16 = 1;

const HEADER_LEN: usize = 64;
// Eight section offsets, CRC32, trailing magic. Three more than the
// original five-offset specification: properties, index specs and index
// files
const FOOTER_LEN: usize = 8 * 8 + 4 + 4;

// Minimum encoded sizes guarding count fields against corrupt preallocation
const MIN_FILE_ENTRY: usize = 42;
const MIN_STATS_ENTRY: usize = 13;
const MIN_DELETE_ENTRY: usize = 21;
const MIN_INDEX_SPEC: usize = 11;
const MIN_INDEX_FILE: usize = MIN_FILE_ENTRY + 8;

// Column stats presence flags
const STAT_MIN: u8 = 1 << 0;
const STAT_MAX: u8 = 1 << 1;
const STAT_BLOOM: u8 = 1 << 2;
const STAT_NDV: u8 = 1 << 3;
const STAT_SIZE: u8 = 1 << 4;
const STAT_KNOWN_MASK: u8 = STAT_MIN | STAT_MAX | STAT_BLOOM | STAT_NDV | STAT_SIZE;

// The strategy encoding is persisted here and in the catalog, so one
// definition lives in zyron-common and this crate names it
pub use zyron_common::{ClusterKey, ClusterMode, ClusterStrategy, ClusteringSchedule};

/// Table properties carrying the clustering policy. The policy lives in
/// the log rather than the catalog so it is versioned with the layout it
/// governs, and a time-travel read sees the policy that was in force
pub const CLUSTERING_MODE_PROPERTY: &str = "clustering_mode";
pub const CLUSTERING_SCHEDULE_PROPERTY: &str = "clustering_schedule";
pub const CLUSTERING_ANCHORS_PROPERTY: &str = "clustering_anchors";

/// The table's clustering specification. An empty key list is a table
/// with no declared or measured ordering
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterSpec {
    /// Monotone spec version, bumped by every clustering change commit
    pub spec_id: u32,
    pub keys: Vec<ClusterKey>,
}

impl ClusterSpec {
    /// The columns files are ordered by, in layout order.
    ///
    /// What a plan is judged against: files are sorted by the first of
    /// these, then by the next within a run of equal values, so the
    /// position of a column in this list is how much the layout does for a
    /// predicate that names it
    pub fn key_columns(&self) -> Vec<u32> {
        self.keys.iter().map(|k| k.column_id).collect()
    }

    /// A table with no clustering, spec zero
    pub fn none() -> Self {
        Self {
            spec_id: 0,
            keys: Vec::new(),
        }
    }

    pub(crate) fn encode_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.spec_id.to_le_bytes());
        buf.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for key in &self.keys {
            buf.extend_from_slice(&key.column_id.to_le_bytes());
            buf.push(key.strategy.to_u8());
            buf.extend_from_slice(&key.param.to_le_bytes());
        }
    }

    pub(crate) fn decode(r: &mut Cursor<'_>) -> Result<Self, ZyronError> {
        let spec_id = r.u32()?;
        let count = r.u32()? as usize;
        r.check_count(count, 9, "cluster key")?;
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            let column_id = r.u32()?;
            let raw = r.u8()?;
            let strategy = ClusterStrategy::from_u8(raw)
                .ok_or_else(|| r.corrupt(format!("unknown cluster strategy {}", raw)))?;
            let param = r.u32()?;
            keys.push(ClusterKey {
                column_id,
                strategy,
                param,
            });
        }
        Ok(Self { spec_id, keys })
    }
}

/// Statistics for one column of one data file
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnStatsEntry {
    pub column_id: u32,
    pub bounds: ColumnBounds,
    /// Opaque value bloom bytes from the segment writer, probed through
    /// the segment bloom reader
    pub bloom: Option<Vec<u8>>,
    /// Distinct values in this file's column, estimated by the writer's
    /// sketch and within a few percent. None for a file written before
    /// the estimate existed, which reads as evidence the clustering
    /// planner does not have rather than as zero distinct values
    pub ndv: Option<u64>,
    /// Bytes this column's segment occupies in the file, padded to the
    /// page boundary the reader seeks to.
    ///
    /// This is what reading the column costs, and columns of one file
    /// differ by more than an order of magnitude, so a per file average is
    /// not a usable stand in. A cost model comparing two access paths
    /// needs it, and needs it without opening a file. None means the
    /// writer did not record it, which reads as evidence a cost model does
    /// not have rather than as a free column
    pub size_bytes: Option<u64>,
}

/// One live data file. The spec calls this a partition entry, it is a
/// data file record, not a partition scheme
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionEntry {
    /// Data file identity, the file path derives from it
    pub partition_id: u64,
    pub size_bytes: u64,
    pub row_count: u64,
    /// Log version whose commit added this file
    pub added_version: u64,
    /// Cluster spec the file was written under, reconstructing the layout
    /// a past version saw
    pub cluster_spec_id: u32,
    /// Sorted by column id. Shared rather than owned because every commit
    /// materializes the next manifest by cloning the previous one, and the
    /// bloom bytes in here are the bulk of an entry: sharing makes that
    /// clone a reference count bump per file instead of a copy of every
    /// bloom in the table. Stats are immutable once a file is written, a
    /// rewrite produces a new file with new stats
    pub column_stats: std::sync::Arc<Vec<ColumnStatsEntry>>,
    /// Active delete predicates the file must be filtered through, each id
    /// resolves in the manifest's delete predicate section
    pub delete_predicate_ids: Vec<u64>,
}

impl PartitionEntry {
    /// Bounds for one column, stats are sorted by column id
    pub fn stats_for(&self, column_id: u32) -> Option<&ColumnStatsEntry> {
        self.column_stats
            .binary_search_by_key(&column_id, |s| s.column_id)
            .ok()
            .map(|i| &self.column_stats[i])
    }
}

impl StatsSource for PartitionEntry {
    fn bounds(&self, column_id: u32) -> Option<&ColumnBounds> {
        self.stats_for(column_id).map(|s| &s.bounds)
    }
}

/// One file's statistics read against the schema that types its columns.
///
/// A bloom probe needs the column's physical type to encode the constant
/// into the same bytes the writer inserted, which a bare `PartitionEntry`
/// does not carry, so pruning through this view skips files a bare entry
/// would have to read.
pub struct FileStats<'a> {
    entry: &'a PartitionEntry,
    schema: &'a LakeSchema,
}

impl<'a> FileStats<'a> {
    pub fn new(entry: &'a PartitionEntry, schema: &'a LakeSchema) -> Self {
        Self { entry, schema }
    }

    pub fn entry(&self) -> &'a PartitionEntry {
        self.entry
    }
}

impl StatsSource for FileStats<'_> {
    fn bounds(&self, column_id: u32) -> Option<&ColumnBounds> {
        self.entry.stats_for(column_id).map(|s| &s.bounds)
    }

    fn may_contain(&self, column_id: u32, value: &LakeValue) -> bool {
        let Some(stats) = self.entry.stats_for(column_id) else {
            return true;
        };
        let Some(bloom) = stats.bloom.as_deref() else {
            return true;
        };
        let Some(column) = self.schema.column_by_id(column_id) else {
            return true;
        };
        let physical = column.physical_type_id();
        let width = physical.fixed_size().unwrap_or(0);
        match value_to_cell(physical, width, value) {
            Some(cell) => might_contain_serialized(bloom, cell.as_slice()),
            // A constant with no provable stored form prunes nothing
            None => true,
        }
    }
}

/// One active predicate-based delete
#[derive(Debug, Clone, PartialEq)]
pub struct DeletePredicate {
    pub id: u64,
    /// Original SQL text, retained for display and diagnostics
    pub sql: String,
    pub predicate: LakePredicate,
    pub created_version: u64,
    /// Rows this predicate deletes that are still physically present.
    ///
    /// Counted when the delete ran, where the rows were being examined
    /// anyway, so nothing has to read a data file to find out how much of
    /// the table is waiting to be rewritten. Files the predicate covered
    /// whole were removed there and then and are not in here
    pub pending_rows: u64,
}

/// Complete table state at one log version
#[derive(Debug, Clone, PartialEq)]
pub struct ManifestFile {
    /// Log version this manifest materializes
    pub snapshot_id: u64,
    /// Previous checkpoint's version, zero for the first
    pub parent_snapshot_id: u64,
    /// Commit timestamp, microseconds since the epoch
    pub timestamp_us: i64,
    pub schema: LakeSchema,
    pub cluster_spec: ClusterSpec,
    /// Live data files sorted by partition id
    pub entries: Vec<PartitionEntry>,
    /// Active delete predicates sorted by id
    pub delete_predicates: Vec<DeletePredicate>,
    /// Table properties, cumulative over every SetProperty commit
    pub properties: BTreeMap<String, String>,
    /// Declared secondary indexes sorted by index id
    pub indexes: Vec<LakeIndexSpec>,
    /// Live index files sorted by index id then partition id. They name
    /// the same directory the data files do and are versioned with them,
    /// which is what makes an index readable at a past version
    pub index_files: Vec<IndexFileEntry>,
}

impl ManifestFile {
    /// Checks every invariant the codec and readers rely on
    pub fn validate(&self) -> Result<(), ZyronError> {
        self.schema.validate()?;
        for pair in self.entries.windows(2) {
            if pair[0].partition_id >= pair[1].partition_id {
                return Err(ZyronError::Internal(format!(
                    "manifest entries not strictly sorted at partition {:#x}",
                    pair[1].partition_id
                )));
            }
        }
        for pair in self.delete_predicates.windows(2) {
            if pair[0].id >= pair[1].id {
                return Err(ZyronError::Internal(format!(
                    "manifest delete predicates not strictly sorted at id {}",
                    pair[1].id
                )));
            }
        }
        for entry in &self.entries {
            for pair in entry.column_stats.windows(2) {
                if pair[0].column_id >= pair[1].column_id {
                    return Err(ZyronError::Internal(format!(
                        "partition {:#x} column stats not strictly sorted at column {}",
                        entry.partition_id, pair[1].column_id
                    )));
                }
            }
            for stat in entry.column_stats.iter() {
                if stat.bounds.null_count > entry.row_count {
                    return Err(ZyronError::Internal(format!(
                        "partition {:#x} column {} null count {} exceeds row count {}",
                        entry.partition_id, stat.column_id, stat.bounds.null_count, entry.row_count
                    )));
                }
            }
            for id in &entry.delete_predicate_ids {
                if self.predicate_by_id(*id).is_none() {
                    return Err(ZyronError::Internal(format!(
                        "partition {:#x} references missing delete predicate {}",
                        entry.partition_id, id
                    )));
                }
            }
        }
        for pair in self.indexes.windows(2) {
            if pair[0].index_id >= pair[1].index_id {
                return Err(ZyronError::Internal(format!(
                    "manifest indexes not strictly sorted at index {}",
                    pair[1].index_id
                )));
            }
        }
        for spec in &self.indexes {
            spec.validate()?;
            for id in &spec.column_ids {
                if self.schema.column_by_id(*id).is_none() {
                    return Err(ZyronError::Internal(format!(
                        "index \"{}\" names column {}, which is not in the schema",
                        spec.name, id
                    )));
                }
            }
        }
        for pair in self.index_files.windows(2) {
            let a = (pair[0].index_id, pair[0].file.partition_id);
            let b = (pair[1].index_id, pair[1].file.partition_id);
            if a >= b {
                return Err(ZyronError::Internal(format!(
                    "manifest index files not strictly sorted at index {} partition {:#x}",
                    b.0, b.1
                )));
            }
        }
        for file in &self.index_files {
            if self.index_by_id(file.index_id).is_none() {
                return Err(ZyronError::Internal(format!(
                    "index file {:#x} belongs to index {}, which is not declared",
                    file.file.partition_id, file.index_id
                )));
            }
        }
        Ok(())
    }

    /// One declared index, indexes are sorted by index id
    pub fn index_by_id(&self, index_id: u32) -> Option<&LakeIndexSpec> {
        self.indexes
            .binary_search_by_key(&index_id, |s| s.index_id)
            .ok()
            .map(|i| &self.indexes[i])
    }

    /// One declared index by name, case sensitive
    pub fn index_by_name(&self, name: &str) -> Option<&LakeIndexSpec> {
        self.indexes.iter().find(|s| s.name == name)
    }

    /// Entry for one data file, entries are sorted by partition id
    pub fn entry_for(&self, partition_id: u64) -> Option<&PartitionEntry> {
        self.entries
            .binary_search_by_key(&partition_id, |e| e.partition_id)
            .ok()
            .map(|i| &self.entries[i])
    }

    /// One delete predicate by id, predicates are sorted by id
    pub fn predicate_by_id(&self, id: u64) -> Option<&DeletePredicate> {
        self.delete_predicates
            .binary_search_by_key(&id, |p| p.id)
            .ok()
            .map(|i| &self.delete_predicates[i])
    }

    /// Column ids the `bloom_filter_columns` property names, resolved
    /// against the current schema. A name that is not a column is ignored
    /// rather than failing the write, the property is advisory layout, and
    /// the writer only forces a filter the pruning path would otherwise not
    /// have
    pub fn declared_bloom_columns(&self) -> Vec<u32> {
        let Some(list) = self.properties.get("bloom_filter_columns") else {
            return Vec::new();
        };
        list.split(',')
            .filter_map(|name| self.schema.column_by_name(name.trim()).map(|c| c.id))
            .collect()
    }

    /// The one column whose bloom the layout already makes unnecessary, if
    /// there is one.
    ///
    /// Only the leading key, and only under RangePartition. That strategy
    /// gives each file a contiguous range of the key, so file bounds are
    /// close to disjoint and an equality resolves to a file or two before
    /// any filter is read. Nothing else in the spec qualifies:
    ///
    /// * A key after the leading one is only sorted within a run of equal
    ///   leading values, so its bounds span the domain once across every
    ///   run and a filter still removes files the bounds keep.
    /// * BitInterleave and SpaceFilling interleave every key into one
    ///   ordering, so no single key gets contiguous ranges.
    /// * AntiCluster spreads a key across files on purpose, which makes its
    ///   bounds maximally wide and a filter the only thing that prunes it
    pub fn bloom_redundant_key(&self) -> Option<u32> {
        let key = self.cluster_spec.keys.first()?;
        (key.strategy == ClusterStrategy::RangePartition).then_some(key.column_id)
    }

    /// Declared bloom columns the layout already covers, so no filter is
    /// built for them.
    ///
    /// Reported rather than dropped in silence: the property is something
    /// an operator asked for, and a request quietly not carried out is
    /// worse than one refused
    pub fn redundant_bloom_columns(&self) -> Vec<u32> {
        let Some(redundant) = self.bloom_redundant_key() else {
            return Vec::new();
        };
        self.declared_bloom_columns()
            .into_iter()
            .filter(|c| *c == redundant)
            .collect()
    }

    /// Columns the writer builds a bloom filter for.
    ///
    /// The declared set minus what the layout already covers. A filter on
    /// the leading range-partitioned key costs bytes in every manifest
    /// entry and removes files the key's own bounds removed first
    pub fn bloom_columns(&self) -> Vec<u32> {
        let redundant = self.bloom_redundant_key();
        self.declared_bloom_columns()
            .into_iter()
            .filter(|c| Some(*c) != redundant)
            .collect()
    }

    /// How the clustering choice interacts with measurement. A table that
    /// never declared one is Auto, so measurement chooses its layout and
    /// keeps revisiting it as the workload moves. An operator who wants the
    /// layout to stay where they put it declares Force
    pub fn clustering_mode(&self) -> ClusterMode {
        self.properties
            .get(CLUSTERING_MODE_PROPERTY)
            .and_then(|v| match v.to_ascii_lowercase().as_str() {
                "force" => Some(ClusterMode::Force),
                "auto" => Some(ClusterMode::Auto),
                "hybrid" => Some(ClusterMode::Hybrid),
                _ => None,
            })
            .unwrap_or(ClusterMode::Auto)
    }

    /// When clustering maintenance may run without being asked.
    ///
    /// A table that never declared one is Continuous, so background passes
    /// remove drift as it appears rather than waiting to be asked. OnDemand
    /// is the opt out, and it means no pass starts unless OPTIMIZE asks for
    /// one
    pub fn clustering_schedule(&self) -> ClusteringSchedule {
        self.properties
            .get(CLUSTERING_SCHEDULE_PROPERTY)
            .and_then(|v| match v.to_ascii_lowercase().as_str() {
                "ondemand" | "on_demand" => Some(ClusteringSchedule::OnDemand),
                "incremental" => Some(ClusteringSchedule::Incremental),
                "continuous" => Some(ClusteringSchedule::Continuous),
                _ => None,
            })
            .unwrap_or(ClusteringSchedule::Continuous)
    }

    /// Key column ids the operator pinned, in declaration order. A
    /// proposal that does not lead with exactly these is illegal, whatever
    /// it would have scored. Under Force the whole spec is the anchor set,
    /// under Auto there are none
    pub fn clustering_anchors(&self) -> Vec<u32> {
        match self.clustering_mode() {
            ClusterMode::Force => self.cluster_spec.keys.iter().map(|k| k.column_id).collect(),
            ClusterMode::Auto => Vec::new(),
            ClusterMode::Hybrid => self
                .properties
                .get(CLUSTERING_ANCHORS_PROPERTY)
                .map(|list| {
                    list.split(',')
                        .filter_map(|name| self.schema.column_by_name(name.trim()).map(|c| c.id))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// One file's statistics paired with this manifest's schema, the form
    /// pruning reads so bounds and value blooms both apply
    pub fn file_stats<'a>(&'a self, entry: &'a PartitionEntry) -> FileStats<'a> {
        FileStats::new(entry, &self.schema)
    }

    /// How this predicate scores against one file, bounds and blooms both
    pub fn prune_file(&self, predicate: &LakePredicate, entry: &PartitionEntry) -> PruneDecision {
        predicate.prune(&self.file_stats(entry))
    }

    /// Files a predicate may match, statistics-pruned. Files the predicate
    /// cannot match are skipped without IO
    pub fn files_matching<'a>(
        &'a self,
        predicate: &'a LakePredicate,
    ) -> impl Iterator<Item = &'a PartitionEntry> {
        self.entries
            .iter()
            .filter(move |e| self.prune_file(predicate, e) != PruneDecision::CannotMatch)
    }

    /// Serializes the whole manifest including header and footer
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_LEN + FOOTER_LEN + 256 * self.entries.len());

        buf.extend_from_slice(&MANIFEST_MAGIC);
        buf.extend_from_slice(&MANIFEST_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&self.schema.schema_id.to_le_bytes());
        buf.extend_from_slice(&self.snapshot_id.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_us.to_le_bytes());
        buf.extend_from_slice(&self.parent_snapshot_id.to_le_bytes());
        buf.extend_from_slice(&self.cluster_spec.spec_id.to_le_bytes());
        buf.resize(HEADER_LEN, 0);

        let schema_off = buf.len() as u64;
        self.schema.encode_into(&mut buf);

        let spec_off = buf.len() as u64;
        self.cluster_spec.encode_into(&mut buf);

        let files_off = buf.len() as u64;
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for entry in &self.entries {
            encode_partition_entry(entry, &mut buf);
        }

        let deletes_off = buf.len() as u64;
        buf.extend_from_slice(&(self.delete_predicates.len() as u32).to_le_bytes());
        for del in &self.delete_predicates {
            encode_delete_predicate(del, &mut buf);
        }

        let props_off = buf.len() as u64;
        buf.extend_from_slice(&(self.properties.len() as u32).to_le_bytes());
        for (key, value) in &self.properties {
            buf.extend_from_slice(&(key.len() as u16).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buf.extend_from_slice(value.as_bytes());
        }

        let index_specs_off = buf.len() as u64;
        buf.extend_from_slice(&(self.indexes.len() as u32).to_le_bytes());
        for spec in &self.indexes {
            encode_index_spec(spec, &mut buf);
        }

        let index_files_off = buf.len() as u64;
        buf.extend_from_slice(&(self.index_files.len() as u64).to_le_bytes());
        for file in &self.index_files {
            encode_index_file(file, &mut buf);
        }

        let footer_off = buf.len() as u64;
        buf.extend_from_slice(&schema_off.to_le_bytes());
        buf.extend_from_slice(&spec_off.to_le_bytes());
        buf.extend_from_slice(&files_off.to_le_bytes());
        buf.extend_from_slice(&deletes_off.to_le_bytes());
        buf.extend_from_slice(&props_off.to_le_bytes());
        buf.extend_from_slice(&index_specs_off.to_le_bytes());
        buf.extend_from_slice(&index_files_off.to_le_bytes());
        buf.extend_from_slice(&footer_off.to_le_bytes());
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf.extend_from_slice(&MANIFEST_MAGIC);
        buf
    }

    /// Parses and validates a whole manifest. `ctx` names the file for
    /// error messages. The checksum is verified before any section parse
    pub fn decode(bytes: &[u8], ctx: &str) -> Result<Self, ZyronError> {
        if bytes.len() < HEADER_LEN + FOOTER_LEN {
            return Err(corrupt(
                ctx,
                format!(
                    "manifest of {} bytes is shorter than header plus footer",
                    bytes.len()
                ),
            ));
        }
        if bytes[..4] != MANIFEST_MAGIC {
            return Err(corrupt(ctx, "bad manifest magic".into()));
        }
        if bytes[bytes.len() - 4..] != MANIFEST_MAGIC {
            return Err(corrupt(ctx, "bad manifest trailing magic".into()));
        }
        let crc_field = bytes.len() - 8;
        let mut crc_bytes = [0u8; 4];
        crc_bytes.copy_from_slice(&bytes[crc_field..crc_field + 4]);
        let stored_crc = u32::from_le_bytes(crc_bytes);
        let actual_crc = crc32fast::hash(&bytes[..crc_field]);
        if stored_crc != actual_crc {
            return Err(corrupt(
                ctx,
                format!(
                    "manifest checksum mismatch, stored {:#010x} computed {:#010x}",
                    stored_crc, actual_crc
                ),
            ));
        }

        let mut h = Cursor::new(&bytes[4..HEADER_LEN], ctx);
        let format_version = h.u16()?;
        if format_version != MANIFEST_FORMAT_VERSION {
            return Err(corrupt(
                ctx,
                format!("unsupported manifest format version {}", format_version),
            ));
        }
        let flags = h.u16()?;
        if flags != 0 {
            return Err(corrupt(
                ctx,
                format!("unknown manifest flags {:#06x}", flags),
            ));
        }
        let header_schema_id = h.u64()?;
        let snapshot_id = h.u64()?;
        let timestamp_us = h.i64()?;
        let parent_snapshot_id = h.u64()?;
        let header_spec_id = h.u32()?;

        let mut f = Cursor::new(&bytes[bytes.len() - FOOTER_LEN..crc_field], ctx);
        let mut offsets = [0u64; 8];
        for slot in &mut offsets {
            *slot = f.u64()?;
        }
        let footer_start = (bytes.len() - FOOTER_LEN) as u64;
        if offsets[0] != HEADER_LEN as u64
            || offsets[7] != footer_start
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(corrupt(
                ctx,
                format!("inconsistent section offsets {:?}", offsets),
            ));
        }

        let section = |i: usize| -> &[u8] { &bytes[offsets[i] as usize..offsets[i + 1] as usize] };

        let (schema, schema_used) = LakeSchema::decode(section(0), ctx)?;
        if schema_used != section(0).len() {
            return Err(corrupt(ctx, "schema section has trailing bytes".into()));
        }
        if schema.schema_id != header_schema_id {
            return Err(corrupt(
                ctx,
                format!(
                    "header schema id {} does not match schema section {}",
                    header_schema_id, schema.schema_id
                ),
            ));
        }

        let mut sc = Cursor::new(section(1), ctx);
        let cluster_spec = ClusterSpec::decode(&mut sc)?;
        if sc.remaining() != 0 {
            return Err(corrupt(
                ctx,
                "cluster spec section has trailing bytes".into(),
            ));
        }
        if cluster_spec.spec_id != header_spec_id {
            return Err(corrupt(
                ctx,
                format!(
                    "header cluster spec id {} does not match section {}",
                    header_spec_id, cluster_spec.spec_id
                ),
            ));
        }

        let mut fr = Cursor::new(section(2), ctx);
        let file_count = fr.u64()? as usize;
        fr.check_count(file_count, MIN_FILE_ENTRY, "file entry")?;
        let mut entries = Vec::with_capacity(file_count);
        for _ in 0..file_count {
            entries.push(decode_partition_entry(&mut fr)?);
        }
        if fr.remaining() != 0 {
            return Err(corrupt(
                ctx,
                "file manifest section has trailing bytes".into(),
            ));
        }

        let mut dr = Cursor::new(section(3), ctx);
        let delete_count = dr.u32()? as usize;
        dr.check_count(delete_count, MIN_DELETE_ENTRY, "delete predicate")?;
        let mut delete_predicates = Vec::with_capacity(delete_count);
        for _ in 0..delete_count {
            delete_predicates.push(decode_delete_predicate(&mut dr)?);
        }
        if dr.remaining() != 0 {
            return Err(corrupt(
                ctx,
                "delete predicate section has trailing bytes".into(),
            ));
        }

        let mut pr = Cursor::new(section(4), ctx);
        let prop_count = pr.u32()? as usize;
        pr.check_count(prop_count, 6, "property")?;
        let mut properties = BTreeMap::new();
        for _ in 0..prop_count {
            let key_len = pr.u16()? as usize;
            let key = pr.utf8(key_len, "property key")?;
            let value_len = pr.u32()? as usize;
            let value = pr.utf8(value_len, "property value")?;
            if properties.insert(key.clone(), value).is_some() {
                return Err(corrupt(ctx, format!("duplicate property key \"{}\"", key)));
            }
        }
        if pr.remaining() != 0 {
            return Err(corrupt(ctx, "property section has trailing bytes".into()));
        }

        let mut ir = Cursor::new(section(5), ctx);
        let index_count = ir.u32()? as usize;
        ir.check_count(index_count, MIN_INDEX_SPEC, "index spec")?;
        let mut indexes = Vec::with_capacity(index_count);
        for _ in 0..index_count {
            indexes.push(decode_index_spec(&mut ir)?);
        }
        if ir.remaining() != 0 {
            return Err(corrupt(ctx, "index spec section has trailing bytes".into()));
        }

        let mut xr = Cursor::new(section(6), ctx);
        let index_file_count = xr.u64()? as usize;
        xr.check_count(index_file_count, MIN_INDEX_FILE, "index file")?;
        let mut index_files = Vec::with_capacity(index_file_count);
        for _ in 0..index_file_count {
            index_files.push(decode_index_file(&mut xr)?);
        }
        if xr.remaining() != 0 {
            return Err(corrupt(ctx, "index file section has trailing bytes".into()));
        }

        let manifest = Self {
            snapshot_id,
            parent_snapshot_id,
            timestamp_us,
            schema,
            cluster_spec,
            entries,
            delete_predicates,
            properties,
            indexes,
            index_files,
        };
        manifest
            .validate()
            .map_err(|e| corrupt(ctx, e.to_string()))?;
        Ok(manifest)
    }
}

/// Serializes one index declaration, shared by the manifest index spec
/// section and AddIndex log entries
pub(crate) fn encode_index_spec(spec: &LakeIndexSpec, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&spec.index_id.to_le_bytes());
    buf.extend_from_slice(&(spec.name.len() as u16).to_le_bytes());
    buf.extend_from_slice(spec.name.as_bytes());
    buf.push(spec.unique as u8);
    buf.extend_from_slice(&(spec.column_ids.len() as u16).to_le_bytes());
    for id in &spec.column_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
}

pub(crate) fn decode_index_spec(r: &mut Cursor<'_>) -> Result<LakeIndexSpec, ZyronError> {
    let index_id = r.u32()?;
    let name_len = r.u16()? as usize;
    let name = r.utf8(name_len, "index name")?;
    let unique = r.u8()? != 0;
    let column_count = r.u16()? as usize;
    r.check_count(column_count, 4, "index column")?;
    let mut column_ids = Vec::with_capacity(column_count);
    for _ in 0..column_count {
        column_ids.push(r.u32()?);
    }
    let spec = LakeIndexSpec {
        index_id,
        name,
        column_ids,
        unique,
    };
    spec.validate()?;
    Ok(spec)
}

/// Serializes one index file record, shared by the manifest index file
/// section and AddIndexFile log entries
pub(crate) fn encode_index_file(file: &IndexFileEntry, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&file.index_id.to_le_bytes());
    buf.extend_from_slice(&(file.covers.len() as u32).to_le_bytes());
    for partition_id in &file.covers {
        buf.extend_from_slice(&partition_id.to_le_bytes());
    }
    encode_partition_entry(&file.file, buf);
}

pub(crate) fn decode_index_file(r: &mut Cursor<'_>) -> Result<IndexFileEntry, ZyronError> {
    let index_id = r.u32()?;
    let cover_count = r.u32()? as usize;
    r.check_count(cover_count, 8, "index coverage")?;
    let mut covers = Vec::with_capacity(cover_count);
    for _ in 0..cover_count {
        covers.push(r.u64()?);
    }
    let file = decode_partition_entry(r)?;
    Ok(IndexFileEntry {
        index_id,
        covers,
        file,
    })
}

/// Serializes one data file record, shared by the manifest file section
/// and AddFile log entries
pub(crate) fn encode_partition_entry(entry: &PartitionEntry, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&entry.partition_id.to_le_bytes());
    buf.extend_from_slice(&entry.size_bytes.to_le_bytes());
    buf.extend_from_slice(&entry.row_count.to_le_bytes());
    buf.extend_from_slice(&entry.added_version.to_le_bytes());
    buf.extend_from_slice(&entry.cluster_spec_id.to_le_bytes());
    buf.extend_from_slice(&(entry.column_stats.len() as u16).to_le_bytes());
    for stat in entry.column_stats.iter() {
        buf.extend_from_slice(&stat.column_id.to_le_bytes());
        let mut flags = 0u8;
        if stat.bounds.min.is_some() {
            flags |= STAT_MIN;
        }
        if stat.bounds.max.is_some() {
            flags |= STAT_MAX;
        }
        if stat.bloom.is_some() {
            flags |= STAT_BLOOM;
        }
        if stat.ndv.is_some() {
            flags |= STAT_NDV;
        }
        if stat.size_bytes.is_some() {
            flags |= STAT_SIZE;
        }
        buf.push(flags);
        buf.extend_from_slice(&stat.bounds.null_count.to_le_bytes());
        if let Some(min) = &stat.bounds.min {
            encode_value(min, buf);
        }
        if let Some(max) = &stat.bounds.max {
            encode_value(max, buf);
        }
        if let Some(bloom) = &stat.bloom {
            buf.extend_from_slice(&(bloom.len() as u32).to_le_bytes());
            buf.extend_from_slice(bloom);
        }
        if let Some(ndv) = stat.ndv {
            buf.extend_from_slice(&ndv.to_le_bytes());
        }
        if let Some(size) = stat.size_bytes {
            buf.extend_from_slice(&size.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(entry.delete_predicate_ids.len() as u16).to_le_bytes());
    for id in &entry.delete_predicate_ids {
        buf.extend_from_slice(&id.to_le_bytes());
    }
}

/// Parses one data file record from a positioned cursor
pub(crate) fn decode_partition_entry(fr: &mut Cursor<'_>) -> Result<PartitionEntry, ZyronError> {
    let partition_id = fr.u64()?;
    let size_bytes = fr.u64()?;
    let row_count = fr.u64()?;
    let added_version = fr.u64()?;
    let cluster_spec_id = fr.u32()?;
    let stats_count = fr.u16()? as usize;
    fr.check_count(stats_count, MIN_STATS_ENTRY, "column stats")?;
    let mut column_stats = Vec::with_capacity(stats_count);
    for _ in 0..stats_count {
        let column_id = fr.u32()?;
        let flags = fr.u8()?;
        if flags & !STAT_KNOWN_MASK != 0 {
            return Err(fr.corrupt(format!(
                "column {} has unknown stats flags {:#04x}",
                column_id, flags
            )));
        }
        let null_count = fr.u64()?;
        let min = if flags & STAT_MIN != 0 {
            Some(decode_value(fr)?)
        } else {
            None
        };
        let max = if flags & STAT_MAX != 0 {
            Some(decode_value(fr)?)
        } else {
            None
        };
        let bloom = if flags & STAT_BLOOM != 0 {
            let len = fr.u32()? as usize;
            Some(fr.take(len)?.to_vec())
        } else {
            None
        };
        let ndv = if flags & STAT_NDV != 0 {
            Some(fr.u64()?)
        } else {
            None
        };
        let size_bytes = if flags & STAT_SIZE != 0 {
            Some(fr.u64()?)
        } else {
            None
        };
        column_stats.push(ColumnStatsEntry {
            ndv,
            column_id,
            bounds: ColumnBounds {
                min,
                max,
                null_count,
                row_count,
            },
            bloom,
            size_bytes,
        });
    }
    let ref_count = fr.u16()? as usize;
    fr.check_count(ref_count, 8, "delete predicate reference")?;
    let mut delete_predicate_ids = Vec::with_capacity(ref_count);
    for _ in 0..ref_count {
        delete_predicate_ids.push(fr.u64()?);
    }
    Ok(PartitionEntry {
        partition_id,
        size_bytes,
        row_count,
        added_version,
        cluster_spec_id,
        column_stats: std::sync::Arc::new(column_stats),
        delete_predicate_ids,
    })
}

/// Serializes one delete predicate record, shared by the manifest delete
/// section and AddDeletePredicate log entries
pub(crate) fn encode_delete_predicate(del: &DeletePredicate, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&del.id.to_le_bytes());
    buf.extend_from_slice(&del.created_version.to_le_bytes());
    buf.extend_from_slice(&del.pending_rows.to_le_bytes());
    buf.extend_from_slice(&(del.sql.len() as u32).to_le_bytes());
    buf.extend_from_slice(del.sql.as_bytes());
    del.predicate.encode_into(buf);
}

/// Parses one delete predicate record from a positioned cursor
pub(crate) fn decode_delete_predicate(dr: &mut Cursor<'_>) -> Result<DeletePredicate, ZyronError> {
    let id = dr.u64()?;
    let created_version = dr.u64()?;
    let pending_rows = dr.u64()?;
    let sql_len = dr.u32()? as usize;
    let sql = dr.utf8(sql_len, "delete predicate sql")?;
    let predicate = LakePredicate::decode_from(dr)?;
    Ok(DeletePredicate {
        id,
        sql,
        predicate,
        created_version,
        pending_rows,
    })
}

// ---------------------------------------------------------------------------
// Auto compaction
// ---------------------------------------------------------------------------

/// Share of files that may sit below a quarter of the target before a
/// compaction runs unasked
pub const DEFAULT_AUTO_COMPACT_SMALL_FILE_RATIO: f64 = 0.25;

/// Share of the table's rows that may be deleted and not yet rewritten
/// before a compaction runs unasked
pub const DEFAULT_AUTO_COMPACT_DEAD_ROW_RATIO: f64 = 0.20;

pub const TARGET_ROWS_PER_FILE_PROPERTY: &str = "target_rows_per_file";
pub const CLUSTER_REPAIR_MAX_INPUTS_PROPERTY: &str = "cluster_repair_max_inputs";
pub const CLUSTER_REPAIR_INTERVAL_SECS_PROPERTY: &str = "cluster_repair_interval_secs";
pub const CLUSTER_REPAIR_URGENCY_THRESHOLD_PROPERTY: &str = "cluster_repair_urgency_threshold";

/// How long a table waits between repair passes when it names no interval
/// of its own
pub const DEFAULT_CLUSTER_REPAIR_INTERVAL_SECS: u64 = 300;

/// Files needing repair before a table stops waiting for its interval.
///
/// Eight is the point where the layout has stopped being a layout: a
/// predicate that should reach one or two files is reaching nine, and every
/// query pays for it until the next pass. Waiting out a five minute clock
/// in that state serves nobody
pub const DEFAULT_CLUSTER_REPAIR_URGENCY_THRESHOLD: usize = 8;
pub const AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY: &str = "auto_compact_small_file_ratio";
pub const AUTO_COMPACT_DEAD_ROW_RATIO_PROPERTY: &str = "auto_compact_dead_row_ratio";

/// Fewest small files worth merging.
///
/// Rewriting one file into one file moves the same rows into the same
/// shape, so a table holding a single small file would trip the ratio on
/// every tick and rewrite it forever
pub const MIN_SMALL_FILES_TO_MERGE: usize = 2;

/// Why a compaction ran without being asked
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionTrigger {
    /// Too many files hold fewer rows than a quarter of the target, so
    /// every scan pays per-file cost for rows that belong in one file
    SmallFiles,
    /// Too much of the table is rows a delete removed logically and no
    /// rewrite has removed physically, so every scan reads and discards
    /// them
    DeadRows,
    /// Both crossed in one check. One compaction answers both, so it runs
    /// once
    Both,
}

impl CompactionTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            CompactionTrigger::SmallFiles => "small_files",
            CompactionTrigger::DeadRows => "dead_rows",
            CompactionTrigger::Both => "small_files_and_dead_rows",
        }
    }
}

impl std::fmt::Display for CompactionTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the manifest says about whether a compaction is due, and the
/// numbers the answer came from.
///
/// The counts are reported whether or not anything tripped, because an
/// operator asking why nothing ran needs the same figures as one asking
/// why something did
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompactionNeed {
    pub trigger: Option<CompactionTrigger>,
    pub small_files: usize,
    pub total_files: usize,
    /// Rows deleted logically and still on disk
    pub pending_deleted_rows: u64,
    pub total_rows: u64,
}

impl ManifestFile {
    /// Identity of everything about this table that changes what a plan
    /// should be.
    ///
    /// Derived rather than stored, so nothing has to remember to bump it.
    /// A field would have to be advanced by every commit that touches the
    /// layout, and the one that forgot would leave plans costed against a
    /// table that no longer exists, which fails as a slow query rather
    /// than as an error. This is a function of the state a plan is costed
    /// against, so two manifests that would cost a plan the same way
    /// produce the same value and nothing can drift.
    ///
    /// What goes into it:
    ///
    /// * The spec, because it decides which columns prune.
    /// * The schema id, because adding or dropping an expression column
    ///   changes which predicates can reach a stored value.
    /// * Mode and schedule, because they decide whether the spec a plan
    ///   was costed against is the one measurement will keep.
    /// * How many files are laid out under the current spec, and how many
    ///   rows they hold. This is why a pass that commits files under an
    ///   unchanged spec still moves the epoch: the layout did not change
    ///   but the data's conformance to it did, and conformance is what
    ///   decides how many files a predicate opens.
    ///
    /// Persisted-safe hashing, because a caller may compare a value taken
    /// before a restart with one taken after
    pub fn clustering_epoch(&self) -> u64 {
        let mut buf = Vec::with_capacity(64 + self.cluster_spec.keys.len() * 9);
        buf.extend_from_slice(&self.cluster_spec.spec_id.to_le_bytes());
        buf.extend_from_slice(&self.schema.schema_id.to_le_bytes());
        for key in &self.cluster_spec.keys {
            buf.extend_from_slice(&key.column_id.to_le_bytes());
            buf.push(key.strategy.to_u8());
            buf.extend_from_slice(&key.param.to_le_bytes());
        }
        buf.push(self.clustering_mode().to_u8());
        buf.push(self.clustering_schedule().to_u8());
        for id in self.clustering_anchors() {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        let conforming = self
            .entries
            .iter()
            .filter(|e| e.cluster_spec_id == self.cluster_spec.spec_id)
            .count() as u64;
        buf.extend_from_slice(&conforming.to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.total_rows().to_le_bytes());
        zyron_common::checksum::hash64(&buf)
    }

    /// Rows across the table's live files, before any delete predicate is
    /// applied
    pub fn total_rows(&self) -> u64 {
        self.entries.iter().map(|e| e.row_count).sum()
    }

    /// Rows a delete removed logically that a rewrite has not removed
    /// physically.
    ///
    /// Exact and free: every predicate recorded what it matched in the
    /// files it did not remove whole, counted while those rows were
    /// already being examined
    pub fn pending_deleted_rows(&self) -> u64 {
        self.delete_predicates
            .iter()
            .map(|p| p.pending_rows)
            .sum::<u64>()
            .min(self.total_rows())
    }

    /// A ratio property, or its default when unset or unreadable.
    ///
    /// An unreadable value falls back rather than failing a maintenance
    /// tick: the property is a threshold, and refusing to maintain a table
    /// because someone typed a word into it would be worse than
    /// maintaining it on the shipped one. Negative and non-finite values
    /// are the same case
    fn ratio_property(&self, key: &str, default: f64) -> f64 {
        self.properties
            .get(key)
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(default)
    }

    /// A whole-number property, or its default when unset or unreadable.
    ///
    /// Same reasoning as `ratio_property`: an unreadable value falls back
    /// rather than stopping maintenance on the table. A statement that sets
    /// one is checked where it is written, so the only way to reach this
    /// fallback is a manifest nobody wrote by hand
    fn count_property(&self, key: &str, default: u64) -> u64 {
        self.properties
            .get(key)
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(default)
    }

    /// Files one repair pass may rewrite.
    ///
    /// A pass reads and rewrites every input, so this is what bounds how
    /// long one pass runs and how much it writes. A table taking constant
    /// small writes wants it high enough to keep up; one that is mostly
    /// read wants it low so a pass never competes with queries for long
    pub fn cluster_repair_max_inputs(&self, fallback: usize) -> usize {
        self.count_property(CLUSTER_REPAIR_MAX_INPUTS_PROPERTY, fallback as u64)
            .max(1) as usize
    }

    /// Seconds between repair passes on this table.
    ///
    /// The node's own cadence is the floor: this can make a table wait
    /// longer between passes, never less. A table that needs attention
    /// sooner than its interval says gets it through the urgency threshold
    /// instead, which is the mechanism for "now" rather than "more often"
    pub fn cluster_repair_interval_secs(&self, fallback: u64) -> u64 {
        self.count_property(CLUSTER_REPAIR_INTERVAL_SECS_PROPERTY, fallback)
    }

    /// Files needing repair before this table stops waiting for its
    /// interval and is passed on the next tick
    pub fn cluster_repair_urgency_threshold(&self) -> usize {
        self.count_property(
            CLUSTER_REPAIR_URGENCY_THRESHOLD_PROPERTY,
            DEFAULT_CLUSTER_REPAIR_URGENCY_THRESHOLD as u64,
        ) as usize
    }

    pub fn auto_compact_small_file_ratio(&self) -> f64 {
        self.ratio_property(
            AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY,
            DEFAULT_AUTO_COMPACT_SMALL_FILE_RATIO,
        )
    }

    pub fn auto_compact_dead_row_ratio(&self) -> f64 {
        self.ratio_property(
            AUTO_COMPACT_DEAD_ROW_RATIO_PROPERTY,
            DEFAULT_AUTO_COMPACT_DEAD_ROW_RATIO,
        )
    }

    /// Rows one data file is aiming for.
    ///
    /// The table property when it names one, otherwise whatever the caller
    /// runs as its default: a node-wide setting is the right answer for a
    /// table that expressed no opinion, and the wrong one for a table that
    /// did
    pub fn target_rows_per_file(&self, fallback: u64) -> u64 {
        self.properties
            .get(TARGET_ROWS_PER_FILE_PROPERTY)
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(fallback)
            .max(1)
    }

    /// Files holding fewer rows than a quarter of the target.
    ///
    /// A quarter rather than the target itself, because a file at
    /// three quarters of the target is the normal result of a bulk load
    /// and rewriting it would cost more than it saves
    pub fn small_file_threshold(target_rows_per_file: u64) -> u64 {
        (target_rows_per_file / 4).max(1)
    }

    /// Whether this table has drifted far enough from its target shape to
    /// be worth rewriting, and the figures behind the answer.
    ///
    /// Reads the manifest and opens nothing, so a maintenance tick can ask
    /// it for every table
    pub fn compaction_need(&self, fallback_rows_per_file: u64) -> CompactionNeed {
        let threshold =
            Self::small_file_threshold(self.target_rows_per_file(fallback_rows_per_file));
        let total_files = self.entries.len();
        let small_files = self
            .entries
            .iter()
            .filter(|e| e.row_count < threshold)
            .count();
        let total_rows = self.total_rows();
        let pending_deleted_rows = self.pending_deleted_rows();

        let small_tripped = small_files >= MIN_SMALL_FILES_TO_MERGE
            && total_files > 0
            && small_files as f64 / total_files as f64 > self.auto_compact_small_file_ratio();
        let dead_tripped = total_rows > 0
            && pending_deleted_rows as f64 / total_rows as f64 > self.auto_compact_dead_row_ratio();

        let trigger = match (small_tripped, dead_tripped) {
            (true, true) => Some(CompactionTrigger::Both),
            (true, false) => Some(CompactionTrigger::SmallFiles),
            (false, true) => Some(CompactionTrigger::DeadRows),
            (false, false) => None,
        };
        CompactionNeed {
            trigger,
            small_files,
            total_files,
            pending_deleted_rows,
            total_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::{CompareOp, LakeValue};
    use crate::schema::LakeColumn;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            3,
            vec![
                LakeColumn {
                    id: 0,
                    name: "id".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "name".into(),
                    type_id: TypeId::Text,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
            ],
        )
        .expect("valid schema")
    }

    fn stats(
        column_id: u32,
        min: i64,
        max: i64,
        null_count: u64,
        row_count: u64,
    ) -> ColumnStatsEntry {
        ColumnStatsEntry {
            ndv: Some(row_count.saturating_sub(null_count)),
            column_id,
            bounds: ColumnBounds {
                min: Some(LakeValue::Int(min)),
                max: Some(LakeValue::Int(max)),
                null_count,
                row_count,
            },
            bloom: None,
            size_bytes: Some(16_384),
        }
    }

    fn sample() -> ManifestFile {
        ManifestFile {
            snapshot_id: 41,
            parent_snapshot_id: 30,
            timestamp_us: 1_754_700_000_000_000,
            schema: schema(),
            cluster_spec: ClusterSpec {
                spec_id: 2,
                keys: vec![ClusterKey {
                    column_id: 0,
                    strategy: ClusterStrategy::RangePartition,
                    param: 16,
                }],
            },
            entries: vec![
                PartitionEntry {
                    partition_id: 0x10,
                    size_bytes: 4096,
                    row_count: 100,
                    added_version: 7,
                    cluster_spec_id: 1,
                    column_stats: std::sync::Arc::new(vec![stats(0, 1, 100, 0, 100)]),
                    delete_predicate_ids: vec![],
                },
                PartitionEntry {
                    partition_id: 0x20,
                    size_bytes: 8192,
                    row_count: 200,
                    added_version: 40,
                    cluster_spec_id: 2,
                    column_stats: std::sync::Arc::new(vec![
                        stats(0, 101, 300, 0, 200),
                        ColumnStatsEntry {
                            ndv: Some(188),
                            column_id: 1,
                            bounds: ColumnBounds {
                                min: Some(LakeValue::Str("alice".into())),
                                max: Some(LakeValue::Str("zoe".into())),
                                null_count: 12,
                                row_count: 200,
                            },
                            bloom: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
                            size_bytes: Some(49_152),
                        },
                    ]),
                    delete_predicate_ids: vec![9],
                },
            ],
            delete_predicates: vec![DeletePredicate {
                id: 9,
                sql: "name = 'deleted'".into(),
                predicate: LakePredicate::Compare {
                    column_id: 1,
                    op: CompareOp::Eq,
                    value: LakeValue::Str("deleted".into()),
                },
                created_version: 41,
                pending_rows: 0,
            }],
            properties: BTreeMap::from([
                ("target_file_size".to_string(), "268435456".to_string()),
                ("time_travel_retention".to_string(), "7d".to_string()),
            ]),
            indexes: vec![
                LakeIndexSpec {
                    index_id: 1,
                    name: "ix_name".into(),
                    column_ids: vec![1],
                    unique: false,
                },
                LakeIndexSpec {
                    index_id: 4,
                    name: "ux_id_name".into(),
                    column_ids: vec![0, 1],
                    unique: true,
                },
            ],
            index_files: vec![IndexFileEntry {
                index_id: 1,
                covers: vec![0x10, 0x11],
                file: PartitionEntry {
                    partition_id: 0x20,
                    size_bytes: 512,
                    row_count: 6,
                    added_version: 41,
                    cluster_spec_id: 0,
                    column_stats: std::sync::Arc::new(vec![ColumnStatsEntry {
                        column_id: 0,
                        bounds: ColumnBounds {
                            min: Some(LakeValue::Str("alice".into())),
                            max: Some(LakeValue::Str("carol".into())),
                            null_count: 1,
                            row_count: 6,
                        },
                        bloom: Some(vec![0x01, 0x02]),
                        ndv: Some(5),
                        size_bytes: Some(16_384),
                    }]),
                    delete_predicate_ids: Vec::new(),
                },
            }],
        }
    }

    #[test]
    fn test_roundtrip_preserves_everything() {
        let m = sample();
        let bytes = m.encode();
        let decoded = ManifestFile::decode(&bytes, "test.zym").expect("decodes");
        assert_eq!(decoded, m);
    }

    #[test]
    fn test_empty_table_roundtrip() {
        let m = ManifestFile {
            snapshot_id: 1,
            parent_snapshot_id: 0,
            timestamp_us: 0,
            schema: schema(),
            cluster_spec: ClusterSpec::none(),
            entries: vec![],
            delete_predicates: vec![],
            properties: BTreeMap::new(),
            indexes: Vec::new(),
            index_files: Vec::new(),
        };
        let bytes = m.encode();
        let decoded = ManifestFile::decode(&bytes, "test.zym").expect("decodes");
        assert_eq!(decoded, m);
    }

    #[test]
    fn test_every_corrupted_byte_is_detected() {
        let m = sample();
        let bytes = m.encode();
        // The checksum covers everything before itself, and the trailing
        // magic is checked directly, so no single byte flip can pass
        for i in 0..bytes.len() {
            let mut bad = bytes.clone();
            bad[i] ^= 0xFF;
            assert!(
                ManifestFile::decode(&bad, "test.zym").is_err(),
                "flip at byte {} must be detected",
                i
            );
        }
    }

    #[test]
    fn test_truncation_is_detected() {
        let bytes = sample().encode();
        for cut in [0, 10, HEADER_LEN, bytes.len() - 1] {
            assert!(ManifestFile::decode(&bytes[..cut], "test.zym").is_err());
        }
    }

    #[test]
    fn test_lookup_and_stats_access() {
        let m = sample();
        assert_eq!(m.entry_for(0x20).map(|e| e.row_count), Some(200));
        assert_eq!(m.entry_for(0x15), None);
        let e = m.entry_for(0x20).expect("exists");
        assert_eq!(
            e.stats_for(1).and_then(|s| s.bloom.as_deref()),
            Some(&[0xDE, 0xAD, 0xBE, 0xEF][..])
        );
        assert_eq!(e.stats_for(7), None);
        assert_eq!(m.predicate_by_id(9).map(|p| p.created_version), Some(41));
        assert_eq!(m.predicate_by_id(8), None);
    }

    #[test]
    fn test_files_matching_prunes_on_stats() {
        let m = sample();
        // id <= 50 excludes the second file whose id range is 101..=300
        let p = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::LtEq,
            value: LakeValue::Int(50),
        };
        let hits: Vec<u64> = m.files_matching(&p).map(|e| e.partition_id).collect();
        assert_eq!(hits, vec![0x10]);
        // A column with no stats prunes nothing
        let q = LakePredicate::Compare {
            column_id: 1,
            op: CompareOp::Eq,
            value: LakeValue::Str("bob".into()),
        };
        let hits: Vec<u64> = m.files_matching(&q).map(|e| e.partition_id).collect();
        assert_eq!(hits, vec![0x10, 0x20]);
    }

    #[test]
    fn test_validate_rejects_broken_invariants() {
        let mut unsorted = sample();
        unsorted.entries.swap(0, 1);
        assert!(unsorted.validate().is_err());

        let mut dangling = sample();
        dangling.entries[1].delete_predicate_ids = vec![777];
        assert!(dangling.validate().is_err());

        let mut bad_nulls = sample();
        std::sync::Arc::make_mut(&mut bad_nulls.entries[0].column_stats)[0]
            .bounds
            .null_count = 500;
        assert!(bad_nulls.validate().is_err());

        let mut unsorted_stats = sample();
        std::sync::Arc::make_mut(&mut unsorted_stats.entries[1].column_stats).swap(0, 1);
        assert!(unsorted_stats.validate().is_err());
    }

    #[test]
    fn test_decode_rejects_wrong_version_and_flags() {
        let bytes = sample().encode();

        let mut wrong_version = bytes.clone();
        wrong_version[4] = 99;
        let crc_field = wrong_version.len() - 8;
        let crc = crc32fast::hash(&wrong_version[..crc_field]);
        wrong_version[crc_field..crc_field + 4].copy_from_slice(&crc.to_le_bytes());
        let err = ManifestFile::decode(&wrong_version, "test.zym").expect_err("rejects");
        assert!(err.to_string().contains("format version"));

        let mut unknown_flags = bytes.clone();
        unknown_flags[6] = 1;
        let crc = crc32fast::hash(&unknown_flags[..crc_field]);
        unknown_flags[crc_field..crc_field + 4].copy_from_slice(&crc.to_le_bytes());
        let err = ManifestFile::decode(&unknown_flags, "test.zym").expect_err("rejects");
        assert!(err.to_string().contains("flags"));
    }

    /// A bloom on the leading range-partitioned key removes files the key's
    /// own bounds removed first, so it is bytes in every manifest entry for
    /// nothing. Nothing else in a spec qualifies, and dropping a filter
    /// that would have pruned is a regression rather than a saving
    #[test]
    fn test_only_the_leading_range_key_makes_a_bloom_redundant() {
        let mut manifest = sample();
        manifest
            .properties
            .insert("bloom_filter_columns".to_string(), "id,name".to_string());

        // No layout at all, so nothing is covered
        manifest.cluster_spec = ClusterSpec::none();
        assert_eq!(manifest.bloom_columns(), vec![0, 1]);
        assert!(manifest.redundant_bloom_columns().is_empty());

        // Leading range key: each file holds a contiguous range of it
        manifest.cluster_spec = ClusterSpec {
            spec_id: 1,
            keys: vec![
                cluster_key(0, ClusterStrategy::RangePartition),
                cluster_key(1, ClusterStrategy::RangePartition),
            ],
        };
        assert_eq!(manifest.redundant_bloom_columns(), vec![0]);
        assert_eq!(
            manifest.bloom_columns(),
            vec![1],
            "a key after the leading one spans the domain once per run of leading values, so \
             its filter still removes files its bounds keep"
        );

        // Z-order interleaves every key, so no single one gets contiguous
        // ranges and every declared filter still earns its bytes
        manifest.cluster_spec = ClusterSpec {
            spec_id: 2,
            keys: vec![
                cluster_key(0, ClusterStrategy::BitInterleave),
                cluster_key(1, ClusterStrategy::BitInterleave),
            ],
        };
        assert!(manifest.redundant_bloom_columns().is_empty());
        assert_eq!(manifest.bloom_columns(), vec![0, 1]);

        // AntiCluster spreads the key on purpose, which makes its bounds
        // maximally wide and the filter the only thing that prunes it
        manifest.cluster_spec = ClusterSpec {
            spec_id: 3,
            keys: vec![cluster_key(0, ClusterStrategy::AntiCluster)],
        };
        assert!(manifest.redundant_bloom_columns().is_empty());
        assert_eq!(manifest.bloom_columns(), vec![0, 1]);
    }

    /// A cluster key nobody declared a filter for is not reported as a
    /// dropped request, because none was made
    #[test]
    fn test_a_cluster_key_with_no_declared_bloom_reports_nothing() {
        let mut manifest = sample();
        manifest
            .properties
            .insert("bloom_filter_columns".to_string(), "name".to_string());
        manifest.cluster_spec = ClusterSpec {
            spec_id: 1,
            keys: vec![cluster_key(0, ClusterStrategy::RangePartition)],
        };
        assert!(manifest.redundant_bloom_columns().is_empty());
        assert_eq!(manifest.bloom_columns(), vec![1]);
    }

    fn cluster_key(column_id: u32, strategy: ClusterStrategy) -> ClusterKey {
        ClusterKey {
            column_id,
            strategy,
            param: 0,
        }
    }

    /// The two thresholds are independent, either one is enough, and both
    /// together still describe one rewrite
    #[test]
    fn test_compaction_need_reads_both_thresholds() {
        // Eight files of ten rows each against a target of four hundred:
        // the small threshold is a hundred, so every file is small
        let mut manifest = sized_manifest(&[10; 8]);
        let need = manifest.compaction_need(400);
        assert_eq!(need.small_files, 8);
        assert_eq!(need.total_files, 8);
        assert_eq!(need.trigger, Some(CompactionTrigger::SmallFiles));

        // Two small files out of ten is a fifth, under the quarter default
        let mut sizes = vec![10, 10];
        sizes.extend(std::iter::repeat_n(200u64, 8));
        manifest = sized_manifest(&sizes);
        let need = manifest.compaction_need(400);
        assert_eq!(need.small_files, 2);
        assert_eq!(
            need.trigger, None,
            "a fifth of the files being small is inside the default"
        );

        // Raising the bar turns the same table into work
        manifest
            .properties
            .insert(AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY.into(), "0.1".into());
        assert_eq!(
            manifest.compaction_need(400).trigger,
            Some(CompactionTrigger::SmallFiles)
        );
    }

    /// Rows a delete removed logically and no rewrite removed physically
    /// are read and discarded by every scan, and the count is exact
    /// because the delete counted them while it had the rows in hand
    #[test]
    fn test_compaction_need_reads_rows_a_delete_left_behind() {
        // A target of a thousand puts the small threshold at 250, so
        // five hundred row files are the shape the table is aiming for and
        // only the delete threshold is in play
        let mut manifest = sized_manifest(&[500, 500]);
        assert_eq!(manifest.total_rows(), 1000);
        assert_eq!(manifest.compaction_need(1000).trigger, None);

        manifest.delete_predicates = vec![DeletePredicate {
            id: 1,
            sql: "x = 1".into(),
            predicate: LakePredicate::IsNull { column_id: 0 },
            created_version: 1,
            pending_rows: 150,
        }];
        assert_eq!(manifest.pending_deleted_rows(), 150);
        assert_eq!(
            manifest.compaction_need(1000).trigger,
            None,
            "fifteen percent is inside the twenty percent default"
        );

        manifest.delete_predicates[0].pending_rows = 250;
        let need = manifest.compaction_need(1000);
        assert_eq!(need.trigger, Some(CompactionTrigger::DeadRows));
        assert_eq!(need.pending_deleted_rows, 250);
        assert_eq!(need.total_rows, 1000);

        // Both at once is still one rewrite
        let mut both = sized_manifest(&[10, 10]);
        both.delete_predicates = manifest.delete_predicates.clone();
        assert_eq!(
            both.compaction_need(4000).trigger,
            Some(CompactionTrigger::Both)
        );
    }

    /// Rewriting one file into one file moves the same rows into the same
    /// shape, so a table holding a single small file would otherwise trip
    /// the ratio on every tick forever
    #[test]
    fn test_one_small_file_is_not_worth_a_rewrite() {
        let manifest = sized_manifest(&[10]);
        let need = manifest.compaction_need(400);
        assert_eq!(need.small_files, 1);
        assert_eq!(need.trigger, None);
    }

    /// A threshold nobody can read falls back to the shipped one rather
    /// than stopping maintenance on the table
    #[test]
    fn test_an_unreadable_threshold_falls_back() {
        let mut manifest = sized_manifest(&[10; 8]);
        manifest
            .properties
            .insert(AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY.into(), "soon".into());
        assert_eq!(
            manifest.auto_compact_small_file_ratio(),
            DEFAULT_AUTO_COMPACT_SMALL_FILE_RATIO
        );
        manifest
            .properties
            .insert(AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY.into(), "-1".into());
        assert_eq!(
            manifest.auto_compact_small_file_ratio(),
            DEFAULT_AUTO_COMPACT_SMALL_FILE_RATIO
        );
        // A threshold above one can never trip, which is how a table opts
        // out of one of the two triggers
        manifest
            .properties
            .insert(AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY.into(), "2".into());
        assert_eq!(manifest.compaction_need(400).trigger, None);
    }

    /// The table's own target wins over whatever the node runs, because a
    /// node-wide default is the right answer only for a table that
    /// expressed no opinion
    #[test]
    fn test_a_table_target_overrides_the_node_default() {
        let mut manifest = sized_manifest(&[10; 8]);
        assert_eq!(manifest.target_rows_per_file(400), 400);
        manifest
            .properties
            .insert(TARGET_ROWS_PER_FILE_PROPERTY.into(), "20".into());
        assert_eq!(manifest.target_rows_per_file(400), 20);
        // A threshold of five now, so ten-row files are not small
        assert_eq!(manifest.compaction_need(400).small_files, 0);
    }

    fn sized_manifest(row_counts: &[u64]) -> ManifestFile {
        let mut manifest = sample();
        manifest.delete_predicates = Vec::new();
        manifest.entries = row_counts
            .iter()
            .enumerate()
            .map(|(i, rows)| PartitionEntry {
                partition_id: i as u64 + 1,
                size_bytes: rows * 8,
                row_count: *rows,
                added_version: 1,
                cluster_spec_id: 0,
                column_stats: std::sync::Arc::new(Vec::new()),
                delete_predicate_ids: Vec::new(),
            })
            .collect();
        manifest
    }

    /// Everything a plan is costed against moves the epoch, and nothing
    /// else does. A change that moved it for no reason would evict every
    /// cached plan on the node; one that did not move it would keep
    /// serving plans priced against a layout that is gone
    #[test]
    fn test_the_clustering_epoch_moves_with_what_a_plan_is_costed_against() {
        let base = sample();
        let baseline = base.clustering_epoch();
        assert_eq!(
            base.clone().clustering_epoch(),
            baseline,
            "the same state has to produce the same value"
        );

        // The spec, because it decides which columns prune
        let mut changed = base.clone();
        changed.cluster_spec = ClusterSpec {
            spec_id: base.cluster_spec.spec_id + 1,
            keys: base.cluster_spec.keys.clone(),
        };
        assert_ne!(changed.clustering_epoch(), baseline);

        // The keys inside it, at the same spec id
        let mut changed = base.clone();
        changed.cluster_spec.keys = vec![ClusterKey {
            column_id: 1,
            strategy: ClusterStrategy::RangePartition,
            param: 16,
        }];
        assert_ne!(changed.clustering_epoch(), baseline);

        // The strategy, which decides how a key buckets
        let mut changed = base.clone();
        changed.cluster_spec.keys[0].strategy = ClusterStrategy::BitInterleave;
        assert_ne!(changed.clustering_epoch(), baseline);

        // The policy, because it decides whether the spec will survive
        let mut changed = base.clone();
        changed
            .properties
            .insert(CLUSTERING_MODE_PROPERTY.into(), "force".into());
        assert_ne!(changed.clustering_epoch(), baseline);
        let mut changed = base.clone();
        changed
            .properties
            .insert(CLUSTERING_SCHEDULE_PROPERTY.into(), "ondemand".into());
        assert_ne!(changed.clustering_epoch(), baseline);

        // The schema, because adding an expression column changes which
        // predicates can reach a stored value
        let mut changed = base.clone();
        changed.schema.schema_id += 1;
        assert_ne!(changed.clustering_epoch(), baseline);

        // Conformance, which is why a pass that commits files under an
        // unchanged spec still evicts: the layout did not move but how
        // much of the data sits under it did
        let mut changed = base.clone();
        for entry in &mut changed.entries {
            entry.cluster_spec_id = changed.cluster_spec.spec_id;
        }
        assert_ne!(changed.clustering_epoch(), baseline);

        // Rows, because a scan's cost is read off them
        let mut changed = base.clone();
        changed.entries[0].row_count += 1;
        assert_ne!(changed.clustering_epoch(), baseline);
    }

    /// A property nothing costs a plan against must not evict every cached
    /// plan on the node
    #[test]
    fn test_an_unrelated_property_leaves_the_clustering_epoch_alone() {
        let base = sample();
        let baseline = base.clustering_epoch();
        let mut changed = base.clone();
        changed
            .properties
            .insert("bloom_filter_columns".into(), "name".into());
        assert_eq!(changed.clustering_epoch(), baseline);
        changed
            .properties
            .insert(AUTO_COMPACT_DEAD_ROW_RATIO_PROPERTY.into(), "0.5".into());
        assert_eq!(
            changed.clustering_epoch(),
            baseline,
            "a maintenance threshold changes when a rewrite happens, not what a plan costs"
        );
    }
}
