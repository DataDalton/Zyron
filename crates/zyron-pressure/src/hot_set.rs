//! What a node had in memory, in a form another node can read.
//!
//! Removing a node from a mesh throws away everything it had cached. The data
//! survives, so nothing is lost in the sense that matters for correctness, but
//! the queries that were being served out of its buffer pool now go to a
//! survivor that has never read those pages, and the mesh spends the minutes
//! after a scale-in slower than the minutes before it. That is the cost that
//! makes operators turn scale-in off.
//!
//! So a draining node writes down what it was holding and hands it over. Two
//! lists, because they answer different questions:
//!
//! - The page identifiers say what to read. A survivor prefetches them during
//!   the drain window, while the draining node is still serving, so the
//!   handover happens before the traffic moves rather than after.
//! - The query shapes say what mattered. A survivor that cannot afford the
//!   whole page list still knows which query templates the departing node was
//!   serving, which is what its planner and its own prefetch need.
//!
//! The same manifest is what makes scale-to-zero honest. A node that stops
//! entirely has to be able to come back without reading its working set one
//! page fault at a time, and the manifest is the only record of what that
//! working set was.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use zyron_common::error::{Result, ZyronError};
use zyron_common::page::PageId;

/// File name inside the data directory.
pub const HOT_SET_FILE: &str = "hot_set_manifest";

const MAGIC: [u8; 8] = *b"ZYHOTSET";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 4 + 8 + 8 + 4 + 4;

/// Pages a draining node hands over by default.
///
/// Sized so the manifest stays small enough to gossip and large enough to
/// carry a real working set: at 16KB pages this describes a gigabyte of
/// resident data, which is the part of a buffer pool that repeat traffic
/// actually returns to.
pub const DEFAULT_HOT_PAGES: u32 = 65_536;

/// Query shapes handed over by default.
pub const DEFAULT_HOT_QUERIES: u32 = 256;

/// Bound on what an operator may configure, so a manifest cannot grow past
/// what a survivor can prefetch inside a drain window.
pub const MAX_HOT_PAGES: u32 = 1_048_576;
pub const MAX_HOT_QUERIES: u32 = 4_096;

/// One query shape the departing node was serving.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HotQuery {
    /// Structural hash of the plan, stable across parameter values
    pub fingerprint: u64,
    /// Times it ran inside the observation window
    pub executions: u32,
    /// Mean estimated work, in microseconds, so a survivor can rank by cost
    /// rather than by count
    pub mean_work_us: u32,
}

/// What one node was holding.
#[derive(Debug, Clone, PartialEq)]
pub struct HotSetManifest {
    pub node_id: u64,
    pub generated_us: i64,
    /// Resident page identifiers, packed and sorted. Sorted because the
    /// survivor reads them in order, which turns a random read pattern into a
    /// sequential one on every device that cares about the difference
    pub pages: Vec<u64>,
    /// Query shapes, heaviest first
    pub queries: Vec<HotQuery>,
}

impl HotSetManifest {
    pub fn new(node_id: u64, generated_us: i64) -> Self {
        Self {
            node_id,
            generated_us,
            pages: Vec::new(),
            queries: Vec::new(),
        }
    }

    /// Builds a manifest from resident pages and observed query shapes.
    ///
    /// Pages arrive ranked by how hot the pool believes they are and leave
    /// sorted by identifier, because the ranking decides what is kept and the
    /// order decides how fast it reads back.
    pub fn build(
        node_id: u64,
        generated_us: i64,
        ranked_pages: impl IntoIterator<Item = PageId>,
        page_limit: u32,
        mut queries: Vec<HotQuery>,
        query_limit: u32,
    ) -> Self {
        let mut pages: Vec<u64> = ranked_pages
            .into_iter()
            .take(page_limit.min(MAX_HOT_PAGES) as usize)
            .map(|p| p.as_u64())
            .collect();
        pages.sort_unstable();
        pages.dedup();

        queries.sort_unstable_by(|a, b| {
            let a_cost = a.executions as u64 * a.mean_work_us as u64;
            let b_cost = b.executions as u64 * b.mean_work_us as u64;
            b_cost.cmp(&a_cost).then(b.executions.cmp(&a.executions))
        });
        queries.truncate(query_limit.min(MAX_HOT_QUERIES) as usize);

        Self {
            node_id,
            generated_us,
            pages,
            queries,
        }
    }

    pub fn page_ids(&self) -> impl Iterator<Item = PageId> + '_ {
        self.pages.iter().copied().map(PageId::from_u64)
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty() && self.queries.is_empty()
    }

    /// Sorted page identifiers pack tightly as deltas, which is what keeps a
    /// million-page manifest inside a gossip frame.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(HEADER_LEN + self.pages.len() * 3 + self.queries.len() * 16);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        // CRC placeholder, filled once the body is in place
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&self.node_id.to_le_bytes());
        buf.extend_from_slice(&self.generated_us.to_le_bytes());
        buf.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.queries.len() as u32).to_le_bytes());

        let mut previous = 0u64;
        for packed in &self.pages {
            write_varint(&mut buf, packed.wrapping_sub(previous));
            previous = *packed;
        }
        for query in &self.queries {
            buf.extend_from_slice(&query.fingerprint.to_le_bytes());
            buf.extend_from_slice(&query.executions.to_le_bytes());
            buf.extend_from_slice(&query.mean_work_us.to_le_bytes());
        }

        let crc = zyron_common::checksum::hash32(&buf[HEADER_LEN..]);
        buf[12..16].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let bad = |reason: &str| ZyronError::ConfigError(format!("hot set manifest {reason}"));
        if bytes.len() < HEADER_LEN {
            return Err(bad("is shorter than its header"));
        }
        if bytes[..8] != MAGIC {
            return Err(bad("does not carry the expected magic"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| bad("header"))?);
        if version != VERSION {
            // No back-compat by design. The manifest describes a page layout,
            // and reading one written under different rules would prefetch
            // the wrong pages while reporting success
            return Err(bad(&format!(
                "is version {version}, this build writes {VERSION}"
            )));
        }
        let stored_crc = u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| bad("crc"))?);
        if zyron_common::checksum::hash32(&bytes[HEADER_LEN..]) != stored_crc {
            return Err(bad("failed its checksum"));
        }
        let node_id = u64::from_le_bytes(bytes[16..24].try_into().map_err(|_| bad("node"))?);
        let generated_us = i64::from_le_bytes(bytes[24..32].try_into().map_err(|_| bad("time"))?);
        let page_count =
            u32::from_le_bytes(bytes[32..36].try_into().map_err(|_| bad("page count"))?) as usize;
        let query_count =
            u32::from_le_bytes(bytes[36..40].try_into().map_err(|_| bad("query count"))?) as usize;
        if page_count > MAX_HOT_PAGES as usize {
            return Err(bad(&format!(
                "declares {page_count} pages, the bound is {MAX_HOT_PAGES}"
            )));
        }
        if query_count > MAX_HOT_QUERIES as usize {
            return Err(bad(&format!(
                "declares {query_count} queries, the bound is {MAX_HOT_QUERIES}"
            )));
        }

        let mut off = HEADER_LEN;
        let mut pages = Vec::with_capacity(page_count);
        let mut previous = 0u64;
        for _ in 0..page_count {
            let (delta, used) = read_varint(bytes, off).ok_or_else(|| bad("ends inside a page"))?;
            off += used;
            previous = previous.wrapping_add(delta);
            pages.push(previous);
        }

        let mut queries = Vec::with_capacity(query_count);
        for _ in 0..query_count {
            if off + 16 > bytes.len() {
                return Err(bad("ends inside a query"));
            }
            let fingerprint =
                u64::from_le_bytes(bytes[off..off + 8].try_into().map_err(|_| bad("query"))?);
            let executions = u32::from_le_bytes(
                bytes[off + 8..off + 12]
                    .try_into()
                    .map_err(|_| bad("query"))?,
            );
            let mean_work_us = u32::from_le_bytes(
                bytes[off + 12..off + 16]
                    .try_into()
                    .map_err(|_| bad("query"))?,
            );
            off += 16;
            queries.push(HotQuery {
                fingerprint,
                executions,
                mean_work_us,
            });
        }

        Ok(Self {
            node_id,
            generated_us,
            pages,
            queries,
        })
    }

    /// Writes the manifest where a resuming node will look for it.
    ///
    /// Temp and rename, so a node that dies mid-write leaves the previous
    /// manifest intact rather than a truncated one. A truncated manifest is
    /// worse than none: scale-to-zero would see a file, allow the shutdown,
    /// and the resume would prefetch a fraction of the working set.
    pub fn persist(&self, data_dir: &Path) -> Result<()> {
        fs::create_dir_all(data_dir)?;
        let path = manifest_path(data_dir);
        let temp = path.with_extension("tmp");
        let bytes = self.encode();
        {
            let mut file = fs::File::create(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &path)?;
        Ok(())
    }

    /// Reads a persisted manifest.
    ///
    /// Absent and unreadable are different answers on purpose. A node with no
    /// manifest starts cold, which is expected on a first boot. A node with a
    /// manifest it cannot read has lost the record of its working set while
    /// something told it that record existed, and scale-to-zero may have been
    /// allowed on the strength of that file.
    pub fn load(data_dir: &Path) -> Result<Option<Self>> {
        let path = manifest_path(data_dir);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        Self::decode(&bytes).map(Some)
    }

    /// Whether a readable manifest exists, which is one of the two conditions
    /// on going to zero.
    pub fn exists(data_dir: &Path) -> bool {
        manifest_path(data_dir).is_file()
    }
}

pub fn manifest_path(data_dir: &Path) -> PathBuf {
    data_dir.join(HOT_SET_FILE)
}

/// What a survivor did with a manifest it was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefetchReport {
    /// Pages the manifest named
    pub requested: u64,
    /// Pages read in, which is the number that moves the hit rate
    pub loaded: u64,
    /// Pages already resident, so the handover cost nothing for these
    pub already_resident: u64,
    /// Pages the pool declined, because prefetching must never evict
    /// something the survivor is using to make room for something it might
    pub declined: u64,
    /// Pages that could not be read
    pub failed: u64,
}

impl PrefetchReport {
    pub fn coverage(&self) -> f64 {
        if self.requested == 0 {
            return 1.0;
        }
        (self.loaded + self.already_resident) as f64 / self.requested as f64
    }
}

fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn read_varint(bytes: &[u8], mut off: usize) -> Option<(u64, usize)> {
    let start = off;
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes.get(off)?;
        off += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((value, off - start));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> HotSetManifest {
        HotSetManifest::build(
            42,
            1_700_000_000_000_000,
            (0..5000u64).map(|n| PageId::new(n as u32 % 3, n)),
            DEFAULT_HOT_PAGES,
            vec![
                HotQuery {
                    fingerprint: 0xAAAA,
                    executions: 10,
                    mean_work_us: 1000,
                },
                HotQuery {
                    fingerprint: 0xBBBB,
                    executions: 1000,
                    mean_work_us: 5,
                },
                HotQuery {
                    fingerprint: 0xCCCC,
                    executions: 3,
                    mean_work_us: 900_000,
                },
            ],
            DEFAULT_HOT_QUERIES,
        )
    }

    #[test]
    fn a_manifest_round_trips() {
        let original = manifest();
        let decoded = HotSetManifest::decode(&original.encode()).expect("decode");
        assert_eq!(decoded, original);
    }

    /// Sorted deltas are the reason a large manifest is transferable at all.
    #[test]
    fn sorted_pages_encode_far_smaller_than_their_raw_width() {
        let dense = HotSetManifest::build(
            1,
            0,
            (0..100_000u64).map(|n| PageId::new(0, n)),
            MAX_HOT_PAGES,
            Vec::new(),
            0,
        );
        let encoded = dense.encode().len();
        let raw = 100_000 * 8;
        assert!(
            encoded < raw / 3,
            "delta encoding gained nothing: {encoded} against {raw}"
        );
        assert_eq!(
            HotSetManifest::decode(&dense.encode())
                .expect("decode")
                .pages,
            dense.pages
        );
    }

    /// The heaviest shapes survive the cut, not the most frequent ones. A
    /// survivor that warms the wrong queries has spent the drain window on
    /// point lookups it would have served anyway.
    #[test]
    fn queries_are_ranked_by_total_work() {
        let ranked = HotSetManifest::build(
            1,
            0,
            std::iter::empty(),
            0,
            vec![
                HotQuery {
                    fingerprint: 1,
                    executions: 10_000,
                    mean_work_us: 1,
                },
                HotQuery {
                    fingerprint: 2,
                    executions: 5,
                    mean_work_us: 2_000_000,
                },
            ],
            1,
        );
        assert_eq!(ranked.queries.len(), 1);
        assert_eq!(ranked.queries[0].fingerprint, 2);
    }

    #[test]
    fn limits_are_enforced_on_the_way_in() {
        let capped = HotSetManifest::build(
            1,
            0,
            (0..1000u64).map(|n| PageId::new(0, n)),
            10,
            Vec::new(),
            0,
        );
        assert_eq!(capped.pages.len(), 10);
    }

    /// Duplicated identifiers would make the survivor read the same page
    /// twice and report a coverage it did not achieve.
    #[test]
    fn duplicate_pages_collapse() {
        let doubled = HotSetManifest::build(
            1,
            0,
            [PageId::new(0, 7), PageId::new(0, 7), PageId::new(1, 7)],
            100,
            Vec::new(),
            0,
        );
        assert_eq!(doubled.pages.len(), 2);
    }

    #[test]
    fn a_corrupted_manifest_is_refused_rather_than_half_read() {
        let mut bytes = manifest().encode();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(HotSetManifest::decode(&bytes).is_err());

        let mut wrong_version = manifest().encode();
        wrong_version[8] = 9;
        assert!(HotSetManifest::decode(&wrong_version).is_err());
    }

    #[test]
    fn a_declared_count_past_the_bound_is_refused() {
        let mut bytes = manifest().encode();
        bytes[32..36].copy_from_slice(&(MAX_HOT_PAGES + 1).to_le_bytes());
        let crc = zyron_common::checksum::hash32(&bytes[HEADER_LEN..]);
        bytes[12..16].copy_from_slice(&crc.to_le_bytes());
        assert!(HotSetManifest::decode(&bytes).is_err());
    }

    #[test]
    fn persist_and_load_agree() {
        let dir =
            std::env::temp_dir().join(format!("zyron_hot_set_{}_{}", std::process::id(), line!()));
        let _ = fs::remove_dir_all(&dir);
        let original = manifest();
        original.persist(&dir).expect("persist");
        assert!(HotSetManifest::exists(&dir));
        assert_eq!(HotSetManifest::load(&dir).expect("load"), Some(original));
        let empty = dir.join("empty");
        assert_eq!(HotSetManifest::load(&empty).expect("absent"), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn coverage_counts_pages_that_were_already_there() {
        let report = PrefetchReport {
            requested: 100,
            loaded: 60,
            already_resident: 30,
            declined: 10,
            failed: 0,
        };
        assert!((report.coverage() - 0.9).abs() < 1e-9);
        assert_eq!(PrefetchReport::default().coverage(), 1.0);
    }
}
