//! Persisted calibration, keyed by hardware fingerprint.
//!
//! Measuring what a row costs takes real traffic, and a node that has to start
//! from nothing every boot spends its first minutes costing plans against
//! values it has not verified. So what it learned is written down under the
//! fingerprint of the machine it learned it on, and read back on the next
//! start.
//!
//! Keying by fingerprint rather than by node is what makes it more than a
//! restart optimisation. Two nodes of the same instance type hash the same, so
//! a fresh node can adopt what a sibling already measured and serve calibrated
//! from its first query. That is the difference between a cold start that is
//! slow and a cold start that is only cold.
//!
//! Stored as a node-local file rather than a catalog table. It holds no user
//! data, wants no transaction, and has to be readable before the catalog is
//! open, which is the same set of properties that put the node identity and
//! the peer registry in files beside it.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::capability::{HardwareFingerprint, OperatorCoefficients, OperatorKind};
use zyron_common::error::{Result, ZyronError};

/// File name inside the data directory.
pub const CALIBRATION_FILE: &str = "calibration_cache";

const MAGIC: [u8; 8] = *b"ZYCALIB\x00";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 4 + 4;

/// Fingerprints kept. A node sees one shape of machine, or a handful across a
/// fleet it has gossiped with, so this is far above what any real deployment
/// reaches and exists to bound the file rather than to ration it.
pub const MAX_FINGERPRINTS: usize = 64;

/// What one hardware shape was measured to cost.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationEntry {
    pub fingerprint: HardwareFingerprint,
    pub coefficients: OperatorCoefficients,
    /// When this was last written, used to decide what to evict
    pub updated_us: i64,
    /// Which node measured it, so a view can say where an inherited
    /// calibration came from
    pub source_node_id: u64,
}

/// Every hardware shape this node knows the cost of.
#[derive(Debug, Clone, Default)]
pub struct CalibrationCache {
    entries: HashMap<HardwareFingerprint, CalibrationEntry>,
}

impl CalibrationCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the cache. A missing file is an empty cache, not an error: a node
    /// starting for the first time has nothing to read and that is normal.
    ///
    /// A corrupt file is also an empty cache rather than a refusal to start.
    /// Losing calibration costs a node the first minutes of its measurements,
    /// which it will make again, and refusing to boot over it would turn a
    /// performance hint into an outage.
    pub fn load(data_dir: &Path) -> Self {
        let path = calibration_path(data_dir);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => return Self::new(),
        };
        match Self::decode(&bytes) {
            Ok(cache) => cache,
            Err(e) => {
                tracing_warn(&format!(
                    "calibration cache at {} is unreadable ({}), starting from measurement",
                    path.display(),
                    e
                ));
                Self::new()
            }
        }
    }

    /// Writes the cache through a temp file and a rename, so a crash mid-write
    /// leaves the previous cache rather than a torn one.
    pub fn persist(&self, data_dir: &Path) -> Result<()> {
        fs::create_dir_all(data_dir)?;
        let path = calibration_path(data_dir);
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

    /// What this hardware shape was measured to cost, if anything has.
    pub fn get(&self, fingerprint: HardwareFingerprint) -> Option<&CalibrationEntry> {
        self.entries.get(&fingerprint)
    }

    /// Records what this node measured, replacing any previous entry for the
    /// same hardware.
    ///
    /// Replaces rather than merges, because the node seeded its accumulator
    /// from this entry at startup: the value being written already contains
    /// the one on disk, and merging would count the same evidence twice.
    pub fn set(&mut self, entry: CalibrationEntry) {
        self.entries.insert(entry.fingerprint, entry);
        self.evict_to_bound();
    }

    /// Folds in what a peer measured for some hardware shape.
    ///
    /// Merges rather than replaces, weighting each side by the evidence behind
    /// it, because the two nodes measured different traffic and neither
    /// reading supersedes the other.
    pub fn merge_peer(&mut self, entry: CalibrationEntry) {
        match self.entries.get_mut(&entry.fingerprint) {
            Some(existing) => {
                existing.coefficients.merge(&entry.coefficients);
                if entry.updated_us > existing.updated_us {
                    existing.updated_us = entry.updated_us;
                }
            }
            None => {
                self.entries.insert(entry.fingerprint, entry);
            }
        }
        self.evict_to_bound();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every entry, oldest first.
    pub fn entries(&self) -> Vec<&CalibrationEntry> {
        let mut out: Vec<&CalibrationEntry> = self.entries.values().collect();
        out.sort_by_key(|e| e.updated_us);
        out
    }

    /// Drops the least recently updated entries once the bound is passed. The
    /// hardware a node last ran on is the hardware it is most likely to see
    /// again, so recency is the right thing to keep.
    fn evict_to_bound(&mut self) {
        while self.entries.len() > MAX_FINGERPRINTS {
            let Some(oldest) = self
                .entries
                .values()
                .min_by_key(|e| e.updated_us)
                .map(|e| e.fingerprint)
            else {
                return;
            };
            self.entries.remove(&oldest);
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut entries = self.entries();
        entries.sort_by_key(|e| e.fingerprint.0);
        let mut buf = Vec::with_capacity(HEADER_LEN + entries.len() * 256);
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        // CRC placeholder, filled once the body is in place
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for entry in entries {
            buf.extend_from_slice(&entry.fingerprint.0.to_le_bytes());
            buf.extend_from_slice(&entry.updated_us.to_le_bytes());
            buf.extend_from_slice(&entry.source_node_id.to_le_bytes());
            buf.extend_from_slice(&(OperatorKind::COUNT as u16).to_le_bytes());
            for kind in OperatorKind::ALL {
                let i = kind.index();
                buf.extend_from_slice(&entry.coefficients.ns_per_unit[i].to_le_bytes());
                buf.extend_from_slice(&entry.coefficients.sample_count[i].to_le_bytes());
            }
        }
        let crc = zyron_common::checksum::hash32(&buf[HEADER_LEN..]);
        buf[12..16].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let bad = |reason: &str| ZyronError::ConfigError(format!("calibration cache {reason}"));
        if bytes.len() < HEADER_LEN {
            return Err(bad("is shorter than its header"));
        }
        if bytes[..8] != MAGIC {
            return Err(bad("does not carry the expected magic"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| bad("header"))?);
        if version != VERSION {
            // No back-compat by design. A format change means the values were
            // measured under different rules, and reading them as though they
            // were not would silently misprice every plan
            return Err(bad(&format!(
                "is version {version}, this build writes {VERSION}"
            )));
        }
        let stored_crc = u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| bad("crc"))?);
        let count =
            u32::from_le_bytes(bytes[16..20].try_into().map_err(|_| bad("count"))?) as usize;
        if zyron_common::checksum::hash32(&bytes[HEADER_LEN..]) != stored_crc {
            return Err(bad("failed its checksum"));
        }
        if count > MAX_FINGERPRINTS {
            return Err(bad(&format!(
                "declares {count} entries, the bound is {MAX_FINGERPRINTS}"
            )));
        }

        let mut cache = Self::new();
        let mut off = HEADER_LEN;
        for _ in 0..count {
            let need = 8 + 8 + 8 + 2;
            if off + need > bytes.len() {
                return Err(bad("ends inside an entry header"));
            }
            let fingerprint = HardwareFingerprint(u64::from_le_bytes(
                bytes[off..off + 8].try_into().map_err(|_| bad("entry"))?,
            ));
            off += 8;
            let updated_us =
                i64::from_le_bytes(bytes[off..off + 8].try_into().map_err(|_| bad("entry"))?);
            off += 8;
            let source_node_id =
                u64::from_le_bytes(bytes[off..off + 8].try_into().map_err(|_| bad("entry"))?);
            off += 8;
            let kinds =
                u16::from_le_bytes(bytes[off..off + 2].try_into().map_err(|_| bad("entry"))?)
                    as usize;
            off += 2;
            if kinds != OperatorKind::COUNT {
                return Err(bad(&format!(
                    "holds {kinds} operator kinds, this build prices {}",
                    OperatorKind::COUNT
                )));
            }
            if off + kinds * 16 > bytes.len() {
                return Err(bad("ends inside an entry body"));
            }
            let mut coefficients = OperatorCoefficients::cold_start();
            for kind in OperatorKind::ALL {
                let i = kind.index();
                let ns =
                    f64::from_le_bytes(bytes[off..off + 8].try_into().map_err(|_| bad("entry"))?);
                off += 8;
                let samples =
                    u64::from_le_bytes(bytes[off..off + 8].try_into().map_err(|_| bad("entry"))?);
                off += 8;
                // A value that is not a positive finite number would poison
                // every estimate made against it, so the cold start value is
                // kept instead and the sample count with it
                if ns.is_finite() && ns > 0.0 {
                    coefficients.ns_per_unit[i] = ns;
                    coefficients.sample_count[i] = samples;
                }
            }
            cache.entries.insert(
                fingerprint,
                CalibrationEntry {
                    fingerprint,
                    coefficients,
                    updated_us,
                    source_node_id,
                },
            );
        }
        Ok(cache)
    }
}

/// Where the cache lives inside a data directory.
pub fn calibration_path(data_dir: &Path) -> PathBuf {
    data_dir.join(CALIBRATION_FILE)
}

/// Logging without taking a tracing dependency in this crate.
fn tracing_warn(message: &str) {
    eprintln!("warning: {message}");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(fp: u64, sort_ns: f64, samples: u64, at: i64) -> CalibrationEntry {
        let mut coefficients = OperatorCoefficients::cold_start();
        coefficients.ns_per_unit[OperatorKind::Sort.index()] = sort_ns;
        coefficients.sample_count[OperatorKind::Sort.index()] = samples;
        CalibrationEntry {
            fingerprint: HardwareFingerprint(fp),
            coefficients,
            updated_us: at,
            source_node_id: 7,
        }
    }

    #[test]
    fn a_missing_file_is_an_empty_cache_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = CalibrationCache::load(dir.path());
        assert!(cache.is_empty());
    }

    #[test]
    fn what_was_written_is_what_comes_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cache = CalibrationCache::new();
        cache.set(entry(0xABCD, 12.5, 50_000, 1_000));
        cache.set(entry(0x1234, 99.25, 7, 2_000));
        cache.persist(dir.path()).expect("persist");

        let read = CalibrationCache::load(dir.path());
        assert_eq!(read.len(), 2);
        let found = read.get(HardwareFingerprint(0xABCD)).expect("entry");
        assert!((found.coefficients.get(OperatorKind::Sort) - 12.5).abs() < 1e-9);
        assert_eq!(found.coefficients.samples(OperatorKind::Sort), 50_000);
        assert_eq!(found.source_node_id, 7);
        assert_eq!(found.updated_us, 1_000);
    }

    #[test]
    fn a_fingerprint_nobody_measured_is_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 5.0, 100, 1));
        cache.persist(dir.path()).expect("persist");
        let read = CalibrationCache::load(dir.path());
        assert!(read.get(HardwareFingerprint(999)).is_none());
    }

    /// The node seeded itself from disk at startup, so writing back must
    /// replace. Merging would fold the same evidence in twice and the sample
    /// count would run away from the number of rows actually measured.
    #[test]
    fn a_local_write_replaces_rather_than_accumulating() {
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 1_000, 1));
        cache.set(entry(1, 20.0, 1_200, 2));
        let found = cache.get(HardwareFingerprint(1)).expect("entry");
        assert!((found.coefficients.get(OperatorKind::Sort) - 20.0).abs() < 1e-9);
        assert_eq!(
            found.coefficients.samples(OperatorKind::Sort),
            1_200,
            "the local write accumulated instead of replacing"
        );
    }

    /// A peer measured different traffic, so neither reading supersedes the
    /// other and both count toward the answer.
    #[test]
    fn a_peer_report_merges_weighted_by_evidence() {
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 100, 1));
        cache.merge_peer(entry(1, 20.0, 300, 5));
        let found = cache.get(HardwareFingerprint(1)).expect("entry");
        // Three quarters of the evidence says twenty
        assert!((found.coefficients.get(OperatorKind::Sort) - 17.5).abs() < 1e-9);
        assert_eq!(found.coefficients.samples(OperatorKind::Sort), 400);
        assert_eq!(found.updated_us, 5);
    }

    #[test]
    fn a_peer_report_for_unseen_hardware_is_adopted_whole() {
        let mut cache = CalibrationCache::new();
        cache.merge_peer(entry(42, 3.5, 9_000, 10));
        let found = cache.get(HardwareFingerprint(42)).expect("entry");
        assert!((found.coefficients.get(OperatorKind::Sort) - 3.5).abs() < 1e-9);
    }

    #[test]
    fn corruption_costs_the_measurements_not_the_boot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 1_000, 1));
        cache.persist(dir.path()).expect("persist");

        let path = calibration_path(dir.path());
        let mut bytes = fs::read(&path).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&path, &bytes).expect("write");

        // Empty rather than a panic or a refusal to start
        let read = CalibrationCache::load(dir.path());
        assert!(read.is_empty());
    }

    #[test]
    fn a_truncated_file_does_not_panic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 1_000, 1));
        cache.persist(dir.path()).expect("persist");

        let path = calibration_path(dir.path());
        let bytes = fs::read(&path).expect("read");
        for cut in [0, 4, HEADER_LEN, HEADER_LEN + 5, bytes.len() - 1] {
            fs::write(&path, &bytes[..cut]).expect("write");
            let _ = CalibrationCache::load(dir.path());
        }
    }

    #[test]
    fn a_file_from_another_format_version_is_refused() {
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 1_000, 1));
        let mut bytes = cache.encode();
        bytes[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        let crc = zyron_common::checksum::hash32(&bytes[HEADER_LEN..]);
        bytes[12..16].copy_from_slice(&crc.to_le_bytes());
        assert!(CalibrationCache::decode(&bytes).is_err());
    }

    #[test]
    fn a_nonsense_coefficient_falls_back_rather_than_poisoning_estimates() {
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 1_000, 1));
        let mut bytes = cache.encode();
        // Overwrite the first coefficient with a NaN
        let first = HEADER_LEN + 8 + 8 + 8 + 2;
        bytes[first..first + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        let crc = zyron_common::checksum::hash32(&bytes[HEADER_LEN..]);
        bytes[12..16].copy_from_slice(&crc.to_le_bytes());

        let read = CalibrationCache::decode(&bytes).expect("decodes");
        let found = read.get(HardwareFingerprint(1)).expect("entry");
        let first_kind = OperatorKind::ALL[0];
        assert!(
            found.coefficients.get(first_kind).is_finite(),
            "a NaN survived into the cost model"
        );
        assert_eq!(
            found.coefficients.get(first_kind),
            first_kind.cold_start_ns_per_unit() as f64
        );
    }

    #[test]
    fn the_file_is_bounded_by_the_fingerprint_cap() {
        let mut cache = CalibrationCache::new();
        for i in 0..(MAX_FINGERPRINTS as u64 + 20) {
            cache.set(entry(i, 1.0, 1, i as i64));
        }
        assert_eq!(cache.len(), MAX_FINGERPRINTS);
        // The oldest went, the newest stayed
        assert!(cache.get(HardwareFingerprint(0)).is_none());
        assert!(
            cache
                .get(HardwareFingerprint(MAX_FINGERPRINTS as u64 + 19))
                .is_some()
        );
    }

    #[test]
    fn a_rewrite_leaves_no_temp_file_behind() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut cache = CalibrationCache::new();
        cache.set(entry(1, 10.0, 1_000, 1));
        cache.persist(dir.path()).expect("persist");
        cache.set(entry(2, 20.0, 1_000, 2));
        cache.persist(dir.path()).expect("persist");
        let temp = calibration_path(dir.path()).with_extension("tmp");
        assert!(!temp.exists(), "a temp file survived the rename");
        assert_eq!(CalibrationCache::load(dir.path()).len(), 2);
    }
}
