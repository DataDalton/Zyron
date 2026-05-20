//! Catalog checkpoint marker file.
//!
//! A single-purpose 24-byte file written atomically by Catalog::checkpoint
//! that records the highest WAL LSN whose effects are guaranteed reflected
//! in storage pages on disk. On reopen, Catalog::new reads this marker
//! before deciding whether to scan the WAL for unflushed DDL: if the marker
//! covers the current WAL frontier, recovery is a no-op and reopen is
//! constant-time.
//!
//! File format (little-endian, 24 bytes total):
//!
//! | offset | size | field            | notes                                |
//! |--------|------|------------------|--------------------------------------|
//! | 0      | 8    | magic            | b"ZYCATCKP"                          |
//! | 8      | 4    | version          | format version, starts at 1          |
//! | 12     | 8    | last_applied_lsn | highest LSN reflected in storage     |
//! | 20     | 4    | checksum         | zyron_common::hash32 over bytes 0..20|
//!
//! Atomic update: writer writes the new payload to `catalog.checkpoint.new`,
//! fsyncs, then renames over `catalog.checkpoint`. A crash mid-write leaves
//! either the old marker intact or the staging file orphaned. Readers
//! detect a corrupt or absent marker and fall back to the full WAL scan,
//! which is still correct.

use std::io::Write;
use std::path::{Path, PathBuf};

use zyron_common::{Result, ZyronError, hash32};

const MAGIC: [u8; 8] = *b"ZYCATCKP";
const VERSION: u32 = 1;
const MARKER_LEN: usize = 24;
const FILE_NAME: &str = "catalog.checkpoint";
const STAGING_NAME: &str = "catalog.checkpoint.new";

/// Decoded catalog checkpoint marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogCheckpoint {
    pub last_applied_lsn: u64,
}

fn marker_path(dir: &Path) -> PathBuf {
    dir.join(FILE_NAME)
}

fn staging_path(dir: &Path) -> PathBuf {
    dir.join(STAGING_NAME)
}

fn encode(ckpt: &CatalogCheckpoint) -> [u8; MARKER_LEN] {
    let mut buf = [0u8; MARKER_LEN];
    buf[0..8].copy_from_slice(&MAGIC);
    buf[8..12].copy_from_slice(&VERSION.to_le_bytes());
    buf[12..20].copy_from_slice(&ckpt.last_applied_lsn.to_le_bytes());
    let checksum = hash32(&buf[0..20]);
    buf[20..24].copy_from_slice(&checksum.to_le_bytes());
    buf
}

fn decode(buf: &[u8]) -> Option<CatalogCheckpoint> {
    if buf.len() != MARKER_LEN {
        return None;
    }
    if buf[0..8] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if version != VERSION {
        return None;
    }
    let last_applied_lsn = u64::from_le_bytes([
        buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
    ]);
    let expected = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let actual = hash32(&buf[0..20]);
    if expected != actual {
        return None;
    }
    Some(CatalogCheckpoint { last_applied_lsn })
}

/// Reads the catalog checkpoint marker. Returns Ok(None) when the file is
/// absent, malformed, or corrupt; the caller falls back to a full WAL scan.
pub fn read(dir: &Path) -> Result<Option<CatalogCheckpoint>> {
    let path = marker_path(dir);
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(ZyronError::IoError(format!(
                "read {}: {}",
                path.display(),
                e
            )));
        }
    };
    Ok(decode(&data))
}

/// Writes the catalog checkpoint marker atomically. The new payload lands
/// on a staging file, is fsynced, then renamed over the canonical name so
/// readers never observe a half-written marker.
pub fn write_atomic(dir: &Path, ckpt: &CatalogCheckpoint) -> Result<()> {
    let target = marker_path(dir);
    let staging = staging_path(dir);
    let buf = encode(ckpt);
    {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&staging)
            .map_err(|e| ZyronError::IoError(format!("open {}: {}", staging.display(), e)))?;
        f.write_all(&buf)
            .map_err(|e| ZyronError::IoError(format!("write {}: {}", staging.display(), e)))?;
        f.sync_all()
            .map_err(|e| ZyronError::IoError(format!("fsync {}: {}", staging.display(), e)))?;
    }
    std::fs::rename(&staging, &target).map_err(|e| {
        ZyronError::IoError(format!(
            "rename {} -> {}: {}",
            staging.display(),
            target.display(),
            e
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_through_disk() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(read(tmp.path()).unwrap().is_none());
        let ckpt = CatalogCheckpoint {
            last_applied_lsn: 4_294_967_300u64,
        };
        write_atomic(tmp.path(), &ckpt).unwrap();
        let recovered = read(tmp.path()).unwrap().unwrap();
        assert_eq!(recovered, ckpt);
    }

    #[test]
    fn missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(read(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn corrupted_marker_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let ckpt = CatalogCheckpoint {
            last_applied_lsn: 42,
        };
        write_atomic(tmp.path(), &ckpt).unwrap();
        // Corrupt the LSN bytes but leave CRC stale, so the CRC mismatches.
        let path = marker_path(tmp.path());
        let mut data = std::fs::read(&path).unwrap();
        data[15] ^= 0xFF;
        std::fs::write(&path, data).unwrap();
        assert!(read(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn wrong_magic_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let path = marker_path(tmp.path());
        std::fs::write(&path, b"not_a_zyron_catalog_ckpt").unwrap();
        assert!(read(tmp.path()).unwrap().is_none());
    }
}
