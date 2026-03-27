#![allow(non_snake_case)]
//! Physical backup and restore for ZyronDB.
//!
//! Copies data files and WAL segments to a destination directory, records
//! checksums in a TOML manifest, and can restore from that manifest with
//! full integrity verification.

use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use zyron_common::{Result, ZyronError};
use zyron_wal::checksum::WalHasher;

/// Size of the read/write buffer used when copying files and computing checksums.
const COPY_BUFFER_SIZE: usize = 64 * 1024;

/// Server version string embedded in every backup manifest.
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single file recorded in the backup manifest.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackupFileEntry {
    /// Path relative to the backup root (e.g. "data/00000001.zyheap").
    pub relativePath: String,
    /// File size in bytes.
    pub size: u64,
    /// Hex-encoded 32-bit checksum of the file contents (WAL two-lane hasher).
    pub checksum: String,
}

/// Top-level manifest written to `manifest.toml` inside every backup directory.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackupManifest {
    /// Manifest format version. Always 1 for this implementation.
    pub version: u32,
    /// ZyronDB server version that created the backup.
    pub serverVersion: String,
    /// ISO 8601 timestamp when the backup was created.
    pub createdAt: String,
    /// Absolute path of the original data directory.
    pub dataDir: String,
    /// Checkpoint LSN captured at the start of the backup.
    pub backupLsn: u64,
    /// First WAL segment included in the backup.
    pub walStartSegment: u32,
    /// Last WAL segment included in the backup.
    pub walEndSegment: u32,
    /// List of every file stored in the backup.
    pub files: Vec<BackupFileEntry>,
}

// ---------------------------------------------------------------------------
// BackupManager
// ---------------------------------------------------------------------------

/// Creates physical backups of a ZyronDB data directory and WAL.
pub struct BackupManager;

impl BackupManager {
    /// Run a full physical backup.
    ///
    /// Copies all data files (.zyheap, .zyridx, .zyr, .fsm) from `dataDir`
    /// and all WAL segments (.wal) from `walDir` into `destDir`. Writes a
    /// TOML manifest containing file checksums and metadata.
    pub fn backup(
        dataDir: &Path,
        walDir: &Path,
        destDir: &Path,
        checkpointLsn: u64,
    ) -> Result<BackupManifest> {
        // Create output directory structure.
        let dataDest = destDir.join("data");
        let walDest = destDir.join("wal");
        fs::create_dir_all(&dataDest)
            .map_err(|e| ZyronError::IoError(format!("failed to create data backup dir: {}", e)))?;
        fs::create_dir_all(&walDest)
            .map_err(|e| ZyronError::IoError(format!("failed to create wal backup dir: {}", e)))?;

        let mut files: Vec<BackupFileEntry> = Vec::new();

        // Copy data files.
        let dataExtensions = &[".zyheap", ".zyridx", ".zyr", ".fsm"];
        let dataFiles = Self::collectFiles(dataDir, dataExtensions)?;
        for srcPath in &dataFiles {
            let relativeFromData = srcPath
                .strip_prefix(dataDir)
                .map_err(|e| ZyronError::Internal(format!("strip_prefix failed: {}", e)))?;
            let dstPath = dataDest.join(relativeFromData);
            if let Some(parent) = dstPath.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    ZyronError::IoError(format!("failed to create parent dir: {}", e))
                })?;
            }
            let (size, checksum) = Self::copyWithChecksum(srcPath, &dstPath)?;
            let relativePath = Path::new("data")
                .join(relativeFromData)
                .to_string_lossy()
                .replace('\\', "/");
            files.push(BackupFileEntry {
                relativePath,
                size,
                checksum,
            });
        }

        // Copy WAL segments.
        let walExtensions = &[".wal"];
        let walFiles = Self::collectFiles(walDir, walExtensions)?;

        let mut walStartSegment: u32 = u32::MAX;
        let mut walEndSegment: u32 = 0;

        for srcPath in &walFiles {
            let relativeFromWal = srcPath
                .strip_prefix(walDir)
                .map_err(|e| ZyronError::Internal(format!("strip_prefix failed: {}", e)))?;
            let dstPath = walDest.join(relativeFromWal);
            if let Some(parent) = dstPath.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    ZyronError::IoError(format!("failed to create parent dir: {}", e))
                })?;
            }
            let (size, checksum) = Self::copyWithChecksum(srcPath, &dstPath)?;
            let relativePath = Path::new("wal")
                .join(relativeFromWal)
                .to_string_lossy()
                .replace('\\', "/");
            files.push(BackupFileEntry {
                relativePath,
                size,
                checksum,
            });

            // Try to extract segment number from file stem for range tracking.
            if let Some(stem) = srcPath.file_stem().and_then(|s| s.to_str()) {
                if let Ok(segNum) = stem.parse::<u32>() {
                    if segNum < walStartSegment {
                        walStartSegment = segNum;
                    }
                    if segNum > walEndSegment {
                        walEndSegment = segNum;
                    }
                }
            }
        }

        // If no WAL files were found, default both segment bounds to zero.
        if walStartSegment == u32::MAX {
            walStartSegment = 0;
        }

        let manifest = BackupManifest {
            version: 1,
            serverVersion: SERVER_VERSION.to_string(),
            createdAt: formatTimestamp(),
            dataDir: dataDir.to_string_lossy().replace('\\', "/"),
            backupLsn: checkpointLsn,
            walStartSegment,
            walEndSegment,
            files,
        };

        // Write manifest to TOML.
        let manifestToml = toml::to_string(&manifest).map_err(|e| {
            ZyronError::Internal(format!("failed to serialize backup manifest: {}", e))
        })?;
        let manifestPath = destDir.join("manifest.toml");
        fs::write(&manifestPath, manifestToml.as_bytes())
            .map_err(|e| ZyronError::IoError(format!("failed to write manifest.toml: {}", e)))?;

        Ok(manifest)
    }

    /// Copy a single file from `src` to `dst`, computing a checksum using the
    /// WAL two-lane hasher along the way. Returns (fileSize, hexChecksum).
    fn copyWithChecksum(src: &Path, dst: &Path) -> Result<(u64, String)> {
        let srcFile = fs::File::open(src)
            .map_err(|e| ZyronError::IoError(format!("failed to open {}: {}", src.display(), e)))?;
        let dstFile = fs::File::create(dst).map_err(|e| {
            ZyronError::IoError(format!("failed to create {}: {}", dst.display(), e))
        })?;

        let mut reader = BufReader::with_capacity(COPY_BUFFER_SIZE, srcFile);
        let mut writer = BufWriter::with_capacity(COPY_BUFFER_SIZE, dstFile);
        let mut hasher = WalHasher::new(0);
        let mut buffer = [0u8; COPY_BUFFER_SIZE];
        let mut totalBytes: u64 = 0;

        loop {
            let bytesRead = reader.read(&mut buffer).map_err(|e| {
                ZyronError::IoError(format!("read error for {}: {}", src.display(), e))
            })?;
            if bytesRead == 0 {
                break;
            }
            hasher.write_payload(&buffer[..bytesRead]);
            writer.write_all(&buffer[..bytesRead]).map_err(|e| {
                ZyronError::IoError(format!("write error for {}: {}", dst.display(), e))
            })?;
            totalBytes += bytesRead as u64;
        }

        writer.flush().map_err(|e| {
            ZyronError::IoError(format!("flush error for {}: {}", dst.display(), e))
        })?;

        let hexChecksum = format!("{:08x}", hasher.finish());

        Ok((totalBytes, hexChecksum))
    }

    /// Recursively walk `dir` and return paths of files whose extension matches
    /// any entry in `extensions` (each entry should include the leading dot).
    fn collectFiles(dir: &Path, extensions: &[&str]) -> Result<Vec<PathBuf>> {
        let mut result = Vec::new();
        if !dir.exists() {
            return Ok(result);
        }
        Self::collectFilesRecursive(dir, extensions, &mut result)?;
        result.sort();
        Ok(result)
    }

    fn collectFilesRecursive(
        dir: &Path,
        extensions: &[&str],
        out: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let entries = fs::read_dir(dir).map_err(|e| {
            ZyronError::IoError(format!("failed to read dir {}: {}", dir.display(), e))
        })?;
        for entry in entries {
            let entry = entry
                .map_err(|e| ZyronError::IoError(format!("failed to read dir entry: {}", e)))?;
            let path = entry.path();
            if path.is_dir() {
                Self::collectFilesRecursive(&path, extensions, out)?;
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let dotExt = format!(".{}", ext);
                    if extensions.iter().any(|&e| e == dotExt) {
                        out.push(path);
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RestoreManager
// ---------------------------------------------------------------------------

/// Restores a ZyronDB backup from a manifest directory.
pub struct RestoreManager;

impl RestoreManager {
    /// Restore a backup into `targetDir`.
    ///
    /// Reads the manifest from `backupDir/manifest.toml`, verifies all file
    /// checksums, then copies every file into `targetDir` preserving the
    /// relative directory structure.
    pub fn restore(backupDir: &Path, targetDir: &Path) -> Result<()> {
        // Read and parse the manifest.
        let manifestPath = backupDir.join("manifest.toml");
        let manifestBytes = fs::read(&manifestPath).map_err(|e| {
            ZyronError::IoError(format!("failed to read {}: {}", manifestPath.display(), e))
        })?;
        let manifestStr = String::from_utf8(manifestBytes).map_err(|e| {
            ZyronError::Internal(format!("manifest.toml is not valid UTF-8: {}", e))
        })?;
        let manifest: BackupManifest = toml::from_str(&manifestStr)
            .map_err(|e| ZyronError::Internal(format!("failed to parse manifest.toml: {}", e)))?;

        // Validate all checksums before copying anything.
        Self::validateManifest(&manifest, backupDir)?;

        // Create target directory.
        fs::create_dir_all(targetDir).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to create target dir {}: {}",
                targetDir.display(),
                e
            ))
        })?;

        // Copy each file into the target directory.
        for entry in &manifest.files {
            let srcPath = backupDir.join(&entry.relativePath);
            let dstPath = targetDir.join(&entry.relativePath);
            if let Some(parent) = dstPath.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| ZyronError::IoError(format!("failed to create dir: {}", e)))?;
            }
            fs::copy(&srcPath, &dstPath).map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to copy {} to {}: {}",
                    srcPath.display(),
                    dstPath.display(),
                    e
                ))
            })?;
        }

        Ok(())
    }

    /// Verify that every file in the manifest exists in `backupDir` and its
    /// xxh3-64 checksum matches the recorded value.
    fn validateManifest(manifest: &BackupManifest, backupDir: &Path) -> Result<()> {
        for entry in &manifest.files {
            let filePath = backupDir.join(&entry.relativePath);
            if !filePath.exists() {
                return Err(ZyronError::IoError(format!(
                    "backup file missing: {}",
                    entry.relativePath
                )));
            }
            let computedChecksum = checksumFile(&filePath)?;
            if computedChecksum != entry.checksum {
                return Err(ZyronError::IoError(format!(
                    "checksum mismatch for {}: expected {}, got {}",
                    entry.relativePath, entry.checksum, computedChecksum
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a hex checksum of a file using the WAL two-lane hasher.
fn checksumFile(path: &Path) -> Result<String> {
    let file = fs::File::open(path)
        .map_err(|e| ZyronError::IoError(format!("failed to open {}: {}", path.display(), e)))?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_SIZE, file);
    let mut hasher = WalHasher::new(0);
    let mut buffer = [0u8; COPY_BUFFER_SIZE];
    loop {
        let bytesRead = reader.read(&mut buffer).map_err(|e| {
            ZyronError::IoError(format!("read error for {}: {}", path.display(), e))
        })?;
        if bytesRead == 0 {
            break;
        }
        hasher.write_payload(&buffer[..bytesRead]);
    }
    Ok(format!("{:08x}", hasher.finish()))
}

/// Format the current wall-clock time as an ISO 8601 string: YYYY-MM-DDTHH:MM:SSZ.
///
/// Uses manual arithmetic over the Unix epoch to avoid pulling in chrono or
/// similar datetime libraries.
pub fn formatTimestamp() -> String {
    let epochSecs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Days and intra-day time.
    let secsInDay: u64 = 86400;
    let mut totalDays = (epochSecs / secsInDay) as i64;
    let daySeconds = (epochSecs % secsInDay) as u32;
    let hours = daySeconds / 3600;
    let minutes = (daySeconds % 3600) / 60;
    let seconds = daySeconds % 60;

    // Convert total days since 1970-01-01 to year/month/day.
    // Shift epoch to 0000-03-01 for simpler leap year handling.
    totalDays += 719_468; // days from 0000-03-01 to 1970-01-01
    let era = if totalDays >= 0 {
        totalDays / 146_097
    } else {
        (totalDays - 146_096) / 146_097
    };
    let dayOfEra = (totalDays - era * 146_097) as u64;
    let yearOfEra = (dayOfEra - dayOfEra / 1460 + dayOfEra / 36524 - dayOfEra / 146_096) / 365;
    let year = yearOfEra as i64 + era * 400;
    let dayOfYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100);
    let mp = (5 * dayOfYear + 2) / 153;
    let day = dayOfYear - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let adjustedYear = if month <= 2 { year + 1 } else { year };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        adjustedYear, month, day, hours, minutes, seconds
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a temporary directory structure, run a backup, verify the
    /// manifest, then restore into a fresh directory and compare file contents.
    #[test]
    fn testBackupAndRestore() {
        let tmpDir = tempfile::tempdir().expect("tempdir creation failed");
        let base = tmpDir.path();

        // Set up source directories.
        let dataDir = base.join("data");
        let walDir = base.join("wal");
        let backupDir = base.join("backup");
        let restoreDir = base.join("restore");

        fs::create_dir_all(&dataDir).expect("create data dir");
        fs::create_dir_all(&walDir).expect("create wal dir");

        // Write test data files.
        let heapContent = b"heap file contents for testing";
        let walContent = b"wal segment contents for testing";
        fs::write(dataDir.join("00000001.zyheap"), heapContent).expect("write heap");
        fs::write(walDir.join("00000001.wal"), walContent).expect("write wal");

        // Also write a non-matching file that should be excluded.
        fs::write(dataDir.join("readme.txt"), b"ignore me").expect("write txt");

        // Run backup.
        let manifest = BackupManager::backup(&dataDir, &walDir, &backupDir, 42)
            .expect("backup should succeed");

        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.backupLsn, 42);
        assert_eq!(manifest.files.len(), 2);

        // The manifest file should exist on disk.
        assert!(backupDir.join("manifest.toml").exists());

        // File entries should have non-empty checksums.
        for entry in &manifest.files {
            assert!(!entry.checksum.is_empty());
            assert!(entry.size > 0);
        }

        // Run restore.
        RestoreManager::restore(&backupDir, &restoreDir).expect("restore should succeed");

        // Verify restored file contents match originals.
        let restoredHeap =
            fs::read(restoreDir.join("data/00000001.zyheap")).expect("read restored heap");
        assert_eq!(restoredHeap.as_slice(), heapContent);

        let restoredWal = fs::read(restoreDir.join("wal/00000001.wal")).expect("read restored wal");
        assert_eq!(restoredWal.as_slice(), walContent);
    }

    /// Backup, corrupt one of the copied files, then verify that manifest
    /// validation detects the mismatch.
    #[test]
    fn testValidateManifestCorrupt() {
        let tmpDir = tempfile::tempdir().expect("tempdir creation failed");
        let base = tmpDir.path();

        let dataDir = base.join("data");
        let walDir = base.join("wal");
        let backupDir = base.join("backup");

        fs::create_dir_all(&dataDir).expect("create data dir");
        fs::create_dir_all(&walDir).expect("create wal dir");

        fs::write(dataDir.join("table.zyheap"), b"original data").expect("write heap");

        let manifest = BackupManager::backup(&dataDir, &walDir, &backupDir, 100)
            .expect("backup should succeed");

        // Corrupt the backed-up file.
        let corruptedPath = backupDir.join("data/table.zyheap");
        fs::write(&corruptedPath, b"corrupted data").expect("corrupt file");

        // Validation should fail.
        let result = RestoreManager::validateManifest(&manifest, &backupDir);
        assert!(result.is_err());
        let errMsg = format!("{}", result.err().expect("should be an error"));
        assert!(errMsg.contains("checksum mismatch"));
    }

    /// Verify that formatTimestamp produces a valid ISO 8601 string.
    #[test]
    fn testFormatTimestamp() {
        let ts = formatTimestamp();
        // Should match pattern YYYY-MM-DDTHH:MM:SSZ (20 chars).
        assert_eq!(ts.len(), 20, "timestamp should be 20 chars: {}", ts);
        assert!(ts.ends_with('Z'), "timestamp should end with Z: {}", ts);
        assert_eq!(&ts[4..5], "-", "expected dash at position 4: {}", ts);
        assert_eq!(&ts[7..8], "-", "expected dash at position 7: {}", ts);
        assert_eq!(&ts[10..11], "T", "expected T at position 10: {}", ts);
        assert_eq!(&ts[13..14], ":", "expected colon at position 13: {}", ts);
        assert_eq!(&ts[16..17], ":", "expected colon at position 16: {}", ts);

        // Year should be reasonable (2020-2099).
        let year: u32 = ts[..4].parse().expect("year should be numeric");
        assert!(year >= 2020 && year <= 2099, "unexpected year: {}", year);
    }
}
