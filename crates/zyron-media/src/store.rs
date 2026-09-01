//! Content addressed media store
//!
//! Objects live under <root>/media/<first two hex chars>/<sha256 hex>. The
//! address is the sha256 of the uncompressed payload. Each object file starts
//! with a small self describing header so reads know whether to decompress.
//! Reference counts are kept in memory and made durable through an append
//! only log of signed deltas that is compacted on open

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::error::{MediaError, MediaResult};

const OBJECT_MAGIC: &[u8; 4] = b"ZYMO";
const OBJECT_VERSION: u8 = 1;
const OBJECT_HEADER_LEN: usize = 6;

const REF_RECORD_LEN: usize = 36;
const REF_LOG_NAME: &str = "refs.zyref";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Result of storing one payload
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredObject {
    pub sha256: [u8; 32],
    /// Bytes stored on disk after optional compression, header excluded
    pub stored_len: u64,
    pub compressed: bool,
}

/// Content addressed store rooted at one directory
pub struct MediaStore {
    media_dir: PathBuf,
    ref_log: Mutex<fs::File>,
    ref_counts: scc::HashMap<[u8; 32], i64>,
}

impl MediaStore {
    /// Opens the store, creating directories and loading reference counts
    pub fn open(root: PathBuf) -> MediaResult<Self> {
        let media_dir = root.join("media");
        fs::create_dir_all(&media_dir)?;

        let log_path = media_dir.join(REF_LOG_NAME);
        let ref_counts = scc::HashMap::new();
        let mut record_count: u64 = 0;
        if log_path.exists() {
            let raw = fs::read(&log_path)?;
            if raw.len() % REF_RECORD_LEN != 0 {
                return Err(MediaError::CorruptObject(format!(
                    "refcount log length {} is not a multiple of the record size",
                    raw.len()
                )));
            }
            for record in raw.chunks_exact(REF_RECORD_LEN) {
                let mut sha = [0u8; 32];
                sha.copy_from_slice(&record[..32]);
                let mut delta_bytes = [0u8; 4];
                delta_bytes.copy_from_slice(&record[32..36]);
                let delta = i32::from_le_bytes(delta_bytes) as i64;
                apply_delta(&ref_counts, &sha, delta);
                record_count += 1;
            }
        }

        let mut live: u64 = 0;
        ref_counts.iter_sync(|_, _| {
            live += 1;
            true
        });

        // Compact when the log has grown well past the live entry set
        if record_count > live.saturating_mul(2) + 16 {
            let mut compacted = Vec::with_capacity(live as usize * REF_RECORD_LEN);
            ref_counts.iter_sync(|sha, count| {
                compacted.extend_from_slice(sha);
                compacted.extend_from_slice(&(*count as i32).to_le_bytes());
                true
            });
            let tmp = media_dir.join(format!(
                ".{REF_LOG_NAME}.tmp-{}-{}",
                std::process::id(),
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            {
                let mut f = fs::File::create(&tmp)?;
                f.write_all(&compacted)?;
                f.sync_all()?;
            }
            replace_file(&tmp, &log_path)?;
            sync_dir(&media_dir)?;
        }

        let ref_log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        Ok(MediaStore {
            media_dir,
            ref_log: Mutex::new(ref_log),
            ref_counts,
        })
    }

    /// Stores a payload, compressing with lz4 frames when asked and smaller
    ///
    /// The address is the sha256 of the uncompressed payload, so the same
    /// bytes stored twice land on the same object file
    pub fn put(&self, bytes: &[u8], compress: bool) -> MediaResult<StoredObject> {
        let sha = hash_bytes(bytes);
        let path = self.object_path(&sha);

        if path.exists() {
            return self.stat_existing(&sha, &path);
        }

        let mut compressed = false;
        let mut stored: &[u8] = bytes;
        let compressed_buf;
        if compress {
            let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
            encoder
                .write_all(bytes)
                .map_err(|e| MediaError::CorruptObject(format!("lz4 compression failed: {e}")))?;
            let encoded = encoder
                .finish()
                .map_err(|e| MediaError::CorruptObject(format!("lz4 compression failed: {e}")))?;
            if encoded.len() < bytes.len() {
                compressed = true;
                compressed_buf = encoded;
                stored = &compressed_buf;
            }
        }

        let shard = path.parent().ok_or_else(|| {
            MediaError::CorruptObject("object path has no parent directory".to_string())
        })?;
        fs::create_dir_all(shard)?;

        let tmp = shard.join(format!(
            ".tmp-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(OBJECT_MAGIC)?;
            f.write_all(&[OBJECT_VERSION, u8::from(compressed)])?;
            f.write_all(stored)?;
            f.sync_all()?;
        }
        match fs::rename(&tmp, &path) {
            Ok(()) => {}
            Err(err) => {
                // A concurrent put of the same content can win the rename
                let _ = fs::remove_file(&tmp);
                if path.exists() {
                    return self.stat_existing(&sha, &path);
                }
                return Err(err.into());
            }
        }
        sync_dir(shard)?;

        Ok(StoredObject {
            sha256: sha,
            stored_len: stored.len() as u64,
            compressed,
        })
    }

    /// Reads a payload back, decompressing and verifying the content hash
    pub fn get(&self, sha256: &[u8; 32]) -> MediaResult<Vec<u8>> {
        let path = self.object_path(sha256);
        let raw = match fs::read(&path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(MediaError::ObjectNotFound(hex::encode(sha256)));
            }
            Err(err) => return Err(err.into()),
        };
        let (compressed, body) = parse_object(&raw, sha256)?;

        let payload = if compressed {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(body);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out).map_err(|e| {
                MediaError::CorruptObject(format!(
                    "lz4 decompression failed for {}: {e}",
                    hex::encode(sha256)
                ))
            })?;
            out
        } else {
            body.to_vec()
        };

        let actual = hash_bytes(&payload);
        if &actual != sha256 {
            return Err(MediaError::IntegrityMismatch {
                expected: hex::encode(sha256),
                actual: hex::encode(actual),
            });
        }
        Ok(payload)
    }

    /// Reports whether an object file exists for the hash
    pub fn contains(&self, sha256: &[u8; 32]) -> bool {
        self.object_path(sha256).exists()
    }

    /// Uncompressed payload length of a stored object
    pub fn len_of(&self, sha256: &[u8; 32]) -> MediaResult<u64> {
        let path = self.object_path(sha256);
        let raw = match fs::read(&path) {
            Ok(raw) => raw,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(MediaError::ObjectNotFound(hex::encode(sha256)));
            }
            Err(err) => return Err(err.into()),
        };
        let (compressed, body) = parse_object(&raw, sha256)?;
        if !compressed {
            return Ok(body.len() as u64);
        }
        let mut decoder = lz4_flex::frame::FrameDecoder::new(body);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(|e| {
            MediaError::CorruptObject(format!(
                "lz4 decompression failed for {}: {e}",
                hex::encode(sha256)
            ))
        })?;
        Ok(out.len() as u64)
    }

    /// Adds one reference to a stored object
    pub fn add_ref(&self, sha256: &[u8; 32]) -> MediaResult<i64> {
        let mut log = self.ref_log.lock();
        self.append_delta(&mut log, sha256, 1)?;
        Ok(apply_delta(&self.ref_counts, sha256, 1))
    }

    /// Drops one reference, deleting the object file when the count reaches zero
    ///
    /// Returns true when this release deleted the object
    pub fn release(&self, sha256: &[u8; 32]) -> MediaResult<bool> {
        let mut log = self.ref_log.lock();
        let current = self.count_of(sha256);
        if current <= 0 {
            return Err(MediaError::ReleaseUnreferenced(hex::encode(sha256)));
        }
        self.append_delta(&mut log, sha256, -1)?;
        let remaining = apply_delta(&self.ref_counts, sha256, -1);
        if remaining > 0 {
            return Ok(false);
        }
        self.ref_counts.remove_sync(sha256);
        let path = self.object_path(sha256);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
        Ok(true)
    }

    /// Current in memory reference count for a hash
    pub fn count_of(&self, sha256: &[u8; 32]) -> i64 {
        let mut count = 0;
        self.ref_counts.read_sync(sha256, |_, v| count = *v);
        count
    }

    fn stat_existing(&self, sha: &[u8; 32], path: &Path) -> MediaResult<StoredObject> {
        let raw = fs::read(path)?;
        let (compressed, body) = parse_object(&raw, sha)?;
        Ok(StoredObject {
            sha256: *sha,
            stored_len: body.len() as u64,
            compressed,
        })
    }

    fn append_delta(&self, log: &mut fs::File, sha256: &[u8; 32], delta: i32) -> MediaResult<()> {
        let mut record = [0u8; REF_RECORD_LEN];
        record[..32].copy_from_slice(sha256);
        record[32..36].copy_from_slice(&delta.to_le_bytes());
        log.write_all(&record)?;
        log.sync_data()?;
        Ok(())
    }

    fn object_path(&self, sha256: &[u8; 32]) -> PathBuf {
        let hex = hex::encode(sha256);
        self.media_dir.join(&hex[..2]).join(hex)
    }
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut sha = [0u8; 32];
    sha.copy_from_slice(&digest);
    sha
}

fn parse_object<'a>(raw: &'a [u8], sha256: &[u8; 32]) -> MediaResult<(bool, &'a [u8])> {
    if raw.len() < OBJECT_HEADER_LEN || &raw[..4] != OBJECT_MAGIC {
        return Err(MediaError::CorruptObject(format!(
            "object {} has no ZYMO header",
            hex::encode(sha256)
        )));
    }
    if raw[4] != OBJECT_VERSION {
        return Err(MediaError::CorruptObject(format!(
            "object {} has unknown version {}",
            hex::encode(sha256),
            raw[4]
        )));
    }
    let compressed = match raw[5] {
        0 => false,
        1 => true,
        other => {
            return Err(MediaError::CorruptObject(format!(
                "object {} has invalid compression flag {other}",
                hex::encode(sha256)
            )));
        }
    };
    Ok((compressed, &raw[OBJECT_HEADER_LEN..]))
}

fn apply_delta(counts: &scc::HashMap<[u8; 32], i64>, sha: &[u8; 32], delta: i64) -> i64 {
    match counts.entry_sync(*sha) {
        scc::hash_map::Entry::Occupied(mut occ) => {
            let next = *occ.get() + delta;
            if next == 0 {
                let _ = occ.remove();
                0
            } else {
                occ.insert(next);
                next
            }
        }
        scc::hash_map::Entry::Vacant(vac) => {
            if delta != 0 {
                vac.insert_entry(delta);
            }
            delta
        }
    }
}

fn replace_file(tmp: &Path, dest: &Path) -> MediaResult<()> {
    // Windows rename refuses to overwrite, remove the destination first
    if dest.exists() {
        fs::remove_file(dest)?;
    }
    fs::rename(tmp, dest)?;
    Ok(())
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> MediaResult<()> {
    let handle = fs::File::open(dir)?;
    handle.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> MediaResult<()> {
    // Directory handles cannot be fsynced through std on this platform,
    // NTFS journals the rename metadata itself
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_store(dir: &Path) -> MediaStore {
        MediaStore::open(dir.to_path_buf()).expect("open store")
    }

    #[test]
    fn put_get_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let payload = b"the payload bytes".to_vec();
        let stored = store.put(&payload, false).expect("put");
        assert!(!stored.compressed);
        assert_eq!(stored.stored_len, payload.len() as u64);
        assert!(store.contains(&stored.sha256));
        assert_eq!(store.len_of(&stored.sha256).expect("len"), 17);
        assert_eq!(store.get(&stored.sha256).expect("get"), payload);
    }

    #[test]
    fn compression_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let payload = vec![42u8; 100_000];
        let stored = store.put(&payload, true).expect("put");
        assert!(stored.compressed);
        assert!(stored.stored_len < payload.len() as u64);
        assert_eq!(store.len_of(&stored.sha256).expect("len"), 100_000);
        assert_eq!(store.get(&stored.sha256).expect("get"), payload);
    }

    #[test]
    fn incompressible_payload_stays_raw() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let payload: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect();
        let stored = store.put(&payload, true).expect("put");
        assert!(!stored.compressed);
        assert_eq!(store.get(&stored.sha256).expect("get"), payload);
    }

    #[test]
    fn second_put_dedupes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let payload = b"same bytes twice".to_vec();
        let first = store.put(&payload, false).expect("first put");
        let second = store.put(&payload, false).expect("second put");
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.stored_len, second.stored_len);
        let shard = dir
            .path()
            .join("media")
            .join(&hex::encode(first.sha256)[..2]);
        let files = fs::read_dir(shard).expect("read shard").count();
        assert_eq!(files, 1);
    }

    #[test]
    fn refcount_release_deletes_at_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let stored = store.put(b"refcounted", false).expect("put");
        store.add_ref(&stored.sha256).expect("ref 1");
        store.add_ref(&stored.sha256).expect("ref 2");
        assert!(!store.release(&stored.sha256).expect("release 1"));
        assert!(store.contains(&stored.sha256));
        assert!(store.release(&stored.sha256).expect("release 2"));
        assert!(!store.contains(&stored.sha256));
        assert!(store.release(&stored.sha256).is_err());
    }

    #[test]
    fn refcounts_survive_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sha;
        {
            let store = open_store(dir.path());
            let stored = store.put(b"durable refs", false).expect("put");
            sha = stored.sha256;
            store.add_ref(&sha).expect("ref");
            store.add_ref(&sha).expect("ref");
        }
        let store = open_store(dir.path());
        assert_eq!(store.count_of(&sha), 2);
        assert!(!store.release(&sha).expect("release"));
        assert!(store.release(&sha).expect("release"));
    }

    #[test]
    fn tampered_object_fails_integrity() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let stored = store.put(b"protect these bytes", false).expect("put");
        let hex_name = hex::encode(stored.sha256);
        let path = dir
            .path()
            .join("media")
            .join(&hex_name[..2])
            .join(&hex_name);
        let mut raw = fs::read(&path).expect("read object");
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;
        fs::write(&path, raw).expect("write tampered");
        match store.get(&stored.sha256) {
            Err(MediaError::IntegrityMismatch { .. }) => {}
            other => panic!("expected integrity mismatch, got {other:?}"),
        }
    }

    #[test]
    fn missing_object_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_store(dir.path());
        let sha = [9u8; 32];
        assert!(matches!(
            store.get(&sha),
            Err(MediaError::ObjectNotFound(_))
        ));
        assert!(!store.contains(&sha));
    }
}
