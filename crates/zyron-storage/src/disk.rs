//! Disk manager for async page-level file I/O.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::page::{PAGE_SIZE, PageId, stamp_page_checksum, verify_page_checksum};
use zyron_common::{Result, ZyronError};

/// How often page reads verify the stored page checksum.
///
/// Every write path stamps unconditionally, this only controls the read
/// side. Sampled exists for benchmark investigations that need to isolate
/// verification cost, Off for tests that measure raw I/O. Server configs
/// reject Off, an unverified production read path hides corruption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageChecksumVerify {
    /// Verify every page read
    Always,
    /// Verify roughly one in a hundred page reads
    Sampled,
    /// Never verify
    Off,
}

impl PageChecksumVerify {
    /// Parses the config vocabulary. The accepted strings live here alone,
    /// so the config validator and the server construction that maps the
    /// setting onto a DiskManager cannot drift apart
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "always" => Some(Self::Always),
            "sampled" => Some(Self::Sampled),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

/// Configuration for the disk manager.
#[derive(Debug, Clone)]
pub struct DiskManagerConfig {
    /// Base directory for data files.
    pub data_dir: PathBuf,
    /// Enable fsync after writes.
    pub fsync_enabled: bool,
    /// Read-side page checksum verification policy.
    pub page_checksum_verify: PageChecksumVerify,
}

impl Default for DiskManagerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            fsync_enabled: true,
            page_checksum_verify: PageChecksumVerify::Always,
        }
    }
}

/// One sampled verification per this many reads when the policy is Sampled
const CHECKSUM_SAMPLE_INTERVAL: u64 = 100;

/// Stripes of per-page latches per file, a power of two so the stripe
/// index is a mask. 64 stripes keep same-stripe collisions between
/// distinct pages rare at realistic I/O concurrency while costing a fixed
/// 512 bytes of lock words per open file
const PAGE_LATCH_STRIPES: usize = 64;

/// Manages async reading and writing pages to disk files.
///
/// Each file_id maps to a separate data file. File 0 is typically
/// the main heap file, while higher file IDs are used for indexes.
pub struct DiskManager {
    /// Configuration.
    config: DiskManagerConfig,
    /// Open files keyed by file_id, one entry and one operating system
    /// handle per file serving every caller. scc::HashMap gives lock-free
    /// lookup, and because all page I/O is positional the entry itself
    /// needs no lock to read or write through.
    files: scc::HashMap<u32, std::sync::Arc<FileEntry>>,
    /// Page reads issued, drives the Sampled verification policy.
    reads_issued: AtomicU64,
}

/// An open data file.
///
/// Page reads and writes address the file by offset (`pread` / `pwrite`),
/// which never touches the file cursor, so any number of threads can read
/// and write the same file at once without coordinating. The page count
/// and length that a lock previously guarded are atomics instead, updated
/// with the monotonic operations their meaning already implied. Growth is
/// `fetch_max` so a late small allocation can never shrink a file another
/// caller already extended.
struct FileEntry {
    /// One handle, shared by the async callers and by the background
    /// writer thread. Opening a path costs milliseconds on some platforms,
    /// so it is done once when the file is first referenced.
    file: std::sync::Arc<std::fs::File>,
    /// Pages the file is known to hold, which bounds what `read_page`
    /// accepts.
    num_pages: AtomicU64,
    /// Bytes the file has been extended to. Kept beside `num_pages` so a
    /// grow is one compare rather than a metadata call.
    len_bytes: AtomicU64,
    /// Held shared by page I/O and exclusively by operations that change
    /// the file's extent underneath it, which is truncation and deletion.
    /// Page I/O never contends with page I/O, only with those two.
    extent: parking_lot::RwLock<()>,
    /// Per-page latches, striped by page number. A page write holds its
    /// stripe exclusively and a page read holds it shared, because 16KB
    /// positional I/O is not atomic on either supported platform and a
    /// read overlapping an in-flight write of the same page would observe
    /// a torn mix of old and new bytes, which checksum verification then
    /// reports as corruption of a page that is fine on disk. Acquired
    /// after `extent` everywhere, held only across the I/O call itself
    page_latches: [parking_lot::RwLock<()>; PAGE_LATCH_STRIPES],
}

impl FileEntry {
    /// The latch stripe covering `page_num`
    #[inline]
    fn page_latch(&self, page_num: u64) -> &parking_lot::RwLock<()> {
        &self.page_latches[(page_num as usize) & (PAGE_LATCH_STRIPES - 1)]
    }
}

impl DiskManager {
    /// Creates a new disk manager.
    pub async fn new(config: DiskManagerConfig) -> Result<Self> {
        tokio::fs::create_dir_all(&config.data_dir).await?;

        Ok(Self {
            config,
            files: scc::HashMap::new(),
            reads_issued: AtomicU64::new(0),
        })
    }

    /// Whether the next page read verifies its checksum, applying the
    /// configured policy. Callers that read page bytes outside this manager
    /// (the heap scan path's sync read) ask here so one counter and one
    /// policy govern every read.
    pub fn should_verify_page(&self) -> bool {
        match self.config.page_checksum_verify {
            PageChecksumVerify::Always => true,
            PageChecksumVerify::Off => false,
            PageChecksumVerify::Sampled => {
                self.reads_issued.fetch_add(1, Ordering::Relaxed) % CHECKSUM_SAMPLE_INTERVAL == 0
            }
        }
    }

    /// Returns the entry for a file, opening it the first time it is
    /// referenced. Lookups are lock-free. A race to open settles by letting
    /// the first inserter win, the loser's handle closes harmlessly, so
    /// every caller ends up on the same handle and the same counters.
    ///
    /// This is synchronous by design. Opening resolves a path, which is
    /// expensive enough on some platforms to matter, and doing it once here
    /// keeps that cost off every later read and write from either the async
    /// callers or the background writer thread.
    fn entry(&self, file_id: u32) -> Result<std::sync::Arc<FileEntry>> {
        if let Some(e) = self.files.read_sync(&file_id, |_, e| e.clone()) {
            return Ok(e);
        }
        let path = self.file_path(file_id);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| ZyronError::IoError(format!("open {}: {}", path.display(), e)))?;
        let len = file
            .metadata()
            .map_err(|e| ZyronError::IoError(format!("stat {}: {}", path.display(), e)))?
            .len();
        let entry = std::sync::Arc::new(FileEntry {
            file: std::sync::Arc::new(file),
            num_pages: AtomicU64::new(len / PAGE_SIZE as u64),
            len_bytes: AtomicU64::new(len),
            extent: parking_lot::RwLock::new(()),
            page_latches: std::array::from_fn(|_| parking_lot::RwLock::new(())),
        });
        let _ = self.files.insert_sync(file_id, entry.clone());
        Ok(self
            .files
            .read_sync(&file_id, |_, e| e.clone())
            .unwrap_or(entry))
    }

    /// Grows a file to cover `target_len`, and only ever grows. Concurrent
    /// allocations can complete out of order, so the length each one sets
    /// has to be the high water mark rather than its own value, otherwise a
    /// smaller late call would truncate away another's pages.
    fn grow_to(entry: &FileEntry, target_len: u64) -> Result<()> {
        let previous = entry.len_bytes.fetch_max(target_len, Ordering::AcqRel);
        if previous >= target_len {
            return Ok(());
        }
        entry.file.set_len(target_len).map_err(|e| {
            // Undo the claim so a later caller retries the extension rather
            // than trusting a length the file does not have
            entry.len_bytes.store(previous, Ordering::Release);
            ZyronError::IoError(format!("extend file to {}: {}", target_len, e))
        })
    }

    /// Returns the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.config.data_dir
    }

    /// Generates the file path for a given file ID.
    fn file_path(&self, file_id: u32) -> PathBuf {
        self.config.data_dir.join(format!("{:08}.dat", file_id))
    }

    /// Reads a page from disk.
    pub async fn read_page(&self, page_id: PageId) -> Result<[u8; PAGE_SIZE]> {
        let entry = self.entry(page_id.file_id)?;
        if page_id.page_num >= entry.num_pages.load(Ordering::Acquire) {
            return Err(ZyronError::IoError(format!(
                "page {} does not exist in file {}",
                page_id.page_num, page_id.file_id
            )));
        }
        let offset = page_id.page_num * (PAGE_SIZE as u64);
        let file_id = page_id.file_id;
        let page_num = page_id.page_num;
        let verify = self.should_verify_page();
        tokio::task::spawn_blocking(move || {
            let _extent = entry.extent.read();
            let mut buffer = [0u8; PAGE_SIZE];
            {
                let _page = entry.page_latch(page_num).read();
                positional_read_exact(&entry.file, &mut buffer, offset).map_err(|e| {
                    ZyronError::IoError(format!(
                        "read page {}@{} for file {}: {}",
                        page_num, offset, file_id, e
                    ))
                })?;
            }
            // A mismatch fails the read before any caller can consume the
            // bytes. Recovery has no physical page redo, so the error
            // surfaces to the caller instead of triggering a repair
            if verify {
                verify_page_checksum(&buffer, page_id)?;
            }
            Ok(buffer)
        })
        .await
        .map_err(|e| ZyronError::IoError(format!("read page task: {}", e)))?
    }

    /// Synchronous page read for callers without an async context (the
    /// heap scan path). Takes the same per-page latch as the write paths
    /// and applies the same verification policy as the async read. A page
    /// past the file's current end reads as zeroes, which decodes as an
    /// empty page, the same answer the async read gives for an allocated
    /// never-written page.
    pub fn read_page_sync(&self, page_id: PageId) -> Result<Box<[u8; PAGE_SIZE]>> {
        let entry = self.entry(page_id.file_id)?;
        let offset = page_id.page_num * (PAGE_SIZE as u64);
        let mut buffer: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        {
            let _extent = entry.extent.read();
            let _page = entry.page_latch(page_id.page_num).read();
            positional_read_up_to(&entry.file, &mut *buffer, offset).map_err(|e| {
                ZyronError::IoError(format!(
                    "read page {}@{} for file {}: {}",
                    page_id.page_num, offset, page_id.file_id, e
                ))
            })?;
        }
        if self.should_verify_page() {
            verify_page_checksum(&buffer, page_id)?;
        }
        Ok(buffer)
    }

    /// Writes a page to disk.
    pub async fn write_page(&self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        let entry = self.entry(page_id.file_id)?;
        let offset = page_id.page_num * (PAGE_SIZE as u64);
        let fsync = self.config.fsync_enabled;
        let mut buf = *data;
        let file_id = page_id.file_id;
        let page_num = page_id.page_num;
        let written = tokio::task::spawn_blocking(move || -> Result<()> {
            let _extent = entry.extent.read();
            stamp_page_checksum(&mut buf);
            {
                let _page = entry.page_latch(page_num).write();
                positional_write_all(&entry.file, &buf, offset).map_err(|e| {
                    ZyronError::IoError(format!(
                        "write page {}@{} for file {}: {}",
                        page_num, offset, file_id, e
                    ))
                })?;
            }
            if fsync {
                entry
                    .file
                    .sync_all()
                    .map_err(|e| ZyronError::IoError(format!("fsync file {}: {}", file_id, e)))?;
            }
            // A write past the known end extends the file, so both counters
            // rise to cover it and never fall
            entry.num_pages.fetch_max(page_num + 1, Ordering::AcqRel);
            entry
                .len_bytes
                .fetch_max((page_num + 1) * PAGE_SIZE as u64, Ordering::AcqRel);
            Ok(())
        })
        .await
        .map_err(|e| ZyronError::IoError(format!("write page task: {}", e)))?;
        written
    }

    /// Allocates a new page in the specified file and extends the underlying
    /// file so the page is physically addressable.
    ///
    /// Extending the file here (rather than waiting for `write_page`) keeps
    /// `read_page` honest: any page number less than `handle.num_pages` is
    /// guaranteed to be readable and returns a zero-filled buffer if no
    /// data has been written yet.
    pub async fn allocate_page(&self, file_id: u32) -> Result<PageId> {
        Ok(self.allocate_pages_batch(file_id, 1).await?[0])
    }

    /// Allocates multiple pages in a single operation and extends the file to
    /// the new page count. Every returned page is physically addressable even
    /// before its first `write_page`.
    ///
    /// The page numbers come from one atomic add, so two callers allocating
    /// at once take disjoint ranges without a lock between them.
    pub async fn allocate_pages_batch(&self, file_id: u32, count: u64) -> Result<Vec<PageId>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let entry = self.entry(file_id)?;
        let start_page = entry.num_pages.fetch_add(count, Ordering::AcqRel);
        let target_len = (start_page + count) * (PAGE_SIZE as u64);
        let e2 = entry.clone();
        tokio::task::spawn_blocking(move || {
            let _extent = e2.extent.read();
            Self::grow_to(&e2, target_len)
        })
        .await
        .map_err(|e| ZyronError::IoError(format!("allocate task: {}", e)))??;

        Ok((0..count)
            .map(|i| PageId::new(file_id, start_page + i))
            .collect())
    }

    /// Returns the number of pages in a file.
    pub async fn num_pages(&self, file_id: u32) -> Result<u64> {
        Ok(self.entry(file_id)?.num_pages.load(Ordering::Acquire))
    }

    /// Synchronous page write for the background writer thread.
    /// Uses std::fs::File handles separate from the async path.
    /// The background writer is a dedicated OS thread that can block on I/O.
    pub fn write_page_sync(&self, page_id: PageId, data: &mut [u8; PAGE_SIZE]) -> Result<()> {
        let entry = self.entry(page_id.file_id)?;
        let _extent = entry.extent.read();
        let file = &entry.file;
        let offset = page_id.page_num * (PAGE_SIZE as u64);
        stamp_page_checksum(data);
        {
            let _page = entry.page_latch(page_id.page_num).write();
            positional_write_all(&file, data, offset).map_err(|e| {
                ZyronError::IoError(format!(
                    "write page {}@{} for file {}: {}",
                    page_id.page_num, offset, page_id.file_id, e
                ))
            })?;
        }
        if self.config.fsync_enabled {
            file.sync_all().map_err(|e| {
                ZyronError::IoError(format!("fsync file {}: {}", page_id.file_id, e))
            })?;
        }
        Ok(())
    }

    /// Writes a page synchronously WITHOUT issuing fsync. Used by the
    /// background writer to issue many writes in a batch followed by a
    /// single `fsync_file` per touched file, which is dramatically faster
    /// than one fsync per page on platforms where fsync dominates write
    /// cost (notably Windows).
    pub fn write_page_sync_no_fsync(
        &self,
        page_id: PageId,
        data: &mut [u8; PAGE_SIZE],
    ) -> Result<()> {
        let entry = self.entry(page_id.file_id)?;
        let _extent = entry.extent.read();
        let file = &entry.file;
        let offset = page_id.page_num * (PAGE_SIZE as u64);
        stamp_page_checksum(data);
        {
            let _page = entry.page_latch(page_id.page_num).write();
            positional_write_all(&file, data, offset).map_err(|e| {
                ZyronError::IoError(format!(
                    "write page {}@{} for file {}: {}",
                    page_id.page_num, offset, page_id.file_id, e
                ))
            })?;
        }
        Ok(())
    }

    /// Issues fsync against the file backing `file_id`, with no effect when
    /// the manager is configured fsync-disabled.
    pub fn fsync_file(&self, file_id: u32) -> Result<()> {
        if !self.config.fsync_enabled {
            return Ok(());
        }
        let entry = self.entry(file_id)?;
        entry
            .file
            .sync_all()
            .map_err(|e| ZyronError::IoError(format!("fsync file {}: {}", file_id, e)))
    }

    /// Flushes all pending writes to disk.
    pub async fn flush(&self) -> Result<()> {
        let mut entries = Vec::new();
        self.files
            .iter_async(|&file_id, e| {
                entries.push((file_id, e.clone()));
                true
            })
            .await;

        for (file_id, entry) in entries {
            tokio::task::spawn_blocking(move || {
                entry
                    .file
                    .sync_all()
                    .map_err(|e| ZyronError::IoError(format!("fsync file {}: {}", file_id, e)))
            })
            .await
            .map_err(|e| ZyronError::IoError(format!("flush task: {}", e)))??;
        }

        Ok(())
    }

    /// Truncates a data file to zero pages, removing all stored data.
    /// The file remains on disk but is reset to empty.
    ///
    /// Takes the extent lock exclusively, so no read or write is addressing
    /// the file while its length changes underneath them.
    pub async fn truncate_file(&self, file_id: u32) -> Result<()> {
        let entry = self.entry(file_id)?;
        let fsync = self.config.fsync_enabled;
        tokio::task::spawn_blocking(move || -> Result<()> {
            let _extent = entry.extent.write();
            entry
                .file
                .set_len(0)
                .map_err(|e| ZyronError::IoError(format!("truncate file {}: {}", file_id, e)))?;
            if fsync {
                entry
                    .file
                    .sync_all()
                    .map_err(|e| ZyronError::IoError(format!("fsync file {}: {}", file_id, e)))?;
            }
            entry.num_pages.store(0, Ordering::Release);
            entry.len_bytes.store(0, Ordering::Release);
            Ok(())
        })
        .await
        .map_err(|e| ZyronError::IoError(format!("truncate task: {}", e)))?
    }

    /// Closes a specific file, first extending it to cover every page that
    /// was allocated, so pages allocated but never written survive.
    pub async fn close_file(&self, file_id: u32) -> Result<()> {
        if let Some((_, entry)) = self.files.remove_async(&file_id).await {
            tokio::task::spawn_blocking(move || -> Result<()> {
                let _extent = entry.extent.write();
                let expected = entry.num_pages.load(Ordering::Acquire) * (PAGE_SIZE as u64);
                entry.file.set_len(expected).map_err(|e| {
                    ZyronError::IoError(format!("extend file {} on close: {}", file_id, e))
                })?;
                entry
                    .file
                    .sync_all()
                    .map_err(|e| ZyronError::IoError(format!("fsync file {}: {}", file_id, e)))
            })
            .await
            .map_err(|e| ZyronError::IoError(format!("close task: {}", e)))??;
        }
        Ok(())
    }

    /// Closes all open files.
    pub async fn close_all(&self) -> Result<()> {
        let mut file_ids = Vec::new();
        self.files
            .iter_async(|&file_id, _| {
                file_ids.push(file_id);
                true
            })
            .await;

        for file_id in file_ids {
            self.close_file(file_id).await?;
        }
        Ok(())
    }

    /// Deletes a data file.
    pub async fn delete_file(&self, file_id: u32) -> Result<()> {
        self.close_file(file_id).await?;
        let path = self.file_path(file_id);
        if path.exists() {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }
}

/// Positional `write_all` against a shared `std::fs::File`. Does not move
/// the file's seek cursor, so concurrent callers writing to different
/// offsets of the same file run in parallel without any lock. On Unix
/// this uses `pwrite` via `FileExt::write_at`, on Windows `WriteFile`
/// with an `OVERLAPPED` offset via `FileExt::seek_write`.
#[cfg(unix)]
fn positional_write_all(
    file: &std::fs::File,
    mut data: &[u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !data.is_empty() {
        match file.write_at(data, offset) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                data = &data[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
fn positional_write_all(
    file: &std::fs::File,
    mut data: &[u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        match file.seek_write(data, offset) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                data = &data[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Positional `read_exact` against a shared `std::fs::File`. The read twin
/// of `positional_write_all`, and the same properties: it does not move the
/// file's seek cursor, so one handle serves every reader of a file and two
/// threads reading different regions of it never contend. On Unix this is
/// `pread` via `FileExt::read_at`, on Windows `ReadFile` with an
/// `OVERLAPPED` offset via `FileExt::seek_read`.
///
/// Taking `&File` rather than `&mut File` is the point. A reader that has
/// to seek needs exclusive access, which is what forces a caller holding
/// one shared handle to open the file again per read
#[cfg(unix)]
pub(crate) fn positional_read_exact(
    file: &std::fs::File,
    mut buf: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    while !buf.is_empty() {
        match file.read_at(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn positional_read_exact(
    file: &std::fs::File,
    mut buf: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Positional read that stops at end of file instead of erroring, leaving
/// the rest of the buffer untouched. For reads of pages the file may not
/// have been extended to cover yet, whose missing tail decodes as zeroes
fn positional_read_up_to(
    file: &std::fs::File,
    mut buf: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        #[cfg(unix)]
        let read = file.read_at(buf, offset);
        #[cfg(windows)]
        let read = file.seek_read(buf, offset);
        match read {
            Ok(0) => break,
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn create_test_disk_manager() -> (DiskManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
            page_checksum_verify: PageChecksumVerify::Always,
        };
        let dm = DiskManager::new(config).await.unwrap();
        (dm, dir)
    }

    #[tokio::test]
    async fn test_disk_manager_new() {
        let (dm, _dir) = create_test_disk_manager().await;
        assert!(dm.data_dir().exists());
    }

    #[tokio::test]
    async fn test_disk_manager_allocate_page() {
        let (dm, _dir) = create_test_disk_manager().await;

        let page1 = dm.allocate_page(0).await.unwrap();
        assert_eq!(page1.file_id, 0);
        assert_eq!(page1.page_num, 0);

        let page2 = dm.allocate_page(0).await.unwrap();
        assert_eq!(page2.page_num, 1);

        assert_eq!(dm.num_pages(0).await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_disk_manager_write_read() {
        let (dm, _dir) = create_test_disk_manager().await;

        let page_id = dm.allocate_page(0).await.unwrap();

        // Write data
        let mut data = [0u8; PAGE_SIZE];
        data[0] = 0xAB;
        data[100] = 0xCD;
        data[PAGE_SIZE - 1] = 0xEF;
        dm.write_page(page_id, &data).await.unwrap();

        // Read back
        let read_data = dm.read_page(page_id).await.unwrap();
        assert_eq!(read_data[0], 0xAB);
        assert_eq!(read_data[100], 0xCD);
        assert_eq!(read_data[PAGE_SIZE - 1], 0xEF);
    }

    #[tokio::test]
    async fn test_disk_manager_multiple_files() {
        let (dm, _dir) = create_test_disk_manager().await;

        // Allocate pages in different files
        let page_f0 = dm.allocate_page(0).await.unwrap();
        let page_f1 = dm.allocate_page(1).await.unwrap();
        let page_f2 = dm.allocate_page(2).await.unwrap();

        assert_eq!(page_f0.file_id, 0);
        assert_eq!(page_f1.file_id, 1);
        assert_eq!(page_f2.file_id, 2);

        // Write to each
        let mut data0 = [0u8; PAGE_SIZE];
        data0[0] = 0x00;
        dm.write_page(page_f0, &data0).await.unwrap();

        let mut data1 = [0u8; PAGE_SIZE];
        data1[0] = 0x11;
        dm.write_page(page_f1, &data1).await.unwrap();

        let mut data2 = [0u8; PAGE_SIZE];
        data2[0] = 0x22;
        dm.write_page(page_f2, &data2).await.unwrap();

        // Read back
        assert_eq!(dm.read_page(page_f0).await.unwrap()[0], 0x00);
        assert_eq!(dm.read_page(page_f1).await.unwrap()[0], 0x11);
        assert_eq!(dm.read_page(page_f2).await.unwrap()[0], 0x22);
    }

    #[tokio::test]
    async fn test_disk_manager_read_nonexistent_page() {
        let (dm, _dir) = create_test_disk_manager().await;

        // Allocate one page
        dm.allocate_page(0).await.unwrap();

        // Try to read page that doesn't exist
        let result = dm.read_page(PageId::new(0, 99)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_disk_manager_overwrite_page() {
        let (dm, _dir) = create_test_disk_manager().await;

        let page_id = dm.allocate_page(0).await.unwrap();

        // Write initial data
        let mut data1 = [0u8; PAGE_SIZE];
        data1[0] = 0xAA;
        dm.write_page(page_id, &data1).await.unwrap();

        // Overwrite with new data
        let mut data2 = [0u8; PAGE_SIZE];
        data2[0] = 0xBB;
        dm.write_page(page_id, &data2).await.unwrap();

        // Read should return new data
        let read_data = dm.read_page(page_id).await.unwrap();
        assert_eq!(read_data[0], 0xBB);
    }

    #[tokio::test]
    async fn test_disk_manager_persistence() {
        let dir = tempdir().unwrap();
        let page_id;

        // Write data
        {
            let config = DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: true,
                page_checksum_verify: PageChecksumVerify::Always,
            };
            let dm = DiskManager::new(config).await.unwrap();
            page_id = dm.allocate_page(0).await.unwrap();

            let mut data = [0u8; PAGE_SIZE];
            data[0] = 0xFF;
            dm.write_page(page_id, &data).await.unwrap();
        }

        // Read with new disk manager
        {
            let config = DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: true,
                page_checksum_verify: PageChecksumVerify::Always,
            };
            let dm = DiskManager::new(config).await.unwrap();

            let read_data = dm.read_page(page_id).await.unwrap();
            assert_eq!(read_data[0], 0xFF);
        }
    }

    #[tokio::test]
    async fn test_disk_manager_delete_file() {
        let (dm, dir) = create_test_disk_manager().await;

        dm.allocate_page(0).await.unwrap();
        let file_path = dir.path().join("00000000.dat");
        assert!(file_path.exists());

        dm.delete_file(0).await.unwrap();
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_disk_manager_num_pages() {
        let (dm, _dir) = create_test_disk_manager().await;

        assert_eq!(dm.num_pages(0).await.unwrap(), 0);

        dm.allocate_page(0).await.unwrap();
        assert_eq!(dm.num_pages(0).await.unwrap(), 1);

        dm.allocate_page(0).await.unwrap();
        dm.allocate_page(0).await.unwrap();
        assert_eq!(dm.num_pages(0).await.unwrap(), 3);
    }

    #[tokio::test]
    async fn test_disk_manager_flush() {
        let (dm, _dir) = create_test_disk_manager().await;

        dm.allocate_page(0).await.unwrap();
        dm.allocate_page(1).await.unwrap();

        // Should not panic
        dm.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_disk_manager_close_file() {
        let (dm, _dir) = create_test_disk_manager().await;

        dm.allocate_page(0).await.unwrap();
        dm.close_file(0).await.unwrap();

        // Can reopen and continue
        dm.allocate_page(0).await.unwrap();
        assert_eq!(dm.num_pages(0).await.unwrap(), 2);
    }

    /// Flips one byte of a page on disk behind the manager's back
    fn corrupt_on_disk(dir: &tempfile::TempDir, page_id: PageId, byte_offset: usize) {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
        let path = dir.path().join(format!("{:08}.dat", page_id.file_id));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open data file");
        let pos = page_id.page_num * PAGE_SIZE as u64 + byte_offset as u64;
        file.seek(SeekFrom::Start(pos)).expect("seek");
        let mut b = [0u8; 1];
        file.read_exact(&mut b).expect("read byte");
        b[0] ^= 0x40;
        file.seek(SeekFrom::Start(pos)).expect("seek back");
        file.write_all(&b).expect("write byte");
    }

    #[tokio::test]
    async fn test_page_checksum_detects_on_disk_corruption() {
        let (dm, dir) = create_test_disk_manager().await;
        let page_id = dm.allocate_page(0).await.unwrap();
        let mut data = [0u8; PAGE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 13 + 5) as u8;
        }
        dm.write_page(page_id, &data).await.unwrap();
        assert!(dm.read_page(page_id).await.is_ok());

        // Corrupt one body byte behind the manager's back. The read must
        // fail with the checksum variant, never return the corrupt bytes
        corrupt_on_disk(&dir, page_id, 4096);
        match dm.read_page(page_id).await {
            Err(ZyronError::PageChecksumMismatch {
                file_id,
                page_id: page_num,
                ..
            }) => {
                assert_eq!(file_id, 0);
                assert_eq!(page_num, 0);
            }
            other => panic!("expected PageChecksumMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_page_checksum_every_write_path_stamps() {
        let (dm, _dir) = create_test_disk_manager().await;
        let mut data = [0u8; PAGE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 7 + 1) as u8;
        }

        // Async write, sync write, and batched-fsync sync write must all
        // leave a page the verifying read accepts
        let p0 = dm.allocate_page(0).await.unwrap();
        dm.write_page(p0, &data).await.unwrap();
        let p1 = dm.allocate_page(0).await.unwrap();
        dm.write_page_sync(p1, &mut data.clone()).unwrap();
        let p2 = dm.allocate_page(0).await.unwrap();
        dm.write_page_sync_no_fsync(p2, &mut data.clone()).unwrap();
        for p in [p0, p1, p2] {
            assert!(dm.read_page(p).await.is_ok(), "page {p:?} failed verify");
        }
    }

    #[tokio::test]
    async fn test_page_checksum_allocated_unwritten_page_reads_zeroed() {
        let (dm, _dir) = create_test_disk_manager().await;
        // Allocated but never written: reads back zero-filled and passes
        // verification as a fresh page
        let page_id = dm.allocate_page(0).await.unwrap();
        let data = dm.read_page(page_id).await.unwrap();
        assert!(data.iter().all(|&b| b == 0));
    }

    #[tokio::test]
    async fn test_page_checksum_off_policy_skips_verification() {
        let dir = tempdir().unwrap();
        let dm = DiskManager::new(DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
            page_checksum_verify: PageChecksumVerify::Off,
        })
        .await
        .unwrap();
        let page_id = dm.allocate_page(0).await.unwrap();
        let mut data = [0u8; PAGE_SIZE];
        data[0] = 0xAA;
        dm.write_page(page_id, &data).await.unwrap();
        corrupt_on_disk(&dir, page_id, 8000);
        // Off never verifies, the corrupt page is returned as read
        assert!(dm.read_page(page_id).await.is_ok());
        assert!(!dm.should_verify_page());
    }

    /// Reader and writer of one page share a striped latch, exclusive on
    /// the write side, so a read overlapping an in-flight write of the
    /// same page can never observe a torn mix of two images. Without the
    /// latch this surfaces as a spurious checksum mismatch under load,
    /// 16KB positional I/O is not atomic on either supported platform
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_page_latch_prevents_torn_reads() {
        let (dm, _dir) = create_test_disk_manager().await;
        let dm = std::sync::Arc::new(dm);
        let page_id = dm.allocate_page(0).await.unwrap();
        dm.write_page_sync(page_id, &mut [0x11u8; PAGE_SIZE])
            .unwrap();

        let writer_dm = std::sync::Arc::clone(&dm);
        let writer = std::thread::spawn(move || {
            for i in 0..500u32 {
                let fill = if i % 2 == 0 { 0x22 } else { 0x33 };
                writer_dm
                    .write_page_sync_no_fsync(page_id, &mut [fill; PAGE_SIZE])
                    .unwrap();
            }
        });
        // Sync reads on this thread race the writer thread. Verification is
        // Always in the test config, so a torn image would fail its checksum
        for _ in 0..500 {
            let page = dm
                .read_page_sync(page_id)
                .expect("a concurrent read must never observe a torn page");
            let fill = page[100];
            assert!(fill == 0x11 || fill == 0x22 || fill == 0x33);
        }
        writer.join().expect("writer thread");
    }

    #[tokio::test]
    async fn test_page_checksum_sampled_policy_verifies_periodically() {
        let dir = tempdir().unwrap();
        let dm = DiskManager::new(DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
            page_checksum_verify: PageChecksumVerify::Sampled,
        })
        .await
        .unwrap();
        // Exactly one verification per CHECKSUM_SAMPLE_INTERVAL decisions
        let verified = (0..CHECKSUM_SAMPLE_INTERVAL)
            .filter(|_| dm.should_verify_page())
            .count();
        assert_eq!(verified, 1);
    }
}
