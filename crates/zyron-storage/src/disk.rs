//! Disk manager for page-level file I/O.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use zyron_common::page::{PageId, PAGE_SIZE};
use zyron_common::{Result, ZyronError};

/// Configuration for the disk manager.
#[derive(Debug, Clone)]
pub struct DiskManagerConfig {
    /// Base directory for data files.
    pub data_dir: PathBuf,
    /// Enable fsync after writes.
    pub fsync_enabled: bool,
}

impl Default for DiskManagerConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            fsync_enabled: true,
        }
    }
}

/// Manages reading and writing pages to disk files.
///
/// Each file_id maps to a separate data file. File 0 is typically
/// the main heap file, while higher file IDs are used for indexes.
pub struct DiskManager {
    /// Configuration.
    config: DiskManagerConfig,
    /// Open file handles keyed by file_id.
    files: Mutex<HashMap<u32, FileHandle>>,
}

/// Handle for an open data file.
struct FileHandle {
    /// The file handle.
    file: File,
    /// Path to the file.
    #[allow(dead_code)]
    path: PathBuf,
    /// Number of pages in the file.
    num_pages: u32,
}

impl DiskManager {
    /// Creates a new disk manager.
    pub fn new(config: DiskManagerConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir)?;

        Ok(Self {
            config,
            files: Mutex::new(HashMap::new()),
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

    /// Opens or creates a data file.
    fn open_file(&self, file_id: u32) -> Result<()> {
        let mut files = self.files.lock();

        if files.contains_key(&file_id) {
            return Ok(());
        }

        let path = self.file_path(file_id);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        let file_size = file.metadata()?.len();
        let num_pages = (file_size / PAGE_SIZE as u64) as u32;

        files.insert(
            file_id,
            FileHandle {
                file,
                path,
                num_pages,
            },
        );

        Ok(())
    }

    /// Reads a page from disk.
    pub fn read_page(&self, page_id: PageId) -> Result<[u8; PAGE_SIZE]> {
        self.open_file(page_id.file_id)?;

        let mut files = self.files.lock();
        let handle = files.get_mut(&page_id.file_id).ok_or_else(|| {
            ZyronError::IoError(format!("file {} not open", page_id.file_id))
        })?;

        if page_id.page_num >= handle.num_pages {
            return Err(ZyronError::IoError(format!(
                "page {} does not exist in file {}",
                page_id.page_num, page_id.file_id
            )));
        }

        let offset = (page_id.page_num as u64) * (PAGE_SIZE as u64);
        handle.file.seek(SeekFrom::Start(offset))?;

        let mut buffer = [0u8; PAGE_SIZE];
        handle.file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    /// Writes a page to disk.
    pub fn write_page(&self, page_id: PageId, data: &[u8; PAGE_SIZE]) -> Result<()> {
        self.open_file(page_id.file_id)?;

        let mut files = self.files.lock();
        let handle = files.get_mut(&page_id.file_id).ok_or_else(|| {
            ZyronError::IoError(format!("file {} not open", page_id.file_id))
        })?;

        let offset = (page_id.page_num as u64) * (PAGE_SIZE as u64);
        handle.file.seek(SeekFrom::Start(offset))?;
        handle.file.write_all(data)?;

        if self.config.fsync_enabled {
            handle.file.sync_all()?;
        }

        // Update page count if we extended the file
        if page_id.page_num >= handle.num_pages {
            handle.num_pages = page_id.page_num + 1;
        }

        Ok(())
    }

    /// Allocates a new page in the specified file.
    ///
    /// Returns the PageId of the newly allocated page.
    pub fn allocate_page(&self, file_id: u32) -> Result<PageId> {
        self.open_file(file_id)?;

        let mut files = self.files.lock();
        let handle = files.get_mut(&file_id).ok_or_else(|| {
            ZyronError::IoError(format!("file {} not open", file_id))
        })?;

        let page_num = handle.num_pages;
        let page_id = PageId::new(file_id, page_num);

        // Write an empty page to extend the file
        let offset = (page_num as u64) * (PAGE_SIZE as u64);
        handle.file.seek(SeekFrom::Start(offset))?;
        handle.file.write_all(&[0u8; PAGE_SIZE])?;

        if self.config.fsync_enabled {
            handle.file.sync_all()?;
        }

        handle.num_pages = page_num + 1;

        Ok(page_id)
    }

    /// Returns the number of pages in a file.
    pub fn num_pages(&self, file_id: u32) -> Result<u32> {
        self.open_file(file_id)?;

        let files = self.files.lock();
        let handle = files.get(&file_id).ok_or_else(|| {
            ZyronError::IoError(format!("file {} not open", file_id))
        })?;

        Ok(handle.num_pages)
    }

    /// Flushes all pending writes to disk.
    pub fn flush(&self) -> Result<()> {
        let files = self.files.lock();
        for handle in files.values() {
            handle.file.sync_all()?;
        }
        Ok(())
    }

    /// Closes a specific file.
    pub fn close_file(&self, file_id: u32) -> Result<()> {
        let mut files = self.files.lock();
        if let Some(handle) = files.remove(&file_id) {
            handle.file.sync_all()?;
        }
        Ok(())
    }

    /// Closes all open files.
    pub fn close_all(&self) -> Result<()> {
        let mut files = self.files.lock();
        for (_, handle) in files.drain() {
            handle.file.sync_all()?;
        }
        Ok(())
    }

    /// Deletes a data file.
    pub fn delete_file(&self, file_id: u32) -> Result<()> {
        self.close_file(file_id)?;
        let path = self.file_path(file_id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

impl Drop for DiskManager {
    fn drop(&mut self) {
        let _ = self.close_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_disk_manager() -> (DiskManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        };
        let dm = DiskManager::new(config).unwrap();
        (dm, dir)
    }

    #[test]
    fn test_disk_manager_new() {
        let (dm, _dir) = create_test_disk_manager();
        assert!(dm.data_dir().exists());
    }

    #[test]
    fn test_disk_manager_allocate_page() {
        let (dm, _dir) = create_test_disk_manager();

        let page1 = dm.allocate_page(0).unwrap();
        assert_eq!(page1.file_id, 0);
        assert_eq!(page1.page_num, 0);

        let page2 = dm.allocate_page(0).unwrap();
        assert_eq!(page2.page_num, 1);

        assert_eq!(dm.num_pages(0).unwrap(), 2);
    }

    #[test]
    fn test_disk_manager_write_read() {
        let (dm, _dir) = create_test_disk_manager();

        let page_id = dm.allocate_page(0).unwrap();

        // Write data
        let mut data = [0u8; PAGE_SIZE];
        data[0] = 0xAB;
        data[100] = 0xCD;
        data[PAGE_SIZE - 1] = 0xEF;
        dm.write_page(page_id, &data).unwrap();

        // Read back
        let read_data = dm.read_page(page_id).unwrap();
        assert_eq!(read_data[0], 0xAB);
        assert_eq!(read_data[100], 0xCD);
        assert_eq!(read_data[PAGE_SIZE - 1], 0xEF);
    }

    #[test]
    fn test_disk_manager_multiple_files() {
        let (dm, _dir) = create_test_disk_manager();

        // Allocate pages in different files
        let page_f0 = dm.allocate_page(0).unwrap();
        let page_f1 = dm.allocate_page(1).unwrap();
        let page_f2 = dm.allocate_page(2).unwrap();

        assert_eq!(page_f0.file_id, 0);
        assert_eq!(page_f1.file_id, 1);
        assert_eq!(page_f2.file_id, 2);

        // Write to each
        let mut data0 = [0u8; PAGE_SIZE];
        data0[0] = 0x00;
        dm.write_page(page_f0, &data0).unwrap();

        let mut data1 = [0u8; PAGE_SIZE];
        data1[0] = 0x11;
        dm.write_page(page_f1, &data1).unwrap();

        let mut data2 = [0u8; PAGE_SIZE];
        data2[0] = 0x22;
        dm.write_page(page_f2, &data2).unwrap();

        // Read back
        assert_eq!(dm.read_page(page_f0).unwrap()[0], 0x00);
        assert_eq!(dm.read_page(page_f1).unwrap()[0], 0x11);
        assert_eq!(dm.read_page(page_f2).unwrap()[0], 0x22);
    }

    #[test]
    fn test_disk_manager_read_nonexistent_page() {
        let (dm, _dir) = create_test_disk_manager();

        // Allocate one page
        dm.allocate_page(0).unwrap();

        // Try to read page that doesn't exist
        let result = dm.read_page(PageId::new(0, 99));
        assert!(result.is_err());
    }

    #[test]
    fn test_disk_manager_overwrite_page() {
        let (dm, _dir) = create_test_disk_manager();

        let page_id = dm.allocate_page(0).unwrap();

        // Write initial data
        let mut data1 = [0u8; PAGE_SIZE];
        data1[0] = 0xAA;
        dm.write_page(page_id, &data1).unwrap();

        // Overwrite with new data
        let mut data2 = [0u8; PAGE_SIZE];
        data2[0] = 0xBB;
        dm.write_page(page_id, &data2).unwrap();

        // Read should return new data
        let read_data = dm.read_page(page_id).unwrap();
        assert_eq!(read_data[0], 0xBB);
    }

    #[test]
    fn test_disk_manager_persistence() {
        let dir = tempdir().unwrap();
        let page_id;

        // Write data
        {
            let config = DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: true,
            };
            let dm = DiskManager::new(config).unwrap();
            page_id = dm.allocate_page(0).unwrap();

            let mut data = [0u8; PAGE_SIZE];
            data[0] = 0xFF;
            dm.write_page(page_id, &data).unwrap();
        }

        // Read with new disk manager
        {
            let config = DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: true,
            };
            let dm = DiskManager::new(config).unwrap();

            let read_data = dm.read_page(page_id).unwrap();
            assert_eq!(read_data[0], 0xFF);
        }
    }

    #[test]
    fn test_disk_manager_delete_file() {
        let (dm, dir) = create_test_disk_manager();

        dm.allocate_page(0).unwrap();
        let file_path = dir.path().join("00000000.dat");
        assert!(file_path.exists());

        dm.delete_file(0).unwrap();
        assert!(!file_path.exists());
    }

    #[test]
    fn test_disk_manager_num_pages() {
        let (dm, _dir) = create_test_disk_manager();

        assert_eq!(dm.num_pages(0).unwrap(), 0);

        dm.allocate_page(0).unwrap();
        assert_eq!(dm.num_pages(0).unwrap(), 1);

        dm.allocate_page(0).unwrap();
        dm.allocate_page(0).unwrap();
        assert_eq!(dm.num_pages(0).unwrap(), 3);
    }

    #[test]
    fn test_disk_manager_flush() {
        let (dm, _dir) = create_test_disk_manager();

        dm.allocate_page(0).unwrap();
        dm.allocate_page(1).unwrap();

        // Should not panic
        dm.flush().unwrap();
    }

    #[test]
    fn test_disk_manager_close_file() {
        let (dm, _dir) = create_test_disk_manager();

        dm.allocate_page(0).unwrap();
        dm.close_file(0).unwrap();

        // Can reopen and continue
        dm.allocate_page(0).unwrap();
        assert_eq!(dm.num_pages(0).unwrap(), 2);
    }
}
