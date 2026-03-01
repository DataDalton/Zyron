//! Checkpoint performance profiling test.
//! Measures each stage of the write/load path separately.

use std::sync::Arc;
use std::time::Instant;
use tempfile::tempdir;

use zyron_buffer::BufferPool;
use zyron_common::page::PageId;
use zyron_storage::{BTreeIndex, DiskManager, DiskManagerConfig, TupleId};

#[tokio::test]
async fn profile_checkpoint_stages() {
    const KEY_COUNT: usize = 1_000_000;

    let dir = tempdir().unwrap();
    let checkpoint_dir = dir.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::auto_sized());

    let mut btree = BTreeIndex::create(disk.clone(), pool.clone(), 0, checkpoint_dir.clone())
        .await
        .unwrap();

    for i in 0..KEY_COUNT as u64 {
        let key = i.to_be_bytes();
        let tid = TupleId::new(PageId::new(0, i % 1000), (i % 100) as u16);
        btree.insert_exclusive(&key, tid).unwrap();
    }

    println!("\n=== Checkpoint Stage Profiling (1M keys) ===\n");

    // Stage 1: Buffer allocation
    let page_count = 2873u32; // approximate
    let entry_size = 4 + 16384;
    let total_bytes = 32 + (page_count as usize) * entry_size + 12;

    let t0 = Instant::now();
    let buf = vec![0u8; total_bytes];
    let alloc_time = t0.elapsed();
    drop(buf);
    println!(
        "  Alloc (vec![0u8; {}MB]): {:.2} ms",
        total_bytes / 1024 / 1024,
        alloc_time.as_secs_f64() * 1000.0
    );

    // Stage 2: Full checkpoint write (includes alloc + copy + checksum + write + fsync)
    let ckpt_path = checkpoint_dir.join("index_0.zyridx");
    let t1 = Instant::now();
    btree.force_checkpoint(42000).unwrap();
    let full_write_time = t1.elapsed();
    let file_size = std::fs::metadata(&ckpt_path).unwrap().len();
    println!(
        "  Full write: {:.2} ms ({:.1} MB, {:.0} MB/sec)",
        full_write_time.as_secs_f64() * 1000.0,
        file_size as f64 / 1024.0 / 1024.0,
        (file_size as f64 / 1024.0 / 1024.0) / full_write_time.as_secs_f64()
    );

    // Stage 3: Read the file into memory
    let t2 = Instant::now();
    let data = std::fs::read(&ckpt_path).unwrap();
    let read_time = t2.elapsed();
    println!(
        "  Raw fs::read: {:.2} ms ({:.0} MB/sec)",
        read_time.as_secs_f64() * 1000.0,
        (data.len() as f64 / 1024.0 / 1024.0) / read_time.as_secs_f64()
    );

    // Stage 4: Just the write (no fsync) for comparison
    let tmp_path = checkpoint_dir.join("test_no_fsync.zyridx");
    let buf2 = vec![0xABu8; file_size as usize];
    let t3 = Instant::now();
    std::fs::write(&tmp_path, &buf2).unwrap();
    let write_no_fsync = t3.elapsed();
    println!(
        "  Raw fs::write (no fsync): {:.2} ms ({:.0} MB/sec)",
        write_no_fsync.as_secs_f64() * 1000.0,
        (buf2.len() as f64 / 1024.0 / 1024.0) / write_no_fsync.as_secs_f64()
    );

    // Stage 5: Just fsync
    let file = std::fs::File::open(&tmp_path).unwrap();
    let t4 = Instant::now();
    file.sync_all().unwrap();
    let fsync_time = t4.elapsed();
    println!(
        "  Standalone fsync: {:.2} ms",
        fsync_time.as_secs_f64() * 1000.0
    );

    // Stage 6: Write + fsync together
    let tmp2 = checkpoint_dir.join("test_with_fsync.zyridx");
    let t5 = Instant::now();
    let f = std::fs::File::create(&tmp2).unwrap();
    std::io::Write::write_all(&mut &f, &buf2).unwrap();
    f.sync_all().unwrap();
    let write_with_fsync = t5.elapsed();
    println!(
        "  Write + fsync: {:.2} ms ({:.0} MB/sec)",
        write_with_fsync.as_secs_f64() * 1000.0,
        (buf2.len() as f64 / 1024.0 / 1024.0) / write_with_fsync.as_secs_f64()
    );

    // Stage 7: Full load from checkpoint
    let t6 = Instant::now();
    let loaded = BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
        .await
        .unwrap();
    let full_load_time = t6.elapsed();
    println!(
        "  Full load: {:.2} ms ({:.0} MB/sec)",
        full_load_time.as_secs_f64() * 1000.0,
        (file_size as f64 / 1024.0 / 1024.0) / full_load_time.as_secs_f64()
    );
    drop(loaded);

    println!(
        "\n  File size: {:.2} MB ({} pages)",
        file_size as f64 / 1024.0 / 1024.0,
        page_count
    );
}
