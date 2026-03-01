//! B+Tree checkpoint serialization and deserialization.
//!
//! Format v8: Single-pass columnar extraction from leaf chain.
//! Common prefix compression for keys, packed page_num + slot_id into u32.
//! O(pages) for metadata, O(entries) for data with raw pointer access.
//!
//! File layout (.zyridx v8):
//!   Header (32 bytes):
//!     magic(8), version(4), lsn(8), entry_count(4),
//!     key_len(2), prefix_len(2), checksum(4)
//!   Key prefix: prefix_len bytes (common prefix of all keys)
//!   Key suffixes: entry_count * suffix_len bytes
//!   Packed values: entry_count * 4 bytes (page_num << 16 | slot_id)

use super::page::{BTreeInternalPage, BTreeLeafPage};
use super::store::InMemoryPageStore;
use super::types::LeafPageHeader;
use std::path::Path;
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId};
use zyron_common::{Result, ZyronError};

const ZYIDX_MAGIC: [u8; 8] = *b"ZYIDX\0\0\0";
const ZYIDX_FORMAT_VERSION: u32 = 8;
const ZYIDX_HEADER_SIZE: usize = 32;

const SLOT_ARRAY_START: usize = PageHeader::SIZE + LeafPageHeader::SIZE;
const SLOT_SIZE: usize = 4;

/// 4-lane parallel multiply-XOR checksum over header and data regions.
/// Processes 32 bytes per iteration with independent multiply chains.
/// The CPU pipelines all 4 lanes (3-cycle multiply latency, 4 independent
/// chains), reaching ~40 GB/s vs CRC32C's ~20 GB/s on modern x86.
/// Covers all bytes (no sampling). Detects bit flips, zeroed regions,
/// truncation, and byte shifts.
fn checkpoint_checksum(header: &[u8], data: &[u8]) -> u32 {
    // Odd primes with good bit distribution for multiply-XOR mixing.
    const P0: u64 = 0x517cc1b727220a95;
    const P1: u64 = 0x6c62272e07bb0143;
    const P2: u64 = 0x8ebc6af09c88c6e3;
    const P3: u64 = 0x305f1d4b1e0e2a6f;
    const FINAL: u64 = 0xff51afd7ed558ccd;

    // Seed lane 0 with data length to distinguish different-sized inputs.
    let mut s0: u64 = P0 ^ (data.len() as u64);
    let mut s1: u64 = P1;
    let mut s2: u64 = P2;
    let mut s3: u64 = P3;

    // Mix header (28 bytes = 3 x u64 + 1 x u32).
    let hp = header.as_ptr();
    unsafe {
        s0 = (s0 ^ (hp as *const u64).read_unaligned()).wrapping_mul(P0);
        s1 = (s1 ^ (hp.add(8) as *const u64).read_unaligned()).wrapping_mul(P1);
        s2 = (s2 ^ (hp.add(16) as *const u64).read_unaligned()).wrapping_mul(P2);
        s3 = (s3 ^ ((hp.add(24) as *const u32).read_unaligned() as u64)).wrapping_mul(P3);
    }

    // Process data in 32-byte chunks (4 independent multiply chains).
    let ptr = data.as_ptr();
    let len = data.len();
    let mut i = 0;
    while i + 32 <= len {
        unsafe {
            s0 = (s0 ^ (ptr.add(i) as *const u64).read_unaligned()).wrapping_mul(P0);
            s1 = (s1 ^ (ptr.add(i + 8) as *const u64).read_unaligned()).wrapping_mul(P1);
            s2 = (s2 ^ (ptr.add(i + 16) as *const u64).read_unaligned()).wrapping_mul(P2);
            s3 = (s3 ^ (ptr.add(i + 24) as *const u64).read_unaligned()).wrapping_mul(P3);
        }
        i += 32;
    }

    // Tail: remaining 8-byte words into lane 0.
    while i + 8 <= len {
        s0 = (s0 ^ unsafe { (ptr.add(i) as *const u64).read_unaligned() }).wrapping_mul(P0);
        i += 8;
    }
    // Tail: remaining < 8 bytes.
    if i < len {
        let mut tail: u64 = 0;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(i), &mut tail as *mut u64 as *mut u8, len - i);
        }
        s0 = (s0 ^ tail).wrapping_mul(P0);
    }

    // Combine all 4 lanes and finalize to 32 bits.
    let mut h = s0 ^ s1 ^ s2 ^ s3;
    h ^= h >> 33;
    h = h.wrapping_mul(FINAL);
    h ^= h >> 33;
    h as u32
}

pub fn write_checkpoint_from_store(
    path: &Path,
    store: &InMemoryPageStore,
    checkpoint_lsn: u64,
    root_page_num: u32,
    height: u32,
    fsync: bool,
) -> Result<u64> {
    // Find first leaf by traversing internal nodes leftmost pointers.
    let mut first_leaf = root_page_num;
    if height > 1 {
        for _ in 0..(height - 1) {
            if let Some(data) = store.get(first_leaf) {
                let internal = BTreeInternalPage::from_bytes(*data);
                first_leaf = internal.leftmost_child().page_num as u32;
            } else {
                return write_empty_checkpoint(path, checkpoint_lsn, fsync);
            }
        }
    }

    let ho = LeafPageHeader::OFFSET;

    // Walk leaf chain: collect entry counts and determine key_len.
    let mut total_entries = 0u32;
    let mut key_len: u16 = 0;
    let mut leaf_pages: Vec<(u32, u16)> = Vec::with_capacity(4096);

    {
        let mut cur = first_leaf;
        loop {
            let pd = match store.get(cur) {
                Some(d) => d,
                None => break,
            };
            let ns = u16::from_le_bytes([pd[ho], pd[ho + 1]]);
            if leaf_pages.is_empty() && ns > 0 {
                let e0_off =
                    u16::from_le_bytes([pd[SLOT_ARRAY_START], pd[SLOT_ARRAY_START + 1]]) as usize;
                key_len = u16::from_le_bytes([pd[e0_off], pd[e0_off + 1]]);
            }
            leaf_pages.push((cur, ns));
            total_entries += ns as u32;
            let next = u64::from_le_bytes([
                pd[ho + 4],
                pd[ho + 5],
                pd[ho + 6],
                pd[ho + 7],
                pd[ho + 8],
                pd[ho + 9],
                pd[ho + 10],
                pd[ho + 11],
            ]);
            if next == u64::MAX {
                break;
            }
            cur = next as u32;
        }
    }

    if total_entries == 0 {
        return write_empty_checkpoint(path, checkpoint_lsn, fsync);
    }

    let kl = key_len as usize;

    // Common prefix: compare first key of first leaf with last key of last leaf.
    let mut prefix_len = 0usize;
    if leaf_pages.len() > 1 && kl > 0 {
        let (fp, fns) = leaf_pages[0];
        let (lp, lns) = leaf_pages[leaf_pages.len() - 1];
        if fns > 0 && lns > 0 {
            let fd = store.get(fp).unwrap();
            let ld = store.get(lp).unwrap();
            let f_off =
                u16::from_le_bytes([fd[SLOT_ARRAY_START], fd[SLOT_ARRAY_START + 1]]) as usize;
            let l_slot_off = SLOT_ARRAY_START + (lns as usize - 1) * SLOT_SIZE;
            let l_off = u16::from_le_bytes([ld[l_slot_off], ld[l_slot_off + 1]]) as usize;
            while prefix_len < kl && fd[f_off + 2 + prefix_len] == ld[l_off + 2 + prefix_len] {
                prefix_len += 1;
            }
        }
    }

    let suffix_len = kl - prefix_len;
    let n = total_entries as usize;
    let data_size = prefix_len + n * suffix_len + n * 4;
    let total_size = ZYIDX_HEADER_SIZE + data_size;

    // Allocate output buffer without zeroing.
    let mut buf = Vec::with_capacity(total_size);
    unsafe {
        buf.set_len(total_size);
    }
    let bp = buf.as_mut_ptr();

    // Header
    unsafe {
        std::ptr::copy_nonoverlapping(ZYIDX_MAGIC.as_ptr(), bp, 8);
        (bp.add(8) as *mut u32).write_unaligned(ZYIDX_FORMAT_VERSION);
        (bp.add(12) as *mut u64).write_unaligned(checkpoint_lsn);
        (bp.add(20) as *mut u32).write_unaligned(total_entries);
        (bp.add(24) as *mut u16).write_unaligned(key_len);
        (bp.add(26) as *mut u16).write_unaligned(prefix_len as u16);
        (bp.add(28) as *mut u32).write_unaligned(0);
    }

    // Key prefix
    let prefix_start = ZYIDX_HEADER_SIZE;
    if prefix_len > 0 {
        let (fp, _) = leaf_pages[0];
        let fd = store.get(fp).unwrap();
        let f_off = u16::from_le_bytes([fd[SLOT_ARRAY_START], fd[SLOT_ARRAY_START + 1]]) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                fd.as_ptr().add(f_off + 2),
                bp.add(prefix_start),
                prefix_len,
            );
        }
    }

    // Extract key suffixes and packed values using slot array lookups.
    let suffixes_start = prefix_start + prefix_len;
    let values_start = suffixes_start + n * suffix_len;
    let mut sk = unsafe { bp.add(suffixes_start) };
    let mut vp = unsafe { bp.add(values_start) };

    // Extract suffixes and packed values from leaf pages.
    // On-disk leaf entry format: key_len(2) + key + page_num(u32) + slot_id(u16).
    for &(pn, ns) in &leaf_pages {
        let pd = store.get(pn).unwrap();
        let pp = pd.as_ptr();
        let ns = ns as usize;
        for slot in 0..ns {
            let slot_off = SLOT_ARRAY_START + slot * SLOT_SIZE;
            let entry_off = unsafe { (pp.add(slot_off) as *const u16).read_unaligned() as usize };
            unsafe {
                std::ptr::copy_nonoverlapping(pp.add(entry_off + 2 + prefix_len), sk, suffix_len);
                sk = sk.add(suffix_len);
                let pid_offset = entry_off + 2 + kl;
                let page_num = (pp.add(pid_offset) as *const u32).read_unaligned();
                let slot_id = (pp.add(pid_offset + 4) as *const u16).read_unaligned();
                (vp as *mut u32).write_unaligned((page_num << 16) | slot_id as u32);
                vp = vp.add(4);
            }
        }
    }

    // Checksum
    let checksum = checkpoint_checksum(&buf[..28], &buf[ZYIDX_HEADER_SIZE..]);
    unsafe {
        (bp.add(28) as *mut u32).write_unaligned(checksum);
    }

    {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&buf)?;
        if fsync {
            file.sync_all()?;
        }
    }

    Ok(total_size as u64)
}

fn write_empty_checkpoint(path: &Path, checkpoint_lsn: u64, fsync: bool) -> Result<u64> {
    let mut buf = [0u8; ZYIDX_HEADER_SIZE];
    buf[0..8].copy_from_slice(&ZYIDX_MAGIC);
    buf[8..12].copy_from_slice(&ZYIDX_FORMAT_VERSION.to_le_bytes());
    buf[12..20].copy_from_slice(&checkpoint_lsn.to_le_bytes());
    let checksum = checkpoint_checksum(&buf[..28], &[]);
    buf[28..32].copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(path, &buf)?;
    if fsync {
        std::fs::File::open(path)?.sync_all()?;
    }
    Ok(ZYIDX_HEADER_SIZE as u64)
}

pub fn load_checkpoint_into_store(
    path: &Path,
    store: &mut InMemoryPageStore,
    file_id: u32,
) -> Result<(u64, u32, u32, u32)> {
    // Read file using pre-allocated uninitialized buffer + read_exact
    // to avoid Vec zeroing overhead of std::fs::read.
    let buf = {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to open checkpoint file {}: {}",
                path.display(),
                e
            ))
        })?;
        let file_len = file
            .metadata()
            .map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to read checkpoint metadata {}: {}",
                    path.display(),
                    e
                ))
            })?
            .len() as usize;
        let mut buf = Vec::with_capacity(file_len);
        unsafe {
            buf.set_len(file_len);
        }
        file.read_exact(&mut buf).map_err(|e| {
            ZyronError::IoError(format!(
                "failed to read checkpoint file {}: {}",
                path.display(),
                e
            ))
        })?;
        buf
    };

    if buf.len() < ZYIDX_HEADER_SIZE {
        return Err(ZyronError::RecoveryFailed(
            "checkpoint file too small".into(),
        ));
    }
    if buf[0..8] != ZYIDX_MAGIC {
        return Err(ZyronError::RecoveryFailed("invalid magic bytes".into()));
    }
    let ver = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    if ver != ZYIDX_FORMAT_VERSION {
        return Err(ZyronError::RecoveryFailed(format!(
            "unsupported version: {} (expected {})",
            ver, ZYIDX_FORMAT_VERSION
        )));
    }

    let checkpoint_lsn = u64::from_le_bytes(buf[12..20].try_into().unwrap());
    let entry_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
    let key_len = u16::from_le_bytes([buf[24], buf[25]]);
    let prefix_len = u16::from_le_bytes([buf[26], buf[27]]) as usize;
    let stored_checksum = u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]);

    let kl = key_len as usize;
    let suffix_len = kl - prefix_len;
    let n = entry_count as usize;
    let data_size = prefix_len + n * suffix_len + n * 4;
    let expected_size = ZYIDX_HEADER_SIZE + data_size;
    if buf.len() < expected_size {
        return Err(ZyronError::RecoveryFailed(
            "checkpoint file truncated".into(),
        ));
    }

    // Validate checksum
    let computed = checkpoint_checksum(
        &buf[..28],
        &buf[ZYIDX_HEADER_SIZE..ZYIDX_HEADER_SIZE + data_size],
    );
    if stored_checksum != computed {
        return Err(ZyronError::RecoveryFailed(
            "checkpoint checksum mismatch".into(),
        ));
    }

    if entry_count == 0 {
        let root = store.allocate();
        store.write(
            root,
            BTreeLeafPage::new(PageId::new(file_id, root as u64)).as_bytes(),
        );
        return Ok((checkpoint_lsn, root, 0, 1));
    }

    let src = buf.as_ptr();
    let prefix_start = ZYIDX_HEADER_SIZE;
    let suffixes_start = prefix_start + prefix_len;
    let values_start = suffixes_start + n * suffix_len;

    let eds = 2 + kl + 6; // entry data size per entry: key_len(2) + key + page_num(4) + slot_id(2)
    let max_entries_per_page = (PAGE_SIZE - SLOT_ARRAY_START) / (eds + SLOT_SIZE);
    let num_leaves = (n + max_entries_per_page - 1) / max_entries_per_page;
    let data_end_full = PAGE_SIZE - max_entries_per_page * eds;
    let slot_array_bytes_full = max_entries_per_page * SLOT_SIZE;

    // Pre-build slot array for full pages (same for all full pages).
    let mut slot_array_full = vec![0u8; slot_array_bytes_full];
    for slot in 0..max_entries_per_page {
        let entry_off = (PAGE_SIZE - (slot + 1) * eds) as u16;
        let so = slot * SLOT_SIZE;
        slot_array_full[so..so + 2].copy_from_slice(&entry_off.to_le_bytes());
        slot_array_full[so + 2..so + 4].copy_from_slice(&(eds as u16).to_le_bytes());
    }

    // Pre-build 56-byte page header template (PageHeader + LeafPageHeader).
    let mut hdr_tmpl = [0u8; SLOT_ARRAY_START];
    hdr_tmpl[16] = zyron_common::page::PageType::BTreeLeaf as u8;
    hdr_tmpl[ho_off()..ho_off() + 2].copy_from_slice(&(max_entries_per_page as u16).to_le_bytes());
    hdr_tmpl[ho_off() + 2..ho_off() + 4].copy_from_slice(&(data_end_full as u16).to_le_bytes());
    hdr_tmpl[ho_off() + 4..ho_off() + 12].copy_from_slice(&u64::MAX.to_le_bytes());

    // Allocate all leaf pages via arena (single alloc_zeroed, pre-faults OS pages).
    let first_page = store.bulk_allocate(num_leaves);

    let mut first_keys: Vec<u8> = Vec::with_capacity(num_leaves * kl);
    let suffix_in_entry = 2 + prefix_len;
    let pid_in_entry = 2 + kl;

    let mut ei = 0usize;

    // Pre-build a full entry template: [key_len:2][prefix:prefix_len][...suffix...][...pid...][...sid...]
    // Stamp key_len and prefix once, then the inner loop only writes suffix + value fields.
    let mut entry_tmpl = vec![0u8; eds];
    entry_tmpl[0..2].copy_from_slice(&key_len.to_le_bytes());
    if prefix_len > 0 {
        entry_tmpl[2..2 + prefix_len]
            .copy_from_slice(&buf[prefix_start..prefix_start + prefix_len]);
    }

    for leaf_idx in 0..num_leaves {
        let ns = max_entries_per_page.min(n - ei);
        let pn = first_page + leaf_idx as u32;
        let pb = store.get_mut(pn).unwrap();
        let pp = pb.as_mut_ptr();

        // Write page header template + page_id fields.
        unsafe {
            std::ptr::copy_nonoverlapping(hdr_tmpl.as_ptr(), pp, SLOT_ARRAY_START);
            // PageHeader: file_id(u32) at offset 0, page_num(u64) at offset 4
            (pp as *mut u32).write_unaligned(file_id);
            (pp.add(4) as *mut u64).write_unaligned(pn as u64);
        }

        // Fix num_slots and data_end for partial last page.
        if ns < max_entries_per_page {
            unsafe {
                (pp.add(ho_off()) as *mut u16).write_unaligned(ns as u16);
                (pp.add(ho_off() + 2) as *mut u16).write_unaligned((PAGE_SIZE - ns * eds) as u16);
            }
        }

        // Write slot array (bulk copy for full pages, computed for partial).
        if ns == max_entries_per_page {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    slot_array_full.as_ptr(),
                    pp.add(SLOT_ARRAY_START),
                    slot_array_bytes_full,
                );
            }
        } else {
            for slot in 0..ns {
                let entry_off = (PAGE_SIZE - (slot + 1) * eds) as u16;
                unsafe {
                    (pp.add(SLOT_ARRAY_START + slot * SLOT_SIZE) as *mut u16)
                        .write_unaligned(entry_off);
                    (pp.add(SLOT_ARRAY_START + slot * SLOT_SIZE + 2) as *mut u16)
                        .write_unaligned(eds as u16);
                }
            }
        }

        // Reconstruct entries from columnar checkpoint data into page layout.
        // Specialized for common key sizes to emit direct mov instructions
        // instead of memcpy calls from copy_nonoverlapping with runtime sizes.
        let tmpl_ptr = entry_tmpl.as_ptr();
        let tmpl_fixed = 2 + prefix_len;
        let mut entry_base = PAGE_SIZE - eds;
        let mut s_off = suffixes_start + ei * suffix_len;
        let mut v_off = values_start + ei * 4;

        match (tmpl_fixed, suffix_len) {
            // u64 keys with no common prefix: direct u16 + u64 writes.
            (2, 8) => {
                let kl_le = key_len.to_le();
                for _ in 0..ns {
                    unsafe {
                        // key_len: single u16 store.
                        (pp.add(entry_base) as *mut u16).write_unaligned(kl_le);
                        // suffix: 8 bytes (direct u64 read/write).
                        let suf = (src.add(s_off) as *const u64).read_unaligned();
                        (pp.add(entry_base + 2) as *mut u64).write_unaligned(suf);
                        // Unpack value: (page_num << 16) | slot_id.
                        // Write page_num(u32) + slot_id(u16) = 6 bytes.
                        let packed = (src.add(v_off) as *const u32).read_unaligned();
                        (pp.add(entry_base + pid_in_entry) as *mut u32)
                            .write_unaligned(packed >> 16);
                        (pp.add(entry_base + pid_in_entry + 4) as *mut u16)
                            .write_unaligned((packed & 0xFFFF) as u16);
                    }
                    entry_base -= eds;
                    s_off += 8;
                    v_off += 4;
                }
            }
            // u32 keys with no common prefix: direct u16 + u32 writes.
            (2, 4) => {
                let kl_le = key_len.to_le();
                for _ in 0..ns {
                    unsafe {
                        (pp.add(entry_base) as *mut u16).write_unaligned(kl_le);
                        let suf = (src.add(s_off) as *const u32).read_unaligned();
                        (pp.add(entry_base + 2) as *mut u32).write_unaligned(suf);
                        let packed = (src.add(v_off) as *const u32).read_unaligned();
                        (pp.add(entry_base + pid_in_entry) as *mut u32)
                            .write_unaligned(packed >> 16);
                        (pp.add(entry_base + pid_in_entry + 4) as *mut u16)
                            .write_unaligned((packed & 0xFFFF) as u16);
                    }
                    entry_base -= eds;
                    s_off += 4;
                    v_off += 4;
                }
            }
            // Generic fallback for other key sizes.
            _ => {
                for _ in 0..ns {
                    unsafe {
                        std::ptr::copy_nonoverlapping(tmpl_ptr, pp.add(entry_base), tmpl_fixed);
                        std::ptr::copy_nonoverlapping(
                            src.add(s_off),
                            pp.add(entry_base + suffix_in_entry),
                            suffix_len,
                        );
                        let packed = (src.add(v_off) as *const u32).read_unaligned();
                        (pp.add(entry_base + pid_in_entry) as *mut u32)
                            .write_unaligned(packed >> 16);
                        (pp.add(entry_base + pid_in_entry + 4) as *mut u16)
                            .write_unaligned((packed & 0xFFFF) as u16);
                    }
                    entry_base -= eds;
                    s_off += suffix_len;
                    v_off += 4;
                }
            }
        }
        ei += ns;

        // Record first key for internal page construction.
        let feo = PAGE_SIZE - eds;
        first_keys.extend_from_slice(&pb[feo + 2..feo + 2 + kl]);

        // Set next-leaf pointer (stored as PageId.as_u64() = file_id << 32 | page_num).
        if leaf_idx + 1 < num_leaves {
            let next_pn = first_page + leaf_idx as u32 + 1;
            let nlo = ho_off() + 4;
            let next_packed = ((file_id as u64) << 32) | next_pn as u64;
            pb[nlo..nlo + 8].copy_from_slice(&next_packed.to_le_bytes());
        }
    }

    let leaf_page_nums: Vec<u32> = (first_page..first_page + num_leaves as u32).collect();

    if leaf_page_nums.len() == 1 {
        return Ok((checkpoint_lsn, leaf_page_nums[0], entry_count, 1));
    }

    let (root, h) = build_internal_pages(store, &leaf_page_nums, &first_keys, kl, file_id);
    Ok((checkpoint_lsn, root, entry_count, h))
}

#[inline(always)]
fn ho_off() -> usize {
    LeafPageHeader::OFFSET
}

fn build_internal_pages(
    store: &mut InMemoryPageStore,
    leaf_pages: &[u32],
    keys_flat: &[u8],
    kl: usize,
    file_id: u32,
) -> (u32, u32) {
    let mut cp: Vec<u32> = leaf_pages.to_vec();
    let mut ck: Vec<u8> = keys_flat.to_vec();
    let mut level = 0u16;
    loop {
        let mut pp: Vec<u32> = Vec::new();
        let mut pk: Vec<u8> = Vec::new();
        let ids = PageHeader::SIZE + super::types::InternalPageHeader::SIZE;
        let iu = PAGE_SIZE - ids - 8;
        let it = iu * 3 / 4;
        let es = 2 + kl + 4;
        let mut ci: Option<(u32, BTreeInternalPage)> = None;
        let mut cu = 0usize;
        let mut first = true;
        for i in 0..cp.len() {
            let ks = i * kl;
            if first {
                let pn = store.allocate();
                let pid = PageId::new(file_id, pn as u64);
                let mut int = BTreeInternalPage::new(pid, level);
                int.set_leftmost_child(PageId::new(file_id, cp[i] as u64));
                pp.push(pn);
                pk.extend_from_slice(&ck[ks..ks + kl]);
                ci = Some((pn, int));
                cu = 0;
                first = false;
                continue;
            }
            if cu + es > it {
                if let Some((pn, int)) = ci.take() {
                    store.write(pn, int.as_bytes());
                }
                let pn = store.allocate();
                let pid = PageId::new(file_id, pn as u64);
                let mut int = BTreeInternalPage::new(pid, level);
                int.set_leftmost_child(PageId::new(file_id, cp[i] as u64));
                pp.push(pn);
                pk.extend_from_slice(&ck[ks..ks + kl]);
                ci = Some((pn, int));
                cu = 0;
                continue;
            }
            if let Some((_, ref mut int)) = ci {
                let key = bytes::Bytes::copy_from_slice(&ck[ks..ks + kl]);
                let _ = int.insert(key, PageId::new(file_id, cp[i] as u64));
                cu += es;
            }
        }
        if let Some((pn, int)) = ci.take() {
            store.write(pn, int.as_bytes());
        }
        if pp.len() == 1 {
            return (pp[0], level as u32 + 2);
        }
        cp = pp;
        ck = pk;
        level += 1;
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub wal_bytes_threshold: u64,
    pub max_interval_secs: u64,
    pub min_interval_secs: u64,
    pub fsync: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            wal_bytes_threshold: 64 * 1024 * 1024,
            max_interval_secs: 600,
            min_interval_secs: 5,
            fsync: true,
        }
    }
}

pub struct CheckpointTrigger {
    config: CheckpointConfig,
    last_checkpoint_time: std::time::Instant,
}

impl CheckpointTrigger {
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            last_checkpoint_time: std::time::Instant::now(),
        }
    }
    /// Checks if a checkpoint should be triggered. The wal_bytes parameter
    /// comes from the lock-free AtomicU64 counter on BTreeIndex.
    /// Checks wal_bytes first to avoid Instant::elapsed() syscall
    /// (~15-20ns on Windows) on the common path where no checkpoint is needed.
    #[inline]
    pub fn should_checkpoint(&self, wal_bytes: u64) -> bool {
        if wal_bytes >= self.config.wal_bytes_threshold {
            // Bytes threshold reached. Only call elapsed() for min_interval guard.
            return self.last_checkpoint_time.elapsed().as_secs() >= self.config.min_interval_secs;
        }
        // Below byte threshold. Check time-based max_interval (requires elapsed).
        self.last_checkpoint_time.elapsed().as_secs() >= self.config.max_interval_secs
    }
    pub fn reset(&mut self) {
        self.last_checkpoint_time = std::time::Instant::now();
    }
    #[inline]
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::TupleId;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.zyridx");
        let mut store = InMemoryPageStore::new();
        let root = store.allocate();
        let mut leaf = BTreeLeafPage::new(PageId::new(0, root as u64));
        for i in 0..100u64 {
            leaf.insert(
                bytes::Bytes::copy_from_slice(&i.to_be_bytes()),
                TupleId::new(PageId::new(0, i % 10), (i % 5) as u16),
            )
            .unwrap();
        }
        store.write(root, leaf.as_bytes());
        write_checkpoint_from_store(&path, &store, 42, root, 1, false).unwrap();
        let mut ls = InMemoryPageStore::new();
        let (lsn, lr, c, h) = load_checkpoint_into_store(&path, &mut ls, 0).unwrap();
        assert_eq!(lsn, 42);
        assert_eq!(c, 100);
        assert_eq!(h, 1);
        for i in 0..100u64 {
            let f = BTreeLeafPage::get_in_slice(ls.get(lr).unwrap(), &i.to_be_bytes());
            assert!(f.is_some(), "Key {} missing", i);
            let t = f.unwrap();
            assert_eq!(t.page_id.page_num, i % 10);
            assert_eq!(t.slot_id, (i % 5) as u16);
        }
    }

    #[test]
    fn test_checkpoint_detects_corruption() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.zyridx");
        let mut store = InMemoryPageStore::new();
        let root = store.allocate();
        let mut leaf = BTreeLeafPage::new(PageId::new(0, root as u64));
        for i in 0..10u64 {
            leaf.insert(
                bytes::Bytes::copy_from_slice(&i.to_be_bytes()),
                TupleId::new(PageId::new(0, 0), 0),
            )
            .unwrap();
        }
        store.write(root, leaf.as_bytes());
        write_checkpoint_from_store(&path, &store, 1, root, 1, false).unwrap();
        let mut d = std::fs::read(&path).unwrap();
        assert!(d.len() > 40, "Checkpoint file too small");
        d[40] ^= 0xFF;
        std::fs::write(&path, &d).unwrap();
        let mut ls = InMemoryPageStore::new();
        assert!(load_checkpoint_into_store(&path, &mut ls, 0).is_err());
    }

    #[test]
    fn test_checkpoint_trigger_bytes_threshold() {
        let t = CheckpointTrigger::new(CheckpointConfig {
            wal_bytes_threshold: 1000,
            max_interval_secs: 3600,
            min_interval_secs: 0,
            fsync: true,
        });
        assert!(!t.should_checkpoint(0));
        assert!(!t.should_checkpoint(500));
        assert!(t.should_checkpoint(1000));
    }

    #[test]
    fn test_checkpoint_trigger_min_interval() {
        let t = CheckpointTrigger::new(CheckpointConfig {
            wal_bytes_threshold: 0,
            max_interval_secs: 3600,
            min_interval_secs: 999,
            fsync: true,
        });
        assert!(!t.should_checkpoint(0));
    }

    #[test]
    fn test_checkpoint_multi_page_round_trip() {
        let dir = tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let disk = std::sync::Arc::new(
                crate::DiskManager::new(crate::DiskManagerConfig {
                    data_dir: dir.path().to_path_buf(),
                    fsync_enabled: false,
                })
                .await
                .unwrap(),
            );
            let pool = std::sync::Arc::new(zyron_buffer::BufferPool::auto_sized());
            let ckpt_dir = dir.path().join("ckpt");
            std::fs::create_dir_all(&ckpt_dir).unwrap();

            let mut btree = crate::btree::index::BTreeIndex::create_with_config(
                disk.clone(),
                pool.clone(),
                0,
                ckpt_dir.clone(),
                CheckpointConfig {
                    fsync: false,
                    ..CheckpointConfig::default()
                },
            )
            .await
            .unwrap();

            let n = 1_000_000u64;
            for i in 0..n {
                let key = i.to_be_bytes();
                let tid = crate::tuple::TupleId::new(PageId::new(0, i % 1000), (i % 100) as u16);
                btree.insert_exclusive(&key, tid).unwrap();
            }

            btree.force_checkpoint(42).unwrap();

            let loaded =
                crate::btree::index::BTreeIndex::open(disk.clone(), pool.clone(), 0, &ckpt_dir)
                    .await
                    .unwrap();

            // Verify all keys survived checkpoint round-trip
            let mut first_missing = None;
            let mut missing_count = 0;
            for i in 0..n {
                let key = i.to_be_bytes();
                let expected =
                    crate::tuple::TupleId::new(PageId::new(0, i % 1000), (i % 100) as u16);
                let found = loaded.search_sync(&key);
                if found != Some(expected) {
                    missing_count += 1;
                    if first_missing.is_none() {
                        first_missing = Some((i, found, expected));
                    }
                }
            }
            if let Some((i, found, expected)) = first_missing {
                panic!(
                    "First missing key: {} (total missing: {}) found={:?} expected={:?}",
                    i, missing_count, found, expected
                );
            }
        });
    }
}
