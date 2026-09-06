//! B+Tree checkpoint serialization and deserialization.
//!
//! Single-pass columnar extraction from the leaf chain, with common prefix
//! compression for keys. O(pages) for metadata, O(entries) for data.
//!
//! File layout (.zyridx):
//!   Header (40 bytes), the first 20 the universal format envelope:
//!     magic(4) = "ZCPT", format_version(4), header_length(4), flags(4),
//!     header_checksum(4), lsn(8), entry_count(4), key_len(2),
//!     prefix_len(2), value_width(2), reserved(2)
//!   Key prefix: prefix_len bytes (common prefix of all keys)
//!   Key suffixes: entry_count * suffix_len bytes
//!   Values: entry_count * value_width bytes of locator payload
//!   Footer: body checksum(4)
//!
//! value_width is the locator payload width shared by every entry: 7 when the
//! index addresses heap rows, 17 otherwise. An index holding both kinds is
//! written entirely in the wide form so the column keeps one stride.

use super::page::{BTreeInternalPage, BTreeLeafPage};
use super::store::{InMemoryPageStore, UninitPage};
use super::types::LeafPageHeader;
use std::path::Path;
use zyron_common::format::envelope::{self, ENVELOPE_FOOTER_LEN, ENVELOPE_HEADER_LEN};
use zyron_common::format::{FormatKind, FormatVersion};
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId};
use zyron_common::profile::{self, Phase};
use zyron_common::{Result, ZyronError};

/// Version the checkpoint file is written at. The envelope in the header
/// carries it, and the format registry declares the same value.
const ZYIDX_FORMAT_VERSION: FormatVersion = crate::format::CHECKPOINT_FORMAT_VERSION;

/// Envelope header plus the checkpoint's own 20-byte header extension.
const ZYIDX_HEADER_SIZE: usize = 40;

/// The checkpoint's own header fields, which the envelope carries as its
/// extension and its header checksum covers.
///
/// ```text
/// [0..8)   checkpoint_lsn u64
/// [8..12)  entry_count u32
/// [12..14) key_len u16
/// [14..16) prefix_len u16
/// [16..18) value_width u16
/// [18..20) reserved
/// ```
fn checkpoint_extension(
    checkpoint_lsn: u64,
    entry_count: u32,
    key_len: u16,
    prefix_len: u16,
    value_width: u16,
) -> [u8; ZYIDX_HEADER_SIZE - ENVELOPE_HEADER_LEN] {
    let mut ext = [0u8; ZYIDX_HEADER_SIZE - ENVELOPE_HEADER_LEN];
    ext[0..8].copy_from_slice(&checkpoint_lsn.to_le_bytes());
    ext[8..12].copy_from_slice(&entry_count.to_le_bytes());
    ext[12..14].copy_from_slice(&key_len.to_le_bytes());
    ext[14..16].copy_from_slice(&prefix_len.to_le_bytes());
    ext[16..18].copy_from_slice(&value_width.to_le_bytes());
    ext
}

const SLOT_ARRAY_START: usize = PageHeader::SIZE + LeafPageHeader::SIZE;
const SLOT_SIZE: usize = 4;
const WIDE_VALUE_WIDTH: usize = zyron_common::RowLocator::MAX_PAYLOAD_LEN;

/// Copies one locator payload. The narrow form is three direct stores, which
/// avoids the memcpy call a runtime-sized copy would emit for seven bytes.
///
/// # Safety
/// `src` and `dst` must both be valid for `width` bytes and must not overlap
#[inline(always)]
unsafe fn copy_value(src: *const u8, dst: *mut u8, width: usize) {
    unsafe {
        if width == zyron_common::RowLocator::NARROW_PAYLOAD_LEN {
            (dst as *mut u32).write_unaligned((src as *const u32).read_unaligned());
            (dst.add(4) as *mut u16).write_unaligned((src.add(4) as *const u16).read_unaligned());
            dst.add(6).write(src.add(6).read());
        } else {
            std::ptr::copy_nonoverlapping(src, dst, width);
        }
    }
}

/// Copies one key suffix. Widths through eight resolve to direct stores, which
/// avoids the memcpy call a runtime-sized copy would emit per entry.
///
/// # Safety
/// `src` and `dst` must both be valid for `width` bytes and must not overlap
#[inline(always)]
unsafe fn copy_suffix(src: *const u8, dst: *mut u8, width: usize) {
    unsafe {
        match width {
            0 => {}
            1 => dst.write(src.read()),
            2 => (dst as *mut u16).write_unaligned((src as *const u16).read_unaligned()),
            3 => {
                (dst as *mut u16).write_unaligned((src as *const u16).read_unaligned());
                dst.add(2).write(src.add(2).read());
            }
            4 => (dst as *mut u32).write_unaligned((src as *const u32).read_unaligned()),
            // Two overlapping four byte moves, the second landing on the last
            // four bytes, which covers five through seven in two stores
            5..=7 => {
                (dst as *mut u32).write_unaligned((src as *const u32).read_unaligned());
                let tail = width - 4;
                (dst.add(tail) as *mut u32)
                    .write_unaligned((src.add(tail) as *const u32).read_unaligned());
            }
            8 => (dst as *mut u64).write_unaligned((src as *const u64).read_unaligned()),
            _ => std::ptr::copy_nonoverlapping(src, dst, width),
        }
    }
}

/// The quantities every leaf run reads, resolved once from the index.
struct GatherLayout {
    /// Key length every entry carries, which is where its locator payload
    /// starts
    kl: usize,
    /// Leading key bytes the whole index shares, stored once and skipped per
    /// entry
    prefix_len: usize,
    /// Key bytes per entry after the shared prefix
    suffix_len: usize,
    /// Stride of the value column
    value_width: usize,
}

/// Why a run of leaf pages stopped.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GatherStop {
    /// Every entry the run was given was copied
    Complete,
    /// An entry's locator payload is a different width than the column's, so
    /// the body is rebuilt with every value at the wide width
    MixedWidths,
    /// A leaf page or a locator payload could not be read
    Unreadable,
}

/// Copies a run of consecutive leaf pages into the checkpoint body.
///
/// `suffixes` and `values` are the two columns, each written front to back at
/// its own fixed stride.
fn gather_leaf_run(
    store: &InMemoryPageStore,
    leaves: &[(u32, u16)],
    layout: &GatherLayout,
    suffixes: &mut [u8],
    values: &mut [u8],
) -> GatherStop {
    let GatherLayout {
        kl,
        prefix_len,
        suffix_len,
        value_width,
    } = *layout;
    let mut sk = suffixes.as_mut_ptr();
    let mut vp = values.as_mut_ptr();
    for &(pn, ns) in leaves {
        let Some(pd) = store.get(pn) else {
            return GatherStop::Unreadable;
        };
        let pp = pd.as_ptr();
        for slot in 0..ns as usize {
            let slot_off = SLOT_ARRAY_START + slot * SLOT_SIZE;
            // SAFETY: the slot array starts at a fixed offset and holds one
            // two byte entry offset per slot, so slot < ns keeps the read
            // inside the page
            let entry_off = unsafe { (pp.add(slot_off) as *const u16).read_unaligned() as usize };
            let pid_offset = entry_off + 2 + kl;
            let entry_width = zyron_common::RowLocator::payload_len_for_tag(pd[pid_offset]);
            if entry_width != value_width && value_width != WIDE_VALUE_WIDTH {
                return GatherStop::MixedWidths;
            }
            // SAFETY: the entry holds key_len(2) + key + payload, so the
            // suffix and the payload are both inside the page, and the two
            // destinations advance by exactly the strides their spans were
            // sized from
            unsafe {
                copy_suffix(pp.add(entry_off + 2 + prefix_len), sk, suffix_len);
                sk = sk.add(suffix_len);
                if entry_width == value_width {
                    copy_value(pp.add(pid_offset), vp, value_width);
                } else {
                    // A narrow entry written into the wide column
                    let Some(loc) = zyron_common::RowLocator::read_payload(&pd[pid_offset..])
                    else {
                        return GatherStop::Unreadable;
                    };
                    loc.write_payload_wide(std::slice::from_raw_parts_mut(vp, WIDE_VALUE_WIDTH));
                }
                vp = vp.add(value_width);
            }
        }
    }
    GatherStop::Complete
}

/// Writes exactly `buf.len()` bytes at `offset` of an already-open file.
///
/// A positioned write rather than a seek and a write, because the writers below
/// run several of these at once and a shared file cursor would make them land on
/// each other's offsets. Windows has no `write_all_at`, so the short write loop
/// is written out.
#[cfg(windows)]
fn write_all_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut buf = buf;
    let mut offset = offset;
    while !buf.is_empty() {
        match file.seek_write(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "the file accepted none of the range",
                ));
            }
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn write_all_at(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(buf, offset)
}

/// How many writers to split a write of `len` bytes over.
///
/// One sequential write hands the file cache the whole body through a single
/// request, and the copy into it runs at a third of what the cache will take
/// from several at once. The gain is in that copy rather than at the device, so
/// this stays well under the core count.
///
/// Below `MIN_SPLIT_BYTES` a single request is already the faster answer and
/// splitting only buys thread starts, so the write stays on the calling thread.
fn write_thread_count(len: usize) -> usize {
    const MIN_SPLIT_BYTES: usize = 32 * 1024 * 1024;
    const MIN_BYTES_PER_WRITER: usize = 4 * 1024 * 1024;
    if len < MIN_SPLIT_BYTES {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (len / MIN_BYTES_PER_WRITER).min(cores).min(8)
}

/// Body bytes one run of leaf pages should cover.
///
/// A run is gathered before any of it is written, so it is the unit the copy
/// runs ahead of the transfer by. Small runs keep the writers fed from the
/// start, large ones keep each transfer sequential, and four megabytes is
/// enough to be a bulk write while leaving dozens of runs across a checkpoint
/// this size.
const BODY_BYTES_PER_RUN: usize = 4 * 1024 * 1024;

/// One run of leaf pages, the span of each column it fills, and where those
/// spans belong in the file.
struct BodyRun<'a> {
    leaves: &'a [(u32, u16)],
    suffixes: &'a mut [u8],
    values: &'a mut [u8],
    suffix_offset: u64,
    value_offset: u64,
}

/// Cuts both columns into one span per run of leaf pages.
///
/// Every entry contributes a fixed stride to each column, so a run's spans and
/// their file offsets follow from the entry counts alone, before a byte is
/// copied. Splitting up front is also what lets the gather hand a finished span
/// to another thread: the spans are disjoint by construction.
fn split_body_runs<'a>(
    leaf_pages: &'a [(u32, u16)],
    layout: &GatherLayout,
    mut suffixes: &'a mut [u8],
    mut values: &'a mut [u8],
    mut suffix_offset: u64,
    mut value_offset: u64,
    pages_per_run: usize,
) -> Vec<BodyRun<'a>> {
    let mut runs = Vec::with_capacity(leaf_pages.len().div_ceil(pages_per_run));
    for leaves in leaf_pages.chunks(pages_per_run) {
        let entries: usize = leaves.iter().map(|&(_, ns)| ns as usize).sum();
        let (s_head, s_rest) = suffixes.split_at_mut(entries * layout.suffix_len);
        suffixes = s_rest;
        let (v_head, v_rest) = values.split_at_mut(entries * layout.value_width);
        values = v_rest;
        runs.push(BodyRun {
            leaves,
            suffix_offset,
            value_offset,
            suffixes: s_head,
            values: v_head,
        });
        suffix_offset += (entries * layout.suffix_len) as u64;
        value_offset += (entries * layout.value_width) as u64;
    }
    runs
}

/// Copies the leaf entries into the two columns and writes each run to the file
/// as soon as it is finished, so the copy and the transfer overlap.
///
/// Both column cursors only move forward, so after a run of leaf pages what the
/// gather produced is a contiguous span of each column, at an offset the entry
/// counts already decided. That is what lets a writer take a span while the
/// gather moves on to the next run. The gather itself stays on this thread.
///
/// The file must already be sized, so no writer extends it. The header, the key
/// prefix and the footer are the caller's to write.
///
/// `hasher` must already hold the key prefix, which is what the body carries
/// ahead of the two columns. Each column span is folded in as it is finished,
/// the suffixes during the gather and the values once it is done, which is the
/// order the body has them in. The values are hashed while the writers are
/// still draining, so the checksum costs the caller nothing it was not already
/// waiting through.
#[allow(clippy::too_many_arguments)]
fn gather_and_write(
    store: &InMemoryPageStore,
    leaf_pages: &[(u32, u16)],
    layout: &GatherLayout,
    suffixes: &mut [u8],
    values: &mut [u8],
    suffix_offset: u64,
    value_offset: u64,
    file: &std::fs::File,
    hasher: &mut zyron_common::checksum::Hasher,
) -> std::io::Result<GatherStop> {
    let body_bytes = suffixes.len() + values.len();
    let writers = write_thread_count(body_bytes);
    // One run when nothing is going to overlap it, so a small checkpoint still
    // goes out in one transfer per column
    let run_count = if writers <= 1 {
        1
    } else {
        body_bytes.div_ceil(BODY_BYTES_PER_RUN).max(1)
    };
    let pages_per_run = leaf_pages.len().div_ceil(run_count).max(1);
    let runs = split_body_runs(
        leaf_pages,
        layout,
        suffixes,
        values,
        suffix_offset,
        value_offset,
        pages_per_run,
    );

    if writers <= 1 {
        let mut value_spans = Vec::with_capacity(runs.len());
        for run in runs {
            match gather_leaf_run(store, run.leaves, layout, run.suffixes, run.values) {
                GatherStop::Complete => {}
                other => return Ok(other),
            }
            let suffixes: &[u8] = run.suffixes;
            let values: &[u8] = run.values;
            hasher.update(suffixes);
            value_spans.push(values);
            write_all_at(file, suffixes, run.suffix_offset)?;
            write_all_at(file, values, run.value_offset)?;
        }
        for span in value_spans {
            hasher.update(span);
        }
        return Ok(GatherStop::Complete);
    }

    let (tx, rx) = std::sync::mpsc::channel::<(&[u8], u64)>();
    let rx = std::sync::Mutex::new(rx);
    let failure: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    let mut stop = GatherStop::Complete;

    std::thread::scope(|scope| {
        for _ in 0..writers {
            let rx = &rx;
            let failure = &failure;
            scope.spawn(move || {
                loop {
                    let next = {
                        let guard = rx.lock().unwrap_or_else(|e| e.into_inner());
                        guard.recv()
                    };
                    let Ok((span, offset)) = next else { break };
                    if let Err(e) = write_all_at(file, span, offset) {
                        let mut slot = failure.lock().unwrap_or_else(|e| e.into_inner());
                        if slot.is_none() {
                            *slot = Some(e);
                        }
                        break;
                    }
                }
            });
        }

        let mut value_spans = Vec::with_capacity(runs.len());
        for run in runs {
            stop = gather_leaf_run(store, run.leaves, layout, run.suffixes, run.values);
            if stop != GatherStop::Complete {
                break;
            }
            // The run is finished, so handing its spans over as shared borrows
            // cannot race the gather, which has moved past them
            let suffixes: &[u8] = run.suffixes;
            let values: &[u8] = run.values;
            hasher.update(suffixes);
            value_spans.push(values);
            let _ = tx.send((suffixes, run.suffix_offset));
            let _ = tx.send((values, run.value_offset));
        }
        // Closes the queue, which is how the writers learn there is no more
        drop(tx);

        // The writers still have most of the body to transfer, so folding the
        // value column in here costs nothing the scope was not already waiting
        // on. Skipped when the gather stopped early, since the body is about to
        // be built again at the wide stride
        if stop == GatherStop::Complete {
            for span in value_spans {
                hasher.update(span);
            }
        }
    });

    if let Some(e) = failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(e);
    }
    Ok(stop)
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

    let _total = profile::scope(Phase::CheckpointWriteTotal);
    let ho = LeafPageHeader::OFFSET;

    // Walk leaf chain: collect entry counts and determine key_len.
    let mut total_entries = 0u32;
    let mut key_len: u16 = 0;
    let mut value_width = WIDE_VALUE_WIDTH;
    let mut leaf_pages: Vec<(u32, u16)> = Vec::with_capacity(4096);

    {
        let _s = profile::scope(Phase::CheckpointScan);
        let mut cur = first_leaf;
        while let Some(pd) = store.get(cur) {
            let ns = u16::from_le_bytes([pd[ho], pd[ho + 1]]);
            if leaf_pages.is_empty() && ns > 0 {
                let e0_off =
                    u16::from_le_bytes([pd[SLOT_ARRAY_START], pd[SLOT_ARRAY_START + 1]]) as usize;
                key_len = u16::from_le_bytes([pd[e0_off], pd[e0_off + 1]]);
                // The whole column takes the first entry's width, and the
                // extraction below verifies every other entry against it
                value_width = zyron_common::RowLocator::payload_len_for_tag(
                    pd[e0_off + 2 + key_len as usize],
                );
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
    let prefix_start = ZYIDX_HEADER_SIZE;

    let io_err = |e: std::io::Error| {
        ZyronError::IoError(format!(
            "failed to write checkpoint file {}: {}",
            path.display(),
            e
        ))
    };

    let (mut buf, total_size, file, footer) = loop {
        let data_size = prefix_len + n * suffix_len + n * value_width;
        let total_size = ZYIDX_HEADER_SIZE + data_size + ENVELOPE_FOOTER_LEN;

        // Every byte of this buffer is written below: the envelope header and
        // its extension, the shared key prefix, both columns, and the footer.
        // A zeroed allocation fills the whole body with bytes the gather
        // overwrites immediately, which at ten million keys is ninety five
        // megabytes written twice
        let mut buf = {
            let _s = profile::scope(Phase::CheckpointAlloc);
            let mut buf: Vec<u8> = Vec::with_capacity(total_size);
            // SAFETY: total_size bytes were just reserved, and the body is
            // read only after a gather that reported every entry copied
            #[allow(clippy::uninit_vec)]
            unsafe {
                buf.set_len(total_size)
            };
            buf
        };

        // Envelope header, then the checkpoint's own header extension
        let extension = checkpoint_extension(
            checkpoint_lsn,
            total_entries,
            key_len,
            prefix_len as u16,
            value_width as u16,
        );
        let header =
            envelope::encode_header(FormatKind::Checkpoint, ZYIDX_FORMAT_VERSION, 0, &extension);
        buf[0..ENVELOPE_HEADER_LEN].copy_from_slice(&header);
        buf[ENVELOPE_HEADER_LEN..ZYIDX_HEADER_SIZE].copy_from_slice(&extension);
        // Key prefix
        if prefix_len > 0 {
            let (fp, _) = leaf_pages[0];
            let Some(fd) = store.get(fp) else {
                return Err(ZyronError::RecoveryFailed(
                    "the first leaf page left the store while the checkpoint was being written"
                        .into(),
                ));
            };
            let f_off =
                u16::from_le_bytes([fd[SLOT_ARRAY_START], fd[SLOT_ARRAY_START + 1]]) as usize;
            buf[prefix_start..prefix_start + prefix_len]
                .copy_from_slice(&fd[f_off + 2..f_off + 2 + prefix_len]);
        }

        // The two columns the entries are copied into. On disk a leaf entry is
        // key_len(2) + key + locator payload, and the checkpoint stores the key
        // past the shared prefix in one column and the raw locator payload in
        // the other
        let suffixes_start = prefix_start + prefix_len;
        let values_start = suffixes_start + n * suffix_len;

        // The envelope footer covers the body, which is the key prefix followed
        // by the two columns. The prefix goes in here and the columns are folded
        // in as the gather finishes them
        let mut hasher = zyron_common::checksum::Hasher::new();
        hasher.update(&buf[prefix_start..suffixes_start]);

        let columns = &mut buf[suffixes_start..total_size - ENVELOPE_FOOTER_LEN];
        let (suffixes, values) = columns.split_at_mut(n * suffix_len);
        let layout = GatherLayout {
            kl,
            prefix_len,
            suffix_len,
            value_width,
        };

        // Sized up front so no writer extends the file, and truncated by the
        // create so a retry at the wide stride leaves nothing of this pass
        let file = std::fs::File::create(path).map_err(io_err)?;
        file.set_len(total_size as u64).map_err(io_err)?;

        let stop = {
            let _s = profile::scope(Phase::CheckpointGather);
            gather_and_write(
                store,
                &leaf_pages,
                &layout,
                suffixes,
                values,
                suffixes_start as u64,
                values_start as u64,
                &file,
                &mut hasher,
            )
            .map_err(io_err)?
        };
        match stop {
            GatherStop::Complete => break (buf, total_size, file, hasher.finish32()),
            GatherStop::Unreadable => {
                return Err(ZyronError::RecoveryFailed(
                    "unreadable locator payload in index page".into(),
                ));
            }
            // This index mixes locator kinds, so the column cannot keep the
            // narrow stride and every value is written at the wide one
            GatherStop::MixedWidths => value_width = WIDE_VALUE_WIDTH,
        }
    };
    // The envelope footer checksum, folded in run by run alongside the gather.
    // The header carries its own checksum, stamped when it was encoded.
    let body_end = total_size - ENVELOPE_FOOTER_LEN;
    buf[body_end..total_size].copy_from_slice(&footer.to_le_bytes());

    // Everything the gather did not already put on disk: the envelope header
    // with its extension, the shared key prefix, and the footer behind them
    let _s = profile::scope(Phase::CheckpointWrite);
    let suffixes_start = prefix_start + prefix_len;
    write_all_at(&file, &buf[..suffixes_start], 0).map_err(io_err)?;
    write_all_at(&file, &buf[body_end..total_size], body_end as u64).map_err(io_err)?;
    if fsync {
        file.sync_all().map_err(io_err)?;
    }

    Ok(total_size as u64)
}

fn write_empty_checkpoint(path: &Path, checkpoint_lsn: u64, fsync: bool) -> Result<u64> {
    let mut buf = [0u8; ZYIDX_HEADER_SIZE + ENVELOPE_FOOTER_LEN];
    let extension = checkpoint_extension(checkpoint_lsn, 0, 0, 0, 0);
    let header =
        envelope::encode_header(FormatKind::Checkpoint, ZYIDX_FORMAT_VERSION, 0, &extension);
    buf[0..ENVELOPE_HEADER_LEN].copy_from_slice(&header);
    buf[ENVELOPE_HEADER_LEN..ZYIDX_HEADER_SIZE].copy_from_slice(&extension);
    let footer = zyron_common::hash32(&[]);
    buf[ZYIDX_HEADER_SIZE..].copy_from_slice(&footer.to_le_bytes());
    std::fs::write(path, buf)?;
    if fsync {
        std::fs::File::open(path)?.sync_all()?;
    }
    Ok(buf.len() as u64)
}

/// Reads exactly `buf.len()` bytes from `offset` of an already-open file.
///
/// A positioned read rather than a seek and a read, because the callers below
/// run several of these at once and a shared file cursor would make them
/// answer each other's offsets. Windows has no `read_exact_at`, so the short
/// read loop is written out.
#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut buf = buf;
    let mut offset = offset;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "the file ended before the range did",
                ));
            }
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

/// How many threads to split a read of `len` bytes over.
///
/// One sequential read keeps a single request outstanding, which is the depth
/// a solid state device is slowest at. Splitting the file into disjoint ranges
/// keeps several in flight. The gain is in the device queue rather than in the
/// CPU, so this stays well under the core count, and a file small enough that
/// one request covers it is read on the calling thread.
fn read_thread_count(len: usize) -> usize {
    const MIN_BYTES_PER_READER: usize = 2 * 1024 * 1024;
    if len < MIN_BYTES_PER_READER * 2 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (len / MIN_BYTES_PER_READER).min(cores).min(8)
}

/// Fills `dst` with the whole contents of `path`.
///
/// Each reader opens its own handle so the ranges cannot disturb one another's
/// file position, which costs one open per reader against a read that is
/// measured in milliseconds.
fn read_whole_file(path: &Path, dst: &mut [u8]) -> Result<()> {
    let io_err = |e: std::io::Error| {
        ZyronError::IoError(format!(
            "failed to read checkpoint file {}: {}",
            path.display(),
            e
        ))
    };
    let readers = read_thread_count(dst.len());
    if readers <= 1 {
        let file = std::fs::File::open(path).map_err(io_err)?;
        return read_exact_at(&file, dst, 0).map_err(io_err);
    }

    let chunk = dst.len().div_ceil(readers);
    let failure: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    std::thread::scope(|scope| {
        for (i, part) in dst.chunks_mut(chunk).enumerate() {
            let failure = &failure;
            scope.spawn(move || {
                let offset = (i * chunk) as u64;
                let outcome =
                    std::fs::File::open(path).and_then(|file| read_exact_at(&file, part, offset));
                if let Err(e) = outcome {
                    let mut slot = failure.lock().unwrap_or_else(|e| e.into_inner());
                    if slot.is_none() {
                        *slot = Some(e);
                    }
                }
            });
        }
    });
    match failure.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some(e) => Err(io_err(e)),
        None => Ok(()),
    }
}

/// Loads a checkpoint into `store`, returning the checkpoint LSN, the root
/// page number, the entry count and the tree height.
///
/// The file is read with positioned reads, its header and footer checksums are
/// verified, and the leaves and internal levels are rebuilt into the store. A
/// missing or corrupt file is the caller's to handle, this returns an error
/// rather than a partial tree
pub fn load_checkpoint_into_store(
    path: &Path,
    store: &mut InMemoryPageStore,
    file_id: u32,
) -> Result<(u64, u32, u32, u32)> {
    let _total = profile::scope(Phase::CheckpointLoadTotal);
    // Read into an uninit Vec, skipping the zero-fill pass that
    // vec![0u8; len] would otherwise force, the reads cover every byte
    // themselves
    let buf = {
        let file_len = std::fs::metadata(path)
            .map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to read checkpoint metadata {}: {}",
                    path.display(),
                    e
                ))
            })?
            .len() as usize;
        let mut buf: Vec<u8> = Vec::with_capacity(file_len);
        let spare = buf.spare_capacity_mut();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(spare.as_mut_ptr() as *mut u8, file_len) };
        {
            let _s = profile::scope(Phase::CheckpointRead);
            read_whole_file(path, slice)?;
        }
        unsafe {
            buf.set_len(file_len);
        }
        buf
    };

    if buf.len() < ZYIDX_HEADER_SIZE + ENVELOPE_FOOTER_LEN {
        return Err(ZyronError::RecoveryFailed(
            "checkpoint file too small".into(),
        ));
    }
    let (header, extension) = envelope::decode_header(&buf[..ZYIDX_HEADER_SIZE])
        .map_err(|e| ZyronError::RecoveryFailed(e.to_string()))?;
    if header.kind != FormatKind::Checkpoint {
        return Err(ZyronError::RecoveryFailed(format!(
            "expected an index checkpoint, found a {} file",
            header.kind
        )));
    }
    if header.header_length as usize != ZYIDX_HEADER_SIZE {
        return Err(ZyronError::RecoveryFailed(format!(
            "checkpoint header declares {} bytes, this binary writes {}",
            header.header_length, ZYIDX_HEADER_SIZE
        )));
    }
    if header.version != ZYIDX_FORMAT_VERSION {
        return Err(ZyronError::RecoveryFailed(format!(
            "checkpoint is at format version {}, this binary writes and reads {}. Upgrade \
             through a release that still reads {} to move the checkpoint forward first",
            header.version, ZYIDX_FORMAT_VERSION, header.version
        )));
    }

    let checkpoint_lsn = u64::from_le_bytes(extension[0..8].try_into().unwrap());
    let entry_count = u32::from_le_bytes(extension[8..12].try_into().unwrap());
    let key_len = u16::from_le_bytes([extension[12], extension[13]]);
    let prefix_len = u16::from_le_bytes([extension[14], extension[15]]) as usize;
    let value_width = u16::from_le_bytes([extension[16], extension[17]]) as usize;
    if entry_count > 0
        && value_width != zyron_common::RowLocator::NARROW_PAYLOAD_LEN
        && value_width != WIDE_VALUE_WIDTH
    {
        return Err(ZyronError::RecoveryFailed(format!(
            "unsupported checkpoint value width: {value_width}"
        )));
    }

    let kl = key_len as usize;
    let suffix_len = kl - prefix_len;
    let n = entry_count as usize;
    let data_size = prefix_len + n * suffix_len + n * value_width;
    let expected_size = ZYIDX_HEADER_SIZE + data_size + ENVELOPE_FOOTER_LEN;
    if buf.len() < expected_size {
        return Err(ZyronError::RecoveryFailed(
            "checkpoint file truncated".into(),
        ));
    }

    // Envelope footer checksum over the body. The header checksum was
    // verified when the envelope was decoded.
    let body_end = ZYIDX_HEADER_SIZE + data_size;
    let stored_checksum = u32::from_le_bytes(
        buf[body_end..body_end + ENVELOPE_FOOTER_LEN]
            .try_into()
            .map_err(|_| ZyronError::RecoveryFailed("checkpoint footer truncated".into()))?,
    );
    let computed = zyron_common::hash32(&buf[ZYIDX_HEADER_SIZE..body_end]);
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

    let prefix_start = ZYIDX_HEADER_SIZE;
    let suffixes_start = prefix_start + prefix_len;
    let values_start = suffixes_start + n * suffix_len;

    let eds = 2 + kl + value_width; // entry data size per entry: key_len(2) + key + locator payload
    let max_entries_per_page = (PAGE_SIZE - SLOT_ARRAY_START) / (eds + SLOT_SIZE);
    let num_leaves = n.div_ceil(max_entries_per_page);
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

    // Page header template, the PageHeader naming the leaf and its format
    // stamp followed by the leaf header, matching BTreeLeafPage::new
    // rebuild_leaf_run overwrites the page id fields for each page
    let mut hdr_tmpl = [0u8; SLOT_ARRAY_START];
    let page_header = PageHeader::new(
        PageId::new(file_id, 0),
        zyron_common::page::PageType::BTreeLeaf,
    );
    hdr_tmpl[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());
    let leaf_header = LeafPageHeader {
        num_slots: max_entries_per_page as u16,
        data_end: data_end_full as u16,
        next_leaf: u64::MAX,
        reserved: 0,
    };
    hdr_tmpl[ho_off()..ho_off() + LeafPageHeader::SIZE].copy_from_slice(&leaf_header.to_bytes());

    // One buffer per leaf, none of them zeroed. Every byte of every page is
    // either written by the rebuild or cleared by it, so the zero fill get_mut
    // does per page is a second pass over bytes about to be overwritten, and
    // at ten million keys that is two hundred megabytes written twice.
    let alloc = profile::scope(Phase::CheckpointAlloc);
    let first_page = store.bulk_allocate(num_leaves);
    // SAFETY: rebuild_leaf_run covers the header, the slot array, the entries
    // and the free space between them, which is all PAGE_SIZE bytes. The runs
    // below partition the pages, so no page is written by two threads
    let pages = unsafe { store.bulk_install_uninit(first_page, num_leaves) };

    let mut first_keys: Vec<u8> = vec![0u8; num_leaves * kl];
    drop(alloc);

    // Pre-build a full entry template: [key_len:2][prefix:prefix_len][...suffix...][...pid...][...sid...]
    // Stamp key_len and prefix once, then the inner loop only writes suffix + value fields.
    let mut entry_tmpl = vec![0u8; eds];
    entry_tmpl[0..2].copy_from_slice(&key_len.to_le_bytes());
    if prefix_len > 0 {
        entry_tmpl[2..2 + prefix_len]
            .copy_from_slice(&buf[prefix_start..prefix_start + prefix_len]);
    }

    let layout = LeafLayout {
        body: &buf,
        suffixes_start,
        values_start,
        suffix_len,
        value_width,
        key_len,
        kl,
        prefix_len,
        eds,
        max_entries_per_page,
        entries_total: n,
        num_leaves,
        file_id,
        first_page,
        hdr_tmpl,
        slot_array_full: &slot_array_full,
        entry_tmpl: &entry_tmpl,
    };

    let rebuild = profile::scope(Phase::CheckpointRebuildLeaves);
    let threads = rebuild_thread_count(num_leaves);
    if threads <= 1 {
        // SAFETY: one run covering every page exactly once
        unsafe { rebuild_leaf_run(&layout, &pages, 0, &mut first_keys) };
    } else {
        let per_run = num_leaves.div_ceil(threads);
        std::thread::scope(|scope| {
            let layout = &layout;
            for (run, (run_pages, run_keys)) in pages
                .chunks(per_run)
                .zip(first_keys.chunks_mut(per_run * kl))
                .enumerate()
            {
                scope.spawn(move || {
                    // SAFETY: chunks partition the pages, so this run owns
                    // every page it is given and shares none of them
                    unsafe { rebuild_leaf_run(layout, run_pages, run * per_run, run_keys) }
                });
            }
        });
    }

    drop(rebuild);

    let leaf_page_nums: Vec<u32> = (first_page..first_page + num_leaves as u32).collect();

    if leaf_page_nums.len() == 1 {
        return Ok((checkpoint_lsn, leaf_page_nums[0], entry_count, 1));
    }

    let _s = profile::scope(Phase::CheckpointBuildInternal);
    let (root, h) = build_internal_pages(store, &leaf_page_nums, &first_keys, kl, file_id);
    Ok((checkpoint_lsn, root, entry_count, h))
}

/// Everything the leaf rebuild needs that is the same for every page.
struct LeafLayout<'a> {
    body: &'a [u8],
    suffixes_start: usize,
    values_start: usize,
    suffix_len: usize,
    value_width: usize,
    key_len: u16,
    kl: usize,
    prefix_len: usize,
    eds: usize,
    max_entries_per_page: usize,
    entries_total: usize,
    num_leaves: usize,
    file_id: u32,
    first_page: u32,
    hdr_tmpl: [u8; SLOT_ARRAY_START],
    slot_array_full: &'a [u8],
    entry_tmpl: &'a [u8],
}

/// Threads to rebuild `num_leaves` pages with.
///
/// One until there is enough work to cover starting them, then half the
/// machine, the same split the column encoder uses. The rebuild is bound by
/// memory bandwidth rather than instructions, so the run length is what
/// decides whether another thread finishes the load sooner or just queues.
fn rebuild_thread_count(num_leaves: usize) -> usize {
    const MIN_LEAVES_PER_RUN: usize = 64;
    if num_leaves < MIN_LEAVES_PER_RUN * 2 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    (cores / 2).max(1).min(num_leaves / MIN_LEAVES_PER_RUN)
}

/// Rebuilds one run of consecutive leaf pages from the columnar body.
///
/// `leaf_start` is the index of the run's first leaf among all of them, which
/// is what decides where its entries begin in the body, what page number it
/// takes, and where its first key lands in `first_keys`. Every leaf but the
/// last is full, so that index alone locates the run and no run needs to know
/// what any other one did.
///
/// # Safety
/// Each page in `pages` must be an uninitialized buffer this run alone owns.
/// All PAGE_SIZE bytes of every one of them are written here.
unsafe fn rebuild_leaf_run(
    layout: &LeafLayout<'_>,
    pages: &[UninitPage],
    leaf_start: usize,
    first_keys: &mut [u8],
) {
    let src = layout.body.as_ptr();
    let kl = layout.kl;
    let eds = layout.eds;
    let key_len = layout.key_len;
    let value_width = layout.value_width;
    let suffix_len = layout.suffix_len;
    let max_entries_per_page = layout.max_entries_per_page;
    let suffix_in_entry = 2 + layout.prefix_len;
    let pid_in_entry = 2 + kl;
    let tmpl_ptr = layout.entry_tmpl.as_ptr();
    let tmpl_fixed = 2 + layout.prefix_len;

    for (i, page) in pages.iter().enumerate() {
        let leaf_idx = leaf_start + i;
        let ei = leaf_idx * max_entries_per_page;
        let ns = max_entries_per_page.min(layout.entries_total - ei);
        let pn = layout.first_page + leaf_idx as u32;
        let pp = page.as_ptr();

        // Write page header template + page_id fields.
        unsafe {
            std::ptr::copy_nonoverlapping(layout.hdr_tmpl.as_ptr(), pp, SLOT_ARRAY_START);
            // PageHeader: file_id(u32) at offset 0, page_num(u64) at offset 4
            (pp as *mut u32).write_unaligned(layout.file_id);
            (pp.add(4) as *mut u64).write_unaligned(pn as u64);
        }

        // Fix num_slots and data_end for partial last page.
        let data_end = PAGE_SIZE - ns * eds;
        if ns < max_entries_per_page {
            unsafe {
                (pp.add(ho_off()) as *mut u16).write_unaligned(ns as u16);
                (pp.add(ho_off() + 2) as *mut u16).write_unaligned(data_end as u16);
            }
        }

        // Write slot array (bulk copy for full pages, computed for partial).
        if ns == max_entries_per_page {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layout.slot_array_full.as_ptr(),
                    pp.add(SLOT_ARRAY_START),
                    ns * SLOT_SIZE,
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

        // Free space between the slot array and the first entry. The buffer
        // arrived uninitialized and may be a recycled page, so this is cleared
        // rather than left holding another page's bytes
        let slots_end = SLOT_ARRAY_START + ns * SLOT_SIZE;
        unsafe { std::ptr::write_bytes(pp.add(slots_end), 0, data_end - slots_end) };

        // Reconstruct entries from columnar checkpoint data into page layout.
        // Checkpoint stores the raw locator payload per entry.
        // Specialized for common key sizes to emit direct mov instructions
        // instead of memcpy calls from copy_nonoverlapping with runtime sizes.
        let mut entry_base = PAGE_SIZE - eds;
        let mut s_off = layout.suffixes_start + ei * suffix_len;
        let mut v_off = layout.values_start + ei * value_width;

        match (tmpl_fixed, suffix_len) {
            // u64 keys with no common prefix: direct u16 + u64 writes.
            (2, 8) => {
                let kl_le = key_len.to_le();
                for _ in 0..ns {
                    unsafe {
                        (pp.add(entry_base) as *mut u16).write_unaligned(kl_le);
                        let suf = (src.add(s_off) as *const u64).read_unaligned();
                        (pp.add(entry_base + 2) as *mut u64).write_unaligned(suf);
                        // Copy the raw locator payload directly.
                        copy_value(
                            src.add(v_off),
                            pp.add(entry_base + pid_in_entry),
                            value_width,
                        );
                    }
                    entry_base -= eds;
                    s_off += 8;
                    v_off += value_width;
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
                        copy_value(
                            src.add(v_off),
                            pp.add(entry_base + pid_in_entry),
                            value_width,
                        );
                    }
                    entry_base -= eds;
                    s_off += 4;
                    v_off += value_width;
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
                        copy_value(
                            src.add(v_off),
                            pp.add(entry_base + pid_in_entry),
                            value_width,
                        );
                    }
                    entry_base -= eds;
                    s_off += suffix_len;
                    v_off += value_width;
                }
            }
        }

        // Record first key for internal page construction.
        let feo = PAGE_SIZE - eds;
        unsafe {
            std::ptr::copy_nonoverlapping(pp.add(feo + 2), first_keys.as_mut_ptr().add(i * kl), kl);
        }

        // Set next-leaf pointer (stored as PageId.as_u64() = file_id << 32 | page_num).
        // The header template leaves it u64::MAX, which is what the last leaf
        // in the chain keeps
        if leaf_idx + 1 < layout.num_leaves {
            let next_packed = ((layout.file_id as u64) << 32) | (pn + 1) as u64;
            unsafe {
                (pp.add(ho_off() + 4) as *mut u64).write_unaligned(next_packed.to_le());
            }
        }
    }
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
        let es = if kl > 8 { 16 + kl } else { 16 };
        let mut ci: Option<(u32, BTreeInternalPage)> = None;
        let mut cu = 0usize;
        let mut first = true;
        for (i, &child_pn) in cp.iter().enumerate() {
            let ks = i * kl;
            if first {
                let pn = store.allocate();
                let pid = PageId::new(file_id, pn as u64);
                let mut int = BTreeInternalPage::new(pid, level);
                int.set_leftmost_child(PageId::new(file_id, child_pn as u64));
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
                int.set_leftmost_child(PageId::new(file_id, child_pn as u64));
                pp.push(pn);
                pk.extend_from_slice(&ck[ks..ks + kl]);
                ci = Some((pn, int));
                cu = 0;
                continue;
            }
            if let Some((_, ref mut int)) = ci {
                let key = bytes::Bytes::copy_from_slice(&ck[ks..ks + kl]);
                int.insert(key, PageId::new(file_id, child_pn as u64))
                    .expect("internal page overflow during checkpoint rebuild");
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
    /// Checks wal_bytes first to avoid the Instant::elapsed() syscall on the
    /// common path where no checkpoint is needed.
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
    use tempfile::tempdir;
    use zyron_common::RowLocator;

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
                RowLocator::Heap {
                    page: PageId::new(0, i % 10),
                    slot: (i % 5) as u16,
                },
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
            assert_eq!(
                f,
                Some(RowLocator::Heap {
                    page: PageId::new(0, i % 10),
                    slot: (i % 5) as u16,
                })
            );
        }
    }

    /// Locators of every kind, so a checkpoint cannot take the narrow stride.
    fn mixed_locator(i: u64) -> RowLocator {
        match i % 3 {
            0 => RowLocator::Heap {
                page: PageId::new(0, i),
                slot: (i % 7) as u16,
            },
            1 => RowLocator::Columnar {
                file_id: i * 3,
                sys_rowid: i * 11,
            },
            _ => RowLocator::Lake {
                file_id: i * 5,
                ordinal: i * 13,
            },
        }
    }

    fn round_trip(entries: usize, locator: impl Fn(u64) -> RowLocator) -> (usize, u64) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mixed.zyridx");
        let mut store = InMemoryPageStore::new();
        let root = store.allocate();
        let mut leaf = BTreeLeafPage::new(PageId::new(0, root as u64));
        let mut written = 0usize;
        for i in 0..entries as u64 {
            if leaf
                .insert(bytes::Bytes::copy_from_slice(&i.to_be_bytes()), locator(i))
                .is_err()
            {
                break;
            }
            written += 1;
        }
        store.write(root, leaf.as_bytes());
        let size = write_checkpoint_from_store(&path, &store, 7, root, 1, false).unwrap();

        let mut ls = InMemoryPageStore::new();
        let (lsn, lr, count, _) = load_checkpoint_into_store(&path, &mut ls, 0).unwrap();
        assert_eq!(lsn, 7);
        assert_eq!(count as usize, written);
        for i in 0..written as u64 {
            let found = BTreeLeafPage::get_in_slice(ls.get(lr).unwrap(), &i.to_be_bytes());
            assert_eq!(found, Some(locator(i)), "entry {i} did not survive");
        }
        (written, size)
    }

    /// The rebuild runs a loop specialized per key width and a generic
    /// fallback, and which one a load takes depends on how much prefix the
    /// keys share. Ascending small integers share their top five bytes and so
    /// always take the fallback, which left the eight byte path covered only
    /// by single-page loads. Keys spread across the whole range share nothing,
    /// and enough of them to fill many leaves puts that path on the parallel
    /// runs as well. Columnar locators widen the value column at the same
    /// time, so the wide stride is covered here rather than only at 300 rows.
    #[test]
    fn test_checkpoint_round_trips_unprefixed_keys_across_runs() {
        let dir = tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let ckpt_dir = dir.path().join("ckpt");
            std::fs::create_dir_all(&ckpt_dir).unwrap();

            let mut btree = crate::btree::index::BTreeIndex::create_with_config(
                0,
                ckpt_dir.clone(),
                CheckpointConfig {
                    fsync: false,
                    ..CheckpointConfig::default()
                },
            )
            .await
            .unwrap();

            // A golden-ratio stride lands keys across the whole u64 range, so
            // the lowest and the highest differ in their first byte and the
            // writer finds no common prefix
            let mut keys: Vec<u64> = (0..200_000u64)
                .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .collect();
            keys.sort_unstable();
            keys.dedup();
            assert_ne!(
                keys[0].to_be_bytes()[0],
                keys[keys.len() - 1].to_be_bytes()[0],
                "wanted keys with no shared prefix"
            );

            let locator = |k: u64| RowLocator::Columnar {
                file_id: k % 97,
                sys_rowid: k,
            };
            for k in &keys {
                btree
                    .insert_exclusive(&k.to_be_bytes(), locator(*k))
                    .unwrap();
            }
            btree.force_checkpoint(11).unwrap();

            let loaded = crate::btree::index::BTreeIndex::open(0, &ckpt_dir)
                .await
                .unwrap();
            for k in &keys {
                assert_eq!(
                    loaded.search_sync(&k.to_be_bytes()),
                    Some(locator(*k)),
                    "key {k:#x} did not survive the rebuild"
                );
            }
        });
    }

    #[test]
    fn checkpoint_round_trips_mixed_locator_kinds() {
        // Forces the writer to abandon the narrow column and rewrite wide
        let (written, mixed_size) = round_trip(300, mixed_locator);
        assert!(written >= 300);

        // The same entry count addressing only heap rows takes the narrow
        // column, so the file is materially smaller
        let (_, heap_size) = round_trip(300, |i| RowLocator::Heap {
            page: PageId::new(0, i),
            slot: (i % 7) as u16,
        });
        assert!(
            heap_size < mixed_size,
            "heap-only checkpoint {heap_size} should be smaller than mixed {mixed_size}"
        );
    }

    /// A body large enough to be written by several writers at once, each
    /// taking a disjoint range of it. Every smaller case in this file is written
    /// by one, so the range arithmetic and the uneven last chunk are covered
    /// only here.
    #[test]
    fn checkpoint_round_trips_a_body_written_by_several_writers() {
        let dir = tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let ckpt_dir = dir.path().join("ckpt");
            std::fs::create_dir_all(&ckpt_dir).unwrap();

            let mut btree = crate::btree::index::BTreeIndex::create_with_config(
                0,
                ckpt_dir.clone(),
                CheckpointConfig {
                    fsync: false,
                    ..CheckpointConfig::default()
                },
            )
            .await
            .unwrap();

            // Columnar locators take the wide value column, so each entry costs
            // enough that this many of them carry the body past the point where
            // the write splits
            let n = 1_800_000u64;
            let locator = |i: u64| RowLocator::Columnar {
                file_id: i % 97,
                sys_rowid: i * 7,
            };
            for i in 0..n {
                btree
                    .insert_exclusive(&i.to_be_bytes(), locator(i))
                    .unwrap();
            }
            btree.force_checkpoint(17).unwrap();

            let path = ckpt_dir.join("index_0.zyridx");
            let size = std::fs::metadata(&path).unwrap().len() as usize;
            let writers = write_thread_count(size);
            assert!(writers > 1, "{size} byte checkpoint went out in one write");
            assert!(
                !size.is_multiple_of(writers),
                "{size} bytes divides evenly over {writers} writers, so the short last chunk is untested"
            );

            let loaded = crate::btree::index::BTreeIndex::open(0, &ckpt_dir)
                .await
                .unwrap();
            assert_eq!(loaded.checkpoint_lsn(), 17);
            for i in 0..n {
                assert_eq!(
                    loaded.search_sync(&i.to_be_bytes()),
                    Some(locator(i)),
                    "entry {i} did not survive the split write"
                );
            }
        });
    }

    /// The narrow value column is abandoned the moment an entry carries a
    /// locator of another width, and the whole body is gathered again at the
    /// wide stride. Widening changes the body length, so the second pass sizes
    /// a new buffer and rewrites both columns. The three hundred entry case
    /// above fits on one leaf page, where nothing about that resizing shows.
    #[test]
    fn checkpoint_rewrites_wide_across_pages_when_widths_are_mixed() {
        let dir = tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let ckpt_dir = dir.path().join("ckpt");
            std::fs::create_dir_all(&ckpt_dir).unwrap();

            let mut btree = crate::btree::index::BTreeIndex::create_with_config(
                0,
                ckpt_dir.clone(),
                CheckpointConfig {
                    fsync: false,
                    ..CheckpointConfig::default()
                },
            )
            .await
            .unwrap();

            // Heap, columnar and lake locators in turn, so the first entry
            // sizes the column narrow and the second contradicts it
            let n = 50_000u64;
            for i in 0..n {
                btree
                    .insert_exclusive(&i.to_be_bytes(), mixed_locator(i))
                    .unwrap();
            }
            btree.force_checkpoint(13).unwrap();

            let path = ckpt_dir.join("index_0.zyridx");
            let mut store = InMemoryPageStore::new();
            let (lsn, _, count, _) = load_checkpoint_into_store(&path, &mut store, 0).unwrap();
            assert_eq!(lsn, 13);
            assert_eq!(count as u64, n);

            let leaf_type = zyron_common::page::PageType::BTreeLeaf as u8;
            let mut leaves = 0usize;
            let mut page_num = 0u32;
            while let Some(page) = store.get(page_num) {
                if page[20] == leaf_type {
                    leaves += 1;
                }
                page_num += 1;
            }
            assert!(
                leaves > 1,
                "{leaves} leaves, the rewrite stayed on one page"
            );

            let loaded = crate::btree::index::BTreeIndex::open(0, &ckpt_dir)
                .await
                .unwrap();
            for i in 0..n {
                assert_eq!(
                    loaded.search_sync(&i.to_be_bytes()),
                    Some(mixed_locator(i)),
                    "entry {i} did not survive the wide rewrite"
                );
            }
        });
    }

    #[test]
    fn checkpoint_round_trips_heap_pages_beyond_32_bits() {
        // A page number past u32 cannot take the narrow form, so these entries
        // must still survive in the wide one
        round_trip(64, |i| RowLocator::Heap {
            page: PageId::new(0, u32::MAX as u64 + 1 + i),
            slot: (i % 7) as u16,
        });
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
                RowLocator::Heap {
                    page: PageId::new(0, 0),
                    slot: 0,
                },
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

    /// The rebuild is handed uninitialized buffers, and a recycled one still
    /// holds the bytes of whatever page had it last. Everything outside the
    /// header, the slot array and the entries has to be cleared, or a leaf
    /// carries data that was never part of this index and two loads of one
    /// checkpoint disagree byte for byte.
    ///
    /// The key count is past the point where the rebuild splits across
    /// threads, so this covers the parallel path as well as the free space.
    #[test]
    fn test_checkpoint_load_leaves_no_uninitialized_bytes() {
        let dir = tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let ckpt_dir = dir.path().join("ckpt");
            std::fs::create_dir_all(&ckpt_dir).unwrap();

            let mut btree = crate::btree::index::BTreeIndex::create_with_config(
                0,
                ckpt_dir.clone(),
                CheckpointConfig {
                    fsync: false,
                    ..CheckpointConfig::default()
                },
            )
            .await
            .unwrap();

            // Enough leaves that rebuild_thread_count asks for more than one
            // run, and a partial last page to cover the trailing free space
            let n = 200_003u64;
            for i in 0..n {
                let key = i.to_be_bytes();
                let tid = RowLocator::Heap {
                    page: PageId::new(0, i % 1000),
                    slot: (i % 100) as u16,
                };
                btree.insert_exclusive(&key, tid).unwrap();
            }
            btree.force_checkpoint(7).unwrap();

            let path = ckpt_dir.join("index_0.zyridx");
            assert!(
                rebuild_thread_count(200_003 / 387) > 1,
                "wanted the parallel path"
            );

            let mut first = InMemoryPageStore::new();
            let (lsn, _, count, _) = load_checkpoint_into_store(&path, &mut first, 0).unwrap();
            assert_eq!(lsn, 7);
            assert_eq!(count as u64, n);

            // A second load into its own store. The buffers behind it are
            // whatever the allocator hands back, so any byte the rebuild
            // leaves alone shows up as a difference here
            let mut second = InMemoryPageStore::new();
            load_checkpoint_into_store(&path, &mut second, 0).unwrap();

            let leaf_type = zyron_common::page::PageType::BTreeLeaf as u8;
            let mut leaves_checked = 0usize;
            let mut page_num = 0u32;
            while let Some(page) = first.get(page_num) {
                assert_eq!(
                    page.as_slice(),
                    second.get(page_num).unwrap().as_slice(),
                    "page {page_num} differs between two loads of one checkpoint"
                );
                if page[20] == leaf_type {
                    let ns = u16::from_le_bytes([page[ho_off()], page[ho_off() + 1]]) as usize;
                    let data_end =
                        u16::from_le_bytes([page[ho_off() + 2], page[ho_off() + 3]]) as usize;
                    let slots_end = SLOT_ARRAY_START + ns * SLOT_SIZE;
                    assert!(
                        page[slots_end..data_end].iter().all(|b| *b == 0),
                        "leaf {page_num} carries {} unwritten bytes of free space",
                        data_end - slots_end
                    );
                    leaves_checked += 1;
                }
                page_num += 1;
            }
            assert!(leaves_checked > 128, "only {leaves_checked} leaves walked");
        });
    }

    #[test]
    fn test_checkpoint_multi_page_round_trip() {
        let dir = tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let ckpt_dir = dir.path().join("ckpt");
            std::fs::create_dir_all(&ckpt_dir).unwrap();

            let mut btree = crate::btree::index::BTreeIndex::create_with_config(
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
                let tid = RowLocator::Heap {
                    page: PageId::new(0, i % 1000),
                    slot: (i % 100) as u16,
                };
                btree.insert_exclusive(&key, tid).unwrap();
            }

            btree.force_checkpoint(42).unwrap();

            let loaded = crate::btree::index::BTreeIndex::open(0, &ckpt_dir)
                .await
                .unwrap();

            // Verify all keys survived checkpoint round-trip
            let mut first_missing = None;
            let mut missing_count = 0;
            for i in 0..n {
                let key = i.to_be_bytes();
                let expected = RowLocator::Heap {
                    page: PageId::new(0, i % 1000),
                    slot: (i % 100) as u16,
                };
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
