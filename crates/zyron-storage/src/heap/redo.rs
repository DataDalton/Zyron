//! Physical redo of heap pages.
//!
//! Every change a heap makes to a page is logged as the exact bytes it wrote
//! and where, from inside the page's frame lock, before the lock is released.
//! A flush copies the page under the exclusive lock and stamps the copy with
//! the newest logged change it holds, so the on-disk page says which records
//! it already reflects. Recovery reads every page record the log holds and
//! replays each one whose LSN is above what the page carried on disk when
//! recovery began, onto the page image, in log order.
//!
//! Records are replayed whatever their transaction did. A page image is put
//! back exactly as it was, and a row an uncommitted transaction wrote stays
//! invisible through that transaction's status, the same way it did before
//! the crash. Appends carry their positions, so two bursts on one page that
//! logged in either order replay to the same image, and the comparison is
//! against the LSN the page carried on disk rather than the running one, so
//! neither is skipped for having been logged after the other.
//!
//! A record naming a page whose file is gone belonged to a table dropped
//! after the record was written and is passed over. A record that does not
//! fit the page it names is refused, never applied around

use std::collections::HashMap;
use std::sync::Arc;

use zyron_buffer::BufferPool;
use zyron_common::page::{PAGE_SIZE, PageId, PageType, page_lsn};
use zyron_common::{Result, ZyronError};
use zyron_wal::record::{LogRecord, LogRecordType, Lsn};
use zyron_wal::writer::WalWriter;

use crate::disk::DiskManager;
use crate::heap::page::{HeapPage, PageVacuum, SlotId};
use crate::tuple::{Tuple, TupleFlags, TupleHeader};

/// Bytes the page id takes at the head of every record
const PAGE_ID_LEN: usize = 12;

/// Bytes a row's header takes in a record, flags, epoch, xmin, xmax, length
const ROW_HEAD_LEN: usize = 2 + 2 + 8 + 8 + 2;

/// One change to one heap page, as the bytes it wrote and where
#[derive(Debug, Clone)]
pub enum PageChange {
    /// Rows appended by one burst, from `first_slot` on, their bytes written
    /// downward from `data_end`
    Append {
        page_id: PageId,
        first_slot: u16,
        data_end: u16,
        tuples: Vec<Tuple>,
    },
    /// Transaction stamps set on rows in place, zero clearing one
    Xmax {
        page_id: PageId,
        stamps: Vec<(u16, u64)>,
    },
    /// Slots emptied and the page compacted
    Prune { page_id: PageId, slots: Vec<u16> },
    /// Slots emptied in place, their bytes left for a later compaction
    Free { page_id: PageId, slots: Vec<u16> },
    /// A row rewritten in place within the footprint it had
    Update {
        page_id: PageId,
        slot: u16,
        tuple: Tuple,
    },
    /// The whole file emptied, a fence in the log. Everything the records
    /// before it put on the file's pages goes with the file's contents, and
    /// the records after it land on an empty file
    Truncate { file_id: u32 },
}

impl PageChange {
    /// The page this change lands on. A truncate names the file's first
    /// page, the file being what it changes
    pub fn page_id(&self) -> PageId {
        match self {
            PageChange::Append { page_id, .. }
            | PageChange::Xmax { page_id, .. }
            | PageChange::Prune { page_id, .. }
            | PageChange::Free { page_id, .. }
            | PageChange::Update { page_id, .. } => *page_id,
            PageChange::Truncate { file_id } => PageId::new(*file_id, 0),
        }
    }

    /// The record type the change is logged as
    pub fn record_type(&self) -> LogRecordType {
        match self {
            PageChange::Append { .. } => LogRecordType::HeapAppend,
            PageChange::Xmax { .. } => LogRecordType::HeapXmax,
            PageChange::Prune { .. } => LogRecordType::HeapPrune,
            PageChange::Free { .. } => LogRecordType::HeapFree,
            PageChange::Update { .. } => LogRecordType::HeapUpdate,
            PageChange::Truncate { .. } => LogRecordType::HeapTruncate,
        }
    }

    /// The record payload
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.encoded_len());
        match self {
            PageChange::Append {
                page_id,
                first_slot,
                data_end,
                tuples,
            } => {
                return encode_append(*page_id, *first_slot, *data_end, tuples);
            }
            _ => put_page_id(&mut buf, self.page_id()),
        }
        match self {
            PageChange::Append { .. } | PageChange::Truncate { .. } => {}
            PageChange::Xmax { stamps, .. } => {
                buf.extend_from_slice(&(stamps.len() as u16).to_le_bytes());
                for (slot, xmax) in stamps {
                    buf.extend_from_slice(&slot.to_le_bytes());
                    buf.extend_from_slice(&xmax.to_le_bytes());
                }
            }
            PageChange::Prune { slots, .. } | PageChange::Free { slots, .. } => {
                buf.extend_from_slice(&(slots.len() as u16).to_le_bytes());
                for slot in slots {
                    buf.extend_from_slice(&slot.to_le_bytes());
                }
            }
            PageChange::Update { slot, tuple, .. } => {
                buf.extend_from_slice(&slot.to_le_bytes());
                put_row(&mut buf, tuple);
            }
        }
        buf
    }

    fn encoded_len(&self) -> usize {
        PAGE_ID_LEN
            + match self {
                PageChange::Append { tuples, .. } => {
                    6 + tuples
                        .iter()
                        .map(|t| ROW_HEAD_LEN + t.data().len())
                        .sum::<usize>()
                }
                PageChange::Xmax { stamps, .. } => 2 + stamps.len() * 10,
                PageChange::Prune { slots, .. } | PageChange::Free { slots, .. } => {
                    2 + slots.len() * 2
                }
                PageChange::Update { tuple, .. } => 2 + ROW_HEAD_LEN + tuple.data().len(),
                PageChange::Truncate { .. } => 0,
            }
    }

    /// Reads a change back from a record's type and payload
    pub fn decode(record_type: LogRecordType, payload: &[u8], lsn: Lsn) -> Result<Self> {
        let mut reader = Reader {
            data: payload,
            at: 0,
            lsn,
        };
        let page_id = reader.page_id()?;
        let change = match record_type {
            LogRecordType::HeapAppend => {
                let first_slot = reader.u16()?;
                let data_end = reader.u16()?;
                let count = reader.u16()? as usize;
                let mut tuples = Vec::with_capacity(count);
                for _ in 0..count {
                    tuples.push(reader.row()?);
                }
                PageChange::Append {
                    page_id,
                    first_slot,
                    data_end,
                    tuples,
                }
            }
            LogRecordType::HeapXmax => {
                let count = reader.u16()? as usize;
                let mut stamps = Vec::with_capacity(count);
                for _ in 0..count {
                    let slot = reader.u16()?;
                    let xmax = reader.u64()?;
                    stamps.push((slot, xmax));
                }
                PageChange::Xmax { page_id, stamps }
            }
            LogRecordType::HeapPrune | LogRecordType::HeapFree => {
                let count = reader.u16()? as usize;
                let mut slots = Vec::with_capacity(count);
                for _ in 0..count {
                    slots.push(reader.u16()?);
                }
                if record_type == LogRecordType::HeapPrune {
                    PageChange::Prune { page_id, slots }
                } else {
                    PageChange::Free { page_id, slots }
                }
            }
            LogRecordType::HeapUpdate => {
                let slot = reader.u16()?;
                let tuple = reader.row()?;
                PageChange::Update {
                    page_id,
                    slot,
                    tuple,
                }
            }
            LogRecordType::HeapTruncate => PageChange::Truncate {
                file_id: page_id.file_id,
            },
            other => {
                return Err(ZyronError::WalCorrupted {
                    lsn: lsn.0,
                    reason: format!("record type {other:?} is not a heap page change"),
                });
            }
        };
        if reader.at != payload.len() {
            return Err(ZyronError::WalCorrupted {
                lsn: lsn.0,
                reason: format!(
                    "a {record_type:?} record carries {} bytes past its change",
                    payload.len() - reader.at
                ),
            });
        }
        Ok(change)
    }

    /// Applies the change to a page image.
    ///
    /// A change that names a slot the page does not hold, or rows that do
    /// not fit it, is refused. The image is not the one the change was made
    /// on, and applying it around would leave a page neither the log nor
    /// the disk describes
    pub fn apply(&self, data: &mut [u8; PAGE_SIZE], lsn: Lsn) -> Result<()> {
        let refused = |what: String| ZyronError::WalCorrupted {
            lsn: lsn.0,
            reason: what,
        };
        match self {
            PageChange::Append {
                first_slot,
                data_end,
                tuples,
                ..
            } => HeapPage::replay_burst_in_slice(&mut data[..], *first_slot, *data_end, tuples)
                .map_err(|e| refused(e.to_string())),
            PageChange::Xmax { stamps, .. } => {
                for &(slot, xmax) in stamps {
                    let stamped = if xmax == 0 {
                        HeapPage::clear_tuple_xmax_in_slice(&mut data[..], SlotId(slot))
                    } else {
                        HeapPage::set_tuple_xmax_in_slice(&mut data[..], SlotId(slot), xmax)
                    };
                    if !stamped {
                        return Err(refused(format!(
                            "a stamp names slot {slot}, which the page does not hold"
                        )));
                    }
                }
                Ok(())
            }
            PageChange::Prune { slots, .. } => {
                let held = HeapPage::heap_header_from_slice(&data[..]).slot_count;
                if let Some(past) = slots.iter().find(|&&slot| slot >= held) {
                    return Err(refused(format!(
                        "a prune names slot {past}, past the {held} slots the page holds"
                    )));
                }
                HeapPage::prune_slots_in_slice(&mut data[..], slots);
                Ok(())
            }
            PageChange::Free { slots, .. } => {
                for &slot in slots {
                    if !HeapPage::delete_tuple_in_slice(&mut data[..], SlotId(slot)) {
                        return Err(refused(format!(
                            "a free names slot {slot}, which the page does not hold"
                        )));
                    }
                }
                Ok(())
            }
            PageChange::Update { slot, tuple, .. } => {
                HeapPage::update_tuple_in_slice(&mut data[..], SlotId(*slot), tuple)
                    .map_err(|e| refused(e.to_string()))
            }
            PageChange::Truncate { file_id } => Err(refused(format!(
                "a truncate of file {file_id} changes the file, not a page image"
            ))),
        }
    }
}

/// The payload of an append record over rows the caller still holds, so
/// the append path logs a burst without copying its rows into a change
pub fn encode_append(page_id: PageId, first_slot: u16, data_end: u16, tuples: &[Tuple]) -> Vec<u8> {
    let rows: usize = tuples.iter().map(|t| ROW_HEAD_LEN + t.data().len()).sum();
    let mut buf = Vec::with_capacity(PAGE_ID_LEN + 6 + rows);
    put_page_id(&mut buf, page_id);
    buf.extend_from_slice(&first_slot.to_le_bytes());
    buf.extend_from_slice(&data_end.to_le_bytes());
    buf.extend_from_slice(&(tuples.len() as u16).to_le_bytes());
    for tuple in tuples {
        put_row(&mut buf, tuple);
    }
    buf
}

/// Logs one page change under `txn_id` and returns the record's position,
/// the LSN the page is stamped with
pub fn log_page_change(wal: &WalWriter, txn_id: u64, change: &PageChange) -> Result<Lsn> {
    wal.log_page_change(txn_id, change.record_type(), &change.encode())
}

/// Logs the fence that empties a heap file, ahead of the file being
/// emptied. The caller waits for the record to be durable before it
/// truncates, so recovery empties the file at the same point in the log
/// whichever side of the truncation the crash fell on
pub fn log_truncate(wal: &WalWriter, file_id: u32) -> Result<Lsn> {
    log_page_change(wal, 0, &PageChange::Truncate { file_id })
}

/// Logs one burst append from the rows the caller holds, under `txn_id`
pub fn log_append(
    wal: &WalWriter,
    txn_id: u64,
    page_id: PageId,
    first_slot: u16,
    data_end: u16,
    tuples: &[Tuple],
) -> Result<Lsn> {
    wal.log_page_change(
        txn_id,
        LogRecordType::HeapAppend,
        &encode_append(page_id, first_slot, data_end, tuples),
    )
}

/// Logs what one vacuum pass changed on a page, the stamps it cleared as one
/// record and the slots it pruned as another, so the two replay in the order
/// the pass made them, and stamps the page with each record as it is
/// written. Called under the page's frame guard, so no copy of the page
/// carries the pass without the stamps that name its records, and the
/// oldest unflushed change the page holds is the first record rather than
/// the last. Returns the newest position written, None when the pass
/// changed nothing
pub fn log_vacuum(
    wal: &WalWriter,
    pool: &BufferPool,
    page_id: PageId,
    changes: &PageVacuum,
) -> Result<Option<Lsn>> {
    let mut newest = None;
    if !changes.cleared.is_empty() {
        let stamps = changes.cleared.iter().map(|&slot| (slot, 0)).collect();
        let lsn = log_page_change(wal, 0, &PageChange::Xmax { page_id, stamps })?;
        pool.mark_dirty_with_lsn(page_id, lsn.0);
        newest = Some(lsn);
    }
    if !changes.pruned.is_empty() {
        let lsn = log_page_change(
            wal,
            0,
            &PageChange::Prune {
                page_id,
                slots: changes.pruned.clone(),
            },
        )?;
        pool.mark_dirty_with_lsn(page_id, lsn.0);
        newest = Some(lsn);
    }
    Ok(newest)
}

/// What one replay did
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RedoStats {
    /// Records applied to a page image
    pub applied: u64,
    /// Records the page already reflected
    pub already_held: u64,
    /// Records naming a file that no longer exists
    pub dropped: u64,
    /// Records that could not be applied and were passed over because the
    /// operator asked for that, each reported as it was skipped
    pub skipped: u64,
}

/// What replay does with a page record it cannot apply
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BadPageRecord {
    /// Replay stops and the error is the startup's, so nothing runs over a
    /// page that is not what the log says
    #[default]
    Stop,
    /// The record is reported and passed over, an operator's way past a
    /// record that would otherwise keep the node from starting, taken
    /// knowing the page it names holds what it held before
    SkipAndReport,
}

/// Replays page records onto their pages, in log order.
///
/// A page is read once from disk on its first record and the LSN it carried
/// then is what every one of its records is compared against, so a burst
/// logged after another burst on the same page is not skipped once the
/// first one has been applied and raised the page. The pages stay dirty in
/// the pool and reach disk through the ordinary flush, stamped with the
/// newest record applied, so a crash before that flush replays the same
/// records again to the same effect
pub async fn apply_page_records(
    disk: &Arc<DiskManager>,
    pool: &Arc<BufferPool>,
    records: &[LogRecord],
    on_bad: BadPageRecord,
) -> Result<RedoStats> {
    let mut stats = RedoStats::default();
    // The LSN each page carried on disk when recovery first read it
    let mut on_disk: HashMap<PageId, u64> = HashMap::new();
    // Files found missing, so a dropped table costs one probe rather than
    // one per record
    let mut missing: HashMap<u32, ()> = HashMap::new();
    for record in records {
        let change = match PageChange::decode(record.record_type, &record.payload, record.lsn) {
            Ok(change) => change,
            Err(e) if on_bad == BadPageRecord::SkipAndReport => {
                tracing::error!(
                    target: "zyron::recovery",
                    lsn = record.lsn.0,
                    record_type = ?record.record_type,
                    error = %e,
                    "a page change record could not be read and was passed over"
                );
                stats.skipped += 1;
                continue;
            }
            Err(e) => return Err(e),
        };
        let page_id = change.page_id();
        if missing.contains_key(&page_id.file_id) {
            stats.dropped += 1;
            continue;
        }
        if !on_disk.contains_key(&page_id) && !disk.file_exists(page_id.file_id) {
            missing.insert(page_id.file_id, ());
            stats.dropped += 1;
            continue;
        }
        if let PageChange::Truncate { file_id } = change {
            // The fence. The pages the earlier records put back are
            // dropped from the pool before the file empties, so no flush
            // can grow it back, and the file's pages are read afresh by
            // the records that follow
            pool.drop_file_pages(file_id);
            disk.truncate_file(file_id).await?;
            on_disk.retain(|page, _| page.file_id != file_id);
            stats.applied += 1;
            continue;
        }
        let (frame, seen) = match pool.fetch_page(page_id) {
            Some(frame) => (frame, on_disk.contains_key(&page_id)),
            None => {
                let image = read_or_blank(disk, page_id).await?;
                (pool.load_page(page_id, &image)?, false)
            }
        };
        if !seen {
            // A page already resident carries its newest change in the
            // frame, which is above whatever its image says once anything
            // has been applied to it, so a replay run twice in one process
            // finds its own work already held rather than doubling it
            let carried = frame.page_lsn();
            on_disk.insert(page_id, carried);
        }
        let carried = on_disk.get(&page_id).copied().ok_or_else(|| {
            ZyronError::Internal("a page's disk position was not recorded".into())
        })?;
        if record.lsn.0 <= carried {
            pool.unpin_page(page_id, false);
            stats.already_held += 1;
            continue;
        }
        let outcome = {
            let mut guard = frame.write_data();
            // A page the log appends to for the first time is blank on disk,
            // and takes a heap page's shape before the rows land
            if matches!(change, PageChange::Append { .. })
                && zyron_common::page::PageHeader::from_bytes(&guard[..]).page_type
                    != PageType::Heap
                && page_lsn(&guard[..]) == 0
            {
                HeapPage::init_fresh_slice_reuse(&mut guard, page_id);
            }
            change.apply(&mut guard, record.lsn)
        };
        match outcome {
            Ok(()) => {
                pool.mark_dirty_with_lsn(page_id, record.lsn.0);
                pool.unpin_page(page_id, true);
                stats.applied += 1;
            }
            Err(e) if on_bad == BadPageRecord::SkipAndReport => {
                pool.unpin_page(page_id, false);
                tracing::error!(
                    target: "zyron::recovery",
                    lsn = record.lsn.0,
                    file_id = page_id.file_id,
                    page = page_id.page_num,
                    error = %e,
                    "a page change could not be applied and was passed over, the page holds \
                     what it held before it"
                );
                stats.skipped += 1;
            }
            Err(e) => {
                pool.unpin_page(page_id, false);
                return Err(e);
            }
        }
    }
    Ok(stats)
}

/// Reads a page for replay, or a blank image for a page the file does not
/// reach yet, which is what a page allocated and never flushed before the
/// crash looks like
async fn read_or_blank(disk: &Arc<DiskManager>, page_id: PageId) -> Result<[u8; PAGE_SIZE]> {
    if page_id.page_num < disk.num_pages(page_id.file_id).await? {
        return disk.read_page(page_id).await;
    }
    // The file is grown to reach the page, so the flush that follows writes
    // inside it and every page below stays addressable
    let short = disk.num_pages(page_id.file_id).await?;
    let needed = page_id.page_num + 1 - short;
    disk.allocate_pages_batch(page_id.file_id, needed).await?;
    Ok([0u8; PAGE_SIZE])
}

fn put_page_id(buf: &mut Vec<u8>, page_id: PageId) {
    buf.extend_from_slice(&page_id.file_id.to_le_bytes());
    buf.extend_from_slice(&page_id.page_num.to_le_bytes());
}

fn put_row(buf: &mut Vec<u8>, tuple: &Tuple) {
    let header = tuple.header();
    buf.extend_from_slice(&header.flags.0.to_le_bytes());
    buf.extend_from_slice(&header.schema_epoch.to_le_bytes());
    buf.extend_from_slice(&header.xmin.to_le_bytes());
    buf.extend_from_slice(&header.xmax.to_le_bytes());
    buf.extend_from_slice(&(tuple.data().len() as u16).to_le_bytes());
    buf.extend_from_slice(tuple.data());
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    lsn: Lsn,
}

impl Reader<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8]> {
        if self.at + len > self.data.len() {
            return Err(ZyronError::WalCorrupted {
                lsn: self.lsn.0,
                reason: "a heap page record ends inside a field".to_string(),
            });
        }
        let out = &self.data[self.at..self.at + len];
        self.at += len;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16> {
        let raw = self.take(2)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let raw = self.take(8)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(raw);
        Ok(u64::from_le_bytes(bytes))
    }

    fn page_id(&mut self) -> Result<PageId> {
        let file_id = self.u32()?;
        let page_num = self.u64()?;
        Ok(PageId::new(file_id, page_num))
    }

    fn row(&mut self) -> Result<Tuple> {
        let flags = TupleFlags(self.u16()?);
        let schema_epoch = self.u16()?;
        let xmin = self.u64()?;
        let xmax = self.u64()?;
        let len = self.u16()? as usize;
        let data = self.take(len)?.to_vec();
        Ok(Tuple::with_header(
            TupleHeader {
                flags,
                data_len: len as u16,
                schema_epoch,
                xmin,
                xmax,
            },
            data,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u8, xmin: u64) -> Tuple {
        Tuple::with_epoch(vec![id; 40], xmin, 3)
    }

    #[test]
    fn test_every_change_reads_back_as_written() {
        let page_id = PageId::new(7, 12);
        let changes = vec![
            PageChange::Append {
                page_id,
                first_slot: 3,
                data_end: 16_000,
                tuples: vec![row(1, 9), row(2, 9)],
            },
            PageChange::Xmax {
                page_id,
                stamps: vec![(3, 11), (0, 0)],
            },
            PageChange::Prune {
                page_id,
                slots: vec![1, 4],
            },
            PageChange::Free {
                page_id,
                slots: vec![2],
            },
            PageChange::Update {
                page_id,
                slot: 5,
                tuple: row(3, 12),
            },
        ];
        for change in changes {
            let payload = change.encode();
            assert_eq!(payload.len(), change.encoded_len());
            let back = PageChange::decode(change.record_type(), &payload, Lsn(1)).expect("decodes");
            assert_eq!(back.page_id(), page_id);
            assert_eq!(back.encode(), payload);
        }
    }

    #[test]
    fn test_a_short_record_is_refused() {
        let change = PageChange::Xmax {
            page_id: PageId::new(1, 1),
            stamps: vec![(3, 11)],
        };
        let payload = change.encode();
        for cut in 0..payload.len() {
            assert!(
                PageChange::decode(LogRecordType::HeapXmax, &payload[..cut], Lsn(5)).is_err(),
                "a record cut at {cut} decoded"
            );
        }
    }

    /// Two bursts on one page replay to the same image whichever order
    /// their records are applied in, because each carries its positions
    #[test]
    fn test_bursts_replay_the_same_in_either_order() {
        let page_id = PageId::new(2, 0);
        let mut live = [0u8; PAGE_SIZE];
        HeapPage::init_fresh_slice_reuse(&mut live, page_id);
        let first: Vec<Tuple> = vec![row(1, 5), row(2, 5)];
        let second: Vec<Tuple> = vec![row(3, 6)];
        let mut ids = Vec::new();
        let a = unsafe {
            HeapPage::insert_tuples_burst_placed(live.as_mut_ptr(), page_id, &first, &mut ids)
        };
        let b = unsafe {
            HeapPage::insert_tuples_burst_placed(live.as_mut_ptr(), page_id, &second, &mut ids)
        };
        let record_a = PageChange::Append {
            page_id,
            first_slot: a.first_slot,
            data_end: a.data_end,
            tuples: first,
        };
        let record_b = PageChange::Append {
            page_id,
            first_slot: b.first_slot,
            data_end: b.data_end,
            tuples: second,
        };

        let mut forward = [0u8; PAGE_SIZE];
        HeapPage::init_fresh_slice_reuse(&mut forward, page_id);
        record_a.apply(&mut forward, Lsn(1)).expect("a");
        record_b.apply(&mut forward, Lsn(2)).expect("b");
        let mut reversed = [0u8; PAGE_SIZE];
        HeapPage::init_fresh_slice_reuse(&mut reversed, page_id);
        record_b.apply(&mut reversed, Lsn(2)).expect("b");
        record_a.apply(&mut reversed, Lsn(1)).expect("a");

        assert_eq!(&forward[..], &live[..]);
        assert_eq!(&reversed[..], &live[..]);
    }

    /// A change that does not fit its page is refused rather than written
    /// past the page
    #[test]
    fn test_a_change_that_does_not_fit_is_refused() {
        let page_id = PageId::new(2, 1);
        let mut page = [0u8; PAGE_SIZE];
        HeapPage::init_fresh_slice_reuse(&mut page, page_id);
        let too_wide = PageChange::Append {
            page_id,
            first_slot: 0,
            data_end: 100,
            tuples: vec![row(1, 5), row(2, 5)],
        };
        assert!(too_wide.apply(&mut page, Lsn(3)).is_err());
        let no_such_slot = PageChange::Xmax {
            page_id,
            stamps: vec![(4, 9)],
        };
        assert!(no_such_slot.apply(&mut page, Lsn(4)).is_err());
    }
}
