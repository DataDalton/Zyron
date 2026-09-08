//! Moving an index checkpoint off storing each row's address twice.
//!
//! An 11.0 checkpoint holds the shared key prefix, then a column of whole keys
//! past that prefix, then a column of locator payloads. A key an index writes
//! ends in the seventeen byte order-preserving suffix naming its row, and the
//! payload names the same row in seven, so 11.0 spends twenty four bytes an
//! entry on one address.
//!
//! 11.1 keeps the seven and drops the seventeen. The key column holds the
//! value alone, the locator sits behind each value rather than in a column of
//! its own, and the load rebuilds the suffix from the locator.
//!
//! The step is a body change. Every key is taken apart into its value and the
//! row it names, the shared prefix is measured again over the values because
//! the old one was measured over whole keys, and the footer checksum is taken
//! again. The entries come out in the order they went in, naming the same
//! rows.

use zyron_common::RowLocator;
use zyron_common::format::envelope::{self, ENVELOPE_FOOTER_LEN, ENVELOPE_HEADER_LEN};
use zyron_common::format::registry::{FormatFixture, FormatMigrator};
use zyron_common::format::{FormatKind, FormatVersion};

use crate::format::{CHECKPOINT_FORMAT_VERSION, CHECKPOINT_OLDEST_READABLE};

/// Envelope header plus the checkpoint's own 20-byte header extension.
const HEADER_SIZE: usize = 40;

/// A checkpoint the 0.14.0 writer produced, so the 11.0 reader and this step
/// are exercised against bytes that writer laid down rather than bytes the
/// current writer round-tripped.
static FIXTURE_11_0: &[u8] = include_bytes!("fixtures/v11_0.zyridx");

inventory::submit! {
    FormatFixture {
        kind: FormatKind::Checkpoint,
        version: CHECKPOINT_OLDEST_READABLE,
        bytes: FIXTURE_11_0,
        path: "crates/zyron-storage/src/btree/fixtures/v11_0.zyridx",
    }
}

inventory::submit! {
    FormatMigrator {
        kind: FormatKind::Checkpoint,
        from: CHECKPOINT_OLDEST_READABLE,
        to: CHECKPOINT_FORMAT_VERSION,
        // A checkpoint is derived from the tree it describes, so going back is
        // rebuilding it rather than transforming the file
        reversible: false,
        forward: checkpoint_11_0_to_11_1,
        backward: None,
        no_body_change: false,
        description: "the per-entry locator column comes out, each key already ending in the suffix that names its row",
    }
}

/// The fields the checkpoint keeps in the envelope's header extension.
struct Extension {
    lsn: u64,
    entry_count: u32,
    key_len: u16,
    prefix_len: u16,
    value_width: u16,
}

fn read_extension(extension: &[u8]) -> Extension {
    Extension {
        lsn: u64::from_le_bytes(extension[0..8].try_into().unwrap_or([0; 8])),
        entry_count: u32::from_le_bytes(extension[8..12].try_into().unwrap_or([0; 4])),
        key_len: u16::from_le_bytes([extension[12], extension[13]]),
        prefix_len: u16::from_le_bytes([extension[14], extension[15]]),
        value_width: u16::from_le_bytes([extension[16], extension[17]]),
    }
}

fn write_extension(
    fields: &Extension,
    value_width: u16,
) -> [u8; HEADER_SIZE - ENVELOPE_HEADER_LEN] {
    let mut ext = [0u8; HEADER_SIZE - ENVELOPE_HEADER_LEN];
    ext[0..8].copy_from_slice(&fields.lsn.to_le_bytes());
    ext[8..12].copy_from_slice(&fields.entry_count.to_le_bytes());
    ext[12..14].copy_from_slice(&fields.key_len.to_le_bytes());
    ext[14..16].copy_from_slice(&fields.prefix_len.to_le_bytes());
    ext[16..18].copy_from_slice(&value_width.to_le_bytes());
    ext
}

/// Reads the header of a checkpoint and returns its fields with the body span
/// each column occupies.
fn parts(file: &[u8], expect: FormatVersion) -> Result<(Extension, usize, usize), String> {
    if file.len() < HEADER_SIZE + ENVELOPE_FOOTER_LEN {
        return Err(format!("{} bytes is shorter than a checkpoint", file.len()));
    }
    let (header, extension) =
        envelope::decode_header(&file[..HEADER_SIZE]).map_err(|e| e.to_string())?;
    if header.kind != FormatKind::Checkpoint {
        return Err(format!(
            "expected a checkpoint, found a {} file",
            header.kind
        ));
    }
    if header.version != expect {
        return Err(format!(
            "expected a checkpoint at {expect}, found one at {}",
            header.version
        ));
    }
    let fields = read_extension(extension);
    let n = fields.entry_count as usize;
    let suffix_len = (fields.key_len as usize)
        .checked_sub(fields.prefix_len as usize)
        .ok_or_else(|| "checkpoint declares a prefix longer than its keys".to_string())?;
    let keys_len = fields.prefix_len as usize + n * suffix_len;
    let values_len = n * fields.value_width as usize;
    if file.len() < HEADER_SIZE + keys_len + values_len + ENVELOPE_FOOTER_LEN {
        return Err("checkpoint is shorter than the body its header describes".to_string());
    }
    Ok((fields, keys_len, values_len))
}

/// Assembles a checkpoint from its header fields and body, stamping the
/// version and taking the footer checksum over what was written.
fn assemble(fields: &Extension, value_width: u16, body: &[u8], version: FormatVersion) -> Vec<u8> {
    let extension = write_extension(fields, value_width);
    let header = envelope::encode_header(FormatKind::Checkpoint, version, 0, &extension);
    let mut out = Vec::with_capacity(HEADER_SIZE + body.len() + ENVELOPE_FOOTER_LEN);
    out.extend_from_slice(&header);
    out.extend_from_slice(&extension);
    out.extend_from_slice(body);
    let mut hasher = zyron_common::checksum::Hasher::new();
    hasher.update(body);
    out.extend_from_slice(&hasher.finish32().to_le_bytes());
    out
}

/// Takes the row-naming suffix out of every stored key.
///
/// 11.0 kept each key whole, suffix and all, and a locator column beside it,
/// which is the same address twice. 11.1 stores the value alone and puts the
/// locator that names its row straight after it, so the seventeen byte suffix
/// is rebuilt on load from the seven byte form rather than carried.
pub fn checkpoint_11_0_to_11_1(file: &[u8]) -> Result<Vec<u8>, String> {
    let (fields, keys_len, _values_len) = parts(file, CHECKPOINT_OLDEST_READABLE)?;
    let n = fields.entry_count as usize;
    let kl = fields.key_len as usize;
    let old_prefix = fields.prefix_len as usize;
    let old_suffix = kl - old_prefix;
    let keys = &file[HEADER_SIZE..HEADER_SIZE + keys_len];
    let width = fields.value_width as usize;

    if n == 0 {
        let empty = Extension {
            prefix_len: 0,
            ..fields
        };
        return Ok(assemble(
            &empty,
            RowLocator::NARROW_PAYLOAD_LEN as u16,
            &[],
            CHECKPOINT_FORMAT_VERSION,
        ));
    }

    // Every key put back together, so the value and the row it names can be
    // told apart
    let vl = kl
        .checked_sub(RowLocator::KEY_SUFFIX_LEN)
        .ok_or_else(|| format!("a {kl} byte key is shorter than the suffix naming its row"))?;
    let mut values: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut locators = Vec::with_capacity(n);
    let mut key = Vec::with_capacity(kl);
    for i in 0..n {
        key.clear();
        key.extend_from_slice(&keys[..old_prefix]);
        let at = old_prefix + i * old_suffix;
        key.extend_from_slice(&keys[at..at + old_suffix]);
        let locator = RowLocator::from_key(&key)
            .ok_or_else(|| format!("entry {i} of the checkpoint holds a key that names no row"))?;
        locators.push(locator);
        values.push(key[..vl].to_vec());
    }

    // The shared prefix is taken over the values alone now, so it is measured
    // again rather than carried across
    let mut prefix_len = 0usize;
    if n > 1 {
        let first = &values[0];
        let last = &values[n - 1];
        while prefix_len < vl && first[prefix_len] == last[prefix_len] {
            prefix_len += 1;
        }
    }

    let width = if width == RowLocator::MAX_PAYLOAD_LEN {
        RowLocator::MAX_PAYLOAD_LEN
    } else {
        RowLocator::NARROW_PAYLOAD_LEN
    };
    let stride = (vl - prefix_len) + width;
    let mut body = Vec::with_capacity(prefix_len + n * stride);
    body.extend_from_slice(&values[0][..prefix_len]);
    let mut payload = [0u8; RowLocator::MAX_PAYLOAD_LEN];
    for (value, locator) in values.iter().zip(locators.iter()) {
        body.extend_from_slice(&value[prefix_len..]);
        if width == RowLocator::MAX_PAYLOAD_LEN {
            locator.write_payload_wide(&mut payload);
            body.extend_from_slice(&payload[..width]);
        } else {
            let written = locator.write_payload(&mut payload);
            body.extend_from_slice(&payload[..written]);
        }
    }

    let moved = Extension {
        prefix_len: prefix_len as u16,
        ..fields
    };
    Ok(assemble(
        &moved,
        width as u16,
        &body,
        CHECKPOINT_FORMAT_VERSION,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_fixture_is_an_11_0_checkpoint_with_a_locator_column() {
        let (header, extension) =
            envelope::decode_header(&FIXTURE_11_0[..HEADER_SIZE]).expect("the fixture decodes");
        assert_eq!(header.kind, FormatKind::Checkpoint);
        assert_eq!(header.version, CHECKPOINT_OLDEST_READABLE);
        let fields = read_extension(extension);
        assert!(fields.entry_count > 0, "the fixture holds no entries");
        assert!(
            fields.value_width == RowLocator::NARROW_PAYLOAD_LEN as u16
                || fields.value_width == RowLocator::MAX_PAYLOAD_LEN as u16,
            "an 11.0 checkpoint carries a locator column, this one declares {}",
            fields.value_width
        );
    }

    #[test]
    fn test_the_step_drops_the_column_and_keeps_the_keys() {
        let moved = checkpoint_11_0_to_11_1(FIXTURE_11_0).expect("the step runs");
        let (header, extension) =
            envelope::decode_header(&moved[..HEADER_SIZE]).expect("the result decodes");
        assert_eq!(header.version, CHECKPOINT_FORMAT_VERSION);
        let after = read_extension(extension);
        let before = read_extension(
            envelope::decode_header(&FIXTURE_11_0[..HEADER_SIZE])
                .expect("fixture")
                .1,
        );
        assert_eq!(after.entry_count, before.entry_count);
        assert_eq!(after.key_len, before.key_len);
        assert_eq!(after.lsn, before.lsn);

        // The suffix naming the row leaves the key column and comes back as
        // the shorter payload, so the file loses ten bytes an entry
        let n = before.entry_count as usize;
        let value_len = before.key_len as usize - RowLocator::KEY_SUFFIX_LEN;
        let stride = value_len - after.prefix_len as usize + after.value_width as usize;
        assert_eq!(
            moved.len(),
            HEADER_SIZE + after.prefix_len as usize + n * stride + ENVELOPE_FOOTER_LEN,
            "the body is not the size the header describes"
        );
        assert!(
            moved.len() < FIXTURE_11_0.len(),
            "the step made the file bigger, {} against {}",
            moved.len(),
            FIXTURE_11_0.len()
        );
    }

    /// Every entry the 11.0 file described survives the step, still naming the
    /// row it named before.
    #[test]
    fn test_every_entry_survives_the_step() {
        let moved = checkpoint_11_0_to_11_1(FIXTURE_11_0).expect("the step runs");
        let before = read_extension(
            envelope::decode_header(&FIXTURE_11_0[..HEADER_SIZE])
                .expect("fixture")
                .1,
        );
        let after = read_extension(
            envelope::decode_header(&moved[..HEADER_SIZE])
                .expect("result")
                .1,
        );
        let n = before.entry_count as usize;
        assert_eq!(after.entry_count, before.entry_count);
        assert_eq!(after.key_len, before.key_len);

        // Read both files back into the row each entry names and compare
        let rows_of = |file: &[u8], ext: &Extension, whole_key: bool| -> Vec<RowLocator> {
            let kl = ext.key_len as usize;
            let p = ext.prefix_len as usize;
            let value_len = if whole_key {
                kl
            } else {
                kl - RowLocator::KEY_SUFFIX_LEN
            };
            let suffix = value_len - p;
            let width = ext.value_width as usize;
            let stride = if whole_key { suffix } else { suffix + width };
            let body = &file[HEADER_SIZE..];
            (0..n)
                .map(|i| {
                    let at = p + i * stride;
                    if whole_key {
                        let mut key = body[..p].to_vec();
                        key.extend_from_slice(&body[at..at + suffix]);
                        RowLocator::from_key(&key).expect("11.0 key names a row")
                    } else {
                        RowLocator::read_payload(&body[at + suffix..at + suffix + width])
                            .expect("11.1 entry names a row")
                    }
                })
                .collect()
        };
        assert_eq!(
            rows_of(&moved, &after, false),
            rows_of(FIXTURE_11_0, &before, true),
            "the step changed which rows the entries name"
        );
    }

    #[test]
    fn test_a_file_shorter_than_a_header_is_refused() {
        let err = checkpoint_11_0_to_11_1(&[0u8; 8]).expect_err("a truncated file is refused");
        assert!(err.contains("shorter than"), "unexpected message: {err}");
    }
}
