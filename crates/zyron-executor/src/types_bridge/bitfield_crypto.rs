//! Bitfield, crypto digest, and encoding decode arms for the types bridge
//!
//! Bitfield values ride in UInt64 columns per the Bitfield physical mapping,
//! digests and decoded bytes ride in Binary columns as Bytea, and the
//! bitfield_to_positions Array result is JSON text bytes in a Binary cell

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        // bitfield mutators, Bitfield out
        "bitfield_set" => bitfield_position_op(
            "bitfield_set(bitfield, int)",
            args,
            zyron_types::bitfield::bitfield_set,
        ),
        "bitfield_clear" => bitfield_position_op(
            "bitfield_clear(bitfield, int)",
            args,
            zyron_types::bitfield::bitfield_clear,
        ),
        "bitfield_toggle" => bitfield_position_op(
            "bitfield_toggle(bitfield, int)",
            args,
            zyron_types::bitfield::bitfield_toggle,
        ),
        "bitfield_test" => bitfield_test_impl(args),
        "bitfield_count" => bitfield_count_impl(args),
        "bitfield_all" => two_bitfield_to_bool(
            "bitfield_all(bitfield, bitfield)",
            args,
            zyron_types::bitfield::bitfield_all,
        ),
        "bitfield_any" => two_bitfield_to_bool(
            "bitfield_any(bitfield, bitfield)",
            args,
            zyron_types::bitfield::bitfield_any,
        ),
        "bitfield_and" => two_bitfield_to_bitfield(
            "bitfield_and(bitfield, bitfield)",
            args,
            zyron_types::bitfield::bitfield_and,
        ),
        "bitfield_or" => two_bitfield_to_bitfield(
            "bitfield_or(bitfield, bitfield)",
            args,
            zyron_types::bitfield::bitfield_or,
        ),
        "bitfield_xor" => two_bitfield_to_bitfield(
            "bitfield_xor(bitfield, bitfield)",
            args,
            zyron_types::bitfield::bitfield_xor,
        ),
        "bitfield_not" => bitfield_not_impl(args),
        "bitfield_to_positions" => bitfield_to_positions_impl(args),
        "bitfield_from_positions" => bitfield_from_positions_impl(args),

        // crypto digests, Bytea out
        "sha256" => one_bytes_to_bytea("sha256(bytes)", args, |b| {
            zyron_types::crypto::sha256(b).to_vec()
        }),
        "sha384" => one_bytes_to_bytea("sha384(bytes)", args, |b| {
            zyron_types::crypto::sha384(b).to_vec()
        }),
        "sha512" => one_bytes_to_bytea("sha512(bytes)", args, |b| {
            zyron_types::crypto::sha512(b).to_vec()
        }),
        "blake3" => one_bytes_to_bytea("blake3(bytes)", args, |b| {
            zyron_types::crypto::blake3_hash(b).to_vec()
        }),
        "hmac_sha256" => hmac_sha256_impl(args),
        "hash_combine" => hash_combine_impl(args),
        "consistent_hash" => consistent_hash_impl(args),

        // encoding decoders, Bytea out, per row parse failure yields NULL
        "hex_decode" => {
            one_string_decode("hex_decode(text)", args, zyron_types::encoding::hex_decode)
        }
        "base58_decode" => one_string_decode(
            "base58_decode(text)",
            args,
            zyron_types::encoding::base58_decode,
        ),
        "base32_decode" => one_string_decode(
            "base32_decode(text)",
            args,
            zyron_types::encoding::base32_decode,
        ),
        "base64url_decode" => one_string_decode(
            "base64url_decode(text)",
            args,
            zyron_types::encoding::base64url_decode,
        ),
        "murmur3_32" => murmur3_32_impl(args),
        "murmur3_128" => murmur3_128_impl(args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn expect_args(sig: &str, args: &[Column], expected: usize) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} argument{}",
            sig,
            expected,
            if expected == 1 { "" } else { "s" }
        )));
    }
    Ok(())
}

// ORs the null bitmaps of every input over the first n rows
fn union_nulls(cols: &[&Column], n: usize) -> NullBitmap {
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if cols.iter().any(|c| c.nulls.is_null(i)) {
            nulls.set_null(i);
        }
    }
    nulls
}

// ---------------------------------------------------------------------------
// bitfield
// ---------------------------------------------------------------------------

// field plus bit position, position outside 0-63 is a structural error
// matching the zyron-types range check, NULL rows skip the check
fn bitfield_position_op<F: Fn(u64, u8) -> Result<u64>>(
    sig: &str,
    args: &[Column],
    f: F,
) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let fields = super::column_ints(&args[0])?;
    let positions = super::column_ints(&args[1])?;
    let n = fields.len().min(positions.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0);
            continue;
        }
        let pos = positions[i];
        if !(0..64).contains(&pos) {
            return Err(ZyronError::ExecutionError(format!(
                "{} bit position {} out of range (0-63)",
                sig, pos
            )));
        }
        data.push(f(fields[i] as u64, pos as u8)?);
    }
    Ok(Column::with_nulls(
        ColumnData::UInt64(data),
        nulls,
        TypeId::Bitfield,
    ))
}

fn bitfield_test_impl(args: &[Column]) -> Result<Column> {
    let sig = "bitfield_test(bitfield, int)";
    expect_args(sig, args, 2)?;
    let fields = super::column_ints(&args[0])?;
    let positions = super::column_ints(&args[1])?;
    let n = fields.len().min(positions.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(false);
            continue;
        }
        let pos = positions[i];
        if !(0..64).contains(&pos) {
            return Err(ZyronError::ExecutionError(format!(
                "{} bit position {} out of range (0-63)",
                sig, pos
            )));
        }
        data.push(zyron_types::bitfield::bitfield_test(
            fields[i] as u64,
            pos as u8,
        )?);
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn bitfield_count_impl(args: &[Column]) -> Result<Column> {
    expect_args("bitfield_count(bitfield)", args, 1)?;
    let fields = super::column_ints(&args[0])?;
    let data: Vec<i32> = fields
        .iter()
        .map(|&f| zyron_types::bitfield::bitfield_count(f as u64) as i32)
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        args[0].nulls.clone(),
        TypeId::Int32,
    ))
}

fn two_bitfield_to_bitfield<F: Fn(u64, u64) -> u64>(
    sig: &str,
    args: &[Column],
    f: F,
) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let a = super::column_ints(&args[0])?;
    let b = super::column_ints(&args[1])?;
    let n = a.len().min(b.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let data: Vec<u64> = (0..n).map(|i| f(a[i] as u64, b[i] as u64)).collect();
    Ok(Column::with_nulls(
        ColumnData::UInt64(data),
        nulls,
        TypeId::Bitfield,
    ))
}

fn two_bitfield_to_bool<F: Fn(u64, u64) -> bool>(
    sig: &str,
    args: &[Column],
    f: F,
) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let a = super::column_ints(&args[0])?;
    let b = super::column_ints(&args[1])?;
    let n = a.len().min(b.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let data: Vec<bool> = (0..n).map(|i| f(a[i] as u64, b[i] as u64)).collect();
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn bitfield_not_impl(args: &[Column]) -> Result<Column> {
    expect_args("bitfield_not(bitfield)", args, 1)?;
    let fields = super::column_ints(&args[0])?;
    let data: Vec<u64> = fields
        .iter()
        .map(|&f| zyron_types::bitfield::bitfield_not(f as u64))
        .collect();
    Ok(Column::with_nulls(
        ColumnData::UInt64(data),
        args[0].nulls.clone(),
        TypeId::Bitfield,
    ))
}

// Array result encoded as JSON text bytes in a Binary cell
fn bitfield_to_positions_impl(args: &[Column]) -> Result<Column> {
    expect_args("bitfield_to_positions(bitfield)", args, 1)?;
    let fields = super::column_ints(&args[0])?;
    let nulls = args[0].nulls.clone();
    let mut data = Vec::with_capacity(fields.len());
    for (i, &f) in fields.iter().enumerate() {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        let positions = zyron_types::bitfield::bitfield_to_positions(f as u64);
        let json = serde_json::to_string(&positions).map_err(|e| {
            ZyronError::ExecutionError(format!(
                "bitfield_to_positions failed to serialize positions {}",
                e
            ))
        })?;
        data.push(json.into_bytes());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// Accepts an Array cell holding JSON text bytes, a malformed cell yields NULL
// in parser style, positions outside 0-63 are ignored matching zyron-types
fn bitfield_from_positions_impl(args: &[Column]) -> Result<Column> {
    expect_args("bitfield_from_positions(array)", args, 1)?;
    let cells = super::column_bytes(&args[0])?;
    let mut nulls = args[0].nulls.clone();
    let mut data = Vec::with_capacity(cells.len());
    for (i, cell) in cells.iter().enumerate() {
        if nulls.is_null(i) {
            data.push(0);
            continue;
        }
        match serde_json::from_slice::<Vec<i64>>(cell) {
            Ok(list) => {
                let byte_positions: Vec<u8> = list
                    .into_iter()
                    .filter(|p| (0..64).contains(p))
                    .map(|p| p as u8)
                    .collect();
                data.push(zyron_types::bitfield::bitfield_from_positions(
                    &byte_positions,
                ));
            }
            Err(_) => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::UInt64(data),
        nulls,
        TypeId::Bitfield,
    ))
}

// ---------------------------------------------------------------------------
// crypto
// ---------------------------------------------------------------------------

fn one_bytes_to_bytea<F: Fn(&[u8]) -> Vec<u8>>(sig: &str, args: &[Column], f: F) -> Result<Column> {
    expect_args(sig, args, 1)?;
    let values = super::column_bytes(&args[0])?;
    let data: Vec<Vec<u8>> = values.iter().map(|b| f(b)).collect();
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        args[0].nulls.clone(),
        TypeId::Bytea,
    ))
}

fn hmac_sha256_impl(args: &[Column]) -> Result<Column> {
    expect_args("hmac_sha256(bytes, bytes)", args, 2)?;
    let data_cells = super::column_bytes(&args[0])?;
    let key_cells = super::column_bytes(&args[1])?;
    let n = data_cells.len().min(key_cells.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        data.push(zyron_types::crypto::hmac_sha256(data_cells[i], key_cells[i]).to_vec());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Bytea,
    ))
}

// u64 inputs and output are bitcast through i64 so the Int64 column carries
// the exact bit pattern
fn hash_combine_impl(args: &[Column]) -> Result<Column> {
    expect_args("hash_combine(int, int)", args, 2)?;
    let a = super::column_ints(&args[0])?;
    let b = super::column_ints(&args[1])?;
    let n = a.len().min(b.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let data: Vec<i64> = (0..n)
        .map(|i| zyron_types::crypto::hash_combine(a[i] as u64, b[i] as u64) as i64)
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        nulls,
        TypeId::Int64,
    ))
}

fn consistent_hash_impl(args: &[Column]) -> Result<Column> {
    let sig = "consistent_hash(bytes, int)";
    expect_args(sig, args, 2)?;
    let keys = super::column_bytes(&args[0])?;
    let buckets = super::column_ints(&args[1])?;
    let n = keys.len().min(buckets.len());
    let nulls = union_nulls(&[&args[0], &args[1]], n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0);
            continue;
        }
        let count = buckets[i];
        if !(0..=u32::MAX as i64).contains(&count) {
            return Err(ZyronError::ExecutionError(format!(
                "{} bucket count {} out of range (0-{})",
                sig,
                count,
                u32::MAX
            )));
        }
        data.push(zyron_types::crypto::consistent_hash(keys[i], count as u32) as i64);
    }
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        nulls,
        TypeId::Int64,
    ))
}

// ---------------------------------------------------------------------------
// encoding
// ---------------------------------------------------------------------------

// parser style, a per row decode failure yields NULL for that row
fn one_string_decode<F: Fn(&str) -> Result<Vec<u8>>>(
    sig: &str,
    args: &[Column],
    f: F,
) -> Result<Column> {
    expect_args(sig, args, 1)?;
    let strings = super::column_strings(&args[0])?;
    let mut nulls = args[0].nulls.clone();
    let mut data = Vec::with_capacity(strings.len());
    for (i, s) in strings.iter().enumerate() {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        match f(s) {
            Ok(bytes) => data.push(bytes),
            Err(_) => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Bytea,
    ))
}

// seed is optional and defaults to 0, an explicit seed is truncated to u32
fn murmur3_32_impl(args: &[Column]) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(
            "murmur3_32(bytes [, seed]) takes 1 or 2 arguments".to_string(),
        ));
    }
    let data_cells = super::column_bytes(&args[0])?;
    let seeds = if args.len() == 2 {
        Some(super::column_ints(&args[1])?)
    } else {
        None
    };
    let n = match &seeds {
        Some(s) => data_cells.len().min(s.len()),
        None => data_cells.len(),
    };
    let arg_refs: Vec<&Column> = args.iter().collect();
    let nulls = union_nulls(&arg_refs, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0);
            continue;
        }
        let seed = seeds.as_ref().map_or(0, |s| s[i] as u32);
        data.push(zyron_types::encoding::murmur3_32(data_cells[i], seed) as i32);
    }
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        nulls,
        TypeId::Int32,
    ))
}

// seed is optional and defaults to 0, the u128 digest is bitcast to i128
fn murmur3_128_impl(args: &[Column]) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(
            "murmur3_128(bytes [, seed]) takes 1 or 2 arguments".to_string(),
        ));
    }
    let data_cells = super::column_bytes(&args[0])?;
    let seeds = if args.len() == 2 {
        Some(super::column_ints(&args[1])?)
    } else {
        None
    };
    let n = match &seeds {
        Some(s) => data_cells.len().min(s.len()),
        None => data_cells.len(),
    };
    let arg_refs: Vec<&Column> = args.iter().collect();
    let nulls = union_nulls(&arg_refs, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0);
            continue;
        }
        let seed = seeds.as_ref().map_or(0, |s| s[i] as u32);
        data.push(zyron_types::encoding::murmur3_128(data_cells[i], seed) as i128);
    }
    Ok(Column::with_nulls(
        ColumnData::Int128(data),
        nulls,
        TypeId::Int128,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u64_col(values: Vec<u64>) -> Column {
        Column::new(ColumnData::UInt64(values), TypeId::Bitfield)
    }

    fn i64_col(values: Vec<i64>) -> Column {
        Column::new(ColumnData::Int64(values), TypeId::Int64)
    }

    fn utf8_col(values: Vec<&str>) -> Column {
        Column::new(
            ColumnData::Utf8(values.into_iter().map(String::from).collect()),
            TypeId::Text,
        )
    }

    #[test]
    fn set_then_test_reads_back_bit() {
        let set = dispatch("bitfield_set", &[u64_col(vec![0]), i64_col(vec![3])], 1)
            .unwrap()
            .unwrap();
        match &set.data {
            ColumnData::UInt64(v) => assert_eq!(v, &vec![8u64]),
            other => panic!("expected UInt64, got {:?}", other),
        }
        let tested = dispatch("bitfield_test", &[set, i64_col(vec![3])], 1)
            .unwrap()
            .unwrap();
        match &tested.data {
            ColumnData::Boolean(v) => assert_eq!(v, &vec![true]),
            other => panic!("expected Boolean, got {:?}", other),
        }
    }

    #[test]
    fn count_returns_popcount_per_row() {
        let out = dispatch(
            "bitfield_count",
            &[u64_col(vec![0b1010_1010, 0, u64::MAX])],
            3,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v, &vec![4, 0, 64]),
            other => panic!("expected Int32, got {:?}", other),
        }
    }

    #[test]
    fn positions_roundtrip_through_json() {
        let json = dispatch("bitfield_to_positions", &[u64_col(vec![0b1010])], 1)
            .unwrap()
            .unwrap();
        match &json.data {
            ColumnData::Binary(v) => assert_eq!(v[0], b"[1,3]".to_vec()),
            other => panic!("expected Binary, got {:?}", other),
        }
        let back = dispatch("bitfield_from_positions", &[json], 1)
            .unwrap()
            .unwrap();
        match &back.data {
            ColumnData::UInt64(v) => assert_eq!(v, &vec![0b1010u64]),
            other => panic!("expected UInt64, got {:?}", other),
        }
    }

    #[test]
    fn null_input_row_propagates_to_output() {
        let a = u64_col(vec![0xFF, 0xFF]);
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let b = Column::with_nulls(
            ColumnData::UInt64(vec![0x0F, 0x0F]),
            nulls,
            TypeId::Bitfield,
        );
        let out = dispatch("bitfield_and", &[a, b], 2).unwrap().unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        match &out.data {
            ColumnData::UInt64(v) => assert_eq!(v[0], 0x0F),
            other => panic!("expected UInt64, got {:?}", other),
        }
    }

    #[test]
    fn sha256_matches_known_vector() {
        let out = dispatch("sha256", &[utf8_col(vec!["hello"])], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                let hex: String = v[0].iter().map(|b| format!("{:02x}", b)).collect();
                assert_eq!(
                    hex,
                    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
                );
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn hex_decode_invalid_row_yields_null() {
        let out = dispatch("hex_decode", &[utf8_col(vec!["deadbeef", "zz"])], 2)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Binary(v) => assert_eq!(v[0], vec![0xDE, 0xAD, 0xBE, 0xEF]),
            other => panic!("expected Binary, got {:?}", other),
        }
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn murmur3_32_seed_defaults_to_zero() {
        let one_arg = dispatch("murmur3_32", &[utf8_col(vec![""])], 1)
            .unwrap()
            .unwrap();
        match &one_arg.data {
            ColumnData::Int32(v) => assert_eq!(v, &vec![0]),
            other => panic!("expected Int32, got {:?}", other),
        }
        let seeded = dispatch(
            "murmur3_32",
            &[utf8_col(vec!["test"]), i64_col(vec![42])],
            1,
        )
        .unwrap()
        .unwrap();
        let unseeded = dispatch("murmur3_32", &[utf8_col(vec!["test"])], 1)
            .unwrap()
            .unwrap();
        match (&seeded.data, &unseeded.data) {
            (ColumnData::Int32(s), ColumnData::Int32(u)) => assert_ne!(s[0], u[0]),
            other => panic!("expected Int32 pair, got {:?}", other),
        }
    }

    #[test]
    fn consistent_hash_stays_in_bucket_range() {
        let out = dispatch(
            "consistent_hash",
            &[utf8_col(vec!["key"]), i64_col(vec![10])],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Int64(v) => assert!((0..10).contains(&v[0])),
            other => panic!("expected Int64, got {:?}", other),
        }
        assert!(
            dispatch(
                "consistent_hash",
                &[utf8_col(vec!["key"]), i64_col(vec![-1])],
                1,
            )
            .unwrap()
            .is_err()
        );
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(dispatch("not_a_function", &[], 1).is_none());
    }
}
