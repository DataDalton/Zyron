//! Temporal encoding honesty-boundary benchmark.
//!
//! The picosecond claim is precise and is measured here, not asserted by
//! hand: a regular/periodic ps series encodes at microsecond-class density,
//! while an irregular ps series carries strictly more information and costs a
//! bounded extra versus an irregular us series of the same shape. The metric
//! is the encoding ratio raw_bits / encoded_bits (higher is denser), which is
//! float-precision safe even when const-step collapses a column to a few
//! bytes total. Baselines are matched: regular ps is compared to regular us,
//! irregular ps to irregular us, so the comparison is like for like.
//!
//! Suite: "temporal". 5-run averaged, JSON+TXT under benchmarks/temporal/.

use zyron_bench_harness::*;
use zyron_common::types::TypeId;
use zyron_storage::columnar::ColumnSegment;
use zyron_storage::encoding::create_encoding;

static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const ROW_COUNT: usize = 200_000;
const MAX_TS_US: i64 = 253_402_300_799_000_000;
const MAX_TS_PS: i128 = MAX_TS_US as i128 * 1_000_000;
const BASE_US: i64 = 1_700_000_000_000_000;

/// Builds a segment from i128 values at the given physical width, round-trips
/// it, and returns the encoding ratio (raw_bits / encoded_bits). Higher is
/// denser. value_size 8 models a us (i64) column, 16 a ps (i128) column.
fn encoding_ratio(values: &[i128], value_size: usize) -> f64 {
    let raw: Vec<[u8; 16]> = values.iter().map(|x| x.to_le_bytes()).collect();
    let refs: Vec<Option<&[u8]>> = raw.iter().map(|b| Some(&b[..value_size])).collect();
    let seg = ColumnSegment::build(0, TypeId::Timestamp, value_size, &refs).expect("build");
    let dec = create_encoding(seg.header.encoding_type)
        .decode(&seg.encoded_data, refs.len(), value_size)
        .expect("decode");
    assert_eq!(dec.len(), refs.len() * value_size, "lossless round trip");
    let raw_bits = (refs.len() * value_size * 8) as f64;
    let enc_bits = (seg.header.encoded_size as f64 * 8.0).max(1.0);
    raw_bits / enc_bits
}

fn decode_rows_per_sec(values: &[i128], value_size: usize) -> f64 {
    let raw: Vec<[u8; 16]> = values.iter().map(|x| x.to_le_bytes()).collect();
    let refs: Vec<Option<&[u8]>> = raw.iter().map(|b| Some(&b[..value_size])).collect();
    let seg = ColumnSegment::build(0, TypeId::Timestamp, value_size, &refs).expect("build");
    let enc = create_encoding(seg.header.encoding_type);
    let t = std::time::Instant::now();
    let d = enc
        .decode(&seg.encoded_data, refs.len(), value_size)
        .expect("decode");
    let secs = t.elapsed().as_secs_f64().max(1e-9);
    std::hint::black_box(&d);
    refs.len() as f64 / secs
}

/// Regular 1-second-step instants as i128 (us magnitude, width 8 reads low 8).
fn regular_us() -> Vec<i128> {
    (0..ROW_COUNT as i128)
        .map(|i| BASE_US as i128 + i * 1_000_000)
        .collect()
}
/// The identical instants in picoseconds (us * 1e6).
fn regular_ps() -> Vec<i128> {
    regular_us().iter().map(|&u| u * 1_000_000).collect()
}

/// Irregular series with the same step distribution in us and ps so the only
/// difference is the resolution, not the shape. Returns (us, ps).
fn irregular_pair() -> (Vec<i128>, Vec<i128>) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut us = Vec::with_capacity(ROW_COUNT);
    let mut ps = Vec::with_capacity(ROW_COUNT);
    let mut a_us: i128 = BASE_US as i128;
    let mut a_ps: i128 = BASE_US as i128 * 1_000_000;
    for _ in 0..ROW_COUNT {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let step_us = 1_000_000 + (state % 500_000) as i128;
        let sub_ps = (state % 1_000_000) as i128; // sub-us entropy, ps only
        a_us += step_us;
        a_ps += step_us * 1_000_000 + sub_ps;
        us.push(a_us);
        ps.push(a_ps);
    }
    (us, ps)
}

#[test]
fn test_regular_ps_series_matches_microsecond_density() {
    zyron_bench_harness::init("temporal");
    let _g = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Regular series: ps vs us encoding ratio ===");

    let us = regular_us();
    let ps = regular_ps();
    let mut us_runs = Vec::new();
    let mut ps_runs = Vec::new();
    for _ in 0..VALIDATION_RUNS {
        us_runs.push(encoding_ratio(&us, 8));
        ps_runs.push(encoding_ratio(&ps, 16));
    }
    let us_avg = us_runs.iter().sum::<f64>() / us_runs.len() as f64;
    validate_metric(
        "regular_density",
        "us regular ratio (raw:encoded)",
        us_runs.clone(),
        100.0,
        true,
    );
    // A regular ps series rides the same const-step / delta machinery, so its
    // ratio stays in the same class as us (at least half the us ratio). Both
    // are enormous because const-step stores the segment in a few bytes.
    let r = validate_metric(
        "regular_density",
        "ps regular ratio (raw:encoded)",
        ps_runs.clone(),
        us_avg * 0.5,
        true,
    );
    assert!(
        r.average >= us_avg * 0.5,
        "regular ps must stay at us-class density: ps {:.0}x vs us {:.0}x",
        r.average,
        us_avg
    );
    assert!(
        r.average >= 100.0,
        "regular ps must compact hard (const-step/delta), got {:.0}x",
        r.average
    );
}

#[test]
fn test_irregular_ps_overhead_bounded_versus_irregular_us() {
    zyron_bench_harness::init("temporal");
    let _g = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Irregular ps vs irregular us (matched shape) ===");

    let (us, ps) = irregular_pair();
    let mut us_runs = Vec::new();
    let mut ps_runs = Vec::new();
    for _ in 0..VALIDATION_RUNS {
        us_runs.push(encoding_ratio(&us, 8));
        ps_runs.push(encoding_ratio(&ps, 16));
    }
    let us_avg = us_runs.iter().sum::<f64>() / us_runs.len() as f64;
    validate_metric(
        "irregular_overhead",
        "us irregular ratio (raw:encoded)",
        us_runs.clone(),
        1.0,
        true,
    );
    let r = validate_metric(
        "irregular_overhead",
        "ps irregular ratio (raw:encoded)",
        ps_runs.clone(),
        us_avg / 2.5,
        true,
    );
    // Documented boundary: irregular high-precision data is ~1.5-2x a us
    // column, never unbounded. As a ratio that means ps stays within 1/2.5 of
    // the us ratio (headroom over the 2x claim) and still beats raw 128-bit.
    assert!(
        r.average >= us_avg / 2.5,
        "irregular ps overhead must stay bounded vs irregular us: ps {:.2}x vs us {:.2}x",
        r.average,
        us_avg
    );
    assert!(
        r.average > 1.0,
        "irregular ps must still beat raw 128-bit storage, got {:.2}x",
        r.average
    );
}

#[test]
fn test_open_interval_ps_sentinel_column_collapses() {
    zyron_bench_harness::init("temporal");
    let _g = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== sys_end ps sentinel column density ===");

    let mut ps = vec![MAX_TS_PS; ROW_COUNT];
    for k in (0..ROW_COUNT).step_by(997) {
        ps[k] = BASE_US as i128 * 1_000_000 + k as i128;
    }
    let mut runs = Vec::new();
    for _ in 0..VALIDATION_RUNS {
        runs.push(encoding_ratio(&ps, 16));
    }
    // ~all rows one sentinel: Dictionary/Constant collapses this to a tiny
    // code stream. Measured behavior is ~16x; require at least 10x (a real
    // collapse), not the raw 128-bit column.
    let r = validate_metric(
        "sentinel_density",
        "sentinel ps ratio (raw:encoded)",
        runs,
        10.0,
        true,
    );
    assert!(
        r.average >= 10.0,
        "sentinel ps column must collapse hard, got {:.1}x",
        r.average
    );
}

#[test]
fn test_full_materialization_ps_cost_versus_us() {
    zyron_bench_harness::init("temporal");
    let _g = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Decode throughput: ps vs us full materialization ===");

    let us = regular_us();
    let ps = regular_ps();
    let mut us_runs = Vec::new();
    let mut ps_runs = Vec::new();
    for _ in 0..VALIDATION_RUNS {
        us_runs.push(decode_rows_per_sec(&us, 8));
        ps_runs.push(decode_rows_per_sec(&ps, 16));
    }
    let us_avg = us_runs.iter().sum::<f64>() / us_runs.len() as f64;
    validate_metric(
        "ps_decode_throughput",
        "us decode rows/sec",
        us_runs.clone(),
        1.0,
        true,
    );
    let r = validate_metric(
        "ps_decode_throughput",
        "ps decode rows/sec",
        ps_runs.clone(),
        us_avg / 3.0,
        true,
    );
    // Honesty: full ps materialization is ~1.5-2x a us column, never beyond
    // ~3x. This is the documented "not zero-impact in all cases" line.
    //
    // The bound is read off the validation above rather than recomputed here.
    // Comparing the averages directly applied the ratio in an unoptimized
    // build too, where the same call had just printed "not applied" and asked
    // for a release run: the two paths do not lose the same proportion of
    // their work to a debug build, so the ratio between them measures the
    // profile rather than the code.
    assert!(
        r.passed,
        "ps full materialization must stay within ~3x us: ps {:.0} vs us {:.0} rows/sec",
        r.average, us_avg
    );
}
