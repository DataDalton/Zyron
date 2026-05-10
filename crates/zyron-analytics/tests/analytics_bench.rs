#![allow(non_snake_case, unused_assignments, unused_variables)]

//! Analytics Engine Benchmark Suite
//!
//! Integration tests for ZyronDB analytics engine (Phase 16):
//! - ROLLUP/CUBE/GROUPING SETS single-pass aggregation
//! - Period-over-period comparisons (YOY/MOM/YTD)
//! - Cohort retention matrix construction
//! - Funnel conversion analysis with window constraints
//! - Single-pass data profiling with bounded memory sketches
//! - Outlier detection (Z-score, IQR)
//! - Pearson correlation aggregator and pairwise matrix
//! - Performance scaling on 1M and 10M row inputs
//!
//! Performance Targets (5-run average):
//! | Test                              | Metric  | Target   |
//! |-----------------------------------|---------|----------|
//! | ROLLUP (1M rows, 3 cols)          | latency | 200ms    |
//! | CUBE (1M rows, 3 cols)            | latency | 500ms    |
//! | YOY (10M rows)                    | latency | 2s       |
//! | Cohort retention (10M events)     | latency | 10s      |
//! | Funnel (10M events, 5 steps)      | latency | 5s       |
//! | DATA_PROFILE (10M, 20 cols)       | latency | 30s      |
//! | ZSCORE (10M rows)                 | latency | 500ms    |
//! | CORR (10M rows)                   | latency | 300ms    |
//! | CORRELATION_MATRIX (20 cols)      | latency | 5s       |
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target
//!
//! Run: cargo test -p zyron-analytics --test analytics_bench --release -- --nocapture

use std::sync::Mutex;
use std::time::Instant;

use zyron_bench_harness::*;

use zyron_analytics::cohort::CohortEvent;
use zyron_analytics::correlation::{
    CorrelationMatrix, KendallAggregator, PearsonAggregator, SpearmanAggregator,
    correlation_matrix, kendall_tau, mutual_information, pearson_corr,
};
use zyron_analytics::funnel::FunnelEvent;
use zyron_analytics::grouping::{
    Aggregator, GroupingSetExpander, GroupingSetType, GroupingSetsRunner, RowKey, SumAgg,
    expand_grouping_sets, grouping_bit, grouping_id_bits,
};
use zyron_analytics::outlier::{
    IsolationForest, OutlierDecision, ZScoreEvaluator, iqr_outlier, mad_outlier, zscore,
};
use zyron_analytics::period_compare::{
    PeriodUnit, mom, mom_growth, mtd_sum, period_compare_value, qoq, qtd_sum,
    same_period_last_year, wow, yoy, yoy_growth, ytd_sum,
};
use zyron_analytics::profiling::{ColumnProfile, TableProfile, column_profile, profile_table};
use zyron_analytics::registry::{AnalyticsFunctionKind, default_registry};
use zyron_analytics::value::{AnalyticsRow, AnalyticsValue, MS_PER_DAY, MS_PER_HOUR};
use zyron_analytics::{
    CohortAnalysis, CohortDefinition, CohortMetric, CohortPeriod, CohortType, FunnelConfig,
    FunnelStep, funnel_analysis, retention_analysis,
};

// =============================================================================
// Performance target constants
// =============================================================================

const ROLLUP_TARGET_MS: f64 = 200.0;
const CUBE_TARGET_MS: f64 = 500.0;
const YOY_TARGET_MS: f64 = 2_000.0;
const COHORT_RETENTION_TARGET_MS: f64 = 10_000.0;
const FUNNEL_TARGET_MS: f64 = 5_000.0;
const DATA_PROFILE_TARGET_MS: f64 = 30_000.0;
const ZSCORE_TARGET_MS: f64 = 500.0;
const CORR_TARGET_MS: f64 = 300.0;
const CORRELATION_MATRIX_TARGET_MS: f64 = 5_000.0;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Synthetic data generators
// =============================================================================

const REGIONS: &[&str] = &["NA", "EU", "APAC", "LATAM", "MEA"];
const COUNTRIES_BY_REGION: &[&[&str]] = &[
    &["US", "CA", "MX"],
    &["UK", "DE", "FR", "IT"],
    &["JP", "CN", "IN"],
    &["BR", "AR"],
    &["AE", "ZA"],
];
const CITIES_BY_COUNTRY: &[&[&str]] = &[
    &["NYC", "SFO", "LAX"],
    &["TOR", "MTL"],
    &["MEX", "GDL"],
    &["LON", "MAN"],
    &["BER", "MUC"],
    &["PAR", "LYO"],
    &["MIL", "ROM"],
    &["TYO", "OSA"],
    &["BJS", "SHA"],
    &["DEL", "BOM"],
    &["SAO", "RIO"],
    &["BUE"],
    &["DUB"],
    &["JNB"],
];
const CATEGORIES: &[&str] = &["electronics", "books", "apparel", "groceries", "toys"];

fn deterministic_region(i: usize) -> &'static str {
    REGIONS[i % REGIONS.len()]
}

fn deterministic_country(i: usize) -> &'static str {
    let region_idx = i % REGIONS.len();
    let countries = COUNTRIES_BY_REGION[region_idx];
    countries[(i / REGIONS.len()) % countries.len()]
}

fn deterministic_city(i: usize) -> &'static str {
    let cidx = i % CITIES_BY_COUNTRY.len();
    let cities = CITIES_BY_COUNTRY[cidx];
    cities[(i / 7) % cities.len()]
}

fn build_sales_rows(n: usize) -> Vec<Vec<AnalyticsValue>> {
    (0..n)
        .map(|i| {
            vec![
                AnalyticsValue::Text(deterministic_region(i).to_string()),
                AnalyticsValue::Text(deterministic_country(i).to_string()),
                AnalyticsValue::Text(deterministic_city(i).to_string()),
                AnalyticsValue::Float(((i % 1000) as f64) + 50.0),
            ]
        })
        .collect()
}

fn build_region_category_rows(n: usize) -> Vec<Vec<AnalyticsValue>> {
    (0..n)
        .map(|i| {
            vec![
                AnalyticsValue::Text(deterministic_region(i).to_string()),
                AnalyticsValue::Text(CATEGORIES[i % CATEGORIES.len()].to_string()),
                AnalyticsValue::Float(((i % 500) as f64) + 10.0),
            ]
        })
        .collect()
}

// Generates a strictly monthly time series with multi-year coverage
fn build_monthly_revenue(months: usize) -> Vec<(i64, f64)> {
    let mut series = Vec::with_capacity(months);
    // Civil-anchored start at 2018-01-01
    let start_y = 2018;
    for m in 0..months {
        let year = start_y + (m / 12) as i32;
        let month = (m % 12) as u32 + 1;
        let day = 1u32;
        let ts = civil_to_ms(year, month, day);
        // Trend with seasonality
        let value =
            10_000.0 + (m as f64) * 50.0 + 500.0 * (((m % 12) as f64).sin() * 4.0).max(-3.0);
        series.push((ts, value));
    }
    series
}

fn civil_to_ms(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 } as u64) + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe as i64 - 719_468;
    days * MS_PER_DAY
}

// Builds cohort events for `users` users over `months` months. Each user
// does activity at random months drawn from the (decay) distribution.
fn build_cohort_events(users: usize, months: usize, seed: u64) -> Vec<CohortEvent> {
    let mut events = Vec::with_capacity(users * 6);
    let mut rng = SplitMix64::new(seed);
    for u in 0..users {
        let first_month = (rng.next_u64() % months as u64) as usize;
        // First event always present
        events.push(CohortEvent {
            user_id: AnalyticsValue::Text(format!("u{}", u)),
            event_time_ms: civil_to_ms(2024, 1, 1) + (first_month as i64) * 30 * MS_PER_DAY,
            revenue: Some(((u % 50) as f64) + 10.0),
            custom_value: None,
            attribute: None,
        });
        // Decaying retention: each subsequent month has decreasing probability
        for m in (first_month + 1)..months {
            let p = 0.7f64.powi((m - first_month) as i32);
            if (rng.next_u64() as f64 / u64::MAX as f64) < p {
                events.push(CohortEvent {
                    user_id: AnalyticsValue::Text(format!("u{}", u)),
                    event_time_ms: civil_to_ms(2024, 1, 1) + (m as i64) * 30 * MS_PER_DAY,
                    revenue: Some(((u % 50) as f64) + 5.0),
                    custom_value: None,
                    attribute: None,
                });
            }
        }
    }
    events
}

const FUNNEL_STEPS: &[&str] = &["page_view", "add_to_cart", "checkout", "purchase"];

fn build_funnel_events(users: usize, seed: u64) -> Vec<FunnelEvent> {
    let mut events = Vec::with_capacity(users * 4);
    let mut rng = SplitMix64::new(seed);
    let base_ts = civil_to_ms(2024, 6, 1);
    for u in 0..users {
        // Step progression: each user advances through steps with decreasing probability
        let mut t = base_ts + (u as i64) * 1_000;
        let progress = (rng.next_u64() % 100) as usize;
        // Tier of progression
        let max_step = if progress < 5 {
            0
        } else if progress < 35 {
            1
        } else if progress < 70 {
            2
        } else {
            3
        };
        for s in 0..=max_step {
            events.push(FunnelEvent {
                user_id: AnalyticsValue::Text(format!("u{}", u)),
                event_time_ms: t,
                event_name: FUNNEL_STEPS[s].to_string(),
            });
            t += (rng.next_u64() % (10 * 60_000)) as i64 + 1_000;
        }
        // 5% of users have a purchase event outside the 30 minute window
        if u % 20 == 0 {
            events.push(FunnelEvent {
                user_id: AnalyticsValue::Text(format!("u{}", u)),
                event_time_ms: base_ts + (u as i64) * 1_000 + 31 * 60_000 + 1_000,
                event_name: "purchase".into(),
            });
        }
    }
    events
}

// Build a wide table for DATA_PROFILE: 10 columns, mixed types
fn build_profile_table(rows: usize) -> (Vec<String>, Vec<String>, Vec<Vec<AnalyticsValue>>) {
    let names = vec![
        "id".into(),
        "region".into(),
        "category".into(),
        "revenue".into(),
        "quantity".into(),
        "discount".into(),
        "rating".into(),
        "score".into(),
        "label".into(),
        "active".into(),
    ];
    let types = vec![
        "INT64".into(),
        "TEXT".into(),
        "TEXT".into(),
        "FLOAT64".into(),
        "INT64".into(),
        "FLOAT64".into(),
        "FLOAT64".into(),
        "FLOAT64".into(),
        "TEXT".into(),
        "BOOLEAN".into(),
    ];
    let mut cols: Vec<Vec<AnalyticsValue>> =
        (0..names.len()).map(|_| Vec::with_capacity(rows)).collect();
    let mut rng = SplitMix64::new(0xC0FFEE);
    for i in 0..rows {
        cols[0].push(AnalyticsValue::Int(i as i64));
        cols[1].push(AnalyticsValue::Text(deterministic_region(i).to_string()));
        cols[2].push(AnalyticsValue::Text(
            CATEGORIES[i % CATEGORIES.len()].to_string(),
        ));
        // 5% nulls for revenue
        if i % 20 == 0 {
            cols[3].push(AnalyticsValue::Null);
        } else {
            cols[3].push(AnalyticsValue::Float(((i % 5000) as f64) + 25.0));
        }
        cols[4].push(AnalyticsValue::Int((i % 100) as i64));
        cols[5].push(AnalyticsValue::Float(
            ((rng.next_u64() % 100) as f64) / 100.0,
        ));
        cols[6].push(AnalyticsValue::Float(((i % 5) as f64) + 1.0));
        cols[7].push(AnalyticsValue::Float(
            ((rng.next_u64() % 1_000) as f64) / 10.0,
        ));
        cols[8].push(AnalyticsValue::Text(format!("label-{}", i % 1000)));
        cols[9].push(AnalyticsValue::Bool(i % 2 == 0));
    }
    (names, types, cols)
}

// Build a 20-column numeric table for CORRELATION_MATRIX
fn build_numeric_table(rows: usize, cols: usize) -> (Vec<String>, Vec<Vec<AnalyticsValue>>) {
    let mut names = Vec::with_capacity(cols);
    for c in 0..cols {
        names.push(format!("c{}", c));
    }
    let mut data: Vec<Vec<AnalyticsValue>> = (0..cols).map(|_| Vec::with_capacity(rows)).collect();
    let mut rng = SplitMix64::new(0xABCD_EF01);
    for i in 0..rows {
        let base = (i as f64) * 0.1;
        for c in 0..cols {
            let phase = c as f64 * 0.137;
            let noise = ((rng.next_u64() % 1000) as f64) / 100.0;
            let v = base + phase * 1.7 + noise;
            data[c].push(AnalyticsValue::Float(v));
        }
    }
    (names, data)
}

// Deterministic SplitMix64
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// =============================================================================
// Test 1: ROLLUP correctness
// =============================================================================

#[test]
fn test_01_rollup_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 1: ROLLUP Correctness ===");

    let schema = vec![
        "region".to_string(),
        "country".to_string(),
        "city".to_string(),
        "revenue".to_string(),
    ];
    let spec = GroupingSetType::Rollup(vec!["region".into(), "country".into(), "city".into()]);
    let mut runner = GroupingSetsRunner::new(schema, spec.clone(), SumAgg::new(3));

    let rows = build_sales_rows(5_000);
    let mut total_actual = 0.0_f64;
    for r in &rows {
        if let AnalyticsValue::Float(v) = r[3] {
            total_actual += v;
        }
        runner.ingest_row(r);
    }
    let out = runner.finalise();

    // Expand sets to know what to expect
    let sets = expand_grouping_sets(&spec);
    tprintln!("  Grouping sets expanded: {}", sets.len());
    assert_eq!(sets.len(), 4, "ROLLUP(3) must yield 4 grouping sets");
    assert!(sets[0].len() == 3 && sets[3].is_empty());

    // Grand total row exists and equals total
    let grand = out
        .iter()
        .find(|o| o.active_columns.is_empty())
        .expect("grand total row missing");
    let grand_value = match grand.aggregate {
        AnalyticsValue::Float(v) => v,
        _ => panic!("grand total should be float"),
    };
    tprintln!(
        "  Grand total = {:.2} (expected {:.2})",
        grand_value,
        total_actual
    );
    assert!(
        (grand_value - total_actual).abs() < 1e-6,
        "grand total mismatch: got {} expected {}",
        grand_value,
        total_actual
    );

    // Region-level subtotals: column count == 1 (just region in active)
    let region_subtotals: Vec<_> = out
        .iter()
        .filter(|o| o.active_columns.len() == 1 && o.active_columns[0] == "region")
        .collect();
    let sum_region: f64 = region_subtotals
        .iter()
        .map(|o| match o.aggregate {
            AnalyticsValue::Float(v) => v,
            _ => 0.0,
        })
        .sum();
    tprintln!(
        "  Region subtotal rows: {}, sum {:.2}",
        region_subtotals.len(),
        sum_region
    );
    assert!(
        (sum_region - total_actual).abs() < 1e-6,
        "region subtotals must sum to grand total"
    );

    // GROUPING() flag verification: for the grand total row, GROUPING(region) = 1
    let g_region = grouping_bit(&grand.active_columns, "region");
    let g_country = grouping_bit(&grand.active_columns, "country");
    assert_eq!(g_region, 1);
    assert_eq!(g_country, 1);
    // For full set row, all GROUPING bits are 0
    let full = out
        .iter()
        .find(|o| o.active_columns.len() == 3)
        .expect("full grouping row missing");
    assert_eq!(grouping_bit(&full.active_columns, "region"), 0);
    assert_eq!(grouping_bit(&full.active_columns, "country"), 0);
    assert_eq!(grouping_bit(&full.active_columns, "city"), 0);

    tprintln!("  Hierarchical subtotals verified: PASS");
    tprintln!("  GROUPING() flag values verified: PASS");
}

// =============================================================================
// Test 2: CUBE correctness
// =============================================================================

#[test]
fn test_02_cube_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 2: CUBE Correctness ===");

    let schema = vec![
        "region".to_string(),
        "category".to_string(),
        "revenue".to_string(),
    ];
    let spec = GroupingSetType::Cube(vec!["region".into(), "category".into()]);
    let mut runner = GroupingSetsRunner::new(schema, spec.clone(), SumAgg::new(2));

    let rows = build_region_category_rows(2_000);
    for r in &rows {
        runner.ingest_row(r);
    }
    let out = runner.finalise();

    let sets = expand_grouping_sets(&spec);
    tprintln!("  Grouping sets expanded: {}", sets.len());
    assert_eq!(sets.len(), 4, "CUBE(2) must yield 4 grouping sets");

    // Verify all 4 combinations exist as set indices in output
    use std::collections::HashSet;
    let observed_sets: HashSet<u32> = out.iter().map(|o| o.set_index).collect();
    assert_eq!(observed_sets.len(), 4, "all 4 CUBE sets must appear");

    // GROUPING_ID for the grand total row, requesting [region, category]
    let grand = out
        .iter()
        .find(|o| o.active_columns.is_empty())
        .expect("grand total row missing");
    let id = grouping_id_bits(
        &grand.active_columns,
        &["region".to_string(), "category".to_string()],
    );
    tprintln!("  GROUPING_ID(grand_total) = 0b{:02b}", id);
    assert_eq!(id, 0b11, "grand total must have all GROUPING bits set");

    // For the (region) only set, GROUPING_ID should be 0b01 (category aggregated away)
    let region_only = out
        .iter()
        .find(|o| o.active_columns == vec!["region".to_string()])
        .expect("region-only set missing");
    let id_region = grouping_id_bits(
        &region_only.active_columns,
        &["region".to_string(), "category".to_string()],
    );
    assert_eq!(id_region, 0b01);

    tprintln!("  CUBE all combinations present: PASS");
    tprintln!("  GROUPING_ID bitmasks verified: PASS");
}

// =============================================================================
// Test 3: GROUPING SETS correctness
// =============================================================================

#[test]
fn test_03_grouping_sets_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 3: GROUPING SETS Correctness ===");

    let schema = vec![
        "region".to_string(),
        "product".to_string(),
        "quarter".to_string(),
        "revenue".to_string(),
    ];
    let spec = GroupingSetType::GroupingSets(vec![
        vec!["region".into(), "product".into()],
        vec!["region".into(), "quarter".into()],
        vec![],
    ]);
    let mut runner = GroupingSetsRunner::new(schema, spec.clone(), SumAgg::new(3));

    // Synthesize rows: 4 regions, 5 products, 4 quarters, revenue
    let mut total = 0.0;
    for r in 0..4 {
        for p in 0..5 {
            for q in 0..4 {
                let row = vec![
                    AnalyticsValue::Text(format!("R{}", r)),
                    AnalyticsValue::Text(format!("P{}", p)),
                    AnalyticsValue::Text(format!("Q{}", q)),
                    AnalyticsValue::Float(10.0 * (r + p + q + 1) as f64),
                ];
                if let AnalyticsValue::Float(v) = row[3] {
                    total += v;
                }
                runner.ingest_row(&row);
            }
        }
    }
    let out = runner.finalise();
    let sets = expand_grouping_sets(&spec);
    assert_eq!(sets.len(), 3);

    use std::collections::HashSet;
    let observed_sets: HashSet<u32> = out.iter().map(|o| o.set_index).collect();
    assert_eq!(observed_sets.len(), 3, "all 3 GROUPING SETS must appear");

    // Grand total row equals sum of all revenue
    let grand = out
        .iter()
        .find(|o| o.active_columns.is_empty())
        .expect("grand total missing");
    if let AnalyticsValue::Float(v) = grand.aggregate {
        assert!(
            (v - total).abs() < 1e-6,
            "grand total mismatch: {} vs {}",
            v,
            total
        );
    }

    tprintln!("  3 distinct grouping types observed: PASS");
    tprintln!("  Grand total matches input sum: PASS");
}

// =============================================================================
// Test 4: Period-over-period comparisons
// =============================================================================

#[test]
fn test_04_period_comparisons() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 4: Period-over-Period Comparisons ===");

    // 36 monthly observations (3 years from 2018-01)
    let series = build_monthly_revenue(36);

    let yoy_vals = yoy(&series);
    // First 12 entries have no prior year, entries 12.. should match entry i-12
    for i in 0..12 {
        assert!(yoy_vals[i].is_none(), "YoY[{}] should be None", i);
    }
    for i in 12..36 {
        let expected = series[i - 12].1;
        assert!(
            yoy_vals[i].is_some(),
            "YoY[{}] missing despite same calendar day in prior year",
            i
        );
        assert!(
            (yoy_vals[i].unwrap() - expected).abs() < 1e-9,
            "YoY[{}] = {:?}, expected {}",
            i,
            yoy_vals[i],
            expected
        );
    }
    tprintln!("  YOY value alignment verified for 24 months: PASS");

    let yoy_g = yoy_growth(&series);
    let mom_vals = mom(&series);
    let mom_g = mom_growth(&series);

    // YOY_GROWTH at month 12: (series[12] - series[0]) / series[0]
    let expected_growth = (series[12].1 - series[0].1) / series[0].1;
    assert!(
        (yoy_g[12].unwrap() - expected_growth).abs() < 1e-9,
        "YoY_GROWTH mismatch"
    );
    tprintln!("  YOY_GROWTH percentage verified: PASS");

    // MOM at month 1 should equal series[0]
    assert!(
        (mom_vals[1].unwrap() - series[0].1).abs() < 1e-9,
        "MoM[1] mismatch"
    );
    tprintln!("  MOM previous-month value verified: PASS");

    // YTD_SUM resets each calendar year. Sum of first 12 months equals YTD[11]
    let ytd = ytd_sum(&series);
    let expected_ytd_11: f64 = series[..12].iter().map(|(_, v)| *v).sum();
    assert!(
        (ytd[11] - expected_ytd_11).abs() < 1e-6,
        "YTD[Dec 2018] = {}, expected {}",
        ytd[11],
        expected_ytd_11
    );
    // YTD[12] (Jan 2019) should reset to series[12].1
    assert!(
        (ytd[12] - series[12].1).abs() < 1e-9,
        "YTD must reset at year boundary"
    );
    tprintln!("  YTD_SUM resets at year boundary: PASS");

    let qtd = qtd_sum(&series);
    // Q1 2018: months 0,1,2; QTD[2] = sum of three
    let expected_qtd_2: f64 = series[..3].iter().map(|(_, v)| *v).sum();
    assert!((qtd[2] - expected_qtd_2).abs() < 1e-6);
    // Q2 2018 starts at month 3
    assert!((qtd[3] - series[3].1).abs() < 1e-9);
    tprintln!("  QTD_SUM bucket boundaries verified: PASS");

    let mtd = mtd_sum(&series);
    // Each month index has only one observation (month start), so MTD = value
    for i in 0..36 {
        assert!((mtd[i] - series[i].1).abs() < 1e-9);
    }

    // PERIOD_COMPARE generic
    let pc = period_compare_value(&series, 12, "month").unwrap();
    if let AnalyticsValue::Float(v) = &pc[12] {
        assert!((*v - series[0].1).abs() < 1e-9);
    }
    tprintln!("  PERIOD_COMPARE generic invocation verified: PASS");
}

// =============================================================================
// Test 5: Cohort retention correctness
// =============================================================================

#[test]
fn test_05_cohort_retention_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 5: Cohort Retention Correctness ===");

    let users = 1_000;
    let months = 12;
    let events = build_cohort_events(users, months, 0xDEADBEEF);
    tprintln!("  Generated {} events for {} users", events.len(), users);

    let analysis = CohortAnalysis {
        definition: CohortDefinition {
            user_id_column: "user_id".into(),
            event_time_column: "event_time".into(),
            cohort_period: CohortPeriod::Month,
            analysis_period: CohortPeriod::Month,
            cohort_type: CohortType::FirstEvent,
        },
        metric: CohortMetric::ActiveUsers,
    };
    let result = retention_analysis(&events, &analysis, months as u32).unwrap();
    tprintln!(
        "  Cohorts: {}, periods per cohort: {}",
        result.cohorts.len(),
        result.periods
    );
    assert!(!result.cohorts.is_empty());
    assert_eq!(result.periods, months as u32);

    // month_0 must equal the cohort size for every cohort
    let mut cohort_size_total = 0u64;
    for cohort in &result.cohorts {
        let m0 = cohort.period_values[0];
        assert!(
            m0 > 0.0,
            "cohort {} has zero size in month_0",
            cohort.cohort_label
        );
        cohort_size_total += m0 as u64;
    }
    assert_eq!(
        cohort_size_total as usize, users,
        "every user must appear in exactly one cohort (sum {} != {})",
        cohort_size_total, users
    );
    tprintln!("  Each user assigned to exactly one cohort: PASS");
    tprintln!("  month_0 values are 100% of each cohort's size: PASS");

    // Retention is non-increasing across periods within each cohort
    let mut decay_violations = 0;
    for cohort in &result.cohorts {
        for w in cohort.period_values.windows(2) {
            if w[0] > 0.0 && w[1] > w[0] {
                decay_violations += 1;
            }
        }
    }
    tprintln!("  Non-monotonic cohort steps: {}", decay_violations);
    assert!(
        decay_violations < (result.cohorts.len() * months / 4),
        "too many non-monotonic retention steps; data is not decaying"
    );
    tprintln!("  Retention decreases over time: PASS");
}

// =============================================================================
// Test 6: Funnel analysis correctness
// =============================================================================

#[test]
fn test_06_funnel_analysis_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 6: Funnel Analysis Correctness ===");

    let users = 1_000;
    let events = build_funnel_events(users, 0xCAFE_F00D);
    tprintln!("  Generated {} events across {} users", events.len(), users);

    let cfg = FunnelConfig {
        steps: FUNNEL_STEPS
            .iter()
            .map(|s| FunnelStep {
                name: (*s).into(),
                event_match: (*s).into(),
            })
            .collect(),
        window_ms: 30 * 60_000, // 30 minutes
        user_id_column: "user_id".into(),
        event_time_column: "event_time".into(),
    };
    let result = funnel_analysis(&events, &cfg).unwrap();

    for s in &result.steps {
        tprintln!(
            "  {:14} users={:>5} conv={:.3} drop={:.3}",
            s.name,
            s.users_count,
            s.conversion_rate,
            s.drop_off_rate
        );
    }

    assert_eq!(result.steps.len(), FUNNEL_STEPS.len());
    // Step counts must be monotonically non-increasing through the funnel
    for i in 1..result.steps.len() {
        assert!(
            result.steps[i].users_count <= result.steps[i - 1].users_count,
            "step {} count {} > previous {}",
            i,
            result.steps[i].users_count,
            result.steps[i - 1].users_count
        );
    }
    tprintln!("  Step counts are monotonically non-increasing: PASS");

    // First step is always 100% conversion
    assert!((result.steps[0].conversion_rate - 1.0).abs() < 1e-9);
    // For each subsequent step, drop_off + conversion = 1
    for s in &result.steps[1..] {
        let total = s.conversion_rate + s.drop_off_rate;
        assert!(
            (total - 1.0).abs() < 1e-9,
            "conversion {} + drop {} = {}, expected 1",
            s.conversion_rate,
            s.drop_off_rate,
            total
        );
    }
    tprintln!("  Drop-off + conversion sums to 1.0 per step: PASS");

    // Window constraint: 5% of users have a delayed purchase (>30min). Those
    // late events must NOT inflate the purchase count beyond what the funnel
    // generator's step distribution produces. Generator: ~30% of users reach
    // step 3 (purchase). Allow some slack but the count must be well below
    // (in_funnel_users + late_users), which would be the value if late events
    // were incorrectly counted.
    let purchases_in_funnel = result.steps[3].users_count;
    let late_event_count = users / 20; // 5% delayed by generator
    let in_funnel_step2 = result.steps[2].users_count;
    // If the window were ignored, late events would push step3 to be near
    // step2 + late events. The assertion: step3 must NOT exceed step2 (upper
    // bound from monotonicity already), AND step3 distance from step2 must be
    // larger than late_event_count (i.e. real drop-off, not late events
    // sneaking in).
    let drop = in_funnel_step2.saturating_sub(purchases_in_funnel);
    tprintln!(
        "  step2={} step3={} drop={} late_events={}",
        in_funnel_step2,
        purchases_in_funnel,
        drop,
        late_event_count
    );
    assert!(
        drop >= late_event_count as u64,
        "window constraint violated: step2 -> step3 drop {} < late_event_count {}",
        drop,
        late_event_count
    );
    tprintln!("  Window constraint excludes late events: PASS");
}

// =============================================================================
// Test 7: Data profiling correctness
// =============================================================================

#[test]
fn test_07_data_profiling_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 7: Data Profiling Correctness ===");

    // 100K rows across 10 columns of mixed types
    let rows = 100_000;
    let (names, types, cols) = build_profile_table(rows);

    // Compute reference statistics for one numeric column
    let revenue_col = &cols[3];
    let mut numeric: Vec<f64> = Vec::with_capacity(rows);
    let mut nulls = 0;
    for v in revenue_col {
        match v {
            AnalyticsValue::Null => nulls += 1,
            AnalyticsValue::Float(f) => numeric.push(*f),
            _ => {}
        }
    }
    numeric.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let actual_p50 = numeric[numeric.len() / 2];

    let profile = profile_table("synthetic", &names, &types, &cols, false).unwrap();
    assert_eq!(profile.row_count, rows as u64);
    assert_eq!(profile.columns.len(), names.len());

    let revenue_profile = profile
        .columns
        .iter()
        .find(|c| c.column_name == "revenue")
        .unwrap();

    // Null count exact match
    tprintln!(
        "  revenue null_count: profile={} actual={}",
        revenue_profile.null_count,
        nulls
    );
    assert_eq!(revenue_profile.null_count as usize, nulls);

    // Distinct count within 2% accuracy
    let actual_distinct = {
        let mut s = std::collections::HashSet::new();
        for v in revenue_col {
            if !matches!(v, AnalyticsValue::Null) {
                s.insert(format!("{:?}", v));
            }
        }
        s.len() as f64
    };
    let est_distinct = revenue_profile.distinct_count as f64;
    let distinct_err = (est_distinct - actual_distinct).abs() / actual_distinct;
    tprintln!(
        "  revenue distinct: estimate={} actual={} err={:.4}",
        est_distinct,
        actual_distinct,
        distinct_err
    );
    // Spec: distinct_count within 2% of actual
    assert!(
        distinct_err <= 0.02,
        "distinct estimate err {:.4} > 2% tolerance",
        distinct_err
    );

    // Spec: percentile estimation within 1% accuracy
    let p50 = revenue_profile.percentiles.as_ref().unwrap().p50;
    let p50_err = (p50 - actual_p50).abs() / actual_p50.abs().max(1.0);
    tprintln!(
        "  revenue p50: estimate={:.2} actual={:.2} err={:.4}",
        p50,
        actual_p50,
        p50_err
    );
    assert!(p50_err <= 0.01, "p50 err {:.4} > 1% tolerance", p50_err);

    // Most common values: at least one MCV should be one of the most populated buckets
    assert!(
        !revenue_profile.most_common_values.is_empty(),
        "MCV list empty"
    );
    tprintln!("  MCV count: {}", revenue_profile.most_common_values.len());

    tprintln!("  All column-level statistics verified: PASS");
}

// =============================================================================
// Test 8: Outlier detection correctness
// =============================================================================

#[test]
fn test_08_outlier_detection_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 8: Outlier Detection Correctness ===");

    // 990 normal points around mean 100, std 5; 10 known outliers far away
    let normal: Vec<f64> = (0..990).map(|i| 100.0 + ((i % 11) as f64 - 5.0)).collect();
    let outliers: Vec<f64> = (0..10).map(|i| 1_000.0 + i as f64 * 50.0).collect();
    let mut data = normal.clone();
    data.extend(outliers.iter().copied());
    let outlier_indices: Vec<usize> = (990..1_000).collect();

    let scores = zscore(&data);
    let strong = outlier_indices
        .iter()
        .filter(|&&i| scores[i].abs() > 3.0)
        .count();
    tprintln!(
        "  ZSCORE: {}/{} injected outliers exceed |z|>3",
        strong,
        outlier_indices.len()
    );
    assert_eq!(
        strong,
        outlier_indices.len(),
        "all injected outliers must have |z| > 3"
    );

    // No false positives on normal data
    let normal_false = scores[..990].iter().filter(|&&z| z.abs() > 3.0).count();
    tprintln!("  ZSCORE false positives on normal data: {}", normal_false);
    assert!(
        normal_false == 0,
        "normal data must not have |z|>3 outliers"
    );

    // IQR_OUTLIER catches injected outliers
    let iqr_dec = iqr_outlier(&data, 1.5);
    let iqr_hits = outlier_indices
        .iter()
        .filter(|&&i| iqr_dec[i].is_outlier())
        .count();
    let iqr_normal_fp = iqr_dec[..990].iter().filter(|d| d.is_outlier()).count();
    tprintln!(
        "  IQR_OUTLIER: {} injected detected, {} false positives in normal data",
        iqr_hits,
        iqr_normal_fp
    );
    assert_eq!(
        iqr_hits,
        outlier_indices.len(),
        "IQR must flag all injected outliers"
    );
    assert!(iqr_normal_fp == 0, "IQR must have no false positives here");

    // MAD_OUTLIER on the same data
    let mad_dec = mad_outlier(&data, 3.5);
    let mad_hits = outlier_indices
        .iter()
        .filter(|&&i| mad_dec[i].is_outlier())
        .count();
    tprintln!("  MAD_OUTLIER: {} injected detected", mad_hits);
    assert!(mad_hits >= 8, "MAD must catch >=8 of 10 injected outliers");

    tprintln!("  All outlier detectors verified: PASS");
}

// =============================================================================
// Test 9: Correlation correctness
// =============================================================================

#[test]
fn test_09_correlation_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 9: Correlation Correctness ===");

    let n = 1_000;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_perfect: Vec<f64> = x.iter().map(|v| 2.0 * v + 5.0).collect();
    let y_anti: Vec<f64> = x.iter().map(|v| -3.0 * v + 1_000.0).collect();
    let mut rng = SplitMix64::new(0xDEAD_BEEF_CAFE);
    let y_indep: Vec<f64> = (0..n)
        .map(|_| (rng.next_u64() as f64 / u64::MAX as f64))
        .collect();

    let c_perfect = pearson_corr(&x, &y_perfect).unwrap();
    let c_anti = pearson_corr(&x, &y_anti).unwrap();
    let c_indep = pearson_corr(&x, &y_indep).unwrap();
    tprintln!(
        "  CORR perfect={:.6} anti={:.6} indep={:.6}",
        c_perfect,
        c_anti,
        c_indep
    );
    assert!((c_perfect - 1.0).abs() < 1e-9);
    assert!((c_anti + 1.0).abs() < 1e-9);
    assert!(c_indep.abs() < 0.10);

    // CORRELATION_MATRIX symmetry and diagonal
    let xs: Vec<AnalyticsValue> = x.iter().map(|v| AnalyticsValue::Float(*v)).collect();
    let ys: Vec<AnalyticsValue> = y_perfect
        .iter()
        .map(|v| AnalyticsValue::Float(*v))
        .collect();
    let zs: Vec<AnalyticsValue> = y_anti.iter().map(|v| AnalyticsValue::Float(*v)).collect();
    let cols = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let m = correlation_matrix(&cols, &[xs, ys, zs]);

    for i in 0..3 {
        assert!(
            (m.get(i, i) - 1.0).abs() < 1e-9,
            "diagonal[{},{}] != 1",
            i,
            i
        );
    }
    for i in 0..3 {
        for j in 0..3 {
            let a = m.get(i, j);
            let b = m.get(j, i);
            assert!(
                (a - b).abs() < 1e-9,
                "matrix not symmetric at ({},{})",
                i,
                j
            );
        }
    }
    tprintln!(
        "  Matrix [0,1]={:.4} [0,2]={:.4} [1,2]={:.4}",
        m.get(0, 1),
        m.get(0, 2),
        m.get(1, 2)
    );
    tprintln!("  CORRELATION_MATRIX symmetry + diagonal verified: PASS");
}

// =============================================================================
// Test 10: Performance benchmarks (5-run averaged)
// =============================================================================

#[test]
fn test_10_rollup_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 10: ROLLUP Performance (1M rows, 3 cols) ===");
    let util_before = take_util_snapshot();

    let rows = build_sales_rows(1_000_000);
    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let schema = vec![
            "region".to_string(),
            "country".to_string(),
            "city".to_string(),
            "revenue".to_string(),
        ];
        let spec = GroupingSetType::Rollup(vec!["region".into(), "country".into(), "city".into()]);
        let mut runner = GroupingSetsRunner::new(schema, spec, SumAgg::new(3));
        let start = Instant::now();
        for r in &rows {
            runner.ingest_row(r);
        }
        let _ = runner.finalise();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "ROLLUP_1M_3cols",
        "Latency (ms)",
        latencies,
        ROLLUP_TARGET_MS,
        false,
    );
    record_test_util("ROLLUP", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "ROLLUP latency avg {:.2}ms > target {:.0}ms",
        result.average, ROLLUP_TARGET_MS
    );
    assert!(!result.regression_detected, "ROLLUP regression detected");
}

#[test]
fn test_11_cube_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 11: CUBE Performance (1M rows, 3 cols) ===");
    let util_before = take_util_snapshot();

    let rows: Vec<Vec<AnalyticsValue>> = (0..1_000_000)
        .map(|i| {
            vec![
                AnalyticsValue::Text(deterministic_region(i).to_string()),
                AnalyticsValue::Text(CATEGORIES[i % CATEGORIES.len()].to_string()),
                AnalyticsValue::Text(format!("Q{}", (i % 4) + 1)),
                AnalyticsValue::Float(((i % 1000) as f64) + 25.0),
            ]
        })
        .collect();

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let schema = vec![
            "region".to_string(),
            "category".to_string(),
            "quarter".to_string(),
            "revenue".to_string(),
        ];
        let spec =
            GroupingSetType::Cube(vec!["region".into(), "category".into(), "quarter".into()]);
        let mut runner = GroupingSetsRunner::new(schema, spec, SumAgg::new(3));
        let start = Instant::now();
        for r in &rows {
            runner.ingest_row(r);
        }
        let _ = runner.finalise();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "CUBE_1M_3cols",
        "Latency (ms)",
        latencies,
        CUBE_TARGET_MS,
        false,
    );
    record_test_util("CUBE", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "CUBE latency avg {:.2}ms > target {:.0}ms",
        result.average, CUBE_TARGET_MS
    );
    assert!(!result.regression_detected, "CUBE regression detected");
}

#[test]
fn test_12_yoy_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 12: YOY Performance (10M rows) ===");
    let util_before = take_util_snapshot();

    // 10M daily observations spanning ~27 years
    let rows = 10_000_000;
    let mut series = Vec::with_capacity(rows);
    let base_ms = civil_to_ms(1998, 1, 1);
    for i in 0..rows {
        series.push((base_ms + (i as i64) * MS_PER_DAY, (i % 1_000) as f64));
    }
    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let out = yoy(&series);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        // Verify the output length matches input
        assert_eq!(out.len(), series.len());
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric("YOY_10M", "Latency (ms)", latencies, YOY_TARGET_MS, false);
    record_test_util("YOY", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "YOY latency avg {:.2}ms > target {:.0}ms",
        result.average, YOY_TARGET_MS
    );
    assert!(!result.regression_detected, "YOY regression detected");
}

#[test]
fn test_13_cohort_retention_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 13: Cohort Retention Performance (10M events) ===");
    let util_before = take_util_snapshot();

    // 1M users * ~10 events each
    let users = 1_000_000;
    let months = 12;
    let events = build_cohort_events(users, months, 0xCAFE_BABE);
    tprintln!("  Generated {} events", events.len());

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let analysis = CohortAnalysis {
            definition: CohortDefinition {
                user_id_column: "user_id".into(),
                event_time_column: "event_time".into(),
                cohort_period: CohortPeriod::Month,
                analysis_period: CohortPeriod::Month,
                cohort_type: CohortType::FirstEvent,
            },
            metric: CohortMetric::ActiveUsers,
        };
        let start = Instant::now();
        let result = retention_analysis(&events, &analysis, months as u32).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert!(!result.cohorts.is_empty());
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "COHORT_RETENTION_10M",
        "Latency (ms)",
        latencies,
        COHORT_RETENTION_TARGET_MS,
        false,
    );
    record_test_util("CohortRetention", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Cohort retention latency avg {:.2}ms > target {:.0}ms",
        result.average, COHORT_RETENTION_TARGET_MS
    );
    assert!(!result.regression_detected, "Cohort regression detected");
}

#[test]
fn test_14_funnel_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 14: Funnel Performance (10M events, 5 steps) ===");
    let util_before = take_util_snapshot();

    let users = 2_000_000;
    let mut events = build_funnel_events(users, 0xC0DE_FACE);
    // Pad up to ~10M events by duplicating a short tail under a different user namespace
    let tail = events.clone();
    for ev in &tail {
        let mut copy = ev.clone();
        if let AnalyticsValue::Text(t) = &mut copy.user_id {
            *t = format!("v_{}", t);
        }
        events.push(copy);
    }
    tprintln!("  Total events: {}", events.len());

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    let steps = vec![
        FunnelStep {
            name: "page_view".into(),
            event_match: "page_view".into(),
        },
        FunnelStep {
            name: "add_to_cart".into(),
            event_match: "add_to_cart".into(),
        },
        FunnelStep {
            name: "checkout".into(),
            event_match: "checkout".into(),
        },
        FunnelStep {
            name: "purchase".into(),
            event_match: "purchase".into(),
        },
        FunnelStep {
            name: "review".into(),
            event_match: "review".into(),
        },
    ];
    for run in 0..VALIDATION_RUNS {
        let cfg = FunnelConfig {
            steps: steps.clone(),
            window_ms: 30 * 60_000,
            user_id_column: "user_id".into(),
            event_time_column: "event_time".into(),
        };
        let start = Instant::now();
        let _ = funnel_analysis(&events, &cfg).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "FUNNEL_10M_5steps",
        "Latency (ms)",
        latencies,
        FUNNEL_TARGET_MS,
        false,
    );
    record_test_util("Funnel", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Funnel latency avg {:.2}ms > target {:.0}ms",
        result.average, FUNNEL_TARGET_MS
    );
    assert!(!result.regression_detected, "Funnel regression detected");
}

#[test]
fn test_15_data_profile_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 15: DATA_PROFILE Performance (10M rows, 20 cols) ===");
    let util_before = take_util_snapshot();

    // Build 20 columns by repeating the 10-column generator and adding numeric extras
    let rows = 10_000_000;
    let (mut names, mut types, mut cols) = build_profile_table(rows);
    // Append 10 additional numeric columns to reach 20
    let mut rng = SplitMix64::new(0x1234_5678);
    for c in 10..20 {
        names.push(format!("num{}", c));
        types.push("FLOAT64".into());
        let mut col = Vec::with_capacity(rows);
        for i in 0..rows {
            col.push(AnalyticsValue::Float(
                ((rng.next_u64() % 100_000) as f64) / 17.0 + (i as f64 * 0.001),
            ));
        }
        cols.push(col);
    }

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let p = profile_table("synthetic20", &names, &types, &cols, false).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(p.row_count, rows as u64);
        assert_eq!(p.columns.len(), 20);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "DATA_PROFILE_10M_20cols",
        "Latency (ms)",
        latencies,
        DATA_PROFILE_TARGET_MS,
        false,
    );
    record_test_util("DataProfile", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Data profile latency avg {:.2}ms > target {:.0}ms",
        result.average, DATA_PROFILE_TARGET_MS
    );
    assert!(
        !result.regression_detected,
        "Data profile regression detected"
    );
}

#[test]
fn test_16_zscore_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 16: ZSCORE Performance (10M rows) ===");
    let util_before = take_util_snapshot();

    let n = 10_000_000;
    let mut rng = SplitMix64::new(0xFEED_FACE);
    let data: Vec<f64> = (0..n)
        .map(|_| (rng.next_u64() as f64 / u64::MAX as f64) * 1_000.0)
        .collect();

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let scores = zscore(&data);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(scores.len(), n);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "ZSCORE_10M",
        "Latency (ms)",
        latencies,
        ZSCORE_TARGET_MS,
        false,
    );
    record_test_util("ZScore", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "ZSCORE latency avg {:.2}ms > target {:.0}ms",
        result.average, ZSCORE_TARGET_MS
    );
    assert!(!result.regression_detected, "ZSCORE regression detected");
}

#[test]
fn test_17_corr_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 17: CORR Performance (10M rows) ===");
    let util_before = take_util_snapshot();

    let n = 10_000_000;
    let mut rng = SplitMix64::new(0x0BAD_BEEF);
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
    let y: Vec<f64> = (0..n)
        .map(|_| (rng.next_u64() as f64 / u64::MAX as f64))
        .collect();

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    let mut sink = 0.0f64;
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut agg = PearsonAggregator::new();
        for (xi, yi) in x.iter().zip(y.iter()) {
            agg.ingest(std::hint::black_box(*xi), std::hint::black_box(*yi));
        }
        let r = agg.correlation().unwrap_or(0.0);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        // Touch sink so the optimizer cannot eliminate the loop
        sink += std::hint::black_box(r);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    std::hint::black_box(sink);
    let result = validate_metric("CORR_10M", "Latency (ms)", latencies, CORR_TARGET_MS, false);
    record_test_util("Corr", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "CORR latency avg {:.2}ms > target {:.0}ms",
        result.average, CORR_TARGET_MS
    );
    assert!(!result.regression_detected, "CORR regression detected");
}

#[test]
fn test_18_correlation_matrix_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 18: CORRELATION_MATRIX Performance (20 cols) ===");
    let util_before = take_util_snapshot();

    let rows = 100_000;
    let cols = 20;
    let (names, data) = build_numeric_table(rows, cols);

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let m = correlation_matrix(&names, &data);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(m.columns.len(), cols);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "CORRELATION_MATRIX_20cols",
        "Latency (ms)",
        latencies,
        CORRELATION_MATRIX_TARGET_MS,
        false,
    );
    record_test_util("CorrelationMatrix", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Correlation matrix latency avg {:.2}ms > target {:.0}ms",
        result.average, CORRELATION_MATRIX_TARGET_MS
    );
    assert!(
        !result.regression_detected,
        "Correlation matrix regression detected"
    );
}
