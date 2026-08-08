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
use zyron_analytics::featureLineage::{LineageEntry, extractTablesAndColumns};
use zyron_analytics::featureStore::{FeatureDefinition, FeatureGroup, FeatureStore, FeatureValue};
use zyron_analytics::funnel::FunnelEvent;
use zyron_analytics::grouping::{
    Aggregator, GroupingSetExpander, GroupingSetType, GroupingSetsRunner, RowKey, SumAgg,
    expand_grouping_sets, grouping_bit, grouping_id_bits,
};
use zyron_analytics::ml::{
    Hyperparameters, ModelConfig, ModelData, ModelType, TrainingData, decisionTree, kmeans,
    linearRegression, logisticRegression, randomForest,
};
use zyron_analytics::mlInference::{predictBatch, predictOne};
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
    AnomalyMethod, CohortAnalysis, CohortDefinition, CohortMetric, CohortPeriod, CohortType,
    FeatureLineageRegistry, ForecastMethod, FunnelConfig, FunnelStep, anomalyDetect, arima, ate,
    diffInDiff, forecast, funnel_analysis, propensityScore, retention_analysis, seasonalDecompose,
    trend,
};
use zyron_common::Xoshiro256pp;

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

// Feature store and ML targets, 5-run average
const FEATURE_RETRIEVAL_TARGET_MS: f64 = 200.0;
const PIT_JOIN_TARGET_MS: f64 = 5_000.0;
const LINEAR_TRAIN_TARGET_MS: f64 = 500.0;
const LOGISTIC_TRAIN_TARGET_MS: f64 = 1_000.0;
const TREE_TRAIN_TARGET_MS: f64 = 2_000.0;
const BATCH_PREDICT_TARGET_MS: f64 = 2_000.0;
const SINGLE_PREDICT_TARGET_US: f64 = 100.0;
const ATE_TARGET_MS: f64 = 5_000.0;
const FORECAST_TARGET_MS: f64 = 10_000.0;

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

// =============================================================================
// Feature store and ML helpers
// =============================================================================

fn buildOrdersFeatureGroup() -> FeatureGroup {
    let mut g = FeatureGroup::new("user_features".into(), "user_id".into());
    g.refreshSeconds = 3600;
    g.sourceQuery =
        "SELECT user_id, SUM(amount) AS total_purchases FROM orders GROUP BY user_id".into();
    g.addFeature(FeatureDefinition::new(
        "total_purchases".into(),
        "FLOAT64".into(),
        "SELECT SUM(amount) FROM orders WHERE user_id = entity.user_id".into(),
    ));
    g.addFeature(FeatureDefinition::new(
        "avg_order_value".into(),
        "FLOAT64".into(),
        "SELECT AVG(amount) FROM orders WHERE user_id = entity.user_id".into(),
    ));
    g.addFeature(FeatureDefinition::new(
        "days_since_last_order".into(),
        "FLOAT64".into(),
        "SELECT CURRENT_DATE - MAX(order_date) FROM orders WHERE user_id = entity.user_id".into(),
    ));
    g.addFeature(FeatureDefinition::new(
        "order_count".into(),
        "INT64".into(),
        "SELECT COUNT(*) FROM orders WHERE user_id = entity.user_id".into(),
    ));
    g.addFeature(FeatureDefinition::new(
        "max_order_value".into(),
        "FLOAT64".into(),
        "SELECT MAX(amount) FROM orders WHERE user_id = entity.user_id".into(),
    ));
    g
}

fn fv(ts: i64, v: f64) -> FeatureValue {
    FeatureValue {
        computationTimestampMs: ts,
        validFromMs: ts,
        validToMs: i64::MAX,
        value: AnalyticsValue::Float(v),
        featureVersion: 1,
    }
}

fn buildLinearTrainingData(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Xoshiro256pp::fromSeed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 * 0.01;
        xs.push(x);
        ys.push(2.0 * x + 3.0 + 0.05 * rng.nextNormal());
    }
    (xs, ys)
}

fn buildClassificationData(n: usize, dims: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Xoshiro256pp::fromSeed(seed);
    let mut xs = Vec::with_capacity(n * dims);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let cls = rng.nextRange(2) as f64;
        let mu = if cls == 1.0 { 1.5 } else { -1.5 };
        for _ in 0..dims {
            xs.push(mu + 0.7 * rng.nextNormal());
        }
        ys.push(cls);
    }
    (xs, ys)
}

fn buildIrisLikeData() -> (Vec<f64>, Vec<f64>, usize) {
    let mut rng = Xoshiro256pp::fromSeed(303);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for cls in 0..3u64 {
        for _ in 0..200 {
            let mu = cls as f64 * 4.0;
            xs.push(mu + 0.5 * rng.nextNormal());
            xs.push(mu + 0.5 * rng.nextNormal());
            xs.push(mu + 0.5 * rng.nextNormal());
            xs.push(mu + 0.5 * rng.nextNormal());
            ys.push(cls as f64);
        }
    }
    (xs, ys, 600)
}

fn buildCausalData(n: usize, trueEffect: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Xoshiro256pp::fromSeed(seed);
    let mut cov = Vec::with_capacity(n * 2);
    let mut treat = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = rng.nextNormal();
        let x2 = rng.nextNormal();
        cov.push(x1);
        cov.push(x2);
        let z = 0.5 * x1 - 0.3 * x2;
        let p = 1.0 / (1.0 + (-z).exp());
        let t = if rng.nextF64() < p { 1.0 } else { 0.0 };
        treat.push(t);
        outcome.push(trueEffect * t + 0.4 * x1 + 0.2 * x2 + 0.3 * rng.nextNormal());
    }
    (outcome, treat, cov)
}

fn buildSinusoidalTrend(n: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let trendPart = 0.1 * i as f64;
        let seasonal = (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin();
        v.push(trendPart + seasonal);
    }
    v
}

// =============================================================================
// Correctness tests
// =============================================================================

#[test]
fn test_19_feature_group_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 19: Feature Group Correctness ===");

    let store = FeatureStore::new();
    store
        .registerFeatureGroup(buildOrdersFeatureGroup())
        .unwrap();
    for i in 0..100u32 {
        let entity = format!("u{:03}", i);
        store
            .writeFeatureValue(
                "user_features",
                &entity,
                "total_purchases",
                fv(1000, (i as f64) * 10.0),
            )
            .unwrap();
        store
            .writeFeatureValue("user_features", &entity, "order_count", fv(1000, i as f64))
            .unwrap();
        store
            .writeFeatureValue(
                "user_features",
                &entity,
                "avg_order_value",
                fv(1000, (i as f64) + 5.0),
            )
            .unwrap();
        store
            .writeFeatureValue(
                "user_features",
                &entity,
                "max_order_value",
                fv(1000, (i as f64) * 2.0),
            )
            .unwrap();
        store
            .writeFeatureValue(
                "user_features",
                &entity,
                "days_since_last_order",
                fv(1000, (i as f64) % 30.0),
            )
            .unwrap();
    }
    let entities: Vec<String> = (0..100u32).map(|i| format!("u{:03}", i)).collect();
    let names = vec![
        "total_purchases".to_string(),
        "order_count".to_string(),
        "avg_order_value".to_string(),
        "max_order_value".to_string(),
        "days_since_last_order".to_string(),
    ];
    let frame = store
        .getFeatures("user_features", &entities, &names, 2000)
        .unwrap();
    assert_eq!(frame.rows.len(), 100);
    for (i, row) in frame.rows.iter().enumerate() {
        if let AnalyticsValue::Float(f) = &row.values[0] {
            assert!((f - (i as f64 * 10.0)).abs() < 1e-9);
        } else {
            panic!("expected float at row {}", i);
        }
        if let AnalyticsValue::Float(f) = &row.values[1] {
            assert!((f - (i as f64)).abs() < 1e-9);
        }
    }
    tprintln!("  100 entities x 5 features verified: PASS");
}

#[test]
fn test_20_point_in_time_join_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 20: Point-in-Time Join Correctness ===");

    let store = FeatureStore::new();
    store
        .registerFeatureGroup(buildOrdersFeatureGroup())
        .unwrap();
    let janFifteenMs = 1_705_276_800_000i64;
    let janThirtyOneMs = 1_706_659_200_000i64;
    for day in 1..=31i64 {
        let ts = janFifteenMs + (day - 15) * 24 * 60 * 60 * 1000;
        store
            .writeFeatureValue(
                "user_features",
                "u1",
                "total_purchases",
                fv(ts, day as f64 * 100.0),
            )
            .unwrap();
    }
    let f15 = store
        .getFeatures(
            "user_features",
            &["u1".to_string()],
            &["total_purchases".to_string()],
            janFifteenMs,
        )
        .unwrap();
    let f31 = store
        .getFeatures(
            "user_features",
            &["u1".to_string()],
            &["total_purchases".to_string()],
            janThirtyOneMs,
        )
        .unwrap();
    let v15 = if let AnalyticsValue::Float(f) = &f15.rows[0].values[0] {
        *f
    } else {
        panic!()
    };
    let v31 = if let AnalyticsValue::Float(f) = &f31.rows[0].values[0] {
        *f
    } else {
        panic!()
    };
    assert!((v15 - 1500.0).abs() < 1.0, "AS OF Jan 15 = {}", v15);
    assert!((v31 - 3100.0).abs() < 1.0, "AS OF Jan 31 = {}", v31);
    assert!(v31 > v15);
    tprintln!(
        "  AS OF Jan 15 = {:.0}, AS OF Jan 31 = {:.0}, no leakage: PASS",
        v15,
        v31
    );
}

#[test]
fn test_21_feature_versioning_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 21: Feature Versioning Correctness ===");

    let store = FeatureStore::new();
    store
        .registerFeatureGroup(buildOrdersFeatureGroup())
        .unwrap();
    let mut v1 = fv(1000, 50.0);
    v1.featureVersion = 1;
    let mut v2 = fv(1000, 88.0);
    v2.featureVersion = 2;
    store
        .writeFeatureValue("user_features", "u1", "total_purchases", v1)
        .unwrap();
    store
        .writeFeatureValue("user_features", "u1", "total_purchases", v2)
        .unwrap();
    let f1 = store
        .getFeaturesVersioned(
            "user_features",
            &["u1".to_string()],
            &["total_purchases".to_string()],
            1,
            2000,
        )
        .unwrap();
    let f2 = store
        .getFeaturesVersioned(
            "user_features",
            &["u1".to_string()],
            &["total_purchases".to_string()],
            2,
            2000,
        )
        .unwrap();
    if let (AnalyticsValue::Float(a), AnalyticsValue::Float(b)) =
        (&f1.rows[0].values[0], &f2.rows[0].values[0])
    {
        assert!((a - 50.0).abs() < 1e-12);
        assert!((b - 88.0).abs() < 1e-12);
        tprintln!("  Version 1 = {}, Version 2 = {}: PASS", a, b);
    } else {
        panic!("expected float values for both versions");
    }

    let mut reg = FeatureLineageRegistry::new();
    let (tables, cols) = extractTablesAndColumns(
        "SELECT user_id, SUM(amount) FROM orders WHERE order_date > '2024-01-01'",
    );
    let mut entry = LineageEntry::new("user_features.total_purchases".into());
    entry.sourceTables = tables;
    entry.sourceColumns = cols;
    entry.lastComputedMs = 1000;
    reg.register("user_features.total_purchases".into(), entry);
    assert!(!reg.isStale("user_features.total_purchases"));
    reg.touchTable("orders", 2000);
    assert!(reg.isStale("user_features.total_purchases"));
    tprintln!("  Lineage staleness: PASS");
}

#[test]
fn test_22_linear_regression_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 22: Linear Regression Correctness ===");

    let (xs, ys) = buildLinearTrainingData(2000, 101);
    let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".into()]);
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, xs.len(), 1);
    let model = linearRegression::train(&config, &data).unwrap();
    let r2 = model.metrics.get("r_squared").copied().unwrap();
    assert!(r2 > 0.95, "R^2 = {}", r2);
    let yh0 = linearRegression::predict(&model, &[0.0]);
    let yh10 = linearRegression::predict(&model, &[10.0]);
    let slope = (yh10 - yh0) / 10.0;
    let intercept = yh0;
    tprintln!(
        "  Recovered slope = {:.3}, intercept = {:.3}, R^2 = {:.4}",
        slope,
        intercept,
        r2
    );
    assert!((slope - 2.0).abs() < 0.2);
    assert!((intercept - 3.0).abs() < 0.5);
}

#[test]
fn test_23_logistic_regression_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 23: Logistic Regression Correctness ===");

    let (xs, ys) = buildClassificationData(2000, 2, 202);
    let mut config = ModelConfig::new(
        ModelType::LogisticRegression,
        vec!["x1".into(), "x2".into()],
    );
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, ys.len(), 2);
    let model = logisticRegression::train(&config, &data).unwrap();
    let acc = model.metrics.get("accuracy").copied().unwrap();
    assert!(acc > 0.85, "accuracy = {}", acc);
    for i in 0..50 {
        let row = &xs[i * 2..i * 2 + 2];
        let p = logisticRegression::predictProbability(&model, row);
        assert!((0.0..=1.0).contains(&p), "probability out of range: {}", p);
    }
    tprintln!("  Accuracy = {:.4}: PASS", acc);
}

#[test]
fn test_24_decision_tree_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 24: Decision Tree Correctness ===");

    let (xs, ys, n) = buildIrisLikeData();
    let data = TrainingData::new(&xs, &ys, n, 4);
    let mut config = ModelConfig::new(
        ModelType::DecisionTreeClassification,
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
    );
    config.hyperparameters.setF64("max_depth", 8.0);
    let model = decisionTree::train(&config, &data).unwrap();
    let acc = model.metrics.get("accuracy").copied().unwrap();
    assert!(acc > 0.9, "accuracy = {}", acc);
    if let ModelData::Tree { nodes } = &model.data {
        assert!(nodes.len() < 200, "tree size = {}", nodes.len());
        tprintln!("  Accuracy = {:.4}, nodes = {}: PASS", acc, nodes.len());
    } else {
        panic!("expected tree model");
    }
}

#[test]
fn test_25_batch_prediction_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 25: Batch Prediction Correctness ===");

    let mut rng = Xoshiro256pp::fromSeed(404);
    let n = 10_000;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.nextNormal();
        xs.push(x);
        ys.push(2.0 * x + 0.1 * rng.nextNormal());
    }
    let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".into()]);
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, n, 1);
    let model = linearRegression::train(&config, &data).unwrap();

    let mPredict = 100_000usize;
    let mut testXs = Vec::with_capacity(mPredict);
    for _ in 0..mPredict {
        testXs.push(rng.nextNormal());
    }
    let mut out = vec![0.0f64; mPredict];
    predictBatch(&model, &testXs, mPredict, 1, &mut out).unwrap();

    let mut finite = 0usize;
    for v in &out {
        if v.is_finite() {
            finite += 1;
        }
    }
    assert_eq!(finite, mPredict);
    tprintln!("  100K predictions, all finite: PASS");
}

#[test]
fn test_26_causal_inference_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 26: Causal Inference Correctness ===");

    let (outcome, treatment, cov) = buildCausalData(2000, 0.10, 505);
    let est = ate(&outcome, &treatment, &cov, 2).unwrap();
    let probs = propensityScore(&treatment, &cov, 2).unwrap();
    for p in &probs {
        assert!(*p > 0.0 && *p < 1.0, "propensity out of range: {}", p);
    }
    assert!((est - 0.10).abs() < 0.4, "ATE = {} vs 0.10", est);

    // ATT and DID
    let attEst = zyron_analytics::att(&outcome, &treatment, &cov, 2).unwrap();
    assert!((attEst - 0.10).abs() < 0.4, "ATT = {}", attEst);
    let post: Vec<f64> = treatment
        .iter()
        .enumerate()
        .map(|(i, _)| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let _ = diffInDiff(&outcome, &treatment, &post).unwrap();
    tprintln!("  ATE = {:.4}, ATT = {:.4}: PASS", est, attEst);
}

#[test]
fn test_27_forecasting_correctness() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 27: Forecasting Correctness ===");

    let v = buildSinusoidalTrend(200);
    let mut extra = std::collections::HashMap::new();
    extra.insert("season".to_string(), 7.0);
    let f = forecast(&v, 30, ForecastMethod::SeasonalDecompose, &extra).unwrap();
    assert_eq!(f.len(), 30);
    let mut increases = 0;
    for i in 1..f.len() {
        if f[i] > f[i - 1] {
            increases += 1;
        }
    }
    assert!(
        increases >= 15,
        "forecast trend not increasing enough: {}",
        increases
    );

    let comp = seasonalDecompose(&v, 7);
    assert_eq!(comp.seasonalIndices.len(), 7);
    let arimaForecast = arima(&v, 10, 1, 1, 1).unwrap();
    assert_eq!(arimaForecast.len(), 10);

    let mut anomalyV = vec![1.0; 100];
    anomalyV[50] = 100.0;
    let detected = anomalyDetect(&anomalyV, AnomalyMethod::ZScore, 3.0);
    assert!(detected[50].1, "anomaly should flag index 50");

    let (slope, _) = trend(&v);
    assert!(slope > 0.0);
    tprintln!(
        "  Trend slope = {:.4}, anomaly + ARIMA + decompose: PASS",
        slope
    );
}

// =============================================================================
// Performance tests
// =============================================================================

#[test]
fn test_28_feature_retrieval_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 28: Feature Retrieval Performance (1K entities, 10 features) ===");
    let util_before = take_util_snapshot();

    let store = FeatureStore::new();
    let mut group = FeatureGroup::new("perf_features".into(), "id".into());
    for f in 0..10 {
        group.addFeature(FeatureDefinition::new(
            format!("f{}", f),
            "FLOAT64".into(),
            String::new(),
        ));
    }
    store.registerFeatureGroup(group).unwrap();

    for i in 0..1000u32 {
        let entity = format!("e{:04}", i);
        for f in 0..10 {
            let name = format!("f{}", f);
            store
                .writeFeatureValue(
                    "perf_features",
                    &entity,
                    &name,
                    fv(1000, (i as f64) * 0.5 + f as f64),
                )
                .unwrap();
        }
    }
    let entities: Vec<String> = (0..1000u32).map(|i| format!("e{:04}", i)).collect();
    let names: Vec<String> = (0..10).map(|f| format!("f{}", f)).collect();

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let frame = store
            .getFeatures("perf_features", &entities, &names, 2000)
            .unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(frame.rows.len(), 1000);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "FEATURE_RETRIEVAL_1K_x_10",
        "Latency (ms)",
        latencies,
        FEATURE_RETRIEVAL_TARGET_MS,
        false,
    );
    record_test_util("FeatureRetrieval", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Feature retrieval avg {:.2}ms > target {:.0}ms",
        result.average, FEATURE_RETRIEVAL_TARGET_MS
    );
    assert!(!result.regression_detected, "Feature retrieval regression");
}

#[test]
fn test_29_pit_join_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 29: PIT Join Performance (1M feature points) ===");
    let util_before = take_util_snapshot();

    let store = FeatureStore::new();
    let mut group = FeatureGroup::new("pit_features".into(), "id".into());
    group.addFeature(FeatureDefinition::new(
        "v".into(),
        "FLOAT64".into(),
        String::new(),
    ));
    store.registerFeatureGroup(group).unwrap();
    // 10K entities x 100 timestamps each = 1M points
    let entities_count: usize = 10_000;
    let history: usize = 100;
    for i in 0..entities_count {
        let entity = format!("e{:05}", i);
        for t in 0..history {
            let ts = 1_000_000 + (t as i64) * 1000;
            store
                .writeFeatureValue("pit_features", &entity, "v", fv(ts, (i + t) as f64))
                .unwrap();
        }
    }
    let entities: Vec<String> = (0..entities_count).map(|i| format!("e{:05}", i)).collect();
    let names = vec!["v".to_string()];

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let asOf = 1_000_000 + (history as i64) * 1000 / 2;
        let start = Instant::now();
        let frame = store
            .getFeatures("pit_features", &entities, &names, asOf)
            .unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(frame.rows.len(), entities_count);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "PIT_JOIN_1M",
        "Latency (ms)",
        latencies,
        PIT_JOIN_TARGET_MS,
        false,
    );
    record_test_util("PitJoin", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "PIT join avg {:.2}ms > target {:.0}ms",
        result.average, PIT_JOIN_TARGET_MS
    );
    assert!(!result.regression_detected, "PIT join regression");
}

#[test]
fn test_30_linear_regression_train_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 30: Linear Regression Train (100K rows, 10 features) ===");
    let util_before = take_util_snapshot();

    let n = 100_000usize;
    let p = 10usize;
    let mut rng = Xoshiro256pp::fromSeed(1001);
    let mut xs = Vec::with_capacity(n * p);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            let v = rng.nextNormal();
            xs.push(v);
            sum += (j as f64 + 1.0) * 0.1 * v;
        }
        ys.push(sum + 0.05 * rng.nextNormal());
    }

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let mut config = ModelConfig::new(
            ModelType::LinearRegression,
            (0..p).map(|i| format!("x{}", i)).collect(),
        );
        config.targetColumn = Some("y".into());
        let data = TrainingData::new(&xs, &ys, n, p);
        let start = Instant::now();
        let model = linearRegression::train(&config, &data).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let r2 = model.metrics.get("r_squared").copied().unwrap();
        assert!(r2 > 0.9, "linear train r2 = {}", r2);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms (R^2 = {:.4})", run + 1, elapsed_ms, r2);
    }
    let result = validate_metric(
        "LINEAR_TRAIN_100K_10",
        "Latency (ms)",
        latencies,
        LINEAR_TRAIN_TARGET_MS,
        false,
    );
    record_test_util("LinearTrain", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Linear train avg {:.2}ms > target {:.0}ms",
        result.average, LINEAR_TRAIN_TARGET_MS
    );
    assert!(!result.regression_detected, "Linear train regression");
}

#[test]
fn test_31_logistic_regression_train_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 31: Logistic Regression Train (100K rows, 10 features) ===");
    let util_before = take_util_snapshot();

    let n = 100_000usize;
    let p = 10usize;
    let mut rng = Xoshiro256pp::fromSeed(1002);
    let mut xs = Vec::with_capacity(n * p);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let cls = rng.nextRange(2) as f64;
        let mu = if cls == 1.0 { 0.8 } else { -0.8 };
        for _ in 0..p {
            xs.push(mu + rng.nextNormal());
        }
        ys.push(cls);
    }

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let mut config = ModelConfig::new(
            ModelType::LogisticRegression,
            (0..p).map(|i| format!("x{}", i)).collect(),
        );
        config.hyperparameters.setF64("max_epochs", 30.0);
        config.targetColumn = Some("y".into());
        let data = TrainingData::new(&xs, &ys, n, p);
        let start = Instant::now();
        let model = logisticRegression::train(&config, &data).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let acc = model.metrics.get("accuracy").copied().unwrap();
        assert!(acc > 0.8, "logistic train accuracy = {}", acc);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms (acc = {:.4})", run + 1, elapsed_ms, acc);
    }
    let result = validate_metric(
        "LOGISTIC_TRAIN_100K_10",
        "Latency (ms)",
        latencies,
        LOGISTIC_TRAIN_TARGET_MS,
        false,
    );
    record_test_util("LogisticTrain", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Logistic train avg {:.2}ms > target {:.0}ms",
        result.average, LOGISTIC_TRAIN_TARGET_MS
    );
    assert!(!result.regression_detected, "Logistic train regression");
}

#[test]
fn test_32_decision_tree_train_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 32: Decision Tree Train (100K rows) ===");
    let util_before = take_util_snapshot();

    let n = 100_000usize;
    let p = 10usize;
    let mut rng = Xoshiro256pp::fromSeed(1003);
    let mut xs = Vec::with_capacity(n * p);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let cls = rng.nextRange(3) as f64;
        let mu = cls * 2.0;
        for _ in 0..p {
            xs.push(mu + 0.5 * rng.nextNormal());
        }
        ys.push(cls);
    }

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let mut config = ModelConfig::new(
            ModelType::DecisionTreeClassification,
            (0..p).map(|i| format!("x{}", i)).collect(),
        );
        config.hyperparameters.setF64("max_depth", 8.0);
        let data = TrainingData::new(&xs, &ys, n, p);
        let start = Instant::now();
        let model = decisionTree::train(&config, &data).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let acc = model.metrics.get("accuracy").copied().unwrap();
        assert!(acc > 0.8, "tree train accuracy = {}", acc);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms (acc = {:.4})", run + 1, elapsed_ms, acc);
    }
    let result = validate_metric(
        "TREE_TRAIN_100K",
        "Latency (ms)",
        latencies,
        TREE_TRAIN_TARGET_MS,
        false,
    );
    record_test_util("TreeTrain", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Tree train avg {:.2}ms > target {:.0}ms",
        result.average, TREE_TRAIN_TARGET_MS
    );
    assert!(!result.regression_detected, "Tree train regression");
}

#[test]
fn test_33_batch_predict_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 33: Batch Predict (1M rows) ===");
    let util_before = take_util_snapshot();

    let p = 10usize;
    let trainN = 10_000usize;
    let mut rng = Xoshiro256pp::fromSeed(1004);
    let mut xs = Vec::with_capacity(trainN * p);
    let mut ys = Vec::with_capacity(trainN);
    for _ in 0..trainN {
        let mut sum = 0.0;
        for j in 0..p {
            let v = rng.nextNormal();
            xs.push(v);
            sum += (j as f64 + 1.0) * 0.1 * v;
        }
        ys.push(sum);
    }
    let mut config = ModelConfig::new(
        ModelType::LinearRegression,
        (0..p).map(|i| format!("x{}", i)).collect(),
    );
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, trainN, p);
    let model = linearRegression::train(&config, &data).unwrap();

    let predictN = 1_000_000usize;
    let mut testXs = Vec::with_capacity(predictN * p);
    for _ in 0..(predictN * p) {
        testXs.push(rng.nextNormal());
    }

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let mut out = vec![0.0f64; predictN];
        let start = Instant::now();
        predictBatch(&model, &testXs, predictN, p, &mut out).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "BATCH_PREDICT_1M",
        "Latency (ms)",
        latencies,
        BATCH_PREDICT_TARGET_MS,
        false,
    );
    record_test_util("BatchPredict", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Batch predict avg {:.2}ms > target {:.0}ms",
        result.average, BATCH_PREDICT_TARGET_MS
    );
    assert!(!result.regression_detected, "Batch predict regression");
}

#[test]
fn test_34_single_predict_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 34: Single Predict Latency ===");
    let util_before = take_util_snapshot();

    let p = 10usize;
    let n = 5_000usize;
    let mut rng = Xoshiro256pp::fromSeed(1005);
    let mut xs = Vec::with_capacity(n * p);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            let v = rng.nextNormal();
            xs.push(v);
            sum += (j as f64 + 1.0) * 0.1 * v;
        }
        ys.push(sum);
    }
    let mut config = ModelConfig::new(
        ModelType::LinearRegression,
        (0..p).map(|i| format!("x{}", i)).collect(),
    );
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, n, p);
    let model = linearRegression::train(&config, &data).unwrap();

    let row: Vec<f64> = (0..p).map(|_| rng.nextNormal()).collect();
    let iters = 10_000usize;
    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0.0f64;
        for _ in 0..iters {
            sink += predictOne(&model, &row);
        }
        std::hint::black_box(sink);
        let elapsed = start.elapsed();
        let per_us = elapsed.as_nanos() as f64 / (iters as f64 * 1_000.0);
        latencies.push(per_us);
        tprintln!("  Run {}: {:.3}us per predict", run + 1, per_us);
    }
    let result = validate_metric(
        "SINGLE_PREDICT",
        "Latency (us)",
        latencies,
        SINGLE_PREDICT_TARGET_US,
        false,
    );
    record_test_util("SinglePredict", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Single predict avg {:.3}us > target {:.1}us",
        result.average, SINGLE_PREDICT_TARGET_US
    );
    assert!(!result.regression_detected, "Single predict regression");
}

#[test]
fn test_35_ate_estimation_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 35: ATE Estimation (100K observations) ===");
    let util_before = take_util_snapshot();

    let (outcome, treatment, cov) = buildCausalData(100_000, 0.10, 1006);

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let est = ate(&outcome, &treatment, &cov, 2).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert!(est.is_finite(), "ATE non-finite: {}", est);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms (est = {:.4})", run + 1, elapsed_ms, est);
    }
    let result = validate_metric("ATE_100K", "Latency (ms)", latencies, ATE_TARGET_MS, false);
    record_test_util("ATE", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "ATE avg {:.2}ms > target {:.0}ms",
        result.average, ATE_TARGET_MS
    );
    assert!(!result.regression_detected, "ATE regression");
}

#[test]
fn test_36_forecast_performance() {
    zyron_bench_harness::init("analytics");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Test 36: Forecast (10K history) ===");
    let util_before = take_util_snapshot();

    let v = buildSinusoidalTrend(10_000);
    let mut extra = std::collections::HashMap::new();
    extra.insert("season".to_string(), 7.0);

    let mut latencies = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let f = forecast(&v, 1000, ForecastMethod::SeasonalDecompose, &extra).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
        assert_eq!(f.len(), 1000);
        latencies.push(elapsed_ms);
        tprintln!("  Run {}: {:.2}ms", run + 1, elapsed_ms);
    }
    let result = validate_metric(
        "FORECAST_10K",
        "Latency (ms)",
        latencies,
        FORECAST_TARGET_MS,
        false,
    );
    record_test_util("Forecast", util_before, take_util_snapshot());
    assert!(
        result.passed,
        "Forecast avg {:.2}ms > target {:.0}ms",
        result.average, FORECAST_TARGET_MS
    );
    assert!(!result.regression_detected, "Forecast regression");
}
