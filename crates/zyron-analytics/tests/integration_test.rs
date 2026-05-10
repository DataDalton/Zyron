// Integration tests across analytics modules

use zyron_analytics::*;

#[test]
fn cube_grouping_set_aggregates_in_one_pass() {
    let schema = vec![
        "region".to_string(),
        "category".to_string(),
        "rev".to_string(),
    ];
    let spec = GroupingSetType::Cube(vec!["region".into(), "category".into()]);
    let mut runner = GroupingSetsRunner::new(schema, spec, grouping::SumAgg::new(2));
    let rows = vec![
        vec![
            AnalyticsValue::Text("US".into()),
            AnalyticsValue::Text("A".into()),
            AnalyticsValue::Float(100.0),
        ],
        vec![
            AnalyticsValue::Text("US".into()),
            AnalyticsValue::Text("B".into()),
            AnalyticsValue::Float(50.0),
        ],
        vec![
            AnalyticsValue::Text("EU".into()),
            AnalyticsValue::Text("A".into()),
            AnalyticsValue::Float(75.0),
        ],
    ];
    for r in &rows {
        runner.ingest_row(r);
    }
    let out = runner.finalise();
    // 4 grouping sets * (3 (region,cat) + 2 (region) + 2 (cat) + 1 ()) = 8 keys
    let total_rev: f64 = out
        .iter()
        .filter(|o| o.active_columns.is_empty())
        .map(|o| match o.aggregate {
            AnalyticsValue::Float(v) => v,
            _ => 0.0,
        })
        .sum();
    assert!((total_rev - 225.0).abs() < 1e-9);
}

#[test]
fn registry_has_all_required_analytics_functions() {
    let r = default_registry();
    for name in [
        "COHORT_RETENTION",
        "FUNNEL_ANALYSIS",
        "DATA_PROFILE",
        "COLUMN_PROFILE",
        "CORRELATION_MATRIX",
        "YOY",
        "YOY_GROWTH",
        "MOM",
        "MOM_GROWTH",
        "WOW",
        "QOQ",
        "SAME_PERIOD_LAST_YEAR",
        "PERIOD_COMPARE",
        "YTD_SUM",
        "QTD_SUM",
        "MTD_SUM",
        "ZSCORE",
        "IQR_OUTLIER",
        "MAD_OUTLIER",
        "CORR",
        "SPEARMAN_CORR",
        "KENDALL_TAU",
        "MUTUAL_INFORMATION",
        "GROUPING",
        "GROUPING_ID",
    ] {
        assert!(r.lookup(name).is_some(), "missing function {}", name);
    }
}

#[test]
fn profile_table_with_correlation_matrix() {
    let names = vec!["x".to_string(), "y".to_string()];
    let types = vec!["FLOAT64".to_string(), "FLOAT64".to_string()];
    let xs: Vec<AnalyticsValue> = (0..1000).map(|i| AnalyticsValue::Float(i as f64)).collect();
    let ys: Vec<AnalyticsValue> = (0..1000)
        .map(|i| AnalyticsValue::Float(2.0 * i as f64 + 1.0))
        .collect();
    let p = profile_table("test", &names, &types, &[xs, ys], true).unwrap();
    assert_eq!(p.row_count, 1000);
    assert_eq!(p.columns.len(), 2);
    let m = p.correlation_matrix.as_ref().unwrap();
    let xy = m.get(0, 1);
    assert!(
        (xy - 1.0).abs() < 1e-6,
        "correlation should be ~1.0, got {}",
        xy
    );
}

#[test]
fn end_to_end_funnel_with_window() {
    let events = vec![
        funnel::FunnelEvent {
            user_id: AnalyticsValue::Text("a".into()),
            event_time_ms: 0,
            event_name: "view".into(),
        },
        funnel::FunnelEvent {
            user_id: AnalyticsValue::Text("a".into()),
            event_time_ms: 10_000,
            event_name: "buy".into(),
        },
        funnel::FunnelEvent {
            user_id: AnalyticsValue::Text("b".into()),
            event_time_ms: 0,
            event_name: "view".into(),
        },
    ];
    let cfg = FunnelConfig {
        steps: vec![
            FunnelStep {
                name: "view".into(),
                event_match: "view".into(),
            },
            FunnelStep {
                name: "buy".into(),
                event_match: "buy".into(),
            },
        ],
        window_ms: 60_000,
        user_id_column: "u".into(),
        event_time_column: "t".into(),
    };
    let r = funnel_analysis(&events, &cfg).unwrap();
    assert_eq!(r.steps[0].users_count, 2);
    assert_eq!(r.steps[1].users_count, 1);
    assert!((r.overall_conversion - 0.5).abs() < 1e-9);
}
