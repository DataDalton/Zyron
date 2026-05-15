#![allow(non_snake_case)]
// Integration tests across analytics modules

use zyron_analytics::ml::{decisionTree, kmeans, linearRegression, logisticRegression, randomForest};
use zyron_analytics::*;
use zyron_common::Xoshiro256pp;

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

// Helpers for feature-store and ML integration tests

fn fv(ts: i64, v: f64) -> FeatureValue {
    FeatureValue {
        computationTimestampMs: ts,
        validFromMs: ts,
        validToMs: i64::MAX,
        value: AnalyticsValue::Float(v),
        featureVersion: 1,
    }
}

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

#[test]
fn featureGroupComputesAndRetrieves() {
    let store = FeatureStore::new();
    store.registerFeatureGroup(buildOrdersFeatureGroup()).unwrap();
    for i in 0..100u32 {
        let entity = format!("u{:03}", i);
        store
            .writeFeatureValue("user_features", &entity, "total_purchases", fv(1000, (i as f64) * 10.0))
            .unwrap();
        store
            .writeFeatureValue("user_features", &entity, "order_count", fv(1000, i as f64))
            .unwrap();
    }
    let entities: Vec<String> = (0..100u32).map(|i| format!("u{:03}", i)).collect();
    let frame = store
        .getFeatures(
            "user_features",
            &entities,
            &["total_purchases".to_string(), "order_count".to_string()],
            2000,
        )
        .unwrap();
    assert_eq!(frame.rows.len(), 100);
    for (i, row) in frame.rows.iter().enumerate() {
        if let AnalyticsValue::Float(f) = &row.values[0] {
            assert!((f - (i as f64 * 10.0)).abs() < 1e-9);
        } else {
            panic!("expected float at row {}", i);
        }
    }
}

#[test]
fn pointInTimeJoinPreventsLeakage() {
    let store = FeatureStore::new();
    store.registerFeatureGroup(buildOrdersFeatureGroup()).unwrap();
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
}

#[test]
fn featureVersioningTracksChanges() {
    let store = FeatureStore::new();
    store.registerFeatureGroup(buildOrdersFeatureGroup()).unwrap();
    let mut v1 = fv(1000, 50.0);
    v1.featureVersion = 1;
    let mut v2 = fv(1000, 88.0);
    v2.featureVersion = 2;
    store.writeFeatureValue("user_features", "u1", "total_purchases", v1).unwrap();
    store.writeFeatureValue("user_features", "u1", "total_purchases", v2).unwrap();
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
    }
}

#[test]
fn lineageDetectsStaleness() {
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
}

#[test]
fn linearRegressionRecoversCoefficients() {
    let mut rng = Xoshiro256pp::fromSeed(101);
    let n = 1000;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 * 0.01;
        xs.push(x);
        ys.push(2.0 * x + 3.0 + 0.05 * rng.nextNormal());
    }
    let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".into()]);
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, n, 1);
    let model = linearRegression::train(&config, &data).unwrap();
    let r2 = model.metrics.get("r_squared").copied().unwrap();
    assert!(r2 > 0.95, "R^2 = {}", r2);
    let yh0 = linearRegression::predict(&model, &[0.0]);
    let yh10 = linearRegression::predict(&model, &[10.0]);
    assert!((yh0 - 3.0).abs() < 0.5, "intercept = {}", yh0);
    assert!((yh10 - 23.0).abs() < 0.5, "slope+intercept = {}", yh10);
}

#[test]
fn logisticRegressionAchievesAccuracy() {
    let mut rng = Xoshiro256pp::fromSeed(202);
    let n = 2000;
    let mut xs = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let cls = rng.nextRange(2) as f64;
        let mu = if cls == 1.0 { 1.5 } else { -1.5 };
        xs.push(mu + 0.6 * rng.nextNormal());
        xs.push(mu + 0.6 * rng.nextNormal());
        ys.push(cls);
    }
    let mut config = ModelConfig::new(
        ModelType::LogisticRegression,
        vec!["x1".into(), "x2".into()],
    );
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, n, 2);
    let model = logisticRegression::train(&config, &data).unwrap();
    let acc = model.metrics.get("accuracy").copied().unwrap();
    assert!(acc > 0.85, "logistic accuracy = {}", acc);
    let p = logisticRegression::predictProbability(&model, &[1.5, 1.5]);
    assert!(p > 0.5);
    assert!((0.0..=1.0).contains(&p));
}

#[test]
fn decisionTreeAchievesAccuracy() {
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
    let data = TrainingData::new(&xs, &ys, ys.len(), 4);
    let mut config = ModelConfig::new(
        ModelType::DecisionTreeClassification,
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
    );
    config.hyperparameters.setF64("max_depth", 8.0);
    let model = decisionTree::train(&config, &data).unwrap();
    let acc = model.metrics.get("accuracy").copied().unwrap();
    assert!(acc > 0.9, "tree accuracy = {}", acc);
    if let ModelData::Tree { nodes } = &model.data {
        assert!(nodes.len() < 100, "tree size = {}", nodes.len());
    }
}

#[test]
fn batchPredictionScalesUp() {
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
    predictBatch(&model, &testXs, mPredict, 1, &mut out);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in &out {
        assert!(v.is_finite());
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    assert!(max > min);
}

#[test]
fn causalAteRecoversTreatmentEffect() {
    let mut rng = Xoshiro256pp::fromSeed(505);
    let n = 2000;
    let mut cov = Vec::with_capacity(n * 2);
    let mut treat = Vec::with_capacity(n);
    let mut out = Vec::with_capacity(n);
    let trueEffect = 0.10;
    for _ in 0..n {
        let x1 = rng.nextNormal();
        let x2 = rng.nextNormal();
        cov.push(x1);
        cov.push(x2);
        let z = 0.5 * x1 - 0.3 * x2;
        let p = 1.0 / (1.0 + (-z).exp());
        let t = if rng.nextF64() < p { 1.0 } else { 0.0 };
        treat.push(t);
        out.push(trueEffect * t + 0.4 * x1 + 0.2 * x2 + 0.3 * rng.nextNormal());
    }
    let est = ate(&out, &treat, &cov, 2).unwrap();
    let attEst = att(&out, &treat, &cov, 2).unwrap();
    assert!((est - trueEffect).abs() < 0.4, "ATE = {} vs {}", est, trueEffect);
    assert!((attEst - trueEffect).abs() < 0.4, "ATT = {}", attEst);
    let probs = propensityScore(&treat, &cov, 2).unwrap();
    for p in probs {
        assert!(p > 0.0 && p < 1.0);
    }
    let ci = ateWithCi(&out, &treat, &cov, 2, 50, 0.05, 7).unwrap();
    assert!(ci.lowerCi <= ci.estimate);
    assert!(ci.upperCi >= ci.estimate);
}

#[test]
fn diffInDiffComputes() {
    let outcome = vec![1.0, 2.0, 1.0, 4.0];
    let treatment = vec![0.0, 0.0, 1.0, 1.0];
    let post = vec![0.0, 1.0, 0.0, 1.0];
    let did = diffInDiff(&outcome, &treatment, &post).unwrap();
    assert!((did - 2.0).abs() < 1e-12);
}

#[test]
fn forecastSinusoidalWithTrend() {
    let n = 200;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let trendPart = 0.1 * i as f64;
        let seasonal = (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin();
        v.push(trendPart + seasonal);
    }
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
    assert!(increases >= 15, "forecast trend not increasing: {}", increases);
    let aFcst = arima(&v, 10, 1, 1, 1).unwrap();
    assert_eq!(aFcst.len(), 10);
}

#[test]
fn anomalyAndDecomposeWork() {
    let mut v = vec![1.0; 100];
    v[50] = 100.0;
    let r = anomalyDetect(&v, AnomalyMethod::ZScore, 3.0);
    assert!(r[50].1);
    let s = seasonalDecompose(&v, 7);
    assert_eq!(s.trend.len(), 100);
    assert_eq!(s.residual.len(), 100);
}

#[test]
fn trendComputesSlope() {
    let v: Vec<f64> = (0..50).map(|i| 3.0 * i as f64 + 7.0).collect();
    let (slope, intercept) = trend(&v);
    assert!((slope - 3.0).abs() < 1e-9);
    assert!((intercept - 7.0).abs() < 1e-9);
}

#[test]
fn modelCacheServesPredictions() {
    let mut rng = Xoshiro256pp::fromSeed(606);
    let n = 500;
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64 * 0.01;
        xs.push(x);
        ys.push(2.0 * x + 3.0 + 0.05 * rng.nextNormal());
    }
    let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".into()]);
    config.targetColumn = Some("y".into());
    let data = TrainingData::new(&xs, &ys, n, 1);
    let model = linearRegression::train(&config, &data).unwrap();
    let cache = modelCache();
    cache.install("test_perf_model".into(), model);
    let h = InferenceHandle::resolve("test_perf_model").unwrap();
    let p = h.predictOne(&[5.0]);
    assert!((p - 13.0).abs() < 0.5, "p = {}", p);
    cache.invalidate("test_perf_model");
}

#[test]
fn kmeansPartitionsTwoBlobs() {
    let mut rng = Xoshiro256pp::fromSeed(707);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..200 {
        xs.push(-5.0 + 0.5 * rng.nextNormal());
        xs.push(-5.0 + 0.5 * rng.nextNormal());
        ys.push(0.0);
    }
    for _ in 0..200 {
        xs.push(5.0 + 0.5 * rng.nextNormal());
        xs.push(5.0 + 0.5 * rng.nextNormal());
        ys.push(1.0);
    }
    let data = TrainingData::new(&xs, &ys, ys.len(), 2);
    let mut config = ModelConfig::new(ModelType::KMeans, vec!["x".into(), "y".into()]);
    config.hyperparameters.setF64("k", 2.0);
    let model = kmeans::train(&config, &data).unwrap();
    assert_eq!(model.weights.len(), 4);
}

#[test]
fn knnAutoPromotesToAnn() {
    // Train with > ANN_PROMOTION_THRESHOLD (10K) rows to force HNSW build
    let mut rng = Xoshiro256pp::fromSeed(909);
    let n = 11_000usize;
    let p = 4usize;
    let mut xs = Vec::with_capacity(n * p);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let cls = rng.nextRange(2) as f64;
        let mu = if cls == 1.0 { 1.5 } else { -1.5 };
        for _ in 0..p {
            xs.push(mu + 0.5 * rng.nextNormal());
        }
        ys.push(cls);
    }
    let data = TrainingData::new(&xs, &ys, n, p);
    let mut config = ModelConfig::new(
        ModelType::KnnClassification,
        (0..p).map(|i| format!("x{}", i)).collect(),
    );
    config.hyperparameters.setF64("k", 5.0);
    let mut model = zyron_analytics::ml::knn::train(&config, &data).unwrap();
    model.modelId = "ann_knn_test".to_string();

    modelCache().install(model.modelId.clone(), model.clone());
    let h = zyron_analytics::InferenceHandle::resolve("ann_knn_test").unwrap();
    // First call triggers HNSW build
    let p1 = h.predictOne(&vec![1.5; p]);
    let p2 = h.predictOne(&vec![-1.5; p]);
    assert!(p1 == 0.0 || p1 == 1.0);
    assert!(p2 == 0.0 || p2 == 1.0);
    // ANN cache should now hold the built index
    let cache = zyron_analytics::ml::annKnn::knnAnnCache();
    assert!(cache.get("ann_knn_test").is_some());
    modelCache().invalidate("ann_knn_test");
    // Invalidation drops the ANN entry too
    assert!(cache.get("ann_knn_test").is_none());
}

#[test]
fn fftAcfMatchesDirect() {
    use zyron_analytics::fft::autocorrelationFft;
    // Generate a series large enough to trigger the FFT path
    let mut rng = Xoshiro256pp::fromSeed(606);
    let n = 8192usize;
    let mut v = Vec::with_capacity(n);
    let mut x = 0.0;
    for _ in 0..n {
        x = 0.6 * x + rng.nextNormal();
        v.push(x);
    }
    // Direct ACF for first few lags
    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = v.iter().map(|y| y - mean).collect();
    let var0: f64 = centered.iter().map(|y| y * y).sum();
    let mut direct = Vec::new();
    for lag in 0..=8 {
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += centered[i] * centered[i + lag];
        }
        direct.push(s / var0);
    }
    let fft = autocorrelationFft(&v, 8);
    for i in 0..=8 {
        assert!(
            (direct[i] - fft[i]).abs() < 1e-6,
            "lag {}: direct={} fft={}",
            i,
            direct[i],
            fft[i]
        );
    }
}

#[test]
fn randomForestClassifies() {
    let mut rng = Xoshiro256pp::fromSeed(808);
    let n = 800;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..n {
        let cls = rng.nextRange(2) as f64;
        let mu = if cls == 1.0 { 2.0 } else { -2.0 };
        xs.push(mu + 0.4 * rng.nextNormal());
        xs.push(mu + 0.4 * rng.nextNormal());
        ys.push(cls);
    }
    let data = TrainingData::new(&xs, &ys, n, 2);
    let mut config = ModelConfig::new(
        ModelType::RandomForestClassification,
        vec!["x1".into(), "x2".into()],
    );
    config.hyperparameters.setF64("n_trees", 16.0);
    let model = randomForest::train(&config, &data).unwrap();
    let acc = model.metrics.get("accuracy").copied().unwrap();
    assert!(acc > 0.95, "rf accuracy = {}", acc);
    let p = predictOne(&model, &[2.0, 2.0]);
    assert!(p == 0.0 || p == 1.0);
}
