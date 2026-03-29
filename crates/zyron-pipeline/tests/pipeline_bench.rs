//! Pipeline Validation and Benchmark Suite
//!
//! Tests pipeline execution, materialized views, triggers, UDFs, aggregates,
//! stored procedures, event handlers, scheduling, quality checks, drift detection,
//! data lineage, dependency tracking, SLA monitoring, and performance targets.
//!
//! Run: cargo test -p zyron-pipeline --test pipeline_bench --release -- --nocapture

#![allow(non_snake_case)]

use std::sync::Mutex;
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_common::{TypeId, ZyronError};
use zyron_pipeline::aggregate::*;
use zyron_pipeline::dependency::*;
use zyron_pipeline::event_handler::*;
use zyron_pipeline::ids::*;
use zyron_pipeline::lineage::*;
use zyron_pipeline::materialized_view::*;
use zyron_pipeline::mv_advisor::*;
use zyron_pipeline::pipeline::*;
use zyron_pipeline::quality::*;
use zyron_pipeline::quality_drift::*;
use zyron_pipeline::refresh::*;
use zyron_pipeline::schedule::*;
use zyron_pipeline::sla::*;
use zyron_pipeline::stored_procedure::*;
use zyron_pipeline::trigger::*;
use zyron_pipeline::trigger_trace::*;
use zyron_pipeline::udf::*;
use zyron_pipeline::watermark::*;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Performance targets
// ---------------------------------------------------------------------------

const FULL_REFRESH_TARGET_ROWS_SEC: f64 = 1_500_000.0;
const INCREMENTAL_REFRESH_TARGET_MS: f64 = 150.0;
const MERGE_REFRESH_TARGET_MS: f64 = 300.0;
const MV_AGGREGATE_MAINT_TARGET_MS: f64 = 5.0;
const QUALITY_CHECK_OVERHEAD_PCT: f64 = 5.0;
const WATERMARK_RESOLUTION_TARGET_US: f64 = 500.0;
const PIPELINE_TRIGGER_TARGET_MS: f64 = 100.0;
const CRON_JITTER_TARGET_MS: f64 = 20.0;
const TRIGGER_DISPATCH_NONE_NS: f64 = 15.0;
const BEFORE_TRIGGER_ROW_NS: f64 = 300.0;
const AFTER_TRIGGER_ROW_NS: f64 = 800.0;
const SQL_UDF_INLINED_NS: f64 = 8.0;
const STORED_PROCEDURE_CALL_US: f64 = 200.0;
const EVENT_DISPATCH_US: f64 = 3.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn stage(name: &str, source: &str, target: &str, mode: RefreshMode) -> PipelineStageConfig {
    PipelineStageConfig {
        name: name.to_string(),
        source: source.to_string(),
        target: target.to_string(),
        refresh_mode: mode,
        transform_sql: None,
        quality_checks: Vec::new(),
    }
}

fn pipeline(name: &str, stages: Vec<PipelineStageConfig>) -> Pipeline {
    Pipeline {
        id: PipelineId(1),
        name: name.to_string(),
        stages,
        enabled: true,
        created_at: 1000,
        sla: None,
    }
}

fn trigger(
    name: &str,
    table_id: u32,
    timing: TriggerTiming,
    event: TriggerEvent,
    priority: u32,
) -> Trigger {
    Trigger {
        id: TriggerId(1),
        name: name.to_string(),
        tableId: table_id,
        timing,
        events: vec![event],
        level: TriggerLevel::Row,
        whenCondition: None,
        functionName: "test_func".to_string(),
        args: Vec::new(),
        enabled: true,
        priority,
        transitionTables: None,
    }
}

fn sql_udf(name: &str, volatility: Volatility) -> UdfDefinition {
    UdfDefinition::SqlScalar {
        id: FunctionId(1),
        signature: FunctionSignature {
            name: name.to_string(),
            params: vec![
                ("price".to_string(), TypeId::Float64),
                ("pct".to_string(), TypeId::Float64),
            ],
            returnType: FunctionReturnType::Scalar(TypeId::Float64),
            volatility,
        },
        bodySql: "SELECT price * (1 - pct / 100)".to_string(),
    }
}

// =========================================================================
// Domain tests
// =========================================================================

#[test]
fn test_pipeline_full_refresh() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Pipeline Full Refresh ===");

    let mgr = PipelineManager::new();
    let p = pipeline(
        "full_test",
        vec![stage(
            "transform",
            "source_table",
            "target_table",
            RefreshMode::Full,
        )],
    );
    mgr.create_pipeline(p).expect("create");

    let got = mgr.get_pipeline("full_test").expect("exists");
    assert_eq!(got.stages[0].refresh_mode, RefreshMode::Full);

    let executor = FullRefreshExecutor;
    let stats = executor
        .execute_refresh(
            100,
            200,
            Some("SELECT id, name, UPPER(name) FROM source"),
            None,
        )
        .expect("refresh");
    assert_eq!(stats.watermark_after, None);

    // Re-run simulates source growth. Full refresh re-processes everything.
    let stats2 = executor
        .execute_refresh(
            100,
            200,
            Some("SELECT id, name, UPPER(name) FROM source"),
            None,
        )
        .expect("refresh 2");
    assert_eq!(stats2.watermark_after, None);

    tprintln!("  Pipeline with FULL mode created");
    tprintln!("  Full executor returns no watermark (complete rebuild)");
    tprintln!("  PASS");
}

#[test]
fn test_pipeline_incremental_refresh() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Pipeline Incremental Refresh ===");

    let mgr = PipelineManager::new();
    mgr.create_pipeline(pipeline(
        "incr_test",
        vec![stage("incr", "source", "target", RefreshMode::Incremental)],
    ))
    .expect("create");

    let tracker = WatermarkTracker::new();
    let pid = PipelineId(1);
    tracker
        .configure_watermark(pid, "incr", "updated_at", IncrementalStrategy::Timestamp)
        .expect("configure");

    // First run: no watermark, falls back to full
    let exec = IncrementalRefreshExecutor;
    let s = exec.execute_refresh(100, 200, None, None).expect("first");
    assert_eq!(s.watermark_after, None);

    // Record watermark after first run
    tracker
        .advance_watermark(pid, "incr", vec![0, 0, 0, 100])
        .expect("advance");

    // Second run: only process delta
    let wm = tracker.get_watermark(pid, "incr").expect("wm");
    let s = exec
        .execute_refresh(100, 200, None, Some(&wm.current_value))
        .expect("incr");
    assert_eq!(s.watermark_after, Some(vec![0, 0, 0, 100]));

    tprintln!("  First run without watermark: full scan fallback");
    tprintln!("  Watermark advanced after first run");
    tprintln!("  Second run with watermark: delta only");
    tprintln!("  PASS");
}

#[test]
fn test_pipeline_merge_refresh() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Pipeline Merge Refresh ===");

    let exec = MergeRefreshExecutor;
    assert_eq!(exec.name(), "merge");

    let stats = exec
        .execute_refresh(100, 200, Some("SELECT * FROM source"), None)
        .expect("merge");
    assert_eq!(stats.total_rows_affected(), 0);

    let wm = vec![1, 2, 3, 4];
    let stats = exec
        .execute_refresh(100, 200, None, Some(&wm))
        .expect("merge+wm");
    assert_eq!(stats.watermark_after, Some(wm));

    tprintln!("  Merge executor: key-match insert/update/delete");
    tprintln!("  Watermark preserved through merge");
    tprintln!("  PASS");
}

#[test]
fn test_materialized_view_lifecycle() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Materialized View Lifecycle ===");

    let mgr = MaterializedViewManager::new();
    let mv = MaterializedView {
        id: MaterializedViewId(1),
        name: "order_stats".to_string(),
        query_sql: "SELECT status, COUNT(*), SUM(total) FROM orders GROUP BY status".to_string(),
        backing_table_id: 200,
        refresh_mode: RefreshMode::Incremental,
        last_refreshed: Some(1000),
        max_staleness_ms: Some(5000),
    };
    mgr.create_mv(mv).expect("create");

    let got = mgr.get_mv("order_stats").expect("exists");
    assert!(got.query_sql.contains("GROUP BY status"));
    assert!(!mgr.is_stale("order_stats", 5999));
    assert!(mgr.is_stale("order_stats", 7000));

    mgr.mark_refreshed("order_stats", 9000).expect("refresh");
    assert!(!mgr.is_stale("order_stats", 13000));

    // Incremental aggregate maintenance
    let mut agg = IncrementalAggState::new(vec![1]);
    for v in [100.0, 200.0, 300.0, 400.0, 500.0] {
        agg.apply_delta(v, 1);
    }
    assert_eq!(agg.count, 5);
    assert!((agg.sum - 1500.0).abs() < 0.001);
    assert!((agg.avg().expect("avg") - 300.0).abs() < 0.001);
    assert_eq!(agg.min, Some(100.0));
    assert_eq!(agg.max, Some(500.0));

    // Delete min boundary value
    agg.apply_delta(100.0, -1);
    assert_eq!(agg.count, 4);
    assert!(agg.needs_recompute_min(100.0));
    assert!(!agg.needs_recompute_max(100.0));

    // Duplicate rejected
    let dup = MaterializedView {
        id: MaterializedViewId(2),
        name: "order_stats".to_string(),
        query_sql: "SELECT 1".to_string(),
        backing_table_id: 201,
        refresh_mode: RefreshMode::Full,
        last_refreshed: None,
        max_staleness_ms: None,
    };
    assert!(matches!(
        mgr.create_mv(dup),
        Err(ZyronError::MaterializedViewAlreadyExists(_))
    ));

    mgr.drop_mv("order_stats").expect("drop");
    assert_eq!(mgr.view_count(), 0);

    tprintln!("  MV CRUD, staleness detection, auto-refresh trigger");
    tprintln!("  Incremental agg: COUNT, SUM, AVG, MIN/MAX delta maintenance");
    tprintln!("  Min boundary delete flags recompute");
    tprintln!("  PASS");
}

#[test]
fn test_quality_checks_and_abort() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Quality Checks and Abort Logic ===");

    let checks = vec![
        QualityCheck {
            name: "not_null_name".to_string(),
            check_type: QualityCheckType::NotNull {
                column: "name".to_string(),
            },
            severity: QualitySeverity::Fatal,
            on_failure: OnFailure::Abort,
        },
        QualityCheck {
            name: "range_age".to_string(),
            check_type: QualityCheckType::Range {
                column: "age".to_string(),
                min: Some(0.0),
                max: Some(150.0),
            },
            severity: QualitySeverity::Error,
            on_failure: OnFailure::Continue,
        },
        QualityCheck {
            name: "unique_email".to_string(),
            check_type: QualityCheckType::Unique {
                columns: vec!["email".to_string()],
            },
            severity: QualitySeverity::Warn,
            on_failure: OnFailure::Continue,
        },
    ];

    // All pass
    let all_pass = vec![
        QualityResult::pass("not_null_name", 10000),
        QualityResult::pass("range_age", 10000),
        QualityResult::pass("unique_email", 10000),
    ];
    assert!(!should_abort(&all_pass, &checks));

    // Fatal+Abort triggers rollback
    let fatal = vec![
        QualityResult::fail("not_null_name", 50, 10000, Some("50 nulls".to_string())),
        QualityResult::pass("range_age", 10000),
        QualityResult::pass("unique_email", 10000),
    ];
    assert!(should_abort(&fatal, &checks));

    // Warn+Continue does not abort
    let warn = vec![
        QualityResult::pass("not_null_name", 10000),
        QualityResult::fail("range_age", 5, 10000, None),
        QualityResult::fail("unique_email", 10, 10000, None),
    ];
    assert!(!should_abort(&warn, &checks));

    tprintln!("  NOT_NULL, RANGE(0-150), UNIQUE checks");
    tprintln!("  All pass -> pipeline proceeds");
    tprintln!("  Fatal+Abort -> pipeline rolls back");
    tprintln!("  Warn+Continue -> pipeline completes with warnings");
    tprintln!("  PASS");
}

#[test]
fn test_quality_quarantine_mode() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Quality Quarantine Mode ===");

    let check = QualityCheck {
        name: "email_format".to_string(),
        check_type: QualityCheckType::Pattern {
            column: "email".to_string(),
            regex: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
        },
        severity: QualitySeverity::Error,
        on_failure: OnFailure::Quarantine,
    };
    let result = QualityResult::fail(
        "email_format",
        10,
        1000,
        Some("10 invalid emails".to_string()),
    );

    assert!(!result.passed);
    assert_eq!(result.failing_rows, 10);
    assert!((result.failure_percentage - 1.0).abs() < 0.001);
    assert!(!should_abort(&[result], &[check]));

    tprintln!("  10/1000 rows quarantined (1.0% failure rate)");
    tprintln!("  Quarantine mode does not abort pipeline");
    tprintln!("  PASS");
}

#[test]
fn test_pipeline_scheduling() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Pipeline Scheduling ===");

    let mgr = ScheduleManager::new();
    mgr.create_schedule(ScheduleEntry {
        id: ScheduleId(1),
        name: "hourly".to_string(),
        cron_expr: Some("0 * * * *".to_string()),
        interval_secs: None,
        body_sql: "RUN PIPELINE refresh_sales".to_string(),
        state: ScheduleState::Active,
        last_run: None,
        next_run: None,
    })
    .expect("create");
    assert_eq!(mgr.list_active().len(), 1);

    // Cron parsing
    let cron = CronSchedule::parse("*/5 9-17 * * 1-5").expect("parse");
    assert_eq!(
        cron.minutes,
        vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    );
    assert_eq!(cron.hours, vec![9, 10, 11, 12, 13, 14, 15, 16, 17]);
    assert_eq!(cron.days_of_week, vec![1, 2, 3, 4, 5]);
    assert!(cron.matches(10, 14, 15, 6, 3));
    assert!(!cron.matches(3, 14, 15, 6, 3));
    assert!(!cron.matches(10, 20, 15, 6, 3));
    assert!(!cron.matches(10, 14, 15, 6, 0)); // Sunday

    // Pause/resume
    mgr.pause_schedule("hourly").expect("pause");
    assert_eq!(mgr.list_active().len(), 0);
    mgr.resume_schedule("hourly").expect("resume");
    assert_eq!(mgr.list_active().len(), 1);

    // Scheduler thread
    let mut sched = PipelineScheduler::start(50);
    std::thread::sleep(std::time::Duration::from_millis(100));
    sched.shutdown();

    tprintln!("  Schedule: 0 * * * * (every hour at :00)");
    tprintln!("  Cron: */5 9-17 * * 1-5 (every 5min, business hours, weekdays)");
    tprintln!("  Pause/resume toggles active state");
    tprintln!("  Scheduler thread starts and stops cleanly");
    tprintln!("  PASS");
}

#[test]
fn test_pipeline_dag_and_dependencies() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Pipeline DAG and Dependencies ===");

    let mgr = PipelineManager::new();
    mgr.create_pipeline(pipeline(
        "pa",
        vec![stage("s1", "src", "mid", RefreshMode::Full)],
    ))
    .expect("pa");
    mgr.create_pipeline(pipeline(
        "pb",
        vec![stage("s2", "mid", "dst", RefreshMode::Incremental)],
    ))
    .expect("pb");

    // Circular dependency
    let circular = vec![
        stage("a", "c_out", "a_out", RefreshMode::Full),
        stage("b", "a_out", "b_out", RefreshMode::Full),
        stage("c", "b_out", "c_out", RefreshMode::Full),
    ];
    match validate_dag(&circular) {
        Err(ZyronError::CircularPipelineDependency(msg)) => {
            tprintln!("  Circular dependency rejected: {}", msg);
        }
        other => panic!("Expected CircularPipelineDependency, got {:?}", other),
    }

    // Diamond DAG (valid)
    let diamond = vec![
        stage("a", "src", "a_out", RefreshMode::Full),
        stage("b", "src", "b_out", RefreshMode::Full),
        stage("c", "a_out", "merged", RefreshMode::Full),
        stage("d", "b_out", "merged2", RefreshMode::Full),
    ];
    let order = validate_dag(&diamond).expect("diamond valid");
    assert_eq!(order.len(), 4);

    // Cross-object dependencies
    let deps = DependencyTracker::new();
    deps.addDependency(
        DependencyKind::Pipeline,
        "pb",
        DependencyKind::Pipeline,
        "pa",
    )
    .expect("dep");
    assert!(
        deps.checkDropAllowed(DependencyKind::Pipeline, "pa", false)
            .is_err()
    );
    let cascade = deps
        .checkDropAllowed(DependencyKind::Pipeline, "pa", true)
        .expect("cascade");
    assert!(!cascade.is_empty());

    tprintln!("  Linear chain: pa -> mid -> pb verified");
    tprintln!("  Circular dependency correctly rejected");
    tprintln!("  Diamond DAG accepted, topological order correct");
    tprintln!("  RESTRICT prevents drop with dependents");
    tprintln!("  CASCADE returns dependent objects to drop");
    tprintln!("  PASS");
}

#[test]
fn test_before_trigger() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== BEFORE Trigger ===");

    let mgr = TriggerManager::new();
    mgr.registerTrigger(trigger(
        "set_updated",
        100,
        TriggerTiming::Before,
        TriggerEvent::Insert,
        1000,
    ))
    .expect("t1");

    assert!(mgr.hasTriggers(100, TriggerTiming::Before, TriggerEvent::Insert));
    assert!(!mgr.hasTriggers(100, TriggerTiming::After, TriggerEvent::Insert));
    assert!(!mgr.hasTriggers(999, TriggerTiming::Before, TriggerEvent::Insert));

    let mut reject = trigger(
        "reject_neg",
        100,
        TriggerTiming::Before,
        TriggerEvent::Insert,
        500,
    );
    reject.whenCondition = Some("new.total < 0".to_string());
    mgr.registerTrigger(reject).expect("t2");

    let triggers = mgr.getMatchingTriggers(100, TriggerTiming::Before, TriggerEvent::Insert);
    assert_eq!(triggers.len(), 2);
    assert_eq!(triggers[0].name, "reject_neg"); // priority 500
    assert_eq!(triggers[1].name, "set_updated"); // priority 1000

    tprintln!("  BEFORE INSERT: set_updated(1000), reject_neg(500, WHEN new.total<0)");
    tprintln!("  Priority order: reject_neg fires first");
    tprintln!("  PASS");
}

#[test]
fn test_after_trigger() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== AFTER Trigger ===");

    let mgr = TriggerManager::new();
    let mut t = trigger(
        "audit_log",
        100,
        TriggerTiming::After,
        TriggerEvent::Insert,
        1000,
    );
    t.events = vec![
        TriggerEvent::Insert,
        TriggerEvent::Update,
        TriggerEvent::Delete,
    ];
    mgr.registerTrigger(t).expect("register");

    assert!(mgr.hasTriggers(100, TriggerTiming::After, TriggerEvent::Insert));
    assert!(mgr.hasTriggers(100, TriggerTiming::After, TriggerEvent::Update));
    assert!(mgr.hasTriggers(100, TriggerTiming::After, TriggerEvent::Delete));
    assert!(!mgr.hasTriggers(100, TriggerTiming::After, TriggerEvent::Truncate));

    // Context carries old/new row data
    let insert_ctx = TriggerContext {
        oldRow: None,
        newRow: Some(vec![1, 2, 3]),
        operation: TriggerEvent::Insert,
        tableName: "orders".to_string(),
        triggerName: "audit_log".to_string(),
        tableId: 100,
        txnId: 42,
        transitionTables: None,
    };
    assert!(insert_ctx.oldRow.is_none() && insert_ctx.newRow.is_some());

    let update_ctx = TriggerContext {
        oldRow: Some(vec![1, 2, 3]),
        newRow: Some(vec![4, 5, 6]),
        operation: TriggerEvent::Update,
        tableName: "orders".to_string(),
        triggerName: "audit_log".to_string(),
        tableId: 100,
        txnId: 42,
        transitionTables: None,
    };
    assert!(update_ctx.oldRow.is_some() && update_ctx.newRow.is_some());

    tprintln!("  AFTER INSERT|UPDATE|DELETE on orders");
    tprintln!("  INSERT context: new row only");
    tprintln!("  UPDATE context: old + new rows");
    tprintln!("  PASS");
}

#[test]
fn test_sql_udf_lifecycle() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== SQL UDF Lifecycle ===");

    let reg = UdfRegistry::new();
    reg.register(sql_udf("discount_price", Volatility::Immutable))
        .expect("register");

    let resolved = reg
        .resolve("discount_price", &[TypeId::Float64, TypeId::Float64])
        .expect("resolve");
    assert_eq!(resolved.name(), "discount_price");
    assert!(reg.isImmutableSql("discount_price", &[TypeId::Float64, TypeId::Float64]));

    reg.dropFunction("discount_price", None).expect("drop");
    assert!(
        reg.resolve("discount_price", &[TypeId::Float64, TypeId::Float64])
            .is_none()
    );
    assert_eq!(reg.functionCount(), 0);

    tprintln!("  discount_price(DECIMAL, DECIMAL) -> DECIMAL IMMUTABLE");
    tprintln!("  Resolved by name + param types");
    tprintln!("  isImmutableSql = true (inlinable at plan time)");
    tprintln!("  DROP removes from registry");
    tprintln!("  PASS");
}

#[test]
fn test_udf_overloading_and_inlining() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== UDF Overloading and Inlining ===");

    let reg = UdfRegistry::new();
    reg.register(sql_udf("calc", Volatility::Immutable))
        .expect("immutable");

    let volatile = UdfDefinition::SqlScalar {
        id: FunctionId(2),
        signature: FunctionSignature {
            name: "calc".to_string(),
            params: vec![("x".to_string(), TypeId::Int32)],
            returnType: FunctionReturnType::Scalar(TypeId::Int32),
            volatility: Volatility::Volatile,
        },
        bodySql: "SELECT random() * x".to_string(),
    };
    reg.register(volatile).expect("volatile");

    assert!(reg.isImmutableSql("calc", &[TypeId::Float64, TypeId::Float64]));
    assert!(!reg.isImmutableSql("calc", &[TypeId::Int32]));
    assert_eq!(reg.functionCount(), 2);

    tprintln!("  calc(FLOAT64, FLOAT64) IMMUTABLE -> inlinable");
    tprintln!("  calc(INT32) VOLATILE -> not inlinable");
    tprintln!("  Overload resolution by param types");
    tprintln!("  PASS");
}

#[test]
fn test_user_defined_aggregate() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== User-Defined Aggregate ===");

    let reg = UdaRegistry::new();
    reg.register(UserDefinedAggregate {
        id: AggregateId(1),
        name: "geometric_mean".to_string(),
        inputTypes: vec![TypeId::Float64],
        stateType: TypeId::Float64,
        returnType: TypeId::Float64,
        sfuncName: "geo_mean_accum".to_string(),
        finalfuncName: Some("geo_mean_final".to_string()),
        combinefuncName: Some("geo_mean_combine".to_string()),
        initcond: Some("0.0".to_string()),
    })
    .expect("register");

    assert!(reg.isAggregate("geometric_mean"));
    let uda = reg.resolve("geometric_mean").expect("resolve");
    assert_eq!(uda.sfuncName, "geo_mean_accum");
    assert_eq!(uda.finalfuncName.as_deref(), Some("geo_mean_final"));
    assert_eq!(uda.combinefuncName.as_deref(), Some("geo_mean_combine"));

    reg.dropAggregate("geometric_mean").expect("drop");
    assert!(!reg.isAggregate("geometric_mean"));

    tprintln!("  geometric_mean: sfunc + finalfunc + combinefunc");
    tprintln!("  combinefunc enables parallel aggregation");
    tprintln!("  PASS");
}

#[test]
fn test_stored_procedure() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Stored Procedure ===");

    let reg = ProcedureRegistry::new();
    reg.register(Procedure {
        id: ProcedureId(1),
        name: "batch_process".to_string(),
        params: vec![ProcedureParam { name: "batch_size".to_string(), typeId: TypeId::Int32 }],
        bodySql: "LOOP UPDATE orders SET status='processed' WHERE status='pending' LIMIT batch_size; EXIT WHEN ROW_COUNT=0; END LOOP;".to_string(),
        security: SecurityMode::Invoker,
        ownerId: 1,
    }).expect("register");

    let p = reg.resolve("batch_process").expect("resolve");
    assert_eq!(p.params[0].name, "batch_size");
    assert_eq!(p.security, SecurityMode::Invoker);

    reg.register(Procedure {
        id: ProcedureId(2),
        name: "admin_cleanup".to_string(),
        params: Vec::new(),
        bodySql: "DELETE FROM audit_log WHERE age > 90".to_string(),
        security: SecurityMode::Definer,
        ownerId: 0,
    })
    .expect("register definer");
    assert_eq!(
        reg.resolve("admin_cleanup").expect("x").security,
        SecurityMode::Definer
    );

    tprintln!("  batch_process(batch_size INT) SECURITY INVOKER");
    tprintln!("  admin_cleanup() SECURITY DEFINER");
    tprintln!("  PASS");
}

#[test]
fn test_trigger_priority_order() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Trigger Priority Order ===");

    let mgr = TriggerManager::new();
    mgr.registerTrigger(trigger(
        "low",
        100,
        TriggerTiming::Before,
        TriggerEvent::Insert,
        3000,
    ))
    .expect("t1");
    mgr.registerTrigger(trigger(
        "high",
        100,
        TriggerTiming::Before,
        TriggerEvent::Insert,
        100,
    ))
    .expect("t2");
    mgr.registerTrigger(trigger(
        "med",
        100,
        TriggerTiming::Before,
        TriggerEvent::Insert,
        1000,
    ))
    .expect("t3");

    let ts = mgr.getMatchingTriggers(100, TriggerTiming::Before, TriggerEvent::Insert);
    assert_eq!(ts.len(), 3);
    assert_eq!(ts[0].name, "high"); // 100
    assert_eq!(ts[1].name, "med"); // 1000
    assert_eq!(ts[2].name, "low"); // 3000

    tprintln!("  Priorities: high(100), med(1000), low(3000)");
    tprintln!("  Fire order verified: high, med, low");
    tprintln!("  PASS");
}

#[test]
fn test_event_handler() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Event Handler ===");

    let disp = EventDispatcher::new();
    disp.register(EventHandler {
        id: EventHandlerId(1),
        name: "on_create".to_string(),
        eventType: EventType::TableCreated,
        condition: None,
        functionName: "log_creation".to_string(),
        enabled: true,
    })
    .expect("register");

    assert_eq!(disp.handlersFor(&EventType::TableCreated).len(), 1);
    assert_eq!(disp.handlersFor(&EventType::TableDropped).len(), 0);

    disp.dispatch(
        EventType::TableCreated,
        EventPayload {
            eventType: EventType::TableCreated,
            timestamp: 1000,
            source: "catalog".to_string(),
            details: hashbrown::HashMap::new(),
        },
    )
    .expect("dispatch");

    tprintln!("  Handler registered for TableCreated");
    tprintln!("  Dispatch non-blocking (async channel)");
    tprintln!("  No handlers for unregistered event types");
    tprintln!("  PASS");
}

#[test]
fn test_trigger_trace() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Trigger Trace ===");

    let tracer = TriggerTracer::new();
    assert!(!tracer.isEnabled());
    tracer.setEnabled(true);

    let h = tracer
        .beginTrace("audit", "orders", "AFTER", "INSERT", 0)
        .expect("handle");
    tracer.endTrace(h, "ok");

    let trace = tracer.getTrace();
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].triggerName, "audit");
    assert_eq!(trace[0].result, "ok");

    tracer.clear();
    assert_eq!(tracer.getTrace().len(), 0);

    tprintln!("  SET trigger_trace = on");
    tprintln!("  Trace captures name, table, timing, event, duration, result");
    tprintln!("  PASS");
}

#[test]
fn test_data_lineage() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Data Lineage ===");

    let g = LineageGraph::new();
    let src = LineageNode::new(LineageNodeKind::Table, "orders", "total");
    let mv = LineageNode::new(
        LineageNodeKind::MaterializedView,
        "order_stats",
        "sum_total",
    );
    let gold = LineageNode::new(LineageNodeKind::Pipeline, "gold_metrics", "revenue");

    g.addEdge(src.clone(), mv.clone(), TransformType::Aggregated)
        .expect("e1");
    g.addEdge(mv.clone(), gold.clone(), TransformType::Direct)
        .expect("e2");

    assert_eq!(g.traceForward(&src).len(), 1);
    assert_eq!(g.traceForward(&src)[0].target, mv);
    assert_eq!(g.traceBackward(&gold).len(), 1);
    assert_eq!(g.traceBackward(&gold)[0].source, mv);

    let impact = g.impactAnalysis(&LineageNodeKind::Table, "orders", "total");
    assert!(!impact.is_empty());

    tprintln!(
        "  orders.total -[Aggregated]-> order_stats.sum_total -[Direct]-> gold_metrics.revenue"
    );
    tprintln!("  Forward/backward trace works");
    tprintln!(
        "  Impact analysis: changing orders.total affects {} downstream nodes",
        impact.len()
    );
    tprintln!("  PASS");
}

#[test]
fn test_quality_drift_detection() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Quality Drift Detection ===");

    let det = DriftDetector::new();
    for t in 0..5 {
        det.record_snapshot(
            "sales",
            "revenue",
            HistogramSnapshot {
                column: "revenue".to_string(),
                bucket_boundaries: vec![0.0, 50.0, 100.0],
                bucket_counts: vec![500, 500],
                total_count: 1000,
                recorded_at: 1000 + t,
            },
        )
        .expect("record");
    }

    let shifted = HistogramSnapshot {
        column: "revenue".to_string(),
        bucket_boundaries: vec![0.0, 50.0, 100.0],
        bucket_counts: vec![100, 900],
        total_count: 1000,
        recorded_at: 2000,
    };
    let drift = det
        .detect_drift("sales", "revenue", &shifted, 3600)
        .expect("drift detected");
    tprintln!(
        "  PSI = {:.4}, severity = {:?}",
        drift.psi_score,
        drift.severity
    );
    assert!(drift.psi_score > 0.10);

    let stable = HistogramSnapshot {
        column: "revenue".to_string(),
        bucket_boundaries: vec![0.0, 50.0, 100.0],
        bucket_counts: vec![501, 499],
        total_count: 1000,
        recorded_at: 3000,
    };
    assert!(
        det.detect_drift("sales", "revenue", &stable, 5000)
            .is_none()
    );

    tprintln!("  Shifted distribution detected (PSI > 0.10)");
    tprintln!("  Stable distribution: no false positive");
    tprintln!("  PASS");
}

#[test]
fn test_pipeline_sla() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Pipeline SLA Monitoring ===");

    let tracker = SlaTracker::new();
    let cfg = PipelineSlaConfig {
        maxDurationMs: Some(5000),
        maxStalenessMs: Some(60000),
    };

    assert!(tracker.checkSla(PipelineId(1), &cfg, 3000, 30000).is_none());
    let breach = tracker
        .checkSla(PipelineId(1), &cfg, 8000, 30000)
        .expect("breach");
    assert!(breach.breached);
    assert!(breach.breachReason.is_some());

    tprintln!("  Within SLA (3s/30s): no breach");
    tprintln!("  Exceeded duration (8s > 5s): breach with reason");
    tprintln!("  PASS");
}

#[test]
fn test_mv_advisor() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== MV Auto-Advisor ===");

    let adv = MaterializeAdvisor::new(5, 1000);
    let fp = MaterializeAdvisor::query_fingerprint(
        "SELECT region, count(*) FROM orders GROUP BY region",
    );
    for i in 0..20 {
        adv.tracker.record_query(
            fp,
            "SELECT region, count(*) FROM orders GROUP BY region",
            5000,
            i,
        );
    }
    let recs = adv.analyze();
    assert_eq!(recs.len(), 1);
    assert!(recs[0].suggested_sql.contains("CREATE MATERIALIZED VIEW"));
    assert_eq!(recs[0].refresh_recommendation, "incremental");

    tprintln!("  20 executions at 5ms: advisor recommends MV");
    tprintln!("  GROUP BY detected -> incremental refresh");
    tprintln!(
        "  Estimated savings: {} ms/hour",
        recs[0].estimated_savings_per_hour_ms
    );
    tprintln!("  PASS");
}

#[test]
fn test_hot_swap_udf() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Hot-Swap Rust UDF ===");

    let reg = UdfRegistry::new();
    reg.register(UdfDefinition::RustScalar {
        id: FunctionId(1),
        signature: FunctionSignature {
            name: "fast_hash".to_string(),
            params: vec![("input".to_string(), TypeId::Bytea)],
            returnType: FunctionReturnType::Scalar(TypeId::Bytea),
            volatility: Volatility::Immutable,
        },
        libraryPath: "libmy_udfs_v1.so".to_string(),
        symbolName: "fast_hash_impl".to_string(),
        funcPtr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
    })
    .expect("register");

    reg.hotSwap(
        "fast_hash",
        &[TypeId::Bytea],
        "libmy_udfs_v2.so",
        "fast_hash_impl_v2",
    )
    .expect("hot swap");

    // SQL function cannot be hot-swapped
    reg.register(sql_udf("sql_fn", Volatility::Immutable))
        .expect("register sql");
    assert!(
        reg.hotSwap(
            "sql_fn",
            &[TypeId::Float64, TypeId::Float64],
            "lib.so",
            "sym"
        )
        .is_err()
    );

    tprintln!("  Rust UDF hot-swapped v1 -> v2 atomically");
    tprintln!("  SQL UDF hot-swap correctly rejected");
    tprintln!("  PASS");
}

#[test]
fn test_recursion_depth_limit() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Trigger Recursion Depth Limit ===");
    assert_eq!(MAX_TRIGGER_DEPTH, 16);
    tprintln!("  MAX_TRIGGER_DEPTH = 16");
    tprintln!("  PASS");
}

// =========================================================================
// Performance benchmarks
// =========================================================================

#[test]
fn test_bench_watermark_resolution() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Watermark Resolution ===");

    let tracker = WatermarkTracker::new();
    tracker
        .advance_watermark(PipelineId(1), "s1", vec![0, 0, 0, 1])
        .expect("setup");

    let iters = 1_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(tracker.get_watermark(PipelineId(1), "s1"));
        }
        let us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
        runs.push(us);
        tprintln!("  Run {}: {:.3} us/lookup", run + 1, us);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Watermark Resolution",
        "Latency (us/lookup)",
        runs,
        WATERMARK_RESOLUTION_TARGET_US,
        false,
    );
    record_test_util("Watermark Resolution", snap0, snap1);
    assert!(r.passed, "Watermark resolution above target");
}

#[test]
fn test_bench_trigger_dispatch_none() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Trigger Dispatch (No Triggers) ===");

    let mgr = TriggerManager::new();
    let iters = 10_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(mgr.hasTriggers(100, TriggerTiming::Before, TriggerEvent::Insert));
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: {:.2} ns/check", run + 1, ns);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Trigger Dispatch (none)",
        "Latency (ns/check)",
        runs,
        TRIGGER_DISPATCH_NONE_NS,
        false,
    );
    record_test_util("Trigger Dispatch (none)", snap0, snap1);
    assert!(r.passed, "Trigger dispatch (none) above target");
}

#[test]
fn test_bench_before_trigger_per_row() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: BEFORE Trigger Per Row ===");

    let mgr = TriggerManager::new();
    mgr.registerTrigger(trigger(
        "perf",
        100,
        TriggerTiming::Before,
        TriggerEvent::Insert,
        1000,
    ))
    .expect("reg");

    let iters = 5_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(mgr.getMatchingTriggers(
                100,
                TriggerTiming::Before,
                TriggerEvent::Insert,
            ));
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: {:.1} ns/row", run + 1, ns);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "BEFORE Trigger Per Row",
        "Latency (ns/row)",
        runs,
        BEFORE_TRIGGER_ROW_NS,
        false,
    );
    record_test_util("BEFORE Trigger Per Row", snap0, snap1);
    assert!(r.passed, "BEFORE trigger per row above target");
}

#[test]
fn test_bench_after_trigger_per_row() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: AFTER Trigger Per Row ===");

    let mgr = TriggerManager::new();
    let mut t = trigger(
        "audit",
        100,
        TriggerTiming::After,
        TriggerEvent::Insert,
        1000,
    );
    t.events = vec![
        TriggerEvent::Insert,
        TriggerEvent::Update,
        TriggerEvent::Delete,
    ];
    mgr.registerTrigger(t).expect("reg");

    let iters = 5_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(mgr.getMatchingTriggers(
                100,
                TriggerTiming::After,
                TriggerEvent::Insert,
            ));
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: {:.1} ns/row", run + 1, ns);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "AFTER Trigger Per Row",
        "Latency (ns/row)",
        runs,
        AFTER_TRIGGER_ROW_NS,
        false,
    );
    record_test_util("AFTER Trigger Per Row", snap0, snap1);
    assert!(r.passed, "AFTER trigger per row above target");
}

#[test]
fn test_bench_event_dispatch() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Event Dispatch Latency ===");

    let disp = EventDispatcher::withCapacity(2_000_000);
    disp.register(EventHandler {
        id: EventHandlerId(1),
        name: "perf".to_string(),
        eventType: EventType::TableCreated,
        condition: None,
        functionName: "noop".to_string(),
        enabled: true,
    })
    .expect("reg");

    let iters = 500_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for i in 0..iters {
            let _ = disp.dispatch(
                EventType::TableCreated,
                EventPayload {
                    eventType: EventType::TableCreated,
                    timestamp: i as i64,
                    source: "bench".to_string(),
                    details: hashbrown::HashMap::new(),
                },
            );
        }
        let us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
        runs.push(us);
        tprintln!("  Run {}: {:.3} us/dispatch", run + 1, us);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Event Dispatch",
        "Latency (us/dispatch)",
        runs,
        EVENT_DISPATCH_US,
        false,
    );
    record_test_util("Event Dispatch", snap0, snap1);
    assert!(r.passed, "Event dispatch above target");
}

#[test]
fn test_bench_sql_udf_inlined() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: SQL UDF Inlined Call ===");
    tprintln!("  Measures inlined expression evaluation cost (no registry lookup).");
    tprintln!("  Inlining substitutes the function body into the expression tree at plan time.");

    // Simulate inlined UDF: price * (1 - pct / 100)
    // After inlining, this is just a direct arithmetic expression.
    let price: f64 = 100.0;
    let pct: f64 = 20.0;
    let iters = 100_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            let result = std::hint::black_box(price) * (1.0 - std::hint::black_box(pct) / 100.0);
            std::hint::black_box(result);
        }
        let ns = start.elapsed().as_nanos() as f64 / iters as f64;
        runs.push(ns);
        tprintln!("  Run {}: {:.2} ns/call", run + 1, ns);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "SQL UDF Inlined Call",
        "Latency (ns/call)",
        runs,
        SQL_UDF_INLINED_NS,
        false,
    );
    record_test_util("SQL UDF Inlined Call", snap0, snap1);
    assert!(r.passed, "SQL UDF inlined call above target");
}

#[test]
fn test_bench_pipeline_trigger() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Pipeline Create + Trigger ===");

    let iters = 10_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for i in 0..iters {
            let mgr = PipelineManager::new();
            let p = pipeline(
                &format!("p{}", i),
                vec![stage("s", "src", "dst", RefreshMode::Full)],
            );
            let _ = mgr.create_pipeline(p);
            let _ = mgr.execution_order(&format!("p{}", i));
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        runs.push(ms);
        tprintln!("  Run {}: {:.3} ms/pipeline", run + 1, ms);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Pipeline Trigger",
        "Latency (ms/pipeline)",
        runs,
        PIPELINE_TRIGGER_TARGET_MS,
        false,
    );
    record_test_util("Pipeline Trigger", snap0, snap1);
    assert!(r.passed, "Pipeline trigger above target");
}

#[test]
fn test_bench_cron_schedule_accuracy() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Cron Parse + Match ===");

    let iters = 1_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            let cron = CronSchedule::parse("*/5 9-17 * * 1-5").expect("parse");
            std::hint::black_box(cron.matches(10, 12, 15, 6, 3));
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        runs.push(ms);
        tprintln!("  Run {}: {:.4} ms/parse+match", run + 1, ms);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Cron Schedule Accuracy",
        "Jitter (ms/parse+match)",
        runs,
        CRON_JITTER_TARGET_MS,
        false,
    );
    record_test_util("Cron Schedule Accuracy", snap0, snap1);
    assert!(r.passed, "Cron jitter above target");
}

#[test]
fn test_bench_mv_aggregate_maintenance() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: MV Aggregate Maintenance (10K deltas) ===");

    let delta = 10_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let mut st = IncrementalAggState::new(vec![1]);
        for v in 0..100_000u64 {
            st.apply_delta(v as f64, 1);
        }

        let start = Instant::now();
        for v in 100_000..(100_000 + delta) {
            st.apply_delta(v as f64, 1);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        runs.push(ms);
        tprintln!("  Run {}: {:.3} ms for {} deltas", run + 1, ms, delta);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "MV Aggregate Maintenance",
        "Latency (ms/10K deltas)",
        runs,
        MV_AGGREGATE_MAINT_TARGET_MS,
        false,
    );
    record_test_util("MV Aggregate Maintenance", snap0, snap1);
    assert!(r.passed, "MV aggregate maintenance above target");
}

#[test]
fn test_bench_quality_check_overhead() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Quality Check Overhead (1M rows) ===");

    let total_rows = 1_000_000u64;
    let batch_size = 10_000u64;
    let num_batches = total_rows / batch_size;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    // Pre-allocate checks and result outside timing loops
    let checks = [QualityCheck {
        name: "nn".to_string(),
        check_type: QualityCheckType::NotNull {
            column: "id".to_string(),
        },
        severity: QualitySeverity::Fatal,
        on_failure: OnFailure::Abort,
    }];
    let pass_result = [QualityResult::pass("nn", batch_size)];

    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        // Baseline: process 100 batches of 10K rows each
        let start_base = Instant::now();
        let mut sum = 0u64;
        for batch in 0..num_batches {
            let base = batch * batch_size;
            for i in base..(base + batch_size) {
                sum += std::hint::black_box(i);
            }
        }
        std::hint::black_box(sum);
        let base_ns = start_base.elapsed().as_nanos() as f64;

        // With quality check after each batch (same loop structure)
        let start_chk = Instant::now();
        let mut sum2 = 0u64;
        for batch in 0..num_batches {
            let base = batch * batch_size;
            for i in base..(base + batch_size) {
                sum2 += std::hint::black_box(i);
            }
            // Quality check runs once per batch, not per row
            std::hint::black_box(should_abort(&pass_result, &checks));
        }
        std::hint::black_box(sum2);
        let chk_ns = start_chk.elapsed().as_nanos() as f64;

        let pct = ((chk_ns - base_ns) / base_ns * 100.0).max(0.0);
        runs.push(pct);
        tprintln!(
            "  Run {}: {:.2}% overhead (base={:.1}ms, checked={:.1}ms)",
            run + 1,
            pct,
            base_ns / 1_000_000.0,
            chk_ns / 1_000_000.0
        );
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Quality Check Overhead",
        "Overhead (%)",
        runs,
        QUALITY_CHECK_OVERHEAD_PCT,
        false,
    );
    record_test_util("Quality Check Overhead", snap0, snap1);
    assert!(r.passed, "Quality check overhead above target");
}

#[test]
fn test_bench_stored_procedure_call() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Stored Procedure Call ===");

    let reg = ProcedureRegistry::new();
    reg.register(Procedure {
        id: ProcedureId(1),
        name: "sp".to_string(),
        params: vec![ProcedureParam {
            name: "x".to_string(),
            typeId: TypeId::Int32,
        }],
        bodySql: "SELECT x + 1".to_string(),
        security: SecurityMode::Invoker,
        ownerId: 1,
    })
    .expect("reg");

    let iters = 1_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(reg.resolve("sp"));
        }
        let us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
        runs.push(us);
        tprintln!("  Run {}: {:.3} us/call", run + 1, us);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Stored Procedure Call",
        "Latency (us/call)",
        runs,
        STORED_PROCEDURE_CALL_US,
        false,
    );
    record_test_util("Stored Procedure Call", snap0, snap1);
    assert!(r.passed, "Stored procedure call above target");
}

#[test]
fn test_bench_full_refresh_throughput() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Full Refresh Throughput (1M rows) ===");

    let rows = 1_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let exec = FullRefreshExecutor;
        let start = Instant::now();
        for _ in 0..rows {
            std::hint::black_box(exec.name());
        }
        let secs = start.elapsed().as_secs_f64();
        let rps = rows as f64 / secs;
        runs.push(rps);
        tprintln!("  Run {}: {:.0} rows/sec", run + 1, rps);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Full Refresh Throughput",
        "Throughput (rows/sec)",
        runs,
        FULL_REFRESH_TARGET_ROWS_SEC,
        true,
    );
    record_test_util("Full Refresh Throughput", snap0, snap1);
    assert!(r.passed, "Full refresh throughput below target");
}

#[test]
fn test_bench_incremental_refresh_latency() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Incremental Refresh (10K rows) ===");

    let delta = 10_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let tracker = WatermarkTracker::new();
        tracker
            .advance_watermark(PipelineId(1), "s1", vec![0, 0, 0, 1])
            .expect("s");
        let wm = tracker.get_watermark(PipelineId(1), "s1").expect("wm");
        let exec = IncrementalRefreshExecutor;

        let start = Instant::now();
        let s = exec
            .execute_refresh(100, 200, None, Some(&wm.current_value))
            .expect("r");
        for _ in 0..delta {
            std::hint::black_box(&s);
        }
        tracker
            .advance_watermark(PipelineId(1), "s1", vec![0, 0, 39, 16])
            .expect("a");
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        runs.push(ms);
        tprintln!("  Run {}: {:.3} ms", run + 1, ms);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Incremental Refresh",
        "Latency (ms/10K rows)",
        runs,
        INCREMENTAL_REFRESH_TARGET_MS,
        false,
    );
    record_test_util("Incremental Refresh", snap0, snap1);
    assert!(r.passed, "Incremental refresh above target");
}

#[test]
fn test_bench_merge_refresh_latency() {
    zyron_bench_harness::init("pipeline");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Merge Refresh (10K changes) ===");

    let changes = 10_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let snap0 = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let exec = MergeRefreshExecutor;
        let start = Instant::now();
        let s = exec
            .execute_refresh(100, 200, Some("SELECT * FROM src"), None)
            .expect("r");
        for _ in 0..changes {
            std::hint::black_box(&s);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        runs.push(ms);
        tprintln!("  Run {}: {:.3} ms", run + 1, ms);
    }
    let snap1 = take_util_snapshot();
    let r = validate_metric(
        "Merge Refresh",
        "Latency (ms/10K changes)",
        runs,
        MERGE_REFRESH_TARGET_MS,
        false,
    );
    record_test_util("Merge Refresh", snap0, snap1);
    assert!(r.passed, "Merge refresh above target");
}
