#![allow(non_snake_case)]
// Production MaterializationExecutor that runs each feature group's
// sourceQuery through the planner+executor and folds the result into
// per-(entity, feature) FeatureValue triples ready for batch upsert

use std::sync::Arc;

use zyron_analytics::value::AnalyticsValue;
use zyron_analytics::{FeatureGroup, FeatureValue};
use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_storage::DiskManager;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;

use crate::background::feature_materialization::MaterializationExecutor;

pub struct PlannerMaterializationExecutor {
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk_manager: Arc<DiskManager>,
    pub txn_manager: Arc<TransactionManager>,
    /// Shared current-thread runtime reused across every materialize call.
    /// Building a tokio runtime per call costs ~1 ms in syscalls, threads,
    /// and reactor setup; reusing one across all refresh ticks keeps that
    /// cost amortized to zero
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PlannerMaterializationExecutor {
    pub fn new(
        catalog: Arc<Catalog>,
        wal: Arc<WalWriter>,
        buffer_pool: Arc<BufferPool>,
        disk_manager: Arc<DiskManager>,
        txn_manager: Arc<TransactionManager>,
    ) -> Self {
        // Multi-thread runtime so parallel planner+executor work inside
        // each materialize() call uses every available core. Two worker
        // threads is enough since the materialization worker fires one
        // group at a time
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("zyron-feature-mat")
            .build()
            .expect("failed to build feature materialization runtime");
        Self {
            catalog,
            wal,
            buffer_pool,
            disk_manager,
            txn_manager,
            runtime: Arc::new(runtime),
        }
    }

    fn runSourceQuery(
        &self,
        group: &FeatureGroup,
    ) -> Result<Vec<(String, AnalyticsValue, Vec<(String, AnalyticsValue)>)>, String> {
        if group.sourceQuery.is_empty() {
            return Ok(Vec::new());
        }
        let mut parser = zyron_parser::Parser::new(&group.sourceQuery)
            .map_err(|e| format!("parse error: {}", e))?;
        let stmt = parser
            .parse_statement()
            .map_err(|e| format!("parse error: {}", e))?;
        let catalog = self.catalog.clone();
        let database_id = zyron_catalog::DatabaseId(1);
        let search_path: Vec<String> = vec!["public".to_string()];

        let wal = self.wal.clone();
        let bp = self.buffer_pool.clone();
        let dm = self.disk_manager.clone();
        let txn_mgr = self.txn_manager.clone();
        let entity_col = group.entityKey.clone();
        let feature_cols: Vec<String> = group.features.iter().map(|f| f.name.clone()).collect();

        self.runtime.block_on(async move {
            let plan = zyron_planner::plan(&catalog, database_id, search_path, stmt, None)
                .await
                .map_err(|e| format!("plan: {}", e))?;
            let schema = plan.output_schema();
            let entityIdx = schema
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(&entity_col))
                .ok_or_else(|| format!("source query missing entity column '{}'", entity_col))?;
            let mut featureIdx: Vec<usize> = Vec::with_capacity(feature_cols.len());
            for f in &feature_cols {
                let idx = schema
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(f))
                    .ok_or_else(|| format!("source query missing feature column '{}'", f))?;
                featureIdx.push(idx);
            }
            let mut read_txn = txn_mgr
                .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
                .map_err(|e| format!("begin txn: {}", e))?;
            let snapshot = read_txn.snapshot.clone();
            let txn_id_u32 =
                u32::try_from(read_txn.txn_id).map_err(|_| "txn id overflow".to_string())?;
            let ctx = Arc::new(zyron_executor::context::ExecutionContext::new(
                catalog.clone(),
                wal,
                bp,
                dm,
                txn_id_u32,
                snapshot,
            ));
            let result = zyron_executor::execute(plan, &ctx).await;
            let _ = txn_mgr.abort(&mut read_txn);
            let batches = result.map_err(|e| format!("execute: {}", e))?;

            let mut out: Vec<(String, AnalyticsValue, Vec<(String, AnalyticsValue)>)> = Vec::new();
            for batch in &batches {
                for r in 0..batch.num_rows {
                    let entity_scalar = batch.column(entityIdx).get_scalar(r);
                    let entity_val = scalarToAnalytics(&entity_scalar);
                    let mut featureVals: Vec<(String, AnalyticsValue)> =
                        Vec::with_capacity(feature_cols.len());
                    for (i, fidx) in featureIdx.iter().enumerate() {
                        let s = batch.column(*fidx).get_scalar(r);
                        featureVals.push((feature_cols[i].clone(), scalarToAnalytics(&s)));
                    }
                    out.push((entity_col.clone(), entity_val, featureVals));
                }
            }
            Ok::<_, String>(out)
        })
    }
}

impl MaterializationExecutor for PlannerMaterializationExecutor {
    fn materialize(
        &self,
        group: &FeatureGroup,
        nowMs: i64,
    ) -> Result<Vec<(String, String, FeatureValue)>, String> {
        let rows = self.runSourceQuery(group)?;
        let mut out: Vec<(String, String, FeatureValue)> =
            Vec::with_capacity(rows.len() * group.features.len());
        for (_entityCol, entityVal, featureVals) in rows {
            let entityKey = analyticsToEntityKey(&entityVal);
            for (fname, fval) in featureVals {
                let fv = FeatureValue {
                    computationTimestampMs: nowMs,
                    validFromMs: nowMs,
                    validToMs: i64::MAX,
                    value: fval,
                    featureVersion: 1,
                };
                out.push((entityKey.clone(), fname, fv));
            }
        }
        Ok(out)
    }
}

fn scalarToAnalytics(v: &zyron_executor::column::ScalarValue) -> AnalyticsValue {
    use zyron_executor::column::ScalarValue;
    match v {
        ScalarValue::Null => AnalyticsValue::Null,
        ScalarValue::Boolean(b) => AnalyticsValue::Bool(*b),
        ScalarValue::Int8(x) => AnalyticsValue::Int(*x as i64),
        ScalarValue::Int16(x) => AnalyticsValue::Int(*x as i64),
        ScalarValue::Int32(x) => AnalyticsValue::Int(*x as i64),
        ScalarValue::Int64(x) => AnalyticsValue::Int(*x),
        ScalarValue::Int128(x) => {
            AnalyticsValue::Int((*x).clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        }
        ScalarValue::UInt8(x) => AnalyticsValue::UInt(*x as u64),
        ScalarValue::UInt16(x) => AnalyticsValue::UInt(*x as u64),
        ScalarValue::UInt32(x) => AnalyticsValue::UInt(*x as u64),
        ScalarValue::UInt64(x) => AnalyticsValue::UInt(*x),
        ScalarValue::Float32(f) => AnalyticsValue::Float(*f as f64),
        ScalarValue::Float64(f) => AnalyticsValue::Float(*f),
        ScalarValue::Utf8(s) => AnalyticsValue::Text(s.clone()),
        _ => AnalyticsValue::Null,
    }
}

fn analyticsToEntityKey(v: &AnalyticsValue) -> String {
    match v {
        AnalyticsValue::Text(s) => s.clone(),
        AnalyticsValue::Int(i) => i.to_string(),
        AnalyticsValue::UInt(u) => u.to_string(),
        AnalyticsValue::Float(f) => f.to_string(),
        AnalyticsValue::Bool(b) => b.to_string(),
        AnalyticsValue::Timestamp(t) => t.to_string(),
        AnalyticsValue::Date(d) => d.to_string(),
        AnalyticsValue::Null => String::new(),
    }
}
