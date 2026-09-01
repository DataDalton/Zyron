// Analytics table-returning function operator.
// Resolves the function name from the analytics registry, scans the
// referenced source table, materialises rows into AnalyticsValue, runs
// the requested analytics function, and emits the result as a DataBatch.

use std::sync::Arc;

use zyron_analytics::{
    AnalyticsFunctionKind, AnalyticsValue, CohortAnalysis, CohortDefinition, CohortEvent,
    CohortMetric, CohortPeriod, CohortType, ColumnProfile, FunnelConfig, FunnelEvent, FunnelStep,
    TableProfile,
};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::LiteralValue;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::context::ExecutionContext;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

pub struct AnalyticsTableFunctionOperator {
    function_name: String,
    named_args: Vec<(String, BoundExpr)>,
    positional_args: Vec<BoundExpr>,
    output_columns: Vec<LogicalColumn>,
    ctx: Arc<ExecutionContext>,
    finished: bool,
}

impl AnalyticsTableFunctionOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        function_name: String,
        named_args: Vec<(String, BoundExpr)>,
        positional_args: Vec<BoundExpr>,
        output_columns: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            function_name,
            named_args,
            positional_args,
            output_columns,
            ctx,
            finished: false,
        }
    }

    fn registry_lookup(&self) -> Result<zyron_analytics::AnalyticsFunction> {
        zyron_analytics::default_registry()
            .lookup(&self.function_name)
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "analytics function '{}' not registered",
                    self.function_name
                ))
            })
    }

    async fn execute(&mut self) -> Result<DataBatch> {
        let func = self.registry_lookup()?;
        if !matches!(func.kind, AnalyticsFunctionKind::TableReturning) {
            return Err(ZyronError::ExecutionError(format!(
                "function '{}' is not a table-returning function",
                self.function_name
            )));
        }
        match self.function_name.as_str() {
            "DATA_PROFILE" => self.run_data_profile().await,
            "COLUMN_PROFILE" => self.run_column_profile().await,
            "CORRELATION_MATRIX" => self.run_correlation_matrix().await,
            "COHORT_RETENTION" => self.run_cohort_retention().await,
            "FUNNEL_ANALYSIS" => self.run_funnel_analysis().await,
            "GET_FEATURES" => self.run_get_features().await,
            "FEATURE_LINEAGE" => self.run_feature_lineage().await,
            "PREDICT_BATCH" => self.run_predict_batch().await,
            "FORECAST" => self.run_forecast().await,
            "ANOMALY_DETECT" => self.run_anomaly_detect().await,
            "ACF" => self.run_acf().await,
            "PACF" => self.run_pacf().await,
            "SEASONALITY_DETECT" => self.run_seasonality_detect().await,
            "CHANGE_POINTS" => self.run_change_points().await,
            "DATE_FEATURES" => self.run_date_features().await,
            "POLYNOMIAL_FEATURES" => self.run_polynomial_features().await,
            "FEATURE_PARITY_CHECK" => self.run_feature_parity_check().await,
            "EXPLAIN_PREDICTION" => self.run_explain_prediction().await,
            "MODEL_LINEAGE" => self.run_model_lineage().await,
            "DETECT_ANOMALIES" => self.run_detect_anomalies().await,
            "DETECT_ANOMALIES_TIMESERIES" => self.run_detect_anomalies_timeseries().await,
            "DETECT_DRIFT" => self.run_detect_drift().await,
            "KMEANS_CLUSTER" => self.run_kmeans_cluster().await,
            "KMEANS_CENTROIDS" => self.run_kmeans_centroids().await,
            "KMEANS_ELBOW" => self.run_kmeans_elbow().await,
            "KMEANS_PREDICT" => self.run_kmeans_predict().await,
            "CAUSAL_IMPACT" => self.run_causal_impact().await,
            "COUNTERFACTUAL" => self.run_counterfactual().await,
            "PROPENSITY_MATCH" => self.run_propensity_match().await,
            "AB_TEST_ANALYSIS" => self.run_ab_test_analysis().await,
            "AB_TEST_SAMPLE_SIZE" => self.run_ab_test_sample_size().await,
            "TARGET_ENCODE" => self.run_target_encode().await,
            "TFIDF" => self.run_tfidf().await,
            "LABEL_ENCODE" => self.run_label_encode().await,
            "ROBUST_SCALE" => self.run_robust_scale().await,
            "ONE_HOT_ENCODE" => self.run_one_hot_encode().await,
            "TEXT_FEATURES" => self.run_text_features().await,
            "SHAPIRO_WILK" => self.run_shapiro_wilk().await,
            "ANDERSON_DARLING" => self.run_anderson_darling().await,
            "KOLMOGOROV_SMIRNOV_1SAMP" => self.run_ks_1samp().await,
            "FIT_DISTRIBUTION" => self.run_fit_distribution().await,
            "LOCAL_OUTLIER_FACTOR" => self.run_local_outlier_factor().await,
            "SEQUENTIAL_ANALYZE" => self.run_sequential_analyze().await,
            "BAYESIAN_ANALYZE" => self.run_bayesian_analyze().await,
            "HYBRID_SEARCH" => self.run_hybrid_search().await,
            other => Err(ZyronError::ExecutionError(format!(
                "analytics function '{}' has no executor dispatch",
                other
            ))),
        }
    }

    fn first_string_arg(&self) -> Result<String> {
        match self.positional_args.first() {
            Some(BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            }) => Ok(s.clone()),
            _ => Err(ZyronError::ExecutionError(format!(
                "{} requires the source table name as the first argument (string literal)",
                self.function_name
            ))),
        }
    }

    /// HYBRID_SEARCH(index_name, query_text, query_vector, k => 10):
    /// searches both halves of a hybrid index and fuses the scores with the
    /// method the index was created with
    async fn run_hybrid_search(&self) -> Result<DataBatch> {
        let index_name = self
            .named_string("index_name")
            .map(Ok)
            .unwrap_or_else(|| self.pos_string_at(0))?;
        let query_text = self
            .named_string("query_text")
            .map(Ok)
            .unwrap_or_else(|| self.pos_string_at(1))?;
        let query_vector_text = self
            .named_string("query_vector")
            .map(Ok)
            .unwrap_or_else(|| self.pos_string_at(2))?;
        let k = self.named_int("k").unwrap_or(10).max(1) as usize;

        let query_vector: Vec<f32> =
            serde_json::from_str(&query_vector_text).map_err(|_| ZyronError::InvalidParameter {
                name: "query_vector".to_string(),
                value: query_vector_text.clone(),
            })?;

        // Resolve the hybrid index by name across the catalog
        let mut found = None;
        for table in self.ctx.catalog.list_all_tables() {
            for idx in self.ctx.catalog.get_indexes_for_table(table.id) {
                if idx.index_type == zyron_catalog::IndexType::Hybrid && idx.name == index_name {
                    found = Some(idx);
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let entry = found.ok_or_else(|| {
            ZyronError::ExecutionError(format!("hybrid index {index_name} does not exist"))
        })?;
        let params = zyron_catalog::index_params::decode_hybrid_params(&entry.parameters)
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "hybrid index {index_name} has no readable fusion parameters"
                ))
            })?;
        if query_vector.len() != params.vector_dims as usize {
            return Err(ZyronError::InvalidParameter {
                name: "query_vector".to_string(),
                value: format!(
                    "{} dimensions, the index stores {}",
                    query_vector.len(),
                    params.vector_dims
                ),
            });
        }

        // A candidate pool larger than k feeds the fusion so a document
        // ranked well by only one side can still reach the fused top k
        let fetch_n = (k * 10).clamp(100, 1000);

        let fts_index = self.ctx.get_fts_index(entry.id).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "hybrid index {index_name} has no live full text half"
            ))
        })?;
        let analyzer = self.ctx.fts_analyzer(entry.id.0);
        let scorer = zyron_search::Bm25Scorer::default();
        let fts_query = zyron_search::FtsQueryParser::parse(&query_text)?;
        let fts_results = fts_index.search(&fts_query, analyzer.as_ref(), &scorer, fetch_n)?;

        let vector_index = self.ctx.get_vector_index(entry.id.0).ok_or_else(|| {
            ZyronError::ExecutionError(format!("hybrid index {index_name} has no live vector half"))
        })?;
        let vec_results = zyron_search::vector::VectorSearch::search(
            vector_index.as_ref(),
            &query_vector,
            fetch_n,
            64,
        )?;

        // Per document raw figures for the output columns
        let mut fts_score_of: std::collections::HashMap<u64, f64> =
            std::collections::HashMap::new();
        for (doc, score) in &fts_results {
            fts_score_of.insert(*doc, *score);
        }
        let mut distance_of: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
        for (doc, dist) in &vec_results {
            distance_of.insert(*doc, *dist as f64);
        }

        let fused: Vec<(u64, f64)> = if params.fusion_method == "linear" {
            zyron_search::vector::HybridSearch::linear_combination(
                &fts_results,
                &vec_results,
                0.5,
                k,
            )
        } else {
            // Reciprocal rank fusion over both ranked lists
            let rrf_k = params.rrf_k.max(1) as f64;
            let mut scores: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
            for (rank, (doc, _)) in fts_results.iter().enumerate() {
                *scores.entry(*doc).or_insert(0.0) += 1.0 / (rrf_k + rank as f64 + 1.0);
            }
            for (rank, (doc, _)) in vec_results.iter().enumerate() {
                *scores.entry(*doc).or_insert(0.0) += 1.0 / (rrf_k + rank as f64 + 1.0);
            }
            let mut fused: Vec<(u64, f64)> = scores.into_iter().collect();
            fused.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            fused.truncate(k);
            fused
        };

        let mut doc_ids = Vec::with_capacity(fused.len());
        let mut scores = Vec::with_capacity(fused.len());
        let mut fts_scores = Vec::with_capacity(fused.len());
        let mut distances = Vec::with_capacity(fused.len());
        for (doc, score) in &fused {
            doc_ids.push(ScalarValue::Int64(*doc as i64));
            scores.push(ScalarValue::Float64(*score));
            fts_scores.push(match fts_score_of.get(doc) {
                Some(s) => ScalarValue::Float64(*s),
                None => ScalarValue::Null,
            });
            distances.push(match distance_of.get(doc) {
                Some(d) => ScalarValue::Float64(*d),
                None => ScalarValue::Null,
            });
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, doc_ids),
            (TypeId::Float64, scores),
            (TypeId::Float64, fts_scores),
            (TypeId::Float64, distances),
        ]))
    }

    fn named_string(&self, name: &str) -> Option<String> {
        self.named_args.iter().find_map(|(n, v)| {
            if n.eq_ignore_ascii_case(name) {
                if let BoundExpr::Literal {
                    value: LiteralValue::String(s),
                    ..
                } = v
                {
                    return Some(s.clone());
                }
            }
            None
        })
    }

    fn named_int(&self, name: &str) -> Option<i64> {
        self.named_args.iter().find_map(|(n, v)| {
            if n.eq_ignore_ascii_case(name) {
                if let BoundExpr::Literal {
                    value: LiteralValue::Integer(i),
                    ..
                } = v
                {
                    return Some(*i);
                }
            }
            None
        })
    }

    fn named_float(&self, name: &str) -> Option<f64> {
        self.named_args.iter().find_map(|(n, v)| {
            if n.eq_ignore_ascii_case(name) {
                match v {
                    BoundExpr::Literal {
                        value: LiteralValue::Float(f),
                        ..
                    } => return Some(*f),
                    BoundExpr::Literal {
                        value: LiteralValue::Integer(i),
                        ..
                    } => return Some(*i as f64),
                    _ => {}
                }
            }
            None
        })
    }

    /// Resolves a table name to its catalog entry, scanning all visible
    /// tables case-insensitively. Returned Arc lets callers read the
    /// column metadata without touching the catalog cache repeatedly.
    fn resolve_table(&self, table_name: &str) -> Result<std::sync::Arc<zyron_catalog::TableEntry>> {
        self.ctx
            .catalog
            .list_all_tables()
            .into_iter()
            .find(|t| t.name.eq_ignore_ascii_case(table_name))
            .ok_or_else(|| ZyronError::TableNotFound(table_name.to_string()))
    }

    async fn run_data_profile(&self) -> Result<DataBatch> {
        let table_name = self.first_string_arg()?;
        // Streaming: build one ColumnProfiler per column up front, then
        // ingest values directly out of the heap scan. Peak memory is
        // bounded by the per-column sketch state (HLL + reservoir + small
        // top-k), independent of the table's row count.
        let table_entry = self.resolve_table(&table_name)?;
        let columns = table_entry.columns.clone();
        let names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();
        let types: Vec<String> = columns
            .iter()
            .map(|c| type_id_to_label(c.type_id))
            .collect();

        let mut profilers: Vec<zyron_analytics::profiling::ColumnProfiler> = names
            .iter()
            .zip(types.iter())
            .map(|(n, t)| zyron_analytics::profiling::ColumnProfiler::new(n.clone(), t.clone()))
            .collect();

        let mut row_count: u64 = 0;
        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            row_count += 1;
            decode_tuple_streaming(view.data, &columns, &mut profilers);
        });

        let column_profiles: Vec<zyron_analytics::profiling::ColumnProfile> =
            profilers.into_iter().map(|p| p.finalise()).collect();
        let profile = zyron_analytics::profiling::TableProfile {
            table_name: table_name.clone(),
            row_count,
            columns: column_profiles,
            correlation_matrix: None,
        };
        Ok(self.encode_table_profile(&profile))
    }

    async fn run_column_profile(&self) -> Result<DataBatch> {
        let table_name = self.first_string_arg()?;
        let column_name = match self.positional_args.get(1) {
            Some(BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            }) => s.clone(),
            _ => {
                return Err(ZyronError::ExecutionError(
                    "COLUMN_PROFILE requires (table_name, column_name) string arguments".into(),
                ));
            }
        };
        // Streaming: only the chosen column is decoded per row; other column
        // bytes are advanced past without materialising AnalyticsValues.
        let table_entry = self.resolve_table(&table_name)?;
        let columns = table_entry.columns.clone();
        let target_idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(&column_name))
            .ok_or_else(|| ZyronError::ColumnNotFound(column_name.clone()))?;
        let target_name = columns[target_idx].name.clone();
        let target_type = type_id_to_label(columns[target_idx].type_id);
        let mut profiler =
            zyron_analytics::profiling::ColumnProfiler::new(target_name, target_type);

        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            decode_one_column_streaming(view.data, &columns, target_idx, &mut profiler);
        });
        let profile = profiler.finalise();
        Ok(self.encode_column_profile_kv(&profile))
    }

    async fn run_correlation_matrix(&self) -> Result<DataBatch> {
        let table_name = self.first_string_arg()?;
        let table_entry = self.resolve_table(&table_name)?;
        let columns = table_entry.columns.clone();
        let all_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();

        // Optional column subset via positional_args[1..]; default = all
        let selected_indices: Vec<usize> = if self.positional_args.len() > 1 {
            let mut out = Vec::new();
            for arg in &self.positional_args[1..] {
                if let BoundExpr::Literal {
                    value: LiteralValue::String(s),
                    ..
                } = arg
                {
                    if let Some(idx) = all_names.iter().position(|n| n.eq_ignore_ascii_case(s)) {
                        out.push(idx);
                    }
                }
            }
            out
        } else {
            (0..all_names.len()).collect()
        };
        let selected_names: Vec<String> = selected_indices
            .iter()
            .map(|&i| all_names[i].clone())
            .collect();

        // Streaming pass: per heap tuple, decode only the selected columns
        // into a small Option<f64> row buffer, then ingest into the
        // correlation builder. No whole-table materialisation.
        let mut builder = zyron_analytics::CorrelationMatrixBuilder::new(selected_names.clone());
        let mut row_buf: Vec<Option<f64>> = vec![None; selected_indices.len()];

        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            decode_selected_columns_to_f64(view.data, &columns, &selected_indices, &mut row_buf);
            builder.ingest_row(&row_buf);
        });

        let m = builder.finalise();
        let n = m.columns.len();
        let mut col_a = Vec::with_capacity(n * n);
        let mut col_b = Vec::with_capacity(n * n);
        let mut corr = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                col_a.push(ScalarValue::Utf8(m.columns[i].clone()));
                col_b.push(ScalarValue::Utf8(m.columns[j].clone()));
                corr.push(ScalarValue::Float64(m.get(i, j)));
            }
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, col_a),
            (TypeId::Varchar, col_b),
            (TypeId::Float64, corr),
        ]))
    }

    async fn run_cohort_retention(&self) -> Result<DataBatch> {
        let table_name = self.first_string_arg()?;
        let user_col = self.named_string("user_id").ok_or_else(|| {
            ZyronError::ExecutionError(
                "COHORT_RETENTION requires user_id => 'column' named argument".into(),
            )
        })?;
        let time_col = self.named_string("event_time").ok_or_else(|| {
            ZyronError::ExecutionError(
                "COHORT_RETENTION requires event_time => 'column' named argument".into(),
            )
        })?;
        let cohort_period = self
            .named_string("cohort_period")
            .as_deref()
            .and_then(CohortPeriod::from_str_ci)
            .unwrap_or(CohortPeriod::Month);
        let analysis_period = self
            .named_string("analysis_period")
            .as_deref()
            .and_then(CohortPeriod::from_str_ci)
            .unwrap_or(cohort_period);
        let periods = self.named_int("periods").unwrap_or(12).max(1) as u32;
        let revenue_col = self.named_string("revenue");

        let table_entry = self.resolve_table(&table_name)?;
        let columns = table_entry.columns.clone();
        let all_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();

        let user_idx = all_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&user_col))
            .ok_or_else(|| ZyronError::ColumnNotFound(user_col.clone()))?;
        let time_idx = all_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&time_col))
            .ok_or_else(|| ZyronError::ColumnNotFound(time_col.clone()))?;
        let revenue_idx = revenue_col
            .as_ref()
            .and_then(|c| all_names.iter().position(|n| n.eq_ignore_ascii_case(c)));

        let analysis = CohortAnalysis {
            definition: CohortDefinition {
                user_id_column: user_col,
                event_time_column: time_col,
                cohort_period,
                analysis_period,
                cohort_type: CohortType::FirstEvent,
            },
            metric: if revenue_idx.is_some() {
                CohortMetric::Revenue
            } else {
                CohortMetric::ActiveUsers
            },
        };
        let mut analyser = zyron_analytics::CohortAnalyser::new(analysis, periods)?;

        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            // Decode just the columns we need for one CohortEvent
            let mut user = AnalyticsValue::Null;
            let mut ts: i64 = 0;
            let mut revenue: Option<f64> = None;
            decode_three_columns(
                view.data,
                &columns,
                user_idx,
                time_idx,
                revenue_idx,
                &mut user,
                &mut ts,
                &mut revenue,
            );
            analyser.ingest(&CohortEvent {
                user_id: user,
                event_time_ms: ts,
                revenue,
                custom_value: None,
                attribute: None,
            });
        });
        let result = analyser.finalise();

        // Output (cohort, period, value)
        let mut col_label = Vec::new();
        let mut col_period = Vec::new();
        let mut col_value = Vec::new();
        for row in &result.cohorts {
            for (p, v) in row.period_values.iter().enumerate() {
                col_label.push(ScalarValue::Utf8(row.cohort_label.clone()));
                col_period.push(ScalarValue::Int32(p as i32));
                col_value.push(ScalarValue::Float64(*v));
            }
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, col_label),
            (TypeId::Int32, col_period),
            (TypeId::Float64, col_value),
        ]))
    }

    async fn run_funnel_analysis(&self) -> Result<DataBatch> {
        let table_name = self.first_string_arg()?;
        let user_col = self.named_string("user_id").ok_or_else(|| {
            ZyronError::ExecutionError(
                "FUNNEL_ANALYSIS requires user_id => 'column' named argument".into(),
            )
        })?;
        let time_col = self.named_string("event_time").ok_or_else(|| {
            ZyronError::ExecutionError(
                "FUNNEL_ANALYSIS requires event_time => 'column' named argument".into(),
            )
        })?;
        let event_col = self.named_string("event_name").ok_or_else(|| {
            ZyronError::ExecutionError(
                "FUNNEL_ANALYSIS requires event_name => 'column' named argument".into(),
            )
        })?;
        // Window in milliseconds (default 24h)
        let window_ms = self
            .named_int("window_ms")
            .unwrap_or(24 * 60 * 60 * 1000)
            .max(1);

        // Steps: collect from positional_args[1..] as string literal sequence
        let mut steps: Vec<FunnelStep> = Vec::new();
        for arg in &self.positional_args[1..] {
            if let BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            } = arg
            {
                steps.push(FunnelStep {
                    name: s.clone(),
                    event_match: s.clone(),
                });
            }
        }
        if steps.is_empty() {
            return Err(ZyronError::ExecutionError(
                "FUNNEL_ANALYSIS requires at least one step (string positional args)".into(),
            ));
        }

        let table_entry = self.resolve_table(&table_name)?;
        let columns = table_entry.columns.clone();
        let all_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();

        let user_idx = all_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&user_col))
            .ok_or_else(|| ZyronError::ColumnNotFound(user_col.clone()))?;
        let time_idx = all_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&time_col))
            .ok_or_else(|| ZyronError::ColumnNotFound(time_col.clone()))?;
        let event_idx = all_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&event_col))
            .ok_or_else(|| ZyronError::ColumnNotFound(event_col.clone()))?;

        let cfg = FunnelConfig {
            steps,
            window_ms,
            user_id_column: user_col,
            event_time_column: time_col,
        };
        let mut analyser = zyron_analytics::FunnelAnalyser::new(cfg)?;

        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            let mut user = AnalyticsValue::Null;
            let mut ts: i64 = 0;
            let mut name = String::new();
            decode_funnel_columns(
                view.data, &columns, user_idx, time_idx, event_idx, &mut user, &mut ts, &mut name,
            );
            analyser.ingest(&FunnelEvent {
                user_id: user,
                event_time_ms: ts,
                event_name: name,
            });
        });
        let result = analyser.finalise();

        let mut col_step = Vec::new();
        let mut col_users = Vec::new();
        let mut col_conv = Vec::new();
        let mut col_drop = Vec::new();
        let mut col_avg = Vec::new();
        for s in &result.steps {
            col_step.push(ScalarValue::Utf8(s.name.clone()));
            col_users.push(ScalarValue::Int64(s.users_count as i64));
            col_conv.push(ScalarValue::Float64(s.conversion_rate));
            col_drop.push(ScalarValue::Float64(s.drop_off_rate));
            col_avg.push(match s.avg_time_to_next_ms {
                Some(v) => ScalarValue::Float64(v),
                None => ScalarValue::Null,
            });
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, col_step),
            (TypeId::Int64, col_users),
            (TypeId::Float64, col_conv),
            (TypeId::Float64, col_drop),
            (TypeId::Float64, col_avg),
        ]))
    }

    fn encode_table_profile(&self, profile: &TableProfile) -> DataBatch {
        let mut col_name = Vec::new();
        let mut col_type = Vec::new();
        let mut col_nulls = Vec::new();
        let mut col_distinct = Vec::new();
        let mut col_mean = Vec::new();
        let mut col_median = Vec::new();
        let mut col_stddev = Vec::new();
        for col in &profile.columns {
            col_name.push(ScalarValue::Utf8(col.column_name.clone()));
            col_type.push(ScalarValue::Utf8(col.data_type.clone()));
            col_nulls.push(ScalarValue::Int64(col.null_count as i64));
            col_distinct.push(ScalarValue::Int64(col.distinct_count as i64));
            col_mean.push(opt_to_scalar(col.mean));
            col_median.push(opt_to_scalar(col.median));
            col_stddev.push(opt_to_scalar(col.stddev));
        }
        scalar_columns_to_batch(vec![
            (TypeId::Varchar, col_name),
            (TypeId::Varchar, col_type),
            (TypeId::Int64, col_nulls),
            (TypeId::Int64, col_distinct),
            (TypeId::Float64, col_mean),
            (TypeId::Float64, col_median),
            (TypeId::Float64, col_stddev),
        ])
    }

    fn encode_column_profile_kv(&self, profile: &ColumnProfile) -> DataBatch {
        let mut kv: Vec<(String, String)> = Vec::new();
        kv.push(("column_name".into(), profile.column_name.clone()));
        kv.push(("data_type".into(), profile.data_type.clone()));
        kv.push(("null_count".into(), profile.null_count.to_string()));
        kv.push(("null_pct".into(), format!("{:.6}", profile.null_pct)));
        kv.push(("distinct_count".into(), profile.distinct_count.to_string()));
        kv.push((
            "distinct_pct".into(),
            format!("{:.6}", profile.distinct_pct),
        ));
        if let Some(v) = profile.mean {
            kv.push(("mean".into(), format!("{}", v)));
        }
        if let Some(v) = profile.median {
            kv.push(("median".into(), format!("{}", v)));
        }
        if let Some(v) = profile.stddev {
            kv.push(("stddev".into(), format!("{}", v)));
        }
        if let Some(v) = profile.variance {
            kv.push(("variance".into(), format!("{}", v)));
        }
        if let Some(v) = profile.skewness {
            kv.push(("skewness".into(), format!("{}", v)));
        }
        if let Some(v) = profile.kurtosis {
            kv.push(("kurtosis".into(), format!("{}", v)));
        }
        if let Some(p) = &profile.percentiles {
            for (k, v) in [
                ("p1", p.p1),
                ("p5", p.p5),
                ("p10", p.p10),
                ("p25", p.p25),
                ("p50", p.p50),
                ("p75", p.p75),
                ("p90", p.p90),
                ("p95", p.p95),
                ("p99", p.p99),
            ] {
                kv.push((k.into(), format!("{}", v)));
            }
        }
        for (val, count) in &profile.most_common_values {
            kv.push((format!("mcv:{:?}", val), format!("count={}", count)));
        }
        for pat in &profile.pattern_frequencies {
            kv.push((format!("pattern:{}", pat.pattern), pat.count.to_string()));
        }
        let mut keys = Vec::with_capacity(kv.len());
        let mut vals = Vec::with_capacity(kv.len());
        for (k, v) in kv {
            keys.push(ScalarValue::Utf8(k));
            vals.push(ScalarValue::Utf8(v));
        }
        scalar_columns_to_batch(vec![(TypeId::Varchar, keys), (TypeId::Varchar, vals)])
    }

    // ---- Feature store and ML ----

    fn pos_strings_at(&self, idx: usize) -> Result<Vec<String>> {
        match self.positional_args.get(idx) {
            Some(BoundExpr::Function { name, args, .. }) if name == "array" => {
                let mut out = Vec::with_capacity(args.len());
                for e in args {
                    if let BoundExpr::Literal {
                        value: LiteralValue::String(s),
                        ..
                    } = e
                    {
                        out.push(s.clone());
                    } else {
                        return Err(ZyronError::ExecutionError(
                            "expected array of string literals".to_string(),
                        ));
                    }
                }
                Ok(out)
            }
            Some(BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            }) => Ok(vec![s.clone()]),
            _ => Err(ZyronError::ExecutionError(format!(
                "{} positional argument {} must be a string array",
                self.function_name, idx
            ))),
        }
    }

    fn pos_string_at(&self, idx: usize) -> Result<String> {
        match self.positional_args.get(idx) {
            Some(BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            }) => Ok(s.clone()),
            _ => Err(ZyronError::ExecutionError(format!(
                "{} positional argument {} must be a string literal",
                self.function_name, idx
            ))),
        }
    }

    fn pos_int_at(&self, idx: usize) -> Result<i64> {
        match self.positional_args.get(idx) {
            Some(BoundExpr::Literal {
                value: LiteralValue::Integer(i),
                ..
            }) => Ok(*i),
            _ => Err(ZyronError::ExecutionError(format!(
                "{} positional argument {} must be an integer literal",
                self.function_name, idx
            ))),
        }
    }

    fn pos_float_at(&self, idx: usize) -> Result<f64> {
        match self.positional_args.get(idx) {
            Some(BoundExpr::Literal {
                value: LiteralValue::Float(f),
                ..
            }) => Ok(*f),
            Some(BoundExpr::Literal {
                value: LiteralValue::Integer(i),
                ..
            }) => Ok(*i as f64),
            _ => Err(ZyronError::ExecutionError(format!(
                "{} positional argument {} must be a numeric literal",
                self.function_name, idx
            ))),
        }
    }

    async fn run_get_features(&self) -> Result<DataBatch> {
        // GET_FEATURES(group_name, [entity_keys], [feature_names], as_of_ms)
        let groupName = self.pos_string_at(0)?;
        let entities = self.pos_strings_at(1)?;
        let features = self.pos_strings_at(2)?;
        let asOf = self.pos_int_at(3).unwrap_or_else(|_| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0)
        });
        let store = zyron_analytics::featureStore();
        let frame = store
            .getFeatures(&groupName, &entities, &features, asOf)
            .map_err(|e| ZyronError::ExecutionError(format!("{}", e)))?;

        let mut entityCol: Vec<ScalarValue> = Vec::new();
        let mut featureCol: Vec<ScalarValue> = Vec::new();
        let mut valueCol: Vec<ScalarValue> = Vec::new();
        for row in &frame.rows {
            for (i, name) in frame.featureNames.iter().enumerate() {
                entityCol.push(ScalarValue::Utf8(row.entityKey.clone()));
                featureCol.push(ScalarValue::Utf8(name.clone()));
                valueCol.push(ScalarValue::Utf8(format!("{:?}", row.values[i])));
            }
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, entityCol),
            (TypeId::Varchar, featureCol),
            (TypeId::Varchar, valueCol),
        ]))
    }

    async fn run_feature_lineage(&self) -> Result<DataBatch> {
        let qualifiedName = self.pos_string_at(0)?;
        let registry = zyron_analytics::featureLineageRegistry();
        let guard = registry.read();
        let entry = guard.get(&qualifiedName).cloned();
        drop(guard);

        let mut sourceTable: Vec<ScalarValue> = Vec::new();
        let mut sourceColumn: Vec<ScalarValue> = Vec::new();
        let mut transform: Vec<ScalarValue> = Vec::new();
        let mut dependency: Vec<ScalarValue> = Vec::new();
        let mut lastComputed: Vec<ScalarValue> = Vec::new();
        if let Some(e) = entry {
            for t in &e.sourceTables {
                sourceTable.push(ScalarValue::Utf8(t.clone()));
                sourceColumn.push(ScalarValue::Utf8(String::new()));
                transform.push(ScalarValue::Utf8(e.transformChain.join(" ; ")));
                dependency.push(ScalarValue::Utf8(e.dependencies.join(",")));
                lastComputed.push(ScalarValue::Int64(e.lastComputedMs));
            }
            for (tab, col) in &e.sourceColumns {
                sourceTable.push(ScalarValue::Utf8(tab.clone()));
                sourceColumn.push(ScalarValue::Utf8(col.clone()));
                transform.push(ScalarValue::Utf8(e.transformChain.join(" ; ")));
                dependency.push(ScalarValue::Utf8(e.dependencies.join(",")));
                lastComputed.push(ScalarValue::Int64(e.lastComputedMs));
            }
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, sourceTable),
            (TypeId::Varchar, sourceColumn),
            (TypeId::Varchar, transform),
            (TypeId::Varchar, dependency),
            (TypeId::Int64, lastComputed),
        ]))
    }

    async fn run_predict_batch(&self) -> Result<DataBatch> {
        // PREDICT_BATCH(model_name, source_table)
        let modelName = self.pos_string_at(0)?;
        let sourceTable = self.pos_string_at(1)?;
        let handle = zyron_analytics::InferenceHandle::resolve(&modelName).ok_or_else(|| {
            ZyronError::ExecutionError(format!("model '{}' not found in cache", modelName))
        })?;

        let tableEntry = self.resolve_table(&sourceTable)?;
        let columns = tableEntry.columns.clone();
        let featureCols = &handle.model.featureColumns;
        let mut featureIdx: Vec<usize> = Vec::with_capacity(featureCols.len());
        for f in featureCols {
            let idx = columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(f))
                .ok_or_else(|| ZyronError::ColumnNotFound(f.clone()))?;
            featureIdx.push(idx);
        }

        let heap_file = self.ctx.get_heap_file(tableEntry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        let mut rowIdx: Vec<ScalarValue> = Vec::new();
        let mut preds: Vec<ScalarValue> = Vec::new();
        let mut featureBuf = vec![0.0f64; featureCols.len()];
        let mut rowNum: i64 = 0;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            decode_row_features(view.data, &columns, &featureIdx, &mut featureBuf);
            let p = handle.predictOne(&featureBuf);
            rowIdx.push(ScalarValue::Int64(rowNum));
            preds.push(ScalarValue::Float64(p));
            rowNum += 1;
        });
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, rowIdx),
            (TypeId::Float64, preds),
        ]))
    }

    async fn run_forecast(&self) -> Result<DataBatch> {
        // FORECAST(table_name, value_col, periods, method)
        let table = self.pos_string_at(0)?;
        let valueCol = self.pos_string_at(1)?;
        let periods = self.pos_int_at(2)? as usize;
        let methodStr = self
            .pos_string_at(3)
            .unwrap_or_else(|_| "linear_trend".into());
        let method = zyron_analytics::ForecastMethod::fromStr(&methodStr).ok_or_else(|| {
            ZyronError::ExecutionError(format!("unknown forecast method '{}'", methodStr))
        })?;
        let values = self.collect_numeric_column(&table, &valueCol).await?;
        let extra = std::collections::HashMap::new();
        let result = zyron_analytics::forecast(&values, periods, method, &extra)?;
        let mut stepCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut valCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for (i, v) in result.iter().enumerate() {
            stepCol.push(ScalarValue::Int64(i as i64));
            valCol.push(ScalarValue::Float64(*v));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, stepCol),
            (TypeId::Float64, valCol),
        ]))
    }

    async fn run_anomaly_detect(&self) -> Result<DataBatch> {
        // ANOMALY_DETECT(table_name, value_col, [method], [threshold])
        let table = self.pos_string_at(0)?;
        let valueCol = self.pos_string_at(1)?;
        let methodStr = self.pos_string_at(2).unwrap_or_else(|_| "zscore".into());
        let threshold = self.pos_float_at(3).unwrap_or(3.0);
        let method = zyron_analytics::AnomalyMethod::fromStr(&methodStr).ok_or_else(|| {
            ZyronError::ExecutionError(format!("unknown anomaly method '{}'", methodStr))
        })?;
        let values = self.collect_numeric_column(&table, &valueCol).await?;
        let result = zyron_analytics::anomalyDetect(&values, method, threshold);
        let mut idxCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut isCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut scoreCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for (i, flag, score) in result {
            idxCol.push(ScalarValue::Int64(i as i64));
            isCol.push(ScalarValue::Boolean(flag));
            scoreCol.push(ScalarValue::Float64(score));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idxCol),
            (TypeId::Boolean, isCol),
            (TypeId::Float64, scoreCol),
        ]))
    }

    async fn run_acf(&self) -> Result<DataBatch> {
        let table = self.pos_string_at(0)?;
        let valueCol = self.pos_string_at(1)?;
        let maxLag = self.pos_int_at(2).unwrap_or(20) as usize;
        let values = self.collect_numeric_column(&table, &valueCol).await?;
        let result = zyron_analytics::acf(&values, maxLag);
        let mut lagCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut valCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for (i, v) in result.iter().enumerate() {
            lagCol.push(ScalarValue::Int64(i as i64));
            valCol.push(ScalarValue::Float64(*v));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, lagCol),
            (TypeId::Float64, valCol),
        ]))
    }

    async fn run_pacf(&self) -> Result<DataBatch> {
        let table = self.pos_string_at(0)?;
        let valueCol = self.pos_string_at(1)?;
        let maxLag = self.pos_int_at(2).unwrap_or(20) as usize;
        let values = self.collect_numeric_column(&table, &valueCol).await?;
        let result = zyron_analytics::pacf(&values, maxLag);
        let mut lagCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut valCol: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for (i, v) in result.iter().enumerate() {
            lagCol.push(ScalarValue::Int64(i as i64));
            valCol.push(ScalarValue::Float64(*v));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, lagCol),
            (TypeId::Float64, valCol),
        ]))
    }

    async fn run_seasonality_detect(&self) -> Result<DataBatch> {
        let table = self.pos_string_at(0)?;
        let valueCol = self.pos_string_at(1)?;
        let maxPeriod = self.pos_int_at(2).unwrap_or(30) as usize;
        let values = self.collect_numeric_column(&table, &valueCol).await?;
        let comps = zyron_analytics::seasonalityDetect(&values, maxPeriod);
        let mut periodCol: Vec<ScalarValue> = Vec::with_capacity(comps.len());
        let mut strengthCol: Vec<ScalarValue> = Vec::with_capacity(comps.len());
        for c in &comps {
            periodCol.push(ScalarValue::Int64(c.period as i64));
            let strength: f64 = c.seasonalIndices.iter().map(|v| v.abs()).sum::<f64>()
                / c.seasonalIndices.len().max(1) as f64;
            strengthCol.push(ScalarValue::Float64(strength));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, periodCol),
            (TypeId::Float64, strengthCol),
        ]))
    }

    async fn run_change_points(&self) -> Result<DataBatch> {
        let table = self.pos_string_at(0)?;
        let valueCol = self.pos_string_at(1)?;
        let threshold = self.pos_float_at(2).unwrap_or(10.0);
        let values = self.collect_numeric_column(&table, &valueCol).await?;
        let points = zyron_analytics::changePoints(&values, threshold);
        let col: Vec<ScalarValue> = points
            .into_iter()
            .map(|p| ScalarValue::Int64(p as i64))
            .collect();
        Ok(scalar_columns_to_batch(vec![(TypeId::Int64, col)]))
    }

    async fn run_date_features(&self) -> Result<DataBatch> {
        // DATE_FEATURES(timestamp_ms_literal)
        let ts = self.pos_int_at(0)?;
        let f = zyron_analytics::ml::transforms::dateFeatures(ts);
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int32, vec![ScalarValue::Int32(f[0] as i32)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[1] as i32)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[2] as i32)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[3] as i32)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[4] as i32)]),
            (TypeId::Boolean, vec![ScalarValue::Boolean(f[5] != 0.0)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[6] as i32)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[7] as i32)]),
            (TypeId::Int32, vec![ScalarValue::Int32(f[8] as i32)]),
        ]))
    }

    async fn run_polynomial_features(&self) -> Result<DataBatch> {
        let degree = self.pos_int_at(0)? as u32;
        let mut features: Vec<f64> = Vec::new();
        for i in 1..self.positional_args.len() {
            features.push(self.pos_float_at(i)?);
        }
        let expanded = zyron_analytics::ml::transforms::polynomialFeatures(&features, degree);
        let mut termCol: Vec<ScalarValue> = Vec::with_capacity(expanded.len());
        let mut valCol: Vec<ScalarValue> = Vec::with_capacity(expanded.len());
        for (i, v) in expanded.iter().enumerate() {
            termCol.push(ScalarValue::Utf8(format!("term_{}", i)));
            valCol.push(ScalarValue::Float64(*v));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, termCol),
            (TypeId::Float64, valCol),
        ]))
    }

    async fn run_feature_parity_check(&self) -> Result<DataBatch> {
        // FEATURE_PARITY_CHECK(group_name, sample_size)
        let groupName = self.pos_string_at(0)?;
        let store = zyron_analytics::featureStore();
        let group = store.group(&groupName).ok_or_else(|| {
            ZyronError::ExecutionError(format!("group '{}' not found", groupName))
        })?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        // Compare offline path (current materialized values) against itself,
        // since the in-process feature store is the single source of truth.
        // External offline/online divergence would surface here when an
        // external pipeline writes to a different backend.
        let names: Vec<String> = group.features.iter().map(|f| f.name.clone()).collect();
        let entities: Vec<String> = Vec::new();
        let frame = store.getFeatures(&groupName, &entities, &names, now)?;
        let parity = zyron_analytics::featureParityCheck(&frame, &frame);
        let entity = vec![ScalarValue::Utf8("__summary__".to_string())];
        let feature = vec![ScalarValue::Utf8("matched_vs_mismatched".to_string())];
        let offline = vec![ScalarValue::Utf8(format!("{}", parity.matched))];
        let online = vec![ScalarValue::Utf8(format!("{}", parity.mismatched))];
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, entity),
            (TypeId::Varchar, feature),
            (TypeId::Varchar, offline),
            (TypeId::Varchar, online),
        ]))
    }

    async fn run_explain_prediction(&self) -> Result<DataBatch> {
        // EXPLAIN_PREDICTION(model_name, feature1, feature2, ...)
        let modelName = self.pos_string_at(0)?;
        let handle = zyron_analytics::InferenceHandle::resolve(&modelName).ok_or_else(|| {
            ZyronError::ExecutionError(format!("model '{}' not found", modelName))
        })?;
        let mut features = Vec::new();
        for i in 1..self.positional_args.len() {
            features.push(self.pos_float_at(i)?);
        }
        let mut nameCol: Vec<ScalarValue> = Vec::new();
        let mut contribCol: Vec<ScalarValue> = Vec::new();
        // Linear-style explanation: weight_j * standardized_feature_j
        // For tree-based models, contribution is the marginal contribution
        // approximated by weight * raw value or zero
        for (j, name) in handle.model.featureColumns.iter().enumerate() {
            let raw = features.get(j).copied().unwrap_or(0.0);
            let mean = handle.model.featureMean.get(j).copied().unwrap_or(0.0);
            let std = handle.model.featureStd.get(j).copied().unwrap_or(1.0);
            let std = if std == 0.0 { 1.0 } else { std };
            let standardized = (raw - mean) / std;
            let weight = handle.model.weights.get(j).copied().unwrap_or(0.0);
            nameCol.push(ScalarValue::Utf8(name.clone()));
            contribCol.push(ScalarValue::Float64(weight * standardized));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, nameCol),
            (TypeId::Float64, contribCol),
        ]))
    }

    async fn run_model_lineage(&self) -> Result<DataBatch> {
        let modelName = self.pos_string_at(0)?;
        let handle = zyron_analytics::InferenceHandle::resolve(&modelName).ok_or_else(|| {
            ZyronError::ExecutionError(format!("model '{}' not found", modelName))
        })?;
        let mut keyCol: Vec<ScalarValue> = Vec::new();
        let mut valCol: Vec<ScalarValue> = Vec::new();
        let pushRow = |k: &str, v: String, kc: &mut Vec<ScalarValue>, vc: &mut Vec<ScalarValue>| {
            kc.push(ScalarValue::Utf8(k.to_string()));
            vc.push(ScalarValue::Utf8(v));
        };
        pushRow(
            "model_type",
            format!("{:?}", handle.model.modelType),
            &mut keyCol,
            &mut valCol,
        );
        pushRow(
            "feature_columns",
            handle.model.featureColumns.join(","),
            &mut keyCol,
            &mut valCol,
        );
        if let Some(t) = &handle.model.targetColumn {
            pushRow("target_column", t.clone(), &mut keyCol, &mut valCol);
        }
        pushRow(
            "training_rows",
            handle.model.trainingRows.to_string(),
            &mut keyCol,
            &mut valCol,
        );
        pushRow(
            "created_at_ms",
            handle.model.createdAtMs.to_string(),
            &mut keyCol,
            &mut valCol,
        );
        for (k, v) in &handle.model.metrics {
            pushRow(
                &format!("metric:{}", k),
                v.to_string(),
                &mut keyCol,
                &mut valCol,
            );
        }
        for (k, v) in &handle.model.hyperparameters.values {
            pushRow(
                &format!("hp:{}", k),
                v.to_string(),
                &mut keyCol,
                &mut valCol,
            );
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, keyCol),
            (TypeId::Varchar, valCol),
        ]))
    }

    // Gathers column names from positional string literals starting at the
    // given index. Each literal may carry a comma separated list
    fn feature_columns_from(&self, start: usize) -> Result<Vec<String>> {
        let mut cols = Vec::new();
        for arg in self.positional_args.iter().skip(start) {
            if let BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            } = arg
            {
                for part in s.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        cols.push(trimmed.to_string());
                    }
                }
            } else {
                break;
            }
        }
        if cols.is_empty() {
            return Err(ZyronError::ExecutionError(format!(
                "{} requires at least one column name argument",
                self.function_name
            )));
        }
        Ok(cols)
    }

    // Scans the table and decodes the requested numeric columns per row.
    // Rows where any requested column is null or non numeric are skipped.
    // Returns the original scan ordinals of the kept rows alongside the rows
    async fn collect_numeric_matrix(
        &self,
        table_name: &str,
        column_names: &[String],
    ) -> Result<(Vec<i64>, Vec<Vec<f64>>)> {
        let table_entry = self.resolve_table(table_name)?;
        let columns = table_entry.columns.clone();
        let mut indices = Vec::with_capacity(column_names.len());
        for name in column_names {
            let idx = columns
                .iter()
                .position(|c| c.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| ZyronError::ColumnNotFound(name.clone()))?;
            indices.push(idx);
        }
        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        let mut row_ids: Vec<i64> = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut buf = vec![0.0f64; indices.len()];
        let mut ordinal: i64 = 0;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            decode_row_features(view.data, &columns, &indices, &mut buf);
            if buf.iter().all(|v| v.is_finite()) {
                row_ids.push(ordinal);
                rows.push(buf.clone());
            }
            ordinal += 1;
        });
        Ok((row_ids, rows))
    }

    // Scans one column as text, null values become empty strings so the
    // output stays aligned with the table scan ordinals
    async fn collect_string_column(
        &self,
        table_name: &str,
        column_name: &str,
    ) -> Result<Vec<String>> {
        let table_entry = self.resolve_table(table_name)?;
        let columns = table_entry.columns.clone();
        let idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(column_name))
            .ok_or_else(|| ZyronError::ColumnNotFound(column_name.to_string()))?;
        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        let mut out: Vec<String> = Vec::new();
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            out.push(decode_column_to_text(view.data, &columns, idx).unwrap_or_default());
        });
        Ok(out)
    }

    // Scans a text column and a numeric column together, skipping rows
    // where the numeric value is null or non numeric. A null text value
    // becomes an empty string
    async fn collect_string_numeric_pairs(
        &self,
        table_name: &str,
        string_column: &str,
        numeric_column: &str,
    ) -> Result<(Vec<i64>, Vec<String>, Vec<f64>)> {
        let table_entry = self.resolve_table(table_name)?;
        let columns = table_entry.columns.clone();
        let str_idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(string_column))
            .ok_or_else(|| ZyronError::ColumnNotFound(string_column.to_string()))?;
        let num_idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(numeric_column))
            .ok_or_else(|| ZyronError::ColumnNotFound(numeric_column.to_string()))?;
        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        let mut row_ids: Vec<i64> = Vec::new();
        let mut strings: Vec<String> = Vec::new();
        let mut numbers: Vec<f64> = Vec::new();
        let mut buf = vec![0.0f64; 1];
        let mut ordinal: i64 = 0;
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            decode_row_features(view.data, &columns, &[num_idx], &mut buf);
            if buf[0].is_finite() {
                row_ids.push(ordinal);
                strings
                    .push(decode_column_to_text(view.data, &columns, str_idx).unwrap_or_default());
                numbers.push(buf[0]);
            }
            ordinal += 1;
        });
        Ok((row_ids, strings, numbers))
    }

    async fn run_detect_anomalies(&self) -> Result<DataBatch> {
        // DETECT_ANOMALIES(table, columns..., method => 'zscore', threshold => 3.0,
        //                  contamination => 0.1, multiplier => 1.5)
        let table = self.pos_string_at(0)?;
        let cols = self.feature_columns_from(1)?;
        let method = self
            .named_string("method")
            .unwrap_or_else(|| "zscore".into());
        let params = zyron_analytics::AnomalyParams {
            contamination: self.named_float("contamination").unwrap_or(0.1),
            threshold: self.named_float("threshold").unwrap_or(3.0),
            multiplier: self.named_float("multiplier").unwrap_or(1.5),
        };
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let result = zyron_analytics::detectAnomalies(&rows, &method, &params)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut flag_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut score_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for (i, flag, score) in result {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            flag_col.push(ScalarValue::Boolean(flag));
            score_col.push(ScalarValue::Float64(score));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Boolean, flag_col),
            (TypeId::Float64, score_col),
        ]))
    }

    async fn run_detect_anomalies_timeseries(&self) -> Result<DataBatch> {
        // DETECT_ANOMALIES_TIMESERIES(table, value_col, method => 'ewma',
        //                             window => 10, period => 7, alpha => 0.3,
        //                             threshold => 3.0)
        let table = self.pos_string_at(0)?;
        let value_col = self.pos_string_at(1)?;
        let method = self.named_string("method").unwrap_or_else(|| "ewma".into());
        let params = zyron_analytics::TimeseriesAnomalyParams {
            window: self.named_int("window").unwrap_or(10).max(1) as usize,
            period: self.named_int("period").unwrap_or(7).max(1) as usize,
            alpha: self.named_float("alpha").unwrap_or(0.3),
            threshold: self.named_float("threshold").unwrap_or(3.0),
        };
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &[value_col]).await?;
        let values: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let result = zyron_analytics::detectAnomaliesTimeseries(&values, &method, &params)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut flag_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut score_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for (i, flag, score) in result {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            flag_col.push(ScalarValue::Boolean(flag));
            score_col.push(ScalarValue::Float64(score));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Boolean, flag_col),
            (TypeId::Float64, score_col),
        ]))
    }

    async fn run_detect_drift(&self) -> Result<DataBatch> {
        // DETECT_DRIFT(ref_table, ref_col, cur_table, cur_col, method => 'ks_test')
        let ref_table = self.pos_string_at(0)?;
        let ref_col = self.pos_string_at(1)?;
        let cur_table = self.pos_string_at(2)?;
        let cur_col = self.pos_string_at(3)?;
        let method = self
            .named_string("method")
            .unwrap_or_else(|| "ks_test".into());
        let reference = self.collect_numeric_column(&ref_table, &ref_col).await?;
        let current = self.collect_numeric_column(&cur_table, &cur_col).await?;
        let drift = zyron_analytics::detectDrift(&reference, &current, &method)?;
        Ok(scalar_columns_to_batch(vec![
            (
                TypeId::Varchar,
                vec![ScalarValue::Utf8(method.to_ascii_lowercase())],
            ),
            (TypeId::Float64, vec![ScalarValue::Float64(drift.statistic)]),
            (
                TypeId::Float64,
                vec![match drift.pValue {
                    Some(p) => ScalarValue::Float64(p),
                    None => ScalarValue::Null,
                }],
            ),
            (TypeId::Boolean, vec![ScalarValue::Boolean(drift.drifted)]),
        ]))
    }

    fn kmeans_hyperparams(&self) -> (usize, usize, u64) {
        let k = self.named_int("k").unwrap_or(8).max(1) as usize;
        let max_iter = self.named_int("max_iter").unwrap_or(100).max(1) as usize;
        let seed = self
            .named_int("seed")
            .map(|s| s.max(0) as u64)
            .unwrap_or(zyron_analytics::DEFAULT_KMEANS_SEED);
        (k, max_iter, seed)
    }

    async fn run_kmeans_cluster(&self) -> Result<DataBatch> {
        // KMEANS_CLUSTER(table, columns..., k => 8, max_iter => 100, seed => 42)
        let table = self.pos_string_at(0)?;
        let cols = self.feature_columns_from(1)?;
        let (k, max_iter, seed) = self.kmeans_hyperparams();
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let out = zyron_analytics::kmeansCluster(&rows, k, max_iter, seed)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(rows.len());
        let mut cluster_col: Vec<ScalarValue> = Vec::with_capacity(rows.len());
        let mut dist_col: Vec<ScalarValue> = Vec::with_capacity(rows.len());
        for (i, (cluster, dist)) in out.assignments.iter().enumerate() {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            cluster_col.push(ScalarValue::Int64(*cluster as i64));
            dist_col.push(ScalarValue::Float64(*dist));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Int64, cluster_col),
            (TypeId::Float64, dist_col),
        ]))
    }

    async fn run_kmeans_centroids(&self) -> Result<DataBatch> {
        // KMEANS_CENTROIDS(table, columns..., k => 8, max_iter => 100, seed => 42)
        let table = self.pos_string_at(0)?;
        let cols = self.feature_columns_from(1)?;
        let (k, max_iter, seed) = self.kmeans_hyperparams();
        let (_, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let centroids = zyron_analytics::kmeansCentroids(&rows, k, max_iter, seed)?;
        let mut cluster_col: Vec<ScalarValue> = Vec::new();
        let mut feature_col: Vec<ScalarValue> = Vec::new();
        let mut value_col: Vec<ScalarValue> = Vec::new();
        for (c, centroid) in centroids.iter().enumerate() {
            for (j, v) in centroid.iter().enumerate() {
                cluster_col.push(ScalarValue::Int64(c as i64));
                feature_col.push(ScalarValue::Utf8(cols[j].clone()));
                value_col.push(ScalarValue::Float64(*v));
            }
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, cluster_col),
            (TypeId::Varchar, feature_col),
            (TypeId::Float64, value_col),
        ]))
    }

    async fn run_kmeans_elbow(&self) -> Result<DataBatch> {
        // KMEANS_ELBOW(table, columns..., k_min => 1, k_max => 8)
        let table = self.pos_string_at(0)?;
        let cols = self.feature_columns_from(1)?;
        let k_min = self.named_int("k_min").unwrap_or(1).max(1) as usize;
        let k_max = self.named_int("k_max").unwrap_or(8).max(1) as usize;
        if k_min > k_max {
            return Err(ZyronError::ExecutionError(format!(
                "KMEANS_ELBOW k_min {} exceeds k_max {}",
                k_min, k_max
            )));
        }
        let (_, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let k_hi = k_max.min(rows.len());
        if k_min > k_hi {
            return Err(ZyronError::ExecutionError(format!(
                "KMEANS_ELBOW k_min {} exceeds the {} usable rows",
                k_min,
                rows.len()
            )));
        }
        let range: Vec<usize> = (k_min..=k_hi).collect();
        let curve = zyron_analytics::kmeansElbow(&rows, &range)?;
        let mut k_col: Vec<ScalarValue> = Vec::with_capacity(curve.len());
        let mut inertia_col: Vec<ScalarValue> = Vec::with_capacity(curve.len());
        for (k, inertia) in curve {
            k_col.push(ScalarValue::Int64(k as i64));
            inertia_col.push(ScalarValue::Float64(inertia));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, k_col),
            (TypeId::Float64, inertia_col),
        ]))
    }

    async fn run_kmeans_predict(&self) -> Result<DataBatch> {
        // KMEANS_PREDICT(table, columns..., centroids => '1.0,2.0;3.0,4.0')
        let table = self.pos_string_at(0)?;
        let cols = self.feature_columns_from(1)?;
        let spec = self.named_string("centroids").ok_or_else(|| {
            ZyronError::ExecutionError(
                "KMEANS_PREDICT requires centroids => 'v1,v2;v3,v4' named argument".into(),
            )
        })?;
        let mut centroids: Vec<Vec<f64>> = Vec::new();
        for row_spec in spec.split(';') {
            let trimmed = row_spec.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut centroid = Vec::new();
            for part in trimmed.split(',') {
                let value: f64 = part.trim().parse().map_err(|_| {
                    ZyronError::ExecutionError(format!(
                        "KMEANS_PREDICT centroid component '{}' is not a number",
                        part.trim()
                    ))
                })?;
                centroid.push(value);
            }
            centroids.push(centroid);
        }
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let preds = zyron_analytics::kmeansPredict(&rows, &centroids)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(preds.len());
        let mut cluster_col: Vec<ScalarValue> = Vec::with_capacity(preds.len());
        for (i, cluster) in preds.iter().enumerate() {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            cluster_col.push(ScalarValue::Int64(*cluster as i64));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Int64, cluster_col),
        ]))
    }

    async fn run_causal_impact(&self) -> Result<DataBatch> {
        // CAUSAL_IMPACT(table, outcome_col, time_col, intervention_index)
        let table = self.pos_string_at(0)?;
        let outcome_col = self.pos_string_at(1)?;
        let time_col = self.pos_string_at(2)?;
        let intervention = match self.named_int("intervention_index") {
            Some(v) => v,
            None => self.pos_int_at(3)?,
        };
        if intervention < 0 {
            return Err(ZyronError::ExecutionError(
                "CAUSAL_IMPACT intervention index must be nonnegative".into(),
            ));
        }
        let (_, rows) = self
            .collect_numeric_matrix(&table, &[outcome_col, time_col])
            .await?;
        let outcome: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let timestamps: Vec<i64> = rows.iter().map(|r| r[1] as i64).collect();
        let points = zyron_analytics::causalImpact(&outcome, &timestamps, intervention as usize)?;
        let mut cols: Vec<Vec<ScalarValue>> = vec![Vec::with_capacity(points.len()); 9];
        for p in &points {
            cols[0].push(ScalarValue::Int64(p.index as i64));
            cols[1].push(ScalarValue::Float64(p.actual));
            cols[2].push(ScalarValue::Float64(p.predicted));
            cols[3].push(ScalarValue::Float64(p.effect));
            cols[4].push(ScalarValue::Float64(p.lower));
            cols[5].push(ScalarValue::Float64(p.upper));
            cols[6].push(ScalarValue::Float64(p.cumulativeEffect));
            cols[7].push(ScalarValue::Float64(p.cumulativeLower));
            cols[8].push(ScalarValue::Float64(p.cumulativeUpper));
        }
        let mut it = cols.into_iter();
        let idx = it.next().unwrap_or_default();
        let rest: Vec<Vec<ScalarValue>> = it.collect();
        let mut batch_cols = vec![(TypeId::Int64, idx)];
        for c in rest {
            batch_cols.push((TypeId::Float64, c));
        }
        Ok(scalar_columns_to_batch(batch_cols))
    }

    async fn run_counterfactual(&self) -> Result<DataBatch> {
        // COUNTERFACTUAL(table, outcome_col, treatment_col, feature_cols...)
        let table = self.pos_string_at(0)?;
        let outcome_col = self.pos_string_at(1)?;
        let treatment_col = self.pos_string_at(2)?;
        let features = self.feature_columns_from(3)?;
        let mut all_cols = vec![outcome_col, treatment_col];
        all_cols.extend(features);
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &all_cols).await?;
        let observed: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let flags: Vec<f64> = rows.iter().map(|r| r[1]).collect();
        let feature_rows: Vec<Vec<f64>> = rows.iter().map(|r| r[2..].to_vec()).collect();
        let result = zyron_analytics::counterfactual(&observed, &flags, &feature_rows)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut observed_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut cf_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        let mut effect_col: Vec<ScalarValue> = Vec::with_capacity(result.len());
        for r in &result {
            idx_col.push(ScalarValue::Int64(row_ids[r.index]));
            observed_col.push(ScalarValue::Float64(r.observed));
            cf_col.push(ScalarValue::Float64(r.counterfactual));
            effect_col.push(ScalarValue::Float64(r.effect));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Float64, observed_col),
            (TypeId::Float64, cf_col),
            (TypeId::Float64, effect_col),
        ]))
    }

    async fn run_propensity_match(&self) -> Result<DataBatch> {
        // PROPENSITY_MATCH(table, treatment_col, feature_cols..., caliper => 0.1)
        let table = self.pos_string_at(0)?;
        let treatment_col = self.pos_string_at(1)?;
        let features = self.feature_columns_from(2)?;
        let caliper = self.named_float("caliper").unwrap_or(0.1);
        let mut all_cols = vec![treatment_col];
        all_cols.extend(features);
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &all_cols).await?;
        let mut treated_rows: Vec<Vec<f64>> = Vec::new();
        let mut treated_ids: Vec<i64> = Vec::new();
        let mut control_rows: Vec<Vec<f64>> = Vec::new();
        let mut control_ids: Vec<i64> = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            if r[0] != 0.0 {
                treated_rows.push(r[1..].to_vec());
                treated_ids.push(row_ids[i]);
            } else {
                control_rows.push(r[1..].to_vec());
                control_ids.push(row_ids[i]);
            }
        }
        let pairs = zyron_analytics::propensityMatch(&treated_rows, &control_rows, caliper)?;
        let mut treated_col: Vec<ScalarValue> = Vec::with_capacity(pairs.len());
        let mut control_col: Vec<ScalarValue> = Vec::with_capacity(pairs.len());
        let mut diff_col: Vec<ScalarValue> = Vec::with_capacity(pairs.len());
        for (ti, ci, diff) in pairs {
            treated_col.push(ScalarValue::Int64(treated_ids[ti]));
            control_col.push(ScalarValue::Int64(control_ids[ci]));
            diff_col.push(ScalarValue::Float64(diff));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, treated_col),
            (TypeId::Int64, control_col),
            (TypeId::Float64, diff_col),
        ]))
    }

    async fn run_ab_test_analysis(&self) -> Result<DataBatch> {
        // AB_TEST_ANALYSIS(table, variant_col, outcome_col, metric_type => 'proportion')
        let table = self.pos_string_at(0)?;
        let variant_col = self.pos_string_at(1)?;
        let outcome_col = self.pos_string_at(2)?;
        let metric_type = self
            .named_string("metric_type")
            .unwrap_or_else(|| "proportion".into());
        let (_, variants, outcomes) = self
            .collect_string_numeric_pairs(&table, &variant_col, &outcome_col)
            .await?;
        let stats = zyron_analytics::abTestAnalysis(&variants, &outcomes, &metric_type)?;
        let opt = |v: Option<f64>| match v {
            Some(f) => ScalarValue::Float64(f),
            None => ScalarValue::Null,
        };
        let mut variant_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        let mut n_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        let mut mean_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        let mut lift_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        let mut lo_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        let mut hi_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        let mut p_out: Vec<ScalarValue> = Vec::with_capacity(stats.len());
        for s in &stats {
            variant_out.push(ScalarValue::Utf8(s.variant.clone()));
            n_out.push(ScalarValue::Int64(s.n as i64));
            mean_out.push(ScalarValue::Float64(s.mean));
            lift_out.push(opt(s.lift));
            lo_out.push(opt(s.ciLower));
            hi_out.push(opt(s.ciUpper));
            p_out.push(opt(s.pValue));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, variant_out),
            (TypeId::Int64, n_out),
            (TypeId::Float64, mean_out),
            (TypeId::Float64, lift_out),
            (TypeId::Float64, lo_out),
            (TypeId::Float64, hi_out),
            (TypeId::Float64, p_out),
        ]))
    }

    async fn run_ab_test_sample_size(&self) -> Result<DataBatch> {
        // AB_TEST_SAMPLE_SIZE(baseline_conversion, mde, power, alpha)
        let baseline = self.pos_float_at(0)?;
        let mde = self.pos_float_at(1)?;
        let power = self.pos_float_at(2)?;
        let alpha = self.pos_float_at(3)?;
        let n = zyron_analytics::abTestSampleSize(baseline, mde, power, alpha)?;
        Ok(scalar_columns_to_batch(vec![(
            TypeId::Int64,
            vec![ScalarValue::Int64(n.ceil() as i64)],
        )]))
    }

    async fn run_target_encode(&self) -> Result<DataBatch> {
        // TARGET_ENCODE(table, category_col, target_col, smoothing => 10.0)
        let table = self.pos_string_at(0)?;
        let category_col = self.pos_string_at(1)?;
        let target_col = self.pos_string_at(2)?;
        let smoothing = self.named_float("smoothing").unwrap_or(10.0);
        let (row_ids, categories, targets) = self
            .collect_string_numeric_pairs(&table, &category_col, &target_col)
            .await?;
        let (encoded, _) = zyron_analytics::targetEncode(&categories, &targets, smoothing)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(encoded.len());
        let mut cat_col: Vec<ScalarValue> = Vec::with_capacity(encoded.len());
        let mut enc_col: Vec<ScalarValue> = Vec::with_capacity(encoded.len());
        for i in 0..encoded.len() {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            cat_col.push(ScalarValue::Utf8(categories[i].clone()));
            enc_col.push(ScalarValue::Float64(encoded[i]));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Varchar, cat_col),
            (TypeId::Float64, enc_col),
        ]))
    }

    async fn run_tfidf(&self) -> Result<DataBatch> {
        // TFIDF(table, text_col)
        let table = self.pos_string_at(0)?;
        let text_col = self.pos_string_at(1)?;
        let docs = self.collect_string_column(&table, &text_col).await?;
        let scored = zyron_analytics::tfidf(&docs);
        let mut doc_col: Vec<ScalarValue> = Vec::new();
        let mut term_col: Vec<ScalarValue> = Vec::new();
        let mut score_col: Vec<ScalarValue> = Vec::new();
        for (doc_idx, terms) in scored.iter().enumerate() {
            for (term, score) in terms {
                doc_col.push(ScalarValue::Int64(doc_idx as i64));
                term_col.push(ScalarValue::Utf8(term.clone()));
                score_col.push(ScalarValue::Float64(*score));
            }
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, doc_col),
            (TypeId::Varchar, term_col),
            (TypeId::Float64, score_col),
        ]))
    }

    async fn run_label_encode(&self) -> Result<DataBatch> {
        // LABEL_ENCODE(table, category_col)
        let table = self.pos_string_at(0)?;
        let category_col = self.pos_string_at(1)?;
        let categories = self.collect_string_column(&table, &category_col).await?;
        let (codes, _) = zyron_analytics::labelEncode(&categories);
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(codes.len());
        let mut cat_col: Vec<ScalarValue> = Vec::with_capacity(codes.len());
        let mut code_col: Vec<ScalarValue> = Vec::with_capacity(codes.len());
        for (i, code) in codes.iter().enumerate() {
            idx_col.push(ScalarValue::Int64(i as i64));
            cat_col.push(ScalarValue::Utf8(categories[i].clone()));
            code_col.push(ScalarValue::Int64(*code));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Varchar, cat_col),
            (TypeId::Int64, code_col),
        ]))
    }

    async fn run_robust_scale(&self) -> Result<DataBatch> {
        // ROBUST_SCALE(table, value_col)
        let table = self.pos_string_at(0)?;
        let value_col = self.pos_string_at(1)?;
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &[value_col]).await?;
        let values: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let (scaled, median, iqr) = zyron_analytics::robustScale(&values)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(scaled.len());
        let mut scaled_col: Vec<ScalarValue> = Vec::with_capacity(scaled.len());
        let mut median_col: Vec<ScalarValue> = Vec::with_capacity(scaled.len());
        let mut iqr_col: Vec<ScalarValue> = Vec::with_capacity(scaled.len());
        for (i, v) in scaled.iter().enumerate() {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            scaled_col.push(ScalarValue::Float64(*v));
            median_col.push(ScalarValue::Float64(median));
            iqr_col.push(ScalarValue::Float64(iqr));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Float64, scaled_col),
            (TypeId::Float64, median_col),
            (TypeId::Float64, iqr_col),
        ]))
    }

    async fn run_one_hot_encode(&self) -> Result<DataBatch> {
        // ONE_HOT_ENCODE(table, category_col)
        let table = self.pos_string_at(0)?;
        let category_col = self.pos_string_at(1)?;
        let categories = self.collect_string_column(&table, &category_col).await?;
        let rows = zyron_analytics::oneHotEncode(&categories);
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(rows.len());
        let mut cat_col: Vec<ScalarValue> = Vec::with_capacity(rows.len());
        let mut ind_col: Vec<ScalarValue> = Vec::with_capacity(rows.len());
        for (row_idx, category, indicator) in rows {
            idx_col.push(ScalarValue::Int64(row_idx as i64));
            cat_col.push(ScalarValue::Utf8(category));
            ind_col.push(ScalarValue::Int64(indicator));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Varchar, cat_col),
            (TypeId::Int64, ind_col),
        ]))
    }

    async fn run_text_features(&self) -> Result<DataBatch> {
        // TEXT_FEATURES(table, text_col)
        let table = self.pos_string_at(0)?;
        let text_col = self.pos_string_at(1)?;
        let texts = self.collect_string_column(&table, &text_col).await?;
        let features = zyron_analytics::textFeatures(&texts);
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        let mut length_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        let mut words_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        let mut chars_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        let mut avg_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        let mut sent_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        let mut punct_col: Vec<ScalarValue> = Vec::with_capacity(features.len());
        for (i, f) in features.iter().enumerate() {
            idx_col.push(ScalarValue::Int64(i as i64));
            length_col.push(ScalarValue::Int64(f.length));
            words_col.push(ScalarValue::Int64(f.wordCount));
            chars_col.push(ScalarValue::Int64(f.charCount));
            avg_col.push(ScalarValue::Float64(f.avgWordLength));
            sent_col.push(ScalarValue::Int64(f.sentenceCount));
            punct_col.push(ScalarValue::Float64(f.punctRatio));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Int64, length_col),
            (TypeId::Int64, words_col),
            (TypeId::Int64, chars_col),
            (TypeId::Float64, avg_col),
            (TypeId::Int64, sent_col),
            (TypeId::Float64, punct_col),
        ]))
    }

    async fn run_shapiro_wilk(&self) -> Result<DataBatch> {
        // SHAPIRO_WILK(table, value_col)
        let table = self.pos_string_at(0)?;
        let value_col = self.pos_string_at(1)?;
        let values = self.collect_numeric_column(&table, &value_col).await?;
        let (w, p) = zyron_analytics::shapiroWilk(&values)?;
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Float64, vec![ScalarValue::Float64(w)]),
            (TypeId::Float64, vec![ScalarValue::Float64(p)]),
        ]))
    }

    async fn run_anderson_darling(&self) -> Result<DataBatch> {
        // ANDERSON_DARLING(table, value_col, dist => 'normal')
        let table = self.pos_string_at(0)?;
        let value_col = self.pos_string_at(1)?;
        let dist = self
            .named_string("dist")
            .or_else(|| self.pos_string_at(2).ok())
            .unwrap_or_else(|| "normal".into());
        let values = self.collect_numeric_column(&table, &value_col).await?;
        let result = zyron_analytics::andersonDarling(&values, &dist)?;
        let mut level_col: Vec<ScalarValue> = Vec::with_capacity(5);
        let mut crit_col: Vec<ScalarValue> = Vec::with_capacity(5);
        let mut a2_col: Vec<ScalarValue> = Vec::with_capacity(5);
        let mut rejected_col: Vec<ScalarValue> = Vec::with_capacity(5);
        for i in 0..5 {
            level_col.push(ScalarValue::Float64(result.significanceLevels[i]));
            crit_col.push(ScalarValue::Float64(result.criticalValues[i]));
            a2_col.push(ScalarValue::Float64(result.a2));
            rejected_col.push(ScalarValue::Boolean(result.a2 > result.criticalValues[i]));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Float64, level_col),
            (TypeId::Float64, crit_col),
            (TypeId::Float64, a2_col),
            (TypeId::Boolean, rejected_col),
        ]))
    }

    async fn run_ks_1samp(&self) -> Result<DataBatch> {
        // KOLMOGOROV_SMIRNOV_1SAMP(table, value_col, dist, params...)
        let table = self.pos_string_at(0)?;
        let value_col = self.pos_string_at(1)?;
        let dist = self.pos_string_at(2)?;
        let mut params = Vec::new();
        for i in 3..self.positional_args.len() {
            params.push(self.pos_float_at(i)?);
        }
        let values = self.collect_numeric_column(&table, &value_col).await?;
        let (d, p) = zyron_analytics::ks1Sample(&values, &dist, &params)?;
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Float64, vec![ScalarValue::Float64(d)]),
            (TypeId::Float64, vec![ScalarValue::Float64(p)]),
        ]))
    }

    async fn run_fit_distribution(&self) -> Result<DataBatch> {
        // FIT_DISTRIBUTION(table, value_col, candidates...)
        let table = self.pos_string_at(0)?;
        let value_col = self.pos_string_at(1)?;
        let mut candidates: Vec<String> = Vec::new();
        for arg in self.positional_args.iter().skip(2) {
            if let BoundExpr::Literal {
                value: LiteralValue::String(s),
                ..
            } = arg
            {
                for part in s.split(',') {
                    let trimmed = part.trim();
                    if !trimmed.is_empty() {
                        candidates.push(trimmed.to_string());
                    }
                }
            }
        }
        let values = self.collect_numeric_column(&table, &value_col).await?;
        let fits = zyron_analytics::fitDistribution(&values, &candidates)?;
        let mut name_col: Vec<ScalarValue> = Vec::with_capacity(fits.len());
        let mut params_col: Vec<ScalarValue> = Vec::with_capacity(fits.len());
        let mut ll_col: Vec<ScalarValue> = Vec::with_capacity(fits.len());
        let mut aic_col: Vec<ScalarValue> = Vec::with_capacity(fits.len());
        for f in &fits {
            let params_text = f
                .params
                .iter()
                .map(|p| format!("{}", p))
                .collect::<Vec<String>>()
                .join(",");
            name_col.push(ScalarValue::Utf8(f.name.clone()));
            params_col.push(ScalarValue::Utf8(params_text));
            ll_col.push(ScalarValue::Float64(f.logLikelihood));
            aic_col.push(ScalarValue::Float64(f.aic));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Varchar, name_col),
            (TypeId::Varchar, params_col),
            (TypeId::Float64, ll_col),
            (TypeId::Float64, aic_col),
        ]))
    }

    async fn run_local_outlier_factor(&self) -> Result<DataBatch> {
        // LOCAL_OUTLIER_FACTOR(table, columns..., k => 5)
        let table = self.pos_string_at(0)?;
        let cols = self.feature_columns_from(1)?;
        let k = self.named_int("k").unwrap_or(5).max(1) as usize;
        let (row_ids, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let scores = zyron_analytics::localOutlierFactor(&rows, k)?;
        let mut idx_col: Vec<ScalarValue> = Vec::with_capacity(scores.len());
        let mut lof_col: Vec<ScalarValue> = Vec::with_capacity(scores.len());
        for (i, s) in scores.iter().enumerate() {
            idx_col.push(ScalarValue::Int64(row_ids[i]));
            lof_col.push(ScalarValue::Float64(*s));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, idx_col),
            (TypeId::Float64, lof_col),
        ]))
    }

    async fn run_sequential_analyze(&self) -> Result<DataBatch> {
        // SEQUENTIAL_ANALYZE(table, successes_a_col, trials_a_col,
        //                    successes_b_col, trials_b_col,
        //                    alpha_spending => 'obrien_fleming', alpha => 0.05)
        // each table row is one look carrying cumulative counts
        let table = self.pos_string_at(0)?;
        let cols = [
            self.pos_string_at(1)?,
            self.pos_string_at(2)?,
            self.pos_string_at(3)?,
            self.pos_string_at(4)?,
        ];
        let spending = self
            .named_string("alpha_spending")
            .unwrap_or_else(|| "obrien_fleming".into());
        let alpha = self.named_float("alpha").unwrap_or(0.05);
        let (_, rows) = self.collect_numeric_matrix(&table, &cols).await?;
        let successes_a: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let trials_a: Vec<f64> = rows.iter().map(|r| r[1]).collect();
        let successes_b: Vec<f64> = rows.iter().map(|r| r[2]).collect();
        let trials_b: Vec<f64> = rows.iter().map(|r| r[3]).collect();
        let looks = zyron_analytics::sequentialAnalyze(
            &successes_a,
            &trials_a,
            &successes_b,
            &trials_b,
            &spending,
            alpha,
        )?;
        let mut look_col: Vec<ScalarValue> = Vec::with_capacity(looks.len());
        let mut z_col: Vec<ScalarValue> = Vec::with_capacity(looks.len());
        let mut boundary_col: Vec<ScalarValue> = Vec::with_capacity(looks.len());
        let mut crossed_col: Vec<ScalarValue> = Vec::with_capacity(looks.len());
        for l in &looks {
            look_col.push(ScalarValue::Int64(l.look as i64));
            z_col.push(ScalarValue::Float64(l.z));
            boundary_col.push(ScalarValue::Float64(l.boundary));
            crossed_col.push(ScalarValue::Boolean(l.crossed));
        }
        Ok(scalar_columns_to_batch(vec![
            (TypeId::Int64, look_col),
            (TypeId::Float64, z_col),
            (TypeId::Float64, boundary_col),
            (TypeId::Boolean, crossed_col),
        ]))
    }

    async fn run_bayesian_analyze(&self) -> Result<DataBatch> {
        // BAYESIAN_ANALYZE(successes_a, trials_a, successes_b, trials_b,
        //                  prior_alpha_a => 1.0, prior_beta_a => 1.0,
        //                  prior_alpha_b => 1.0, prior_beta_b => 1.0)
        let mut counts = [0u64; 4];
        for (slot, i) in counts.iter_mut().zip(0usize..4) {
            let v = self.pos_int_at(i)?;
            if v < 0 {
                return Err(ZyronError::ExecutionError(
                    "BAYESIAN_ANALYZE counts must be nonnegative integers".into(),
                ));
            }
            *slot = v as u64;
        }
        let prior_a = (
            self.named_float("prior_alpha_a").unwrap_or(1.0),
            self.named_float("prior_beta_a").unwrap_or(1.0),
        );
        let prior_b = (
            self.named_float("prior_alpha_b").unwrap_or(1.0),
            self.named_float("prior_beta_b").unwrap_or(1.0),
        );
        let result = zyron_analytics::bayesianAnalyze(
            counts[0], counts[1], counts[2], counts[3], prior_a, prior_b,
        )?;
        Ok(scalar_columns_to_batch(vec![
            (
                TypeId::Float64,
                vec![ScalarValue::Float64(result.probBBeatsA)],
            ),
            (
                TypeId::Float64,
                vec![ScalarValue::Float64(result.expectedLossA)],
            ),
            (
                TypeId::Float64,
                vec![ScalarValue::Float64(result.expectedLossB)],
            ),
            (TypeId::Float64, vec![ScalarValue::Float64(result.meanA)]),
            (TypeId::Float64, vec![ScalarValue::Float64(result.meanB)]),
        ]))
    }

    async fn collect_numeric_column(
        &self,
        table_name: &str,
        column_name: &str,
    ) -> Result<Vec<f64>> {
        let table_entry = self.resolve_table(table_name)?;
        let columns = table_entry.columns.clone();
        let idx = columns
            .iter()
            .position(|c| c.name.eq_ignore_ascii_case(column_name))
            .ok_or_else(|| ZyronError::ColumnNotFound(column_name.to_string()))?;
        let heap_file = self.ctx.get_heap_file(table_entry.id).await?;
        let snapshot = self.ctx.snapshot.clone();
        let guard = heap_file.scan()?;
        let mut values = Vec::new();
        let mut featureBuf = vec![0.0f64; 1];
        guard.for_each(|_tid, view| {
            if view.is_deleted() {
                return;
            }
            if !view.header.is_visible_to(&snapshot) {
                return;
            }
            decode_row_features(view.data, &columns, &[idx], &mut featureBuf);
            values.push(featureBuf[0]);
        });
        Ok(values)
    }
}

impl Operator for AnalyticsTableFunctionOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            let batch = self.execute().await?;
            self.finished = true;
            if batch.num_rows == 0 && batch.num_columns() == 0 {
                return Ok(None);
            }
            // The planner declares this node's schema from the function name
            // and the function builds its own columns from the same name. A
            // disagreement would hand the parent columns it does not expect
            // and be read as wrong values rather than as a mismatch, so it is
            // caught here where the two meet
            if batch.num_columns() != self.output_columns.len() {
                return Err(ZyronError::ExecutionError(format!(
                    "table function \"{}\" produced {} columns, its plan declares {}",
                    self.function_name,
                    batch.num_columns(),
                    self.output_columns.len()
                )));
            }
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

fn opt_to_scalar(v: Option<f64>) -> ScalarValue {
    match v {
        Some(f) => ScalarValue::Float64(f),
        None => ScalarValue::Null,
    }
}

fn type_id_to_label(t: TypeId) -> String {
    match t {
        TypeId::Boolean => "BOOLEAN".into(),
        TypeId::Int8 => "INT8".into(),
        TypeId::Int16 => "INT16".into(),
        TypeId::Int32 => "INT32".into(),
        TypeId::Int64 => "INT64".into(),
        TypeId::Float32 => "FLOAT32".into(),
        TypeId::Float64 => "FLOAT64".into(),
        TypeId::Char | TypeId::Varchar | TypeId::Text => "TEXT".into(),
        TypeId::Date => "DATE".into(),
        TypeId::Timestamp | TypeId::TimestampTz => "TIMESTAMP".into(),
        _ => format!("{:?}", t).to_uppercase(),
    }
}

// Walks the tuple-byte layout, invoking a per-column callback with the
// (column_index, type_id, is_null, value_bytes) for each column. Returns
// after the whole tuple is walked or after the callback signals stop. The
// other streaming decoders below are layered on top of this primitive.
#[inline]
fn walk_tuple_columns<F: FnMut(usize, TypeId, bool, &[u8]) -> std::ops::ControlFlow<()>>(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    mut visit: F,
) {
    let num_cols = columns.len();
    let null_bitmap_len = (num_cols + 7) / 8;
    if data.len() < null_bitmap_len {
        return;
    }
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;
    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if let Some(fixed_size) = col.type_id.fixed_size() {
            if data.len() < offset + fixed_size {
                return;
            }
            let bytes = &data[offset..offset + fixed_size];
            if visit(i, col.type_id, is_null, bytes).is_break() {
                return;
            }
            offset += fixed_size;
        } else {
            if data.len() < offset + 4 {
                return;
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if data.len() < offset + len {
                return;
            }
            let bytes = if is_null {
                &[][..]
            } else {
                &data[offset..offset + len]
            };
            if visit(i, col.type_id, is_null, bytes).is_break() {
                return;
            }
            offset += len;
        }
    }
}

// Decodes selected columns into a Vec<Option<f64>> for the correlation
// streaming path. Non-numeric columns produce None. The output buffer is
// kept by the caller and re-filled per row.
fn decode_selected_columns_to_f64(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    selected_indices: &[usize],
    out: &mut [Option<f64>],
) {
    debug_assert_eq!(out.len(), selected_indices.len());
    for slot in out.iter_mut() {
        *slot = None;
    }
    walk_tuple_columns(data, columns, |i, type_id, is_null, bytes| {
        if let Some(pos) = selected_indices.iter().position(|&s| s == i) {
            if !is_null {
                let v = if type_id.fixed_size().is_some() {
                    decode_fixed_to_analytics(type_id, bytes)
                } else {
                    decode_varlen_to_analytics(type_id, bytes)
                };
                out[pos] = v.as_f64();
            }
        }
        std::ops::ControlFlow::Continue(())
    });
}

// Decode the three columns a cohort event needs (user_id, event_time_ms,
// optional revenue) into the provided slots.
fn decode_three_columns(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    user_idx: usize,
    time_idx: usize,
    revenue_idx: Option<usize>,
    user_out: &mut AnalyticsValue,
    ts_out: &mut i64,
    revenue_out: &mut Option<f64>,
) {
    *user_out = AnalyticsValue::Null;
    *ts_out = 0;
    *revenue_out = None;
    walk_tuple_columns(data, columns, |i, type_id, is_null, bytes| {
        if i == user_idx && !is_null {
            *user_out = if type_id.fixed_size().is_some() {
                decode_fixed_to_analytics(type_id, bytes)
            } else {
                decode_varlen_to_analytics(type_id, bytes)
            };
        } else if i == time_idx && !is_null {
            let v = if type_id.fixed_size().is_some() {
                decode_fixed_to_analytics(type_id, bytes)
            } else {
                decode_varlen_to_analytics(type_id, bytes)
            };
            *ts_out = v.as_timestamp_ms().unwrap_or(0);
        } else if Some(i) == revenue_idx && !is_null {
            let v = if type_id.fixed_size().is_some() {
                decode_fixed_to_analytics(type_id, bytes)
            } else {
                decode_varlen_to_analytics(type_id, bytes)
            };
            *revenue_out = v.as_f64();
        }
        std::ops::ControlFlow::Continue(())
    });
}

// Decode the three columns a funnel event needs (user_id, event_time_ms,
// event_name) into the provided slots. event_name is written into the
// caller's String to reuse its allocation across rows.
fn decode_funnel_columns(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    user_idx: usize,
    time_idx: usize,
    event_idx: usize,
    user_out: &mut AnalyticsValue,
    ts_out: &mut i64,
    name_out: &mut String,
) {
    *user_out = AnalyticsValue::Null;
    *ts_out = 0;
    name_out.clear();
    walk_tuple_columns(data, columns, |i, type_id, is_null, bytes| {
        if i == user_idx && !is_null {
            *user_out = if type_id.fixed_size().is_some() {
                decode_fixed_to_analytics(type_id, bytes)
            } else {
                decode_varlen_to_analytics(type_id, bytes)
            };
        } else if i == time_idx && !is_null {
            let v = if type_id.fixed_size().is_some() {
                decode_fixed_to_analytics(type_id, bytes)
            } else {
                decode_varlen_to_analytics(type_id, bytes)
            };
            *ts_out = v.as_timestamp_ms().unwrap_or(0);
        } else if i == event_idx && !is_null {
            // Reuse the caller's String allocation: clear above + push_str
            // here keeps the buffer stable across heap-scan rows.
            if let Ok(s) = std::str::from_utf8(bytes) {
                name_out.push_str(s);
            }
        }
        std::ops::ControlFlow::Continue(())
    });
}

fn decode_fixed_to_analytics(t: TypeId, b: &[u8]) -> AnalyticsValue {
    match t {
        TypeId::Boolean => AnalyticsValue::Bool(b[0] != 0),
        TypeId::Int8 => AnalyticsValue::Int(b[0] as i8 as i64),
        TypeId::Int16 => AnalyticsValue::Int(i16::from_le_bytes([b[0], b[1]]) as i64),
        TypeId::Int32 => AnalyticsValue::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
        TypeId::Int64 => AnalyticsValue::Int(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])),
        TypeId::UInt8 => AnalyticsValue::UInt(b[0] as u64),
        TypeId::UInt16 => AnalyticsValue::UInt(u16::from_le_bytes([b[0], b[1]]) as u64),
        TypeId::UInt32 => AnalyticsValue::UInt(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64),
        TypeId::UInt64 => AnalyticsValue::UInt(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])),
        TypeId::Float32 => {
            AnalyticsValue::Float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
        }
        TypeId::Float64 => AnalyticsValue::Float(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])),
        TypeId::Date => AnalyticsValue::Date(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        TypeId::Timestamp | TypeId::TimestampTz => AnalyticsValue::Timestamp(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])),
        _ => AnalyticsValue::Null,
    }
}

fn decode_varlen_to_analytics(t: TypeId, b: &[u8]) -> AnalyticsValue {
    match t {
        TypeId::Char | TypeId::Varchar | TypeId::Text => {
            AnalyticsValue::Text(String::from_utf8_lossy(b).into_owned())
        }
        _ => AnalyticsValue::Null,
    }
}

fn analytics_value_to_text(v: &AnalyticsValue) -> Option<String> {
    match v {
        AnalyticsValue::Null => None,
        AnalyticsValue::Bool(b) => Some(b.to_string()),
        AnalyticsValue::Int(i) => Some(i.to_string()),
        AnalyticsValue::UInt(u) => Some(u.to_string()),
        AnalyticsValue::Float(f) => Some(f.to_string()),
        AnalyticsValue::Text(s) => Some(s.clone()),
        AnalyticsValue::Timestamp(t) => Some(t.to_string()),
        AnalyticsValue::Date(d) => Some(d.to_string()),
    }
}

// Decodes one column of a tuple as text, None when the value is null or
// the tuple bytes end before the column
fn decode_column_to_text(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    target_idx: usize,
) -> Option<String> {
    let mut out: Option<String> = None;
    walk_tuple_columns(data, columns, |i, type_id, is_null, bytes| {
        if i == target_idx {
            if !is_null {
                let v = if type_id.fixed_size().is_some() {
                    decode_fixed_to_analytics(type_id, bytes)
                } else {
                    decode_varlen_to_analytics(type_id, bytes)
                };
                out = analytics_value_to_text(&v);
            }
            return std::ops::ControlFlow::Break(());
        }
        std::ops::ControlFlow::Continue(())
    });
    out
}

// Streaming variant: decode each column from the tuple bytes and feed it
// directly into the corresponding ColumnProfiler. Avoids materialising any
// per-row Vec<AnalyticsValue> and any whole-table Vec<Vec<AnalyticsValue>>.
fn decode_tuple_streaming(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    profilers: &mut [zyron_analytics::profiling::ColumnProfiler],
) {
    let num_cols = columns.len();
    let null_bitmap_len = (num_cols + 7) / 8;
    if data.len() < null_bitmap_len {
        return;
    }
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;
    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if let Some(fixed_size) = col.type_id.fixed_size() {
            if data.len() < offset + fixed_size {
                profilers[i].ingest(&AnalyticsValue::Null);
                return;
            }
            if is_null {
                profilers[i].ingest(&AnalyticsValue::Null);
            } else {
                let v = decode_fixed_to_analytics(col.type_id, &data[offset..offset + fixed_size]);
                profilers[i].ingest(&v);
            }
            offset += fixed_size;
        } else {
            if data.len() < offset + 4 {
                profilers[i].ingest(&AnalyticsValue::Null);
                return;
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if is_null {
                profilers[i].ingest(&AnalyticsValue::Null);
                offset += len;
            } else {
                if data.len() < offset + len {
                    profilers[i].ingest(&AnalyticsValue::Null);
                    return;
                }
                let v = decode_varlen_to_analytics(col.type_id, &data[offset..offset + len]);
                profilers[i].ingest(&v);
                offset += len;
            }
        }
    }
}

// Single-column streaming variant: walks the tuple's column layout to find
// the byte range for the target column, decodes only it, and feeds it to
// the profiler. Other columns' bytes are skipped without decoding.
fn decode_one_column_streaming(
    data: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    target_idx: usize,
    profiler: &mut zyron_analytics::profiling::ColumnProfiler,
) {
    let num_cols = columns.len();
    let null_bitmap_len = (num_cols + 7) / 8;
    if data.len() < null_bitmap_len {
        return;
    }
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;
    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if let Some(fixed_size) = col.type_id.fixed_size() {
            if data.len() < offset + fixed_size {
                if i == target_idx {
                    profiler.ingest(&AnalyticsValue::Null);
                }
                return;
            }
            if i == target_idx {
                if is_null {
                    profiler.ingest(&AnalyticsValue::Null);
                } else {
                    let v =
                        decode_fixed_to_analytics(col.type_id, &data[offset..offset + fixed_size]);
                    profiler.ingest(&v);
                }
                // Found target column; safe to stop walking
                return;
            }
            offset += fixed_size;
        } else {
            if data.len() < offset + 4 {
                if i == target_idx {
                    profiler.ingest(&AnalyticsValue::Null);
                }
                return;
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if i == target_idx {
                if is_null {
                    profiler.ingest(&AnalyticsValue::Null);
                } else if data.len() < offset + len {
                    profiler.ingest(&AnalyticsValue::Null);
                } else {
                    let v = decode_varlen_to_analytics(col.type_id, &data[offset..offset + len]);
                    profiler.ingest(&v);
                }
                return;
            }
            if !is_null {
                offset += len;
            } else {
                offset += len;
            }
        }
    }
}

// Decodes the requested columns of one heap tuple into f64 slots through
// the shared tuple walker, so the leading null bitmap and varlen length
// prefixes are honored. A null value, a non numeric column, or a tuple
// that ends early leaves NAN in the slot and the caller drops the row
fn decode_row_features(
    bytes: &[u8],
    columns: &[zyron_catalog::ColumnEntry],
    feature_indices: &[usize],
    out: &mut [f64],
) {
    for slot in out.iter_mut() {
        *slot = f64::NAN;
    }
    let mut remaining = feature_indices.len();
    walk_tuple_columns(bytes, columns, |i, type_id, is_null, value_bytes| {
        for (slot, &idx) in feature_indices.iter().enumerate() {
            if idx != i {
                continue;
            }
            if !is_null {
                let v = if type_id.fixed_size().is_some() {
                    decode_fixed_to_analytics(type_id, value_bytes)
                } else {
                    decode_varlen_to_analytics(type_id, value_bytes)
                };
                if let Some(f) = v.as_f64() {
                    out[slot] = f;
                }
            }
            remaining -= 1;
        }
        if remaining == 0 {
            std::ops::ControlFlow::Break(())
        } else {
            std::ops::ControlFlow::Continue(())
        }
    });
}

fn scalar_columns_to_batch(cols: Vec<(TypeId, Vec<ScalarValue>)>) -> DataBatch {
    let mut output = Vec::with_capacity(cols.len());
    for (type_id, values) in cols {
        let mut data = ColumnData::with_capacity(type_id, values.len());
        let mut nulls = NullBitmap::empty();
        for v in &values {
            nulls.push(v.is_null());
            data.push_scalar(v);
        }
        output.push(Column::with_nulls(data, nulls, type_id));
    }
    DataBatch::new(output)
}
