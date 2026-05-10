// Analytics table-returning function operator.
// Resolves the function name from the analytics registry, scans the
// referenced source table, materialises rows into AnalyticsValue, runs
// the requested analytics function, and emits the result as a DataBatch.

use std::sync::Arc;

use zyron_analytics::{
    AnalyticsFunctionKind, AnalyticsValue, CohortAnalysis, CohortDefinition, CohortEvent,
    CohortMetric, CohortPeriod, CohortType, ColumnProfile, FunnelConfig, FunnelEvent, FunnelStep,
    TableProfile, column_profile, correlation_matrix, funnel_analysis, profile_table,
    retention_analysis,
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
                Ok(None)
            } else {
                Ok(Some(ExecutionBatch::new(batch)))
            }
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
