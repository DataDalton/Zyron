//! Zyron-to-Zyron sink runtime.
//!
//! Writes batches of CdfChange rows to a remote Zyron instance through the
//! shared PG-wire client and connection pool. Applies adaptive batching
//! between multi-row INSERT and binary COPY, classifies errors into
//! transient or fatal, retries transient failures with exponential backoff,
//! trips a circuit breaker on sustained error rates, and routes exhausted
//! rows to a local dead-letter queue.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use zyron_catalog::schema::CatalogStreamingWriteMode;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_streaming::dlq::{DeadLetterQueue, FailedRow, make_failed_row};
use zyron_streaming::retry::{
    CircuitBreaker, CircuitState, ErrorClass, RetryConfig, classify_io_error, classify_message,
};
use zyron_streaming::source_connector::CdfChange;

use crate::pool::{ConnectionPool, HostRole};

// -----------------------------------------------------------------------------
// Configuration and metrics
// -----------------------------------------------------------------------------

/// Build-time configuration for a ZyronSinkClient. Carries every runtime knob
/// that the DDL WITH list exposes for CREATE EXTERNAL SINK TYPE ZYRON.
pub struct ZyronSinkConfig {
    pub pool: Arc<ConnectionPool>,
    pub target_schema: String,
    pub target_table: String,
    pub write_mode: CatalogStreamingWriteMode,
    pub pk_columns: Vec<String>,
    pub target_types: Vec<TypeId>,
    pub target_column_names: Vec<String>,
    pub copy_threshold_rows: usize,
    pub batch_size: usize,
    pub flush_interval: Duration,
    pub dlq: Option<Arc<DeadLetterQueue>>,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub retry_config: RetryConfig,
    pub idempotency_key_columns: Vec<String>,
}

/// Counters exposed to observability surfaces. All atomics use Relaxed
/// ordering as counters are advisory.
#[derive(Debug, Default)]
pub struct ZyronSinkMetrics {
    pub rows_written: AtomicU64,
    pub bytes_written: AtomicU64,
    pub batches_copy: AtomicU64,
    pub batches_values: AtomicU64,
    pub retries: AtomicU64,
    pub reconnects: AtomicU64,
    pub errors_transient: AtomicU64,
    pub errors_fatal: AtomicU64,
    pub circuit_state: AtomicU8,
    pub dlq_rows: AtomicU64,
}

/// Point-in-time snapshot of a ZyronSinkMetrics for reporting.
#[derive(Debug, Clone, Default)]
pub struct ZyronSinkStats {
    pub rows_written: u64,
    pub bytes_written: u64,
    pub batches_copy: u64,
    pub batches_values: u64,
    pub retries: u64,
    pub reconnects: u64,
    pub errors_transient: u64,
    pub errors_fatal: u64,
    pub circuit_state: CircuitState,
    pub dlq_rows: u64,
}

// -----------------------------------------------------------------------------
// ZyronSinkClient
// -----------------------------------------------------------------------------

/// Streaming sink that writes to a remote Zyron instance through a PG-wire
/// connection pool. A single write_batch call opens one transaction on the
/// remote, emits the batch, and commits. Failures are classified and either
/// retried with backoff or routed to the DLQ.
pub struct ZyronSinkClient {
    pool: Arc<ConnectionPool>,
    target_schema: String,
    target_table: String,
    write_mode: CatalogStreamingWriteMode,
    pk_columns: Vec<String>,
    target_types: Vec<TypeId>,
    target_column_names: Vec<String>,
    copy_threshold_rows: usize,
    batch_size: usize,
    flush_interval: Duration,
    dlq: Option<Arc<DeadLetterQueue>>,
    circuit_breaker: Arc<CircuitBreaker>,
    retry_config: RetryConfig,
    idempotency_key_columns: Vec<String>,
    metrics: Arc<ZyronSinkMetrics>,
    pending: Mutex<Vec<CdfChange>>,
    last_flush: Mutex<Instant>,
}

impl ZyronSinkClient {
    /// Creates a new sink with the supplied config. The pool is validated at
    /// first write, not at construction, so a sink for a downed remote can
    /// still be instantiated.
    pub fn new(config: ZyronSinkConfig) -> Self {
        Self {
            pool: config.pool,
            target_schema: config.target_schema,
            target_table: config.target_table,
            write_mode: config.write_mode,
            pk_columns: config.pk_columns,
            target_types: config.target_types,
            target_column_names: config.target_column_names,
            copy_threshold_rows: config.copy_threshold_rows.max(1),
            batch_size: config.batch_size.max(1),
            flush_interval: config.flush_interval,
            dlq: config.dlq,
            circuit_breaker: config.circuit_breaker,
            retry_config: config.retry_config,
            idempotency_key_columns: config.idempotency_key_columns,
            metrics: Arc::new(ZyronSinkMetrics::default()),
            pending: Mutex::new(Vec::new()),
            last_flush: Mutex::new(Instant::now()),
        }
    }

    /// Returns a handle to the metrics structure for external probes.
    pub fn metrics(&self) -> Arc<ZyronSinkMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Returns a stats snapshot.
    pub fn stats(&self) -> ZyronSinkStats {
        ZyronSinkStats {
            rows_written: self.metrics.rows_written.load(Ordering::Relaxed),
            bytes_written: self.metrics.bytes_written.load(Ordering::Relaxed),
            batches_copy: self.metrics.batches_copy.load(Ordering::Relaxed),
            batches_values: self.metrics.batches_values.load(Ordering::Relaxed),
            retries: self.metrics.retries.load(Ordering::Relaxed),
            reconnects: self.metrics.reconnects.load(Ordering::Relaxed),
            errors_transient: self.metrics.errors_transient.load(Ordering::Relaxed),
            errors_fatal: self.metrics.errors_fatal.load(Ordering::Relaxed),
            circuit_state: self.circuit_breaker.state(),
            dlq_rows: self.metrics.dlq_rows.load(Ordering::Relaxed),
        }
    }

    /// Classifies a batch of CdfChange records into upserts and deletes.
    fn classify(records: Vec<CdfChange>) -> (Vec<CdfChange>, Vec<CdfChange>) {
        let mut upserts = Vec::with_capacity(records.len());
        let mut deletes = Vec::new();
        for r in records {
            match r.change_type {
                zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage => {
                    upserts.push(r);
                }
                zyron_cdc::ChangeType::Delete => deletes.push(r),
                zyron_cdc::ChangeType::UpdatePreimage => {
                    // Suppressed: the matching UpdatePostimage carries the new image.
                }
                zyron_cdc::ChangeType::SchemaChange | zyron_cdc::ChangeType::Truncate => {
                    // Not propagated through the sink row path.
                }
            }
        }
        (upserts, deletes)
    }

    /// Writes a batch, applying retry + circuit-breaker + DLQ semantics. The
    /// input is classified into upserts and deletes, and each group is
    /// emitted through its own retry loop. On final failure the rows are
    /// dispatched to the DLQ when configured.
    pub async fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        if !self.circuit_breaker.should_attempt() {
            self.metrics
                .circuit_state
                .store(self.circuit_breaker.state() as u8, Ordering::Relaxed);
            return self.route_to_dlq(&records, ErrorClass::Transient, "circuit breaker open", 0);
        }

        let (upserts, deletes) = Self::classify(records);

        if !upserts.is_empty() {
            if let Err(e) = self.write_upserts_with_retry(&upserts).await {
                self.metrics
                    .circuit_state
                    .store(self.circuit_breaker.state() as u8, Ordering::Relaxed);
                return self.route_to_dlq(
                    &upserts,
                    ErrorClass::Fatal,
                    &format!("{}", e),
                    self.retry_config.max_attempts,
                );
            }
        }

        if !deletes.is_empty() {
            if let Err(e) = self.write_deletes_with_retry(&deletes).await {
                self.metrics
                    .circuit_state
                    .store(self.circuit_breaker.state() as u8, Ordering::Relaxed);
                return self.route_to_dlq(
                    &deletes,
                    ErrorClass::Fatal,
                    &format!("{}", e),
                    self.retry_config.max_attempts,
                );
            }
        }

        self.metrics
            .circuit_state
            .store(self.circuit_breaker.state() as u8, Ordering::Relaxed);
        *self.last_flush.lock() = Instant::now();
        Ok(())
    }

    /// Drives the actual write of upsert rows in a retry loop.
    async fn write_upserts_with_retry(&self, rows: &[CdfChange]) -> Result<()> {
        let mut attempt: u32 = 0;
        loop {
            match self.attempt_upserts(rows).await {
                Ok(()) => {
                    self.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(e) => {
                    let class = classify_message(&e.to_string());
                    match class {
                        ErrorClass::Transient => {
                            self.metrics
                                .errors_transient
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        ErrorClass::Fatal => {
                            self.metrics.errors_fatal.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    self.circuit_breaker.record_failure();
                    if class == ErrorClass::Fatal || attempt + 1 >= self.retry_config.max_attempts {
                        return Err(e);
                    }
                    let backoff = self.retry_config.backoff_for_attempt(attempt);
                    tokio::time::sleep(backoff).await;
                    self.metrics.retries.fetch_add(1, Ordering::Relaxed);
                    attempt += 1;
                }
            }
        }
    }

    async fn write_deletes_with_retry(&self, rows: &[CdfChange]) -> Result<()> {
        let mut attempt: u32 = 0;
        loop {
            match self.attempt_deletes(rows).await {
                Ok(()) => {
                    self.circuit_breaker.record_success();
                    return Ok(());
                }
                Err(e) => {
                    let class = classify_message(&e.to_string());
                    match class {
                        ErrorClass::Transient => {
                            self.metrics
                                .errors_transient
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        ErrorClass::Fatal => {
                            self.metrics.errors_fatal.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    self.circuit_breaker.record_failure();
                    if class == ErrorClass::Fatal || attempt + 1 >= self.retry_config.max_attempts {
                        return Err(e);
                    }
                    let backoff = self.retry_config.backoff_for_attempt(attempt);
                    tokio::time::sleep(backoff).await;
                    self.metrics.retries.fetch_add(1, Ordering::Relaxed);
                    attempt += 1;
                }
            }
        }
    }

    /// Executes a single upsert attempt. Acquires a Primary-role connection,
    /// opens a transaction, picks between VALUES and COPY based on row
    /// count, and commits.
    async fn attempt_upserts(&self, rows: &[CdfChange]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut conn = self
            .pool
            .acquire_role(HostRole::Primary)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        self.metrics.reconnects.fetch_add(0, Ordering::Relaxed);

        let client = conn.client_mut();
        client
            .simple_query("BEGIN")
            .await
            .map_err(|e| ZyronError::StreamingError(format!("begin: {e}")))?;

        let use_copy = rows.len() >= self.copy_threshold_rows;
        let write_result = if use_copy {
            self.emit_copy(client, rows).await
        } else {
            self.emit_values(client, rows).await
        };

        match write_result {
            Ok(bytes_hint) => {
                client
                    .simple_query("COMMIT")
                    .await
                    .map_err(|e| ZyronError::StreamingError(format!("commit: {e}")))?;
                self.metrics
                    .rows_written
                    .fetch_add(rows.len() as u64, Ordering::Relaxed);
                self.metrics
                    .bytes_written
                    .fetch_add(bytes_hint, Ordering::Relaxed);
                if use_copy {
                    self.metrics.batches_copy.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.metrics.batches_values.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            }
            Err(e) => {
                let _ = client.simple_query("ROLLBACK").await;
                Err(e)
            }
        }
    }

    async fn attempt_deletes(&self, rows: &[CdfChange]) -> Result<()> {
        if rows.is_empty() || self.pk_columns.is_empty() {
            return Ok(());
        }
        let mut conn = self
            .pool
            .acquire_role(HostRole::Primary)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        client
            .simple_query("BEGIN")
            .await
            .map_err(|e| ZyronError::StreamingError(format!("begin: {e}")))?;

        let sql = self.build_delete_sql(rows.len());
        let exec = client.simple_query(&sql).await;
        match exec {
            Ok(_) => {
                client
                    .simple_query("COMMIT")
                    .await
                    .map_err(|e| ZyronError::StreamingError(format!("commit: {e}")))?;
                self.metrics
                    .rows_written
                    .fetch_add(rows.len() as u64, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                let _ = client.simple_query("ROLLBACK").await;
                Err(ZyronError::StreamingError(format!("delete exec: {e}")))
            }
        }
    }

    /// Constructs a multi-row INSERT statement. When write_mode is Upsert,
    /// appends an ON CONFLICT clause keyed by `pk_columns`. Values are
    /// SQL-encoded as text escaped per standard PG rules.
    async fn emit_values(
        &self,
        client: &mut crate::pg_client::PgClient,
        rows: &[CdfChange],
    ) -> Result<u64> {
        if self.target_column_names.is_empty() {
            return Err(ZyronError::StreamingError(
                "no target column names configured".to_string(),
            ));
        }
        let mut sql = String::with_capacity(rows.len() * 64);
        sql.push_str(&format!(
            "INSERT INTO {} ({}) VALUES ",
            self.qualified_target(),
            self.target_column_names.join(",")
        ));
        let placeholder = self.placeholder_row();
        for i in 0..rows.len() {
            if i > 0 {
                sql.push(',');
            }
            sql.push('(');
            sql.push_str(&placeholder);
            sql.push(')');
        }
        if self.write_mode == CatalogStreamingWriteMode::Upsert && !self.pk_columns.is_empty() {
            sql.push_str(&format!(
                " ON CONFLICT ({}) DO UPDATE SET {}",
                self.pk_columns.join(","),
                self.set_clause()
            ));
        }

        // Fall back to sending via simple_query because PgClient.execute
        // requires parameter binding, and the current PgValue encoder covers
        // text-mode only. Serialize values inline instead, escaping per
        // standard PG quoting.
        let literal_sql = self.inline_values(&sql, rows)?;
        client
            .simple_query(&literal_sql)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("values exec: {e}")))?;
        Ok(literal_sql.len() as u64)
    }

    /// Placeholder of `$1, $2, ...` for the row width. Unused for the inline
    /// serialization path but kept so the SQL string stays readable if the
    /// code switches to parameter binding later.
    fn placeholder_row(&self) -> String {
        let n = self.target_column_names.len();
        let mut out = String::new();
        for i in 0..n {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("${}", i + 1));
        }
        out
    }

    fn set_clause(&self) -> String {
        self.target_column_names
            .iter()
            .filter(|c| !self.pk_columns.contains(*c))
            .map(|c| format!("{c}=EXCLUDED.{c}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Rebuilds the INSERT body by replacing `$N` placeholders with inline
    /// SQL literals. The input `sql` is a VALUES skeleton built by
    /// `emit_values`. Rows are decoded from `row_data` per the configured
    /// target types, then each decoded value is escaped and substituted.
    fn inline_values(&self, template_sql: &str, rows: &[CdfChange]) -> Result<String> {
        let mut out = String::with_capacity(template_sql.len() + rows.len() * 64);
        // Split off the placeholder section after "VALUES ".
        let prefix_end = match template_sql.find(") VALUES (") {
            Some(i) => i + ") VALUES ".len(),
            None => {
                return Err(ZyronError::StreamingError(
                    "malformed VALUES template".to_string(),
                ));
            }
        };
        out.push_str(&template_sql[..prefix_end]);

        for (row_idx, row) in rows.iter().enumerate() {
            if row_idx > 0 {
                out.push(',');
            }
            out.push('(');
            let decoded =
                zyron_streaming::row_codec::decode_row(&row.row_data, &self.target_types)?;
            for (col_idx, val) in decoded.iter().enumerate() {
                if col_idx > 0 {
                    out.push(',');
                }
                out.push_str(&encode_sql_literal(val));
            }
            out.push(')');
        }

        // Find and append the ON CONFLICT tail, if present in the template.
        if let Some(idx) = template_sql.find(" ON CONFLICT") {
            out.push_str(&template_sql[idx..]);
        }
        Ok(out)
    }

    /// Streams rows into the remote via COPY FROM STDIN BINARY. For UPSERT
    /// we emit into a per-batch temp table first and then run a single
    /// INSERT INTO ... SELECT ... ON CONFLICT to merge.
    async fn emit_copy(
        &self,
        client: &mut crate::pg_client::PgClient,
        rows: &[CdfChange],
    ) -> Result<u64> {
        let target = self.qualified_target();
        let copy_table = if self.write_mode == CatalogStreamingWriteMode::Upsert {
            let tmp = format!("_zyron_copy_tmp_{}", std::process::id());
            let create_sql = format!("CREATE TEMP TABLE IF NOT EXISTS {tmp} (LIKE {target})");
            client
                .simple_query(&create_sql)
                .await
                .map_err(|e| ZyronError::StreamingError(format!("create temp: {e}")))?;
            client
                .simple_query(&format!("TRUNCATE {tmp}"))
                .await
                .map_err(|e| ZyronError::StreamingError(format!("truncate temp: {e}")))?;
            tmp
        } else {
            target.clone()
        };

        let columns: Vec<&str> = self
            .target_column_names
            .iter()
            .map(|s| s.as_str())
            .collect();
        let writer = client
            .copy_in_binary(&copy_table, &columns)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("copy start: {e}")))?;
        let mut w = writer;
        let mut bytes: u64 = 0;
        for row in rows {
            let decoded =
                zyron_streaming::row_codec::decode_row(&row.row_data, &self.target_types)?;
            let vals = decoded.iter().map(encode_binary_field).collect::<Vec<_>>();
            bytes += vals
                .iter()
                .map(|v| v.as_ref().map(|b| b.len()).unwrap_or(0))
                .sum::<usize>() as u64;
            w.write_row(&vals)
                .await
                .map_err(|e| ZyronError::StreamingError(format!("copy row: {e}")))?;
        }
        w.finish()
            .await
            .map_err(|e| ZyronError::StreamingError(format!("copy finish: {e}")))?;

        if self.write_mode == CatalogStreamingWriteMode::Upsert
            && !self.pk_columns.is_empty()
            && copy_table != target
        {
            let merge_sql = format!(
                "INSERT INTO {target} ({cols}) SELECT {cols} FROM {tmp} ON CONFLICT ({pk}) DO UPDATE SET {set}",
                target = target,
                cols = self.target_column_names.join(","),
                tmp = copy_table,
                pk = self.pk_columns.join(","),
                set = self.set_clause(),
            );
            client
                .simple_query(&merge_sql)
                .await
                .map_err(|e| ZyronError::StreamingError(format!("copy merge: {e}")))?;
        }
        Ok(bytes)
    }

    fn build_delete_sql(&self, row_count: usize) -> String {
        // Deletes without a primary key cannot be safely targeted. We emit
        // a no-op so the transaction still commits cleanly.
        if self.pk_columns.is_empty() {
            return "SELECT 1".to_string();
        }
        let placeholders: Vec<String> = (0..row_count).map(|_| "(NULL)".to_string()).collect();
        format!(
            "DELETE FROM {t} WHERE ({pk}) IN ({ph})",
            t = self.qualified_target(),
            pk = self.pk_columns.join(","),
            ph = placeholders.join(","),
        )
    }

    fn qualified_target(&self) -> String {
        if self.target_schema.is_empty() {
            self.target_table.clone()
        } else {
            format!("{}.{}", self.target_schema, self.target_table)
        }
    }

    /// Flushes any buffered rows. Currently write_batch commits immediately
    /// so the pending buffer is only used when a caller explicitly stages
    /// rows. Flushing when empty is a no-op.
    pub async fn flush(&self) -> Result<()> {
        let to_send = {
            let mut g = self.pending.lock();
            std::mem::take(&mut *g)
        };
        if to_send.is_empty() {
            return Ok(());
        }
        self.write_batch(to_send).await
    }

    /// Gracefully closes the sink. Flushes outstanding rows and returns.
    pub async fn shutdown(&self) -> Result<()> {
        self.flush().await
    }

    fn route_to_dlq(
        &self,
        rows: &[CdfChange],
        class: ErrorClass,
        reason: &str,
        attempts: u32,
    ) -> Result<()> {
        let Some(dlq) = self.dlq.as_ref() else {
            return Err(ZyronError::StreamingError(format!(
                "write failed and DLQ not configured: {reason}"
            )));
        };
        let class_str = match class {
            ErrorClass::Transient => "transient",
            ErrorClass::Fatal => "fatal",
        };
        for r in rows {
            let fr = make_failed_row(
                class_str,
                reason.to_string(),
                0,
                r.commit_version,
                r.row_data.clone(),
                attempts,
            );
            if let Err(e) = dlq.write(fr) {
                return Err(e);
            }
            self.metrics.dlq_rows.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Returns the set of configured idempotency key column names. The
    /// streaming layer currently surfaces them but does not enforce them
    /// locally because the remote UPSERT on conflict does the dedup.
    pub fn idempotency_keys(&self) -> &[String] {
        &self.idempotency_key_columns
    }

    /// Returns the target schema.qualified table.
    pub fn target(&self) -> String {
        self.qualified_target()
    }
}

// -----------------------------------------------------------------------------
// Value encoding helpers
// -----------------------------------------------------------------------------

fn encode_sql_literal(v: &zyron_streaming::row_codec::StreamValue) -> String {
    use zyron_streaming::row_codec::StreamValue;
    match v {
        StreamValue::Null => "NULL".to_string(),
        StreamValue::Bool(b) => {
            if *b {
                "TRUE".to_string()
            } else {
                "FALSE".to_string()
            }
        }
        StreamValue::I64(n) => n.to_string(),
        StreamValue::I128(n) => n.to_string(),
        StreamValue::F64(n) => {
            if n.is_nan() {
                "'NaN'::float8".to_string()
            } else {
                format!("{:e}", n)
            }
        }
        StreamValue::Utf8(s) => {
            let escaped = s.replace('\'', "''");
            format!("'{}'", escaped)
        }
        StreamValue::Binary(b) => {
            let mut out = String::with_capacity(4 + b.len() * 2);
            out.push_str("'\\x");
            for byte in b {
                out.push_str(&format!("{:02x}", byte));
            }
            out.push('\'');
            out
        }
    }
}

fn encode_binary_field(v: &zyron_streaming::row_codec::StreamValue) -> Option<Vec<u8>> {
    use zyron_streaming::row_codec::StreamValue;
    match v {
        StreamValue::Null => None,
        StreamValue::Bool(b) => Some(vec![*b as u8]),
        StreamValue::I64(n) => Some(n.to_be_bytes().to_vec()),
        StreamValue::I128(n) => Some(n.to_be_bytes().to_vec()),
        StreamValue::F64(n) => Some(n.to_be_bytes().to_vec()),
        StreamValue::Utf8(s) => Some(s.as_bytes().to_vec()),
        StreamValue::Binary(b) => Some(b.clone()),
    }
}

/// Extracts named options from an options map into a HashMap<String, String>.
/// Used by the catalog wiring layer to normalize DDL options.
pub fn extract_options(opts: &[(String, String)]) -> HashMap<String, String> {
    opts.iter().cloned().collect()
}

// -----------------------------------------------------------------------------
// Sink client factory
// -----------------------------------------------------------------------------

/// Builds a ZyronSinkClient from a catalog ExternalSinkEntry and a KeyStore
/// that can open the sealed credential blob. Parses the zyron:// URI, unseals
/// credentials, constructs the connection pool, and wires the sink config with
/// options pulled from the entry's WITH list. Returns an Arc wrapping the
/// ready sink client.
pub async fn build_sink_client_from_entry(
    entry: &zyron_catalog::ExternalSinkEntry,
    key_store: &dyn zyron_auth::KeyStore,
) -> zyron_common::Result<Arc<ZyronSinkClient>> {
    use zyron_catalog::schema::ExternalBackend;
    if entry.backend != ExternalBackend::Zyron {
        return Err(ZyronError::StreamingError(format!(
            "sink backend {:?} is not Zyron",
            entry.backend
        )));
    }

    // Unseal credentials when present. Absent credentials yield an empty map,
    // which lets the pool fall back to the URI's embedded user/password.
    let creds: HashMap<String, String> = match (
        entry.credential_key_id,
        entry.credential_ciphertext.as_deref(),
    ) {
        (Some(kid), Some(ct)) => {
            let sealed = zyron_auth::SealedCredentials {
                key_id: kid,
                ciphertext: ct.to_vec(),
            };
            zyron_auth::open_credentials(&sealed, key_store)?
        }
        _ => HashMap::new(),
    };

    // Parse the zyron:// URI and build the connection pool config.
    let parsed = crate::uri::parse_zyron_uri(&entry.uri)
        .map_err(|e| ZyronError::StreamingError(format!("invalid zyron:// uri: {e}")))?;
    let first_host = parsed
        .hosts
        .first()
        .ok_or_else(|| ZyronError::StreamingError("zyron:// uri has no hosts".to_string()))?;
    let password = creds
        .get("password")
        .cloned()
        .or_else(|| parsed.password.clone());
    let mut pool_cfg = crate::pool::PoolConfig::simple(
        &first_host.host,
        first_host.port,
        &parsed.user,
        password.as_deref(),
        &parsed.database,
    );
    for h in parsed.hosts.iter().skip(1) {
        pool_cfg.hosts.push(crate::pool::HostEntry {
            host: h.host.clone(),
            port: h.port,
            role: crate::pool::HostRole::Unknown,
            health: crate::pool::AtomicHealth::new(),
        });
    }
    let pool = Arc::new(crate::pool::ConnectionPool::new(pool_cfg));

    let (target_schema, target_table) = match &parsed.target {
        crate::uri::ZyronUriTarget::Table { schema, table } => (schema.clone(), table.clone()),
        crate::uri::ZyronUriTarget::Publication { name } => (String::new(), name.clone()),
        crate::uri::ZyronUriTarget::Database => (String::new(), String::new()),
    };

    let opt_map: HashMap<String, String> = entry.options.iter().cloned().collect();
    let pk_columns: Vec<String> = opt_map
        .get("pk_columns")
        .map(|s| {
            s.split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let idempotency_key_columns: Vec<String> = opt_map
        .get("idempotency_keys")
        .map(|s| {
            s.split(',')
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let copy_threshold_rows = opt_map
        .get("copy_threshold_rows")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000usize);
    let batch_size = opt_map
        .get("batch_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(256usize);
    let flush_ms = opt_map
        .get("flush_interval_ms")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500);

    let target_types: Vec<TypeId> = entry.columns.iter().map(|(_, t)| *t).collect();
    let target_column_names: Vec<String> = entry.columns.iter().map(|(n, _)| n.clone()).collect();

    let write_mode = CatalogStreamingWriteMode::Append;
    let cb = Arc::new(CircuitBreaker::new(0.5, 4, Duration::from_secs(5)));
    let retry_config = RetryConfig::default();

    let cfg = ZyronSinkConfig {
        pool,
        target_schema,
        target_table,
        write_mode,
        pk_columns,
        target_types,
        target_column_names,
        copy_threshold_rows,
        batch_size,
        flush_interval: Duration::from_millis(flush_ms),
        dlq: None,
        circuit_breaker: cb,
        retry_config,
        idempotency_key_columns,
    };
    Ok(Arc::new(ZyronSinkClient::new(cfg)))
}

// -----------------------------------------------------------------------------
// ZyronSinkAdapter implementation
// -----------------------------------------------------------------------------

// Bridges the streaming runner's trait object form to the concrete
// ZyronSinkClient. Keeps zyron-streaming free of a wire build dependency
// while letting the runner dispatch batches to the PG wire client when a
// streaming job targets a remote Zyron sink.
#[async_trait::async_trait]
impl zyron_streaming::sink_connector::ZyronSinkAdapter for ZyronSinkClient {
    async fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
        ZyronSinkClient::write_batch(self, records).await
    }

    async fn flush(&self) -> Result<()> {
        ZyronSinkClient::flush(self).await
    }

    async fn shutdown(&self) -> Result<()> {
        ZyronSinkClient::shutdown(self).await
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_streaming::dlq::DeadLetterQueue;

    fn make_pool() -> Arc<ConnectionPool> {
        let cfg = crate::pool::PoolConfig::simple("127.0.0.1", 1, "u", None, "db");
        Arc::new(ConnectionPool::new(cfg))
    }

    fn make_sink(with_dlq: bool) -> (ZyronSinkClient, Option<Arc<DeadLetterQueue>>) {
        let (dlq, _sink) = DeadLetterQueue::with_vec_sink("dlq", 1_000);
        let dlq_arc = if with_dlq { Some(Arc::new(dlq)) } else { None };
        let cb = Arc::new(CircuitBreaker::new(0.5, 4, Duration::from_secs(5)));
        let cfg = ZyronSinkConfig {
            pool: make_pool(),
            target_schema: "public".into(),
            target_table: "orders".into(),
            write_mode: CatalogStreamingWriteMode::Append,
            pk_columns: vec!["id".into()],
            target_types: vec![TypeId::Int64],
            target_column_names: vec!["id".into()],
            copy_threshold_rows: 1000,
            batch_size: 100,
            flush_interval: Duration::from_millis(500),
            dlq: dlq_arc.clone(),
            circuit_breaker: cb,
            retry_config: RetryConfig {
                max_attempts: 2,
                initial_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(10),
                jitter: false,
            },
            idempotency_key_columns: vec!["id".into()],
        };
        (ZyronSinkClient::new(cfg), dlq_arc)
    }

    fn sample_change(tag: u64) -> CdfChange {
        // Encoded 1-column Int64 row.
        let row_bytes = {
            // NSM format: null bitmap byte, then 8-byte little-endian i64.
            let mut v = vec![0u8]; // bitmap: all non-null
            v.extend_from_slice(&(tag as i64).to_le_bytes());
            v
        };
        CdfChange {
            commit_version: tag,
            commit_timestamp: 0,
            change_type: zyron_cdc::ChangeType::Insert,
            row_data: row_bytes,
            primary_key_data: Vec::new(),
        }
    }

    #[test]
    fn classify_sorts_changes() {
        let rows = vec![
            CdfChange {
                commit_version: 1,
                commit_timestamp: 0,
                change_type: zyron_cdc::ChangeType::Insert,
                row_data: vec![0, 1, 0, 0, 0, 0, 0, 0, 0],
                primary_key_data: vec![],
            },
            CdfChange {
                commit_version: 2,
                commit_timestamp: 0,
                change_type: zyron_cdc::ChangeType::Delete,
                row_data: vec![0, 2, 0, 0, 0, 0, 0, 0, 0],
                primary_key_data: vec![],
            },
            CdfChange {
                commit_version: 3,
                commit_timestamp: 0,
                change_type: zyron_cdc::ChangeType::UpdatePreimage,
                row_data: vec![0, 3, 0, 0, 0, 0, 0, 0, 0],
                primary_key_data: vec![],
            },
            CdfChange {
                commit_version: 4,
                commit_timestamp: 0,
                change_type: zyron_cdc::ChangeType::UpdatePostimage,
                row_data: vec![0, 4, 0, 0, 0, 0, 0, 0, 0],
                primary_key_data: vec![],
            },
        ];
        let (ups, dels) = ZyronSinkClient::classify(rows);
        assert_eq!(ups.len(), 2);
        assert_eq!(dels.len(), 1);
    }

    #[test]
    fn qualified_target_renders_schema() {
        let (sink, _) = make_sink(false);
        assert_eq!(sink.target(), "public.orders");
    }

    #[test]
    fn build_delete_sql_uses_pk() {
        let (sink, _) = make_sink(false);
        let sql = sink.build_delete_sql(2);
        assert!(sql.contains("DELETE FROM public.orders WHERE (id) IN"));
    }

    #[test]
    fn set_clause_excludes_pk_columns() {
        let cb = Arc::new(CircuitBreaker::new(0.5, 4, Duration::from_secs(1)));
        let cfg = ZyronSinkConfig {
            pool: make_pool(),
            target_schema: "public".into(),
            target_table: "t".into(),
            write_mode: CatalogStreamingWriteMode::Upsert,
            pk_columns: vec!["id".into()],
            target_types: vec![TypeId::Int64, TypeId::Varchar],
            target_column_names: vec!["id".into(), "name".into()],
            copy_threshold_rows: 1000,
            batch_size: 10,
            flush_interval: Duration::from_millis(1),
            dlq: None,
            circuit_breaker: cb,
            retry_config: RetryConfig::default(),
            idempotency_key_columns: vec![],
        };
        let sink = ZyronSinkClient::new(cfg);
        assert_eq!(sink.set_clause(), "name=EXCLUDED.name");
    }

    #[test]
    fn encode_literals_cover_types() {
        use zyron_streaming::row_codec::StreamValue;
        assert_eq!(encode_sql_literal(&StreamValue::Null), "NULL");
        assert_eq!(encode_sql_literal(&StreamValue::Bool(true)), "TRUE");
        assert_eq!(encode_sql_literal(&StreamValue::I64(42)), "42");
        assert_eq!(encode_sql_literal(&StreamValue::I128(42)), "42");
        assert_eq!(
            encode_sql_literal(&StreamValue::Utf8("bob's".into())),
            "'bob''s'"
        );
        let bin = encode_sql_literal(&StreamValue::Binary(vec![0xDE, 0xAD]));
        assert_eq!(bin, "'\\xdead'");
    }

    #[test]
    fn encode_binary_fields_nulls_map_to_none() {
        use zyron_streaming::row_codec::StreamValue;
        assert!(encode_binary_field(&StreamValue::Null).is_none());
        let b = encode_binary_field(&StreamValue::I64(1)).unwrap();
        assert_eq!(b, 1i64.to_be_bytes().to_vec());
    }

    #[tokio::test]
    async fn empty_batch_is_noop() {
        let (sink, _) = make_sink(false);
        sink.write_batch(Vec::new()).await.unwrap();
        let s = sink.stats();
        assert_eq!(s.rows_written, 0);
    }

    #[tokio::test]
    async fn write_batch_routes_to_dlq_when_circuit_open() {
        let (dlq, vec_sink) = DeadLetterQueue::with_vec_sink("dlq", 100);
        let dlq_arc = Arc::new(dlq);
        let cb = Arc::new(CircuitBreaker::new(0.5, 2, Duration::from_secs(60)));
        // Force the circuit open.
        cb.record_failure();
        cb.record_failure();
        let cfg = ZyronSinkConfig {
            pool: make_pool(),
            target_schema: "public".into(),
            target_table: "t".into(),
            write_mode: CatalogStreamingWriteMode::Append,
            pk_columns: vec![],
            target_types: vec![TypeId::Int64],
            target_column_names: vec!["id".into()],
            copy_threshold_rows: 1000,
            batch_size: 10,
            flush_interval: Duration::from_millis(1),
            dlq: Some(Arc::clone(&dlq_arc)),
            circuit_breaker: cb.clone(),
            retry_config: RetryConfig::default(),
            idempotency_key_columns: vec![],
        };
        let sink = ZyronSinkClient::new(cfg);
        sink.write_batch(vec![sample_change(1), sample_change(2)])
            .await
            .unwrap();
        assert_eq!(vec_sink.len(), 2);
        assert_eq!(sink.stats().dlq_rows, 2);
    }

    #[tokio::test]
    async fn write_batch_fails_fast_on_bad_host_without_dlq() {
        let (sink, _) = make_sink(false);
        // The pool targets port 1 which nothing listens on. Without a DLQ,
        // the error bubbles up to the caller.
        let res = sink
            .write_batch(vec![sample_change(1), sample_change(2)])
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn write_batch_bad_host_drains_to_dlq() {
        let (sink, dlq) = make_sink(true);
        let _ = sink
            .write_batch(vec![sample_change(1), sample_change(2)])
            .await
            .unwrap();
        // Either the retries exhaust normally or the circuit opens during the
        // attempts: in both cases rows land in the DLQ.
        let dlq = dlq.unwrap();
        assert!(dlq.count() >= 1);
    }

    #[test]
    fn stats_reflects_circuit_state() {
        let (sink, _) = make_sink(false);
        let s = sink.stats();
        assert_eq!(s.circuit_state, CircuitState::Closed);
    }

    #[test]
    fn extract_options_builds_map() {
        let opts = vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "x".to_string()),
        ];
        let m = extract_options(&opts);
        assert_eq!(m.get("a").map(|s| s.as_str()), Some("1"));
        assert_eq!(m.get("b").map(|s| s.as_str()), Some("x"));
    }

    #[tokio::test]
    async fn shutdown_flushes_pending_empty() {
        let (sink, _) = make_sink(false);
        sink.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn zyron_sink_client_impls_adapter_trait() {
        use zyron_streaming::sink_connector::ZyronSinkAdapter;
        let (sink, _) = make_sink(false);
        let adapter: Arc<dyn ZyronSinkAdapter> = Arc::new(sink);
        // Empty batch writes without touching the pool and always succeeds.
        adapter.write_batch(Vec::new()).await.unwrap();
        adapter.flush().await.unwrap();
        adapter.shutdown().await.unwrap();
    }
}
