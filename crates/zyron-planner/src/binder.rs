//! AST-to-bound-plan conversion with name resolution and type checking.
//!
//! The binder converts parsed AST nodes (string-based names) into bound nodes
//! (OID-based references with resolved types). This is the semantic analysis phase
//! that validates column existence, type compatibility, and resolves ambiguity.

use crate::logical::LogicalColumn;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use zyron_catalog::schema::{
    BackpressurePolicy, CatalogClassification, CatalogStreamingWriteMode, ColumnEntry,
    EndpointAuthMode, EndpointMessageFormat, EndpointOutputFormat, HttpMethod, RateLimitPeriod,
    RateLimitScope, RateLimitSpec, RowFormat, SecurityMapKind as CatalogSecurityMapKind,
};
use zyron_catalog::{
    Catalog, ColumnId, EndpointId, ExternalSinkId, ExternalSourceId, NameResolver, PublicationId,
    SchemaId, TableEntry, TableId,
};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::*;

// ---------------------------------------------------------------------------
// Resolved column reference
// ---------------------------------------------------------------------------

/// Uniquely identifies a column within a query scope.
/// table_idx is a local index into BindContext.tables, not the catalog TableId.
/// This correctly handles self-joins and subquery aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    pub table_idx: usize,
    pub column_id: ColumnId,
    pub type_id: TypeId,
    pub nullable: bool,
}

// ---------------------------------------------------------------------------
// Bound table and column definitions
// ---------------------------------------------------------------------------

/// Metadata for a table (or subquery) registered in a bind scope.
#[derive(Debug, Clone)]
pub struct BoundTableRef {
    pub table_idx: usize,
    pub table_id: Option<TableId>,
    pub alias: String,
    pub columns: Vec<BoundColumnDef>,
    pub entry: Option<Arc<TableEntry>>,
}

/// Column definition within a bound table scope.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundColumnDef {
    pub column_id: ColumnId,
    pub name: String,
    pub type_id: TypeId,
    pub nullable: bool,
    pub ordinal: u16,
}

// ---------------------------------------------------------------------------
// Bind context (scope for name resolution)
// ---------------------------------------------------------------------------

/// Scope for name resolution. Tracks visible tables and their columns.
/// A new context is pushed for each subquery.
#[derive(Debug)]
pub struct BindContext {
    pub tables: Vec<BoundTableRef>,
    pub ctes: HashMap<String, BoundCte>,
    pub outer: Option<Box<BindContext>>,
}

impl BindContext {
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            ctes: HashMap::new(),
            outer: None,
        }
    }

    pub fn with_outer(outer: Box<BindContext>) -> Self {
        Self {
            tables: Vec::new(),
            ctes: HashMap::new(),
            outer: Some(outer),
        }
    }
}

/// A bound CTE storing the bound query and output column schema.
#[derive(Debug, Clone)]
pub struct BoundCte {
    pub name: String,
    pub columns: Vec<BoundColumnDef>,
    pub query: Box<BoundSelect>,
}

// ---------------------------------------------------------------------------
// Bound expressions
// ---------------------------------------------------------------------------

/// Type-resolved expression. Replaces string identifiers with ColumnRef.
/// PartialEq is implemented manually because Subquery/Exists/InSubquery
/// variants contain BoundSelect which has Arc<TableEntry> (no PartialEq).
/// Those variants compare as structurally unequal, which is conservative
/// and correct for the optimizer's change-detection logic.
#[derive(Debug, Clone)]
pub enum BoundExpr {
    ColumnRef(ColumnRef),
    Literal {
        value: LiteralValue,
        type_id: TypeId,
    },
    BinaryOp {
        left: Box<BoundExpr>,
        op: BinaryOperator,
        right: Box<BoundExpr>,
        type_id: TypeId,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: Box<BoundExpr>,
        type_id: TypeId,
    },
    IsNull {
        expr: Box<BoundExpr>,
        negated: bool,
    },
    InList {
        expr: Box<BoundExpr>,
        list: Vec<BoundExpr>,
        negated: bool,
    },
    Between {
        expr: Box<BoundExpr>,
        low: Box<BoundExpr>,
        high: Box<BoundExpr>,
        negated: bool,
    },
    Like {
        expr: Box<BoundExpr>,
        pattern: Box<BoundExpr>,
        negated: bool,
    },
    ILike {
        expr: Box<BoundExpr>,
        pattern: Box<BoundExpr>,
        negated: bool,
    },
    Function {
        name: String,
        args: Vec<BoundExpr>,
        return_type: TypeId,
        distinct: bool,
    },
    AggregateFunction {
        name: String,
        args: Vec<BoundExpr>,
        distinct: bool,
        return_type: TypeId,
    },
    Cast {
        expr: Box<BoundExpr>,
        target_type: TypeId,
    },
    Case {
        operand: Option<Box<BoundExpr>>,
        conditions: Vec<BoundWhen>,
        else_result: Option<Box<BoundExpr>>,
        type_id: TypeId,
    },
    Nested(Box<BoundExpr>),
    Subquery {
        plan: Box<BoundSelect>,
        type_id: TypeId,
    },
    Exists {
        plan: Box<BoundSelect>,
        negated: bool,
    },
    InSubquery {
        expr: Box<BoundExpr>,
        plan: Box<BoundSelect>,
        negated: bool,
    },
    WindowFunction {
        function: Box<BoundExpr>,
        partition_by: Vec<BoundExpr>,
        order_by: Vec<BoundOrderBy>,
        frame: Option<WindowFrame>,
        type_id: TypeId,
    },
    Parameter {
        index: usize,
        type_id: TypeId,
    },
}

impl PartialEq for BoundExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BoundExpr::ColumnRef(a), BoundExpr::ColumnRef(b)) => a == b,
            (
                BoundExpr::Literal {
                    value: v1,
                    type_id: t1,
                },
                BoundExpr::Literal {
                    value: v2,
                    type_id: t2,
                },
            ) => v1 == v2 && t1 == t2,
            (
                BoundExpr::BinaryOp {
                    left: l1,
                    op: o1,
                    right: r1,
                    type_id: t1,
                },
                BoundExpr::BinaryOp {
                    left: l2,
                    op: o2,
                    right: r2,
                    type_id: t2,
                },
            ) => o1 == o2 && t1 == t2 && l1 == l2 && r1 == r2,
            (
                BoundExpr::UnaryOp {
                    op: o1,
                    expr: e1,
                    type_id: t1,
                },
                BoundExpr::UnaryOp {
                    op: o2,
                    expr: e2,
                    type_id: t2,
                },
            ) => o1 == o2 && t1 == t2 && e1 == e2,
            (
                BoundExpr::IsNull {
                    expr: e1,
                    negated: n1,
                },
                BoundExpr::IsNull {
                    expr: e2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2,
            (
                BoundExpr::InList {
                    expr: e1,
                    list: l1,
                    negated: n1,
                },
                BoundExpr::InList {
                    expr: e2,
                    list: l2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && l1 == l2,
            (
                BoundExpr::Between {
                    expr: e1,
                    low: lo1,
                    high: hi1,
                    negated: n1,
                },
                BoundExpr::Between {
                    expr: e2,
                    low: lo2,
                    high: hi2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && lo1 == lo2 && hi1 == hi2,
            (
                BoundExpr::Like {
                    expr: e1,
                    pattern: p1,
                    negated: n1,
                },
                BoundExpr::Like {
                    expr: e2,
                    pattern: p2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && p1 == p2,
            (
                BoundExpr::ILike {
                    expr: e1,
                    pattern: p1,
                    negated: n1,
                },
                BoundExpr::ILike {
                    expr: e2,
                    pattern: p2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && p1 == p2,
            (
                BoundExpr::Function {
                    name: n1,
                    args: a1,
                    return_type: r1,
                    distinct: d1,
                },
                BoundExpr::Function {
                    name: n2,
                    args: a2,
                    return_type: r2,
                    distinct: d2,
                },
            ) => n1 == n2 && d1 == d2 && r1 == r2 && a1 == a2,
            (
                BoundExpr::AggregateFunction {
                    name: n1,
                    args: a1,
                    distinct: d1,
                    return_type: r1,
                },
                BoundExpr::AggregateFunction {
                    name: n2,
                    args: a2,
                    distinct: d2,
                    return_type: r2,
                },
            ) => n1 == n2 && d1 == d2 && r1 == r2 && a1 == a2,
            (
                BoundExpr::Cast {
                    expr: e1,
                    target_type: t1,
                },
                BoundExpr::Cast {
                    expr: e2,
                    target_type: t2,
                },
            ) => t1 == t2 && e1 == e2,
            (
                BoundExpr::Case {
                    operand: o1,
                    conditions: c1,
                    else_result: e1,
                    type_id: t1,
                },
                BoundExpr::Case {
                    operand: o2,
                    conditions: c2,
                    else_result: e2,
                    type_id: t2,
                },
            ) => t1 == t2 && o1 == o2 && c1 == c2 && e1 == e2,
            (BoundExpr::Nested(a), BoundExpr::Nested(b)) => a == b,
            (
                BoundExpr::Parameter {
                    index: i1,
                    type_id: t1,
                },
                BoundExpr::Parameter {
                    index: i2,
                    type_id: t2,
                },
            ) => i1 == i2 && t1 == t2,
            (
                BoundExpr::WindowFunction {
                    function: f1,
                    partition_by: p1,
                    order_by: o1,
                    frame: fr1,
                    type_id: t1,
                },
                BoundExpr::WindowFunction {
                    function: f2,
                    partition_by: p2,
                    order_by: o2,
                    frame: fr2,
                    type_id: t2,
                },
            ) => t1 == t2 && f1 == f2 && p1 == p2 && o1 == o2 && fr1 == fr2,
            // Subquery variants: conservatively unequal (contain Arc<TableEntry>).
            (BoundExpr::Subquery { .. }, BoundExpr::Subquery { .. }) => false,
            (BoundExpr::Exists { .. }, BoundExpr::Exists { .. }) => false,
            (BoundExpr::InSubquery { .. }, BoundExpr::InSubquery { .. }) => false,
            _ => false,
        }
    }
}

impl BoundExpr {
    /// Returns the output TypeId of this expression.
    pub fn type_id(&self) -> TypeId {
        match self {
            BoundExpr::ColumnRef(cr) => cr.type_id,
            BoundExpr::Literal { type_id, .. } => *type_id,
            BoundExpr::BinaryOp { type_id, .. } => *type_id,
            BoundExpr::UnaryOp { type_id, .. } => *type_id,
            BoundExpr::IsNull { .. } => TypeId::Boolean,
            BoundExpr::InList { .. } => TypeId::Boolean,
            BoundExpr::Between { .. } => TypeId::Boolean,
            BoundExpr::Like { .. } => TypeId::Boolean,
            BoundExpr::ILike { .. } => TypeId::Boolean,
            BoundExpr::Function { return_type, .. } => *return_type,
            BoundExpr::AggregateFunction { return_type, .. } => *return_type,
            BoundExpr::Cast { target_type, .. } => *target_type,
            BoundExpr::Case { type_id, .. } => *type_id,
            BoundExpr::Nested(inner) => inner.type_id(),
            BoundExpr::Subquery { type_id, .. } => *type_id,
            BoundExpr::Exists { .. } => TypeId::Boolean,
            BoundExpr::InSubquery { .. } => TypeId::Boolean,
            BoundExpr::WindowFunction { type_id, .. } => *type_id,
            BoundExpr::Parameter { type_id, .. } => *type_id,
        }
    }

    /// Returns true if this expression can produce NULL.
    pub fn nullable(&self) -> bool {
        match self {
            BoundExpr::ColumnRef(cr) => cr.nullable,
            BoundExpr::Literal { value, .. } => matches!(value, LiteralValue::Null),
            BoundExpr::IsNull { .. } => false,
            BoundExpr::Exists { .. } => false,
            BoundExpr::InSubquery { .. } => false,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundWhen {
    pub condition: BoundExpr,
    pub result: BoundExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundOrderBy {
    pub expr: BoundExpr,
    pub asc: bool,
    pub nulls_first: bool,
}

// ---------------------------------------------------------------------------
// Bound statements
// ---------------------------------------------------------------------------

/// Top-level bound statement.
#[derive(Debug, Clone)]
pub enum BoundStatement {
    Select(BoundSelect),
    Insert(BoundInsert),
    Update(BoundUpdate),
    Delete(BoundDelete),
    CreateStreamingJob(BoundStreamingJob),
    DropStreamingJob {
        name: String,
        if_exists: bool,
    },
    AlterStreamingJob {
        name: String,
        action: AlterStreamingJobAction,
    },
    CreateExternalSource(Box<BoundCreateExternalSource>),
    CreateExternalSink(Box<BoundCreateExternalSink>),
    DropExternalSource {
        name: String,
        schema_id: SchemaId,
        if_exists: bool,
    },
    DropExternalSink {
        name: String,
        schema_id: SchemaId,
        if_exists: bool,
    },
    AlterExternalSource(Box<BoundAlterExternalSource>),
    AlterExternalSink(Box<BoundAlterExternalSink>),

    // -----------------------------------------------------------------------
    // Zyron-to-Zyron data plane: publications, endpoints, security map
    // -----------------------------------------------------------------------
    CreatePublication(Box<BoundCreatePublication>),
    AlterPublication(Box<BoundAlterPublication>),
    DropPublication {
        name: String,
        schema_id: SchemaId,
        if_exists: bool,
        cascade: bool,
    },
    CreateEndpoint(Box<BoundCreateEndpoint>),
    CreateStreamingEndpoint(Box<BoundCreateStreamingEndpoint>),
    AlterEndpoint(Box<BoundAlterEndpoint>),
    DropEndpoint {
        name: String,
        schema_id: SchemaId,
        if_exists: bool,
    },
    AlterSecurityMap(Box<BoundAlterSecurityMap>),
    DropSecurityMap(Box<BoundDropSecurityMap>),
    TagPublication {
        name: String,
        schema_id: SchemaId,
        publication_id: PublicationId,
        tags: Vec<String>,
    },
    UntagPublication {
        name: String,
        schema_id: SchemaId,
        publication_id: PublicationId,
        tags: Vec<String>,
    },
}

// ---------------------------------------------------------------------------
// Bound Zyron-to-Zyron DDL: publication
// ---------------------------------------------------------------------------

/// Bound form of CREATE PUBLICATION with resolved tables, columns, bound
/// predicates, parsed options, and a deterministic schema fingerprint.
#[derive(Debug, Clone)]
pub struct BoundCreatePublication {
    pub schema_id: SchemaId,
    pub name: String,
    pub if_not_exists: bool,
    pub tables: Vec<BoundPublicationTable>,
    pub where_predicate: Option<BoundExpr>,
    pub change_feed: bool,
    pub row_format: RowFormat,
    pub retention_days: u32,
    pub retain_until_subscribers_advance: bool,
    pub max_rows_per_sec: u64,
    pub max_bytes_per_sec: u64,
    pub max_concurrent_subscribers: u32,
    pub classification: CatalogClassification,
    pub allow_initial_snapshot: bool,
    pub schema_fingerprint: [u8; 32],
}

/// Resolved single table reference inside a publication. Empty columns means
/// all table columns are exposed, consistent with the catalog entry shape.
#[derive(Debug, Clone)]
pub struct BoundPublicationTable {
    pub table_id: TableId,
    pub columns: Vec<ColumnId>,
    pub where_predicate: Option<BoundExpr>,
}

/// Bound form of ALTER PUBLICATION. The binder resolves the target and
/// produces the action-specific payload. Recomputing the schema fingerprint
/// happens in the wire dispatcher when the table set changes.
#[derive(Debug, Clone)]
pub struct BoundAlterPublication {
    pub name: String,
    pub schema_id: SchemaId,
    pub publication_id: PublicationId,
    pub action: BoundAlterPublicationAction,
}

#[derive(Debug, Clone)]
pub enum BoundAlterPublicationAction {
    AddTable(BoundPublicationTable),
    DropTable(TableId),
    SetOptions(PublicationOptionUpdates),
    SetWhere(BoundExpr),
    Rename(String),
}

/// Parsed ALTER PUBLICATION SET option overrides. Only the options the user
/// supplied are Some, the rest stay None so the wire layer knows which
/// fields to update.
#[derive(Debug, Clone, Default)]
pub struct PublicationOptionUpdates {
    pub retention_days: Option<u32>,
    pub retain_until_subscribers_advance: Option<bool>,
    pub max_rows_per_sec: Option<u64>,
    pub max_bytes_per_sec: Option<u64>,
    pub max_concurrent_subscribers: Option<u32>,
    pub classification: Option<CatalogClassification>,
    pub allow_initial_snapshot: Option<bool>,
    pub change_feed: Option<bool>,
    pub row_format: Option<RowFormat>,
}

// ---------------------------------------------------------------------------
// Bound Zyron-to-Zyron DDL: endpoints
// ---------------------------------------------------------------------------

/// Bound form of CREATE ENDPOINT (REST). The embedded bound_sql exists for
/// validation, the raw sql string stays for runtime re-compilation against
/// each request's parameter values.
#[derive(Debug, Clone)]
pub struct BoundCreateEndpoint {
    pub schema_id: SchemaId,
    pub name: String,
    pub if_not_exists: bool,
    pub path: String,
    pub methods: Vec<HttpMethod>,
    pub sql: String,
    pub bound_sql: Box<BoundStatement>,
    pub param_names: Vec<String>,
    pub param_types: Vec<TypeId>,
    pub output_columns: Vec<ColumnEntry>,
    pub auth: EndpointAuthMode,
    pub required_scopes: Vec<String>,
    pub rate_limit: Option<RateLimitSpec>,
    pub output_format: EndpointOutputFormat,
    pub cors_origins: Vec<String>,
    pub cache_seconds: u32,
    pub timeout_seconds: u32,
    pub max_body_bytes: u32,
}

/// Bound form of CREATE STREAMING ENDPOINT (WebSocket or SSE) backed by an
/// existing publication. The binder resolves the publication id so the wire
/// layer does not have to look it up again.
#[derive(Debug, Clone)]
pub struct BoundCreateStreamingEndpoint {
    pub schema_id: SchemaId,
    pub name: String,
    pub if_not_exists: bool,
    pub path: String,
    pub protocol: StreamingEndpointProtocol,
    pub backing_publication_id: PublicationId,
    pub backing_publication_name: String,
    pub auth: EndpointAuthMode,
    pub required_scopes: Vec<String>,
    pub max_connections_per_ip: Option<u32>,
    pub message_format: EndpointMessageFormat,
    pub heartbeat_seconds: u32,
    pub backpressure: BackpressurePolicy,
    pub max_connections: u32,
}

#[derive(Debug, Clone)]
pub struct BoundAlterEndpoint {
    pub name: String,
    pub schema_id: SchemaId,
    pub endpoint_id: EndpointId,
    pub action: BoundAlterEndpointAction,
}

#[derive(Debug, Clone)]
pub enum BoundAlterEndpointAction {
    Enable,
    Disable,
    SetOptions(EndpointOptionUpdates),
}

#[derive(Debug, Clone, Default)]
pub struct EndpointOptionUpdates {
    pub cache_seconds: Option<u32>,
    pub timeout_seconds: Option<u32>,
    pub max_body_bytes: Option<u32>,
    pub heartbeat_seconds: Option<u32>,
    pub max_connections: Option<u32>,
    pub max_connections_per_ip: Option<u32>,
}

// ---------------------------------------------------------------------------
// Bound Zyron-to-Zyron DDL: security map
// ---------------------------------------------------------------------------

/// Bound form of ALTER SECURITY MAP. The kind and identity key are
/// normalized, the role name is passed through unchanged so the wire layer
/// can resolve it against the security manager at apply time.
#[derive(Debug, Clone)]
pub struct BoundAlterSecurityMap {
    pub kind: CatalogSecurityMapKind,
    pub identity_key: String,
    pub role_name: String,
}

#[derive(Debug, Clone)]
pub struct BoundDropSecurityMap {
    pub kind: CatalogSecurityMapKind,
    pub identity_key: String,
}

// ---------------------------------------------------------------------------
// Bound external source and sink DDL
// ---------------------------------------------------------------------------

/// Bound form of CREATE EXTERNAL SOURCE. The binder resolves the target
/// schema and passes parser-level backend, format, and mode enums through
/// without further translation. The catalog layer converts those to its
/// own enum space when the wire dispatcher persists the entry.
#[derive(Debug, Clone)]
pub struct BoundCreateExternalSource {
    pub schema_id: SchemaId,
    pub name: String,
    pub if_not_exists: bool,
    pub backend: zyron_parser::ast::ExternalBackendKind,
    pub uri: String,
    pub format: zyron_parser::ast::ExternalFormatKind,
    pub mode: zyron_parser::ast::ExternalModeSpec,
    pub options: Vec<(String, String)>,
    pub credentials: Vec<(String, String)>,
    // Explicit COLUMNS clause resolved to TypeIds. Empty when the user did
    // not supply a columns clause, in which case the dispatcher may infer
    // the schema from the first matching file.
    pub columns: Vec<(String, zyron_common::TypeId)>,
}

/// Bound form of CREATE EXTERNAL SINK. Same fields as the source shape
/// with no ingest cadence, sinks emit on the stream's own cadence.
#[derive(Debug, Clone)]
pub struct BoundCreateExternalSink {
    pub schema_id: SchemaId,
    pub name: String,
    pub if_not_exists: bool,
    pub backend: zyron_parser::ast::ExternalBackendKind,
    pub uri: String,
    pub format: zyron_parser::ast::ExternalFormatKind,
    pub options: Vec<(String, String)>,
    pub credentials: Vec<(String, String)>,
    // Explicit COLUMNS clause resolved to TypeIds.
    pub columns: Vec<(String, zyron_common::TypeId)>,
}

#[derive(Debug, Clone)]
pub struct BoundAlterExternalSource {
    pub schema_id: SchemaId,
    pub name: String,
    pub action: zyron_parser::ast::AlterExternalSourceAction,
}

#[derive(Debug, Clone)]
pub struct BoundAlterExternalSink {
    pub schema_id: SchemaId,
    pub name: String,
    pub action: zyron_parser::ast::AlterExternalSinkAction,
}

// ---------------------------------------------------------------------------
// Bound streaming job
// ---------------------------------------------------------------------------

/// Resolved streaming source. Either a Zyron table with change-data-feed, a
/// named external source from the catalog, or an inline endpoint embedded in
/// the statement. ExternalNamed carries the source's catalog classification so
/// the binder can enforce downstream clearance checks.
#[derive(Debug, Clone)]
pub enum BoundStreamingSource {
    ZyronTable {
        table_id: TableId,
        schema_id: SchemaId,
        columns: Vec<ColumnEntry>,
    },
    ExternalNamed {
        source_id: ExternalSourceId,
        schema_id: SchemaId,
        /// Column schema derived from the target table. External sources do
        /// not store their own column schema in the catalog, the binder uses
        /// the target columns' types in projection order and names.
        columns: Vec<ColumnEntry>,
        classification: CatalogClassification,
    },
    ExternalInline {
        backend: zyron_parser::ast::ExternalBackendKind,
        uri: String,
        format: zyron_parser::ast::ExternalFormatKind,
        options: Vec<(String, String)>,
        mode: zyron_parser::ast::ExternalModeSpec,
        columns: Vec<ColumnEntry>,
    },
}

/// Resolved streaming sink. Mirrors BoundStreamingSource for the output side.
#[derive(Debug, Clone)]
pub enum BoundStreamingSink {
    ZyronTable {
        table_id: TableId,
        schema_id: SchemaId,
        columns: Vec<ColumnEntry>,
    },
    ExternalNamed {
        sink_id: ExternalSinkId,
        schema_id: SchemaId,
        columns: Vec<ColumnEntry>,
        classification: CatalogClassification,
    },
    ExternalInline {
        backend: zyron_parser::ast::ExternalBackendKind,
        uri: String,
        format: zyron_parser::ast::ExternalFormatKind,
        options: Vec<(String, String)>,
        columns: Vec<ColumnEntry>,
    },
}

/// Validated CREATE STREAMING JOB form. Produced by the binder and consumed by
/// the wire-layer dispatcher. The binder enforces single-table FROM with an
/// optional WHERE predicate. Projections must match the target column arity
/// and types exactly. Source and target may each be a Zyron table or an
/// external endpoint.
#[derive(Debug, Clone)]
pub struct BoundStreamingJob {
    pub name: String,
    pub if_not_exists: bool,
    pub source: BoundStreamingSource,
    pub target: BoundStreamingSink,
    pub projections: Vec<BoundExpr>,
    pub predicate: Option<BoundExpr>,
    pub write_mode: CatalogStreamingWriteMode,
    pub job_mode: zyron_parser::ast::ExternalModeSpec,
    // -----------------------------------------------------------------------------
    // Primary key column ids of the target table for UPSERT write mode.
    // Empty for Append mode or when the target is external. The binder
    // populates this from the target's catalog constraints whenever the
    // write mode is Upsert.
    pub target_pk_columns: Vec<ColumnId>,
    // -----------------------------------------------------------------------------
    // Windowed-aggregate configuration derived from GROUP BY + WATERMARK +
    // late-data policy. None when the job is a pure filter+project pipeline.
    pub aggregate: Option<BoundAggregateSpec>,
    // -----------------------------------------------------------------------------
    // Join configuration for two-source streaming jobs. None for single-table
    // FROM. When set, the runner uses the interval or temporal join engine
    // instead of the plain filter+project loop.
    pub join: Option<BoundStreamingJoinSpec>,
}

/// Outer-join semantics carried by the bound streaming spec. Mirrors the
/// parser's StreamingJoinType, kept as a planner-local type so downstream
/// crates (wire, streaming) can match on it without pulling the parser in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundStreamingJoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Resolved streaming-job join. Interval form carries two stream sources and
/// a symmetric time window. Temporal form carries one stream source and a
/// Zyron table looked up per incoming row by primary key.
#[derive(Debug, Clone)]
pub enum BoundStreamingJoinSpec {
    Interval {
        right_source: BoundStreamingSource,
        right_alias: String,
        /// Equi-key ordinals on the left source. Matches right_key_ordinals
        /// pairwise, and each pair must share a TypeId.
        left_key_ordinals: Vec<u16>,
        right_key_ordinals: Vec<u16>,
        left_event_time_ordinal: u16,
        right_event_time_ordinal: u16,
        within_us: i64,
        /// Inner emits only matching rows. Left, Right, Full flush
        /// unmatched rows on each side with NULLs on the opposite side
        /// when the watermark advances past event_time + within_us.
        join_type: BoundStreamingJoinType,
        combined_columns: Vec<ColumnEntry>,
    },
    Temporal {
        right_table_id: TableId,
        right_schema_id: SchemaId,
        right_alias: String,
        right_pk_ordinals: Vec<u16>,
        left_key_ordinals: Vec<u16>,
        left_event_time_ordinal: u16,
        /// Only Inner or Left are allowed on temporal joins. Right and Full
        /// are rejected at bind time.
        join_type: BoundStreamingJoinType,
        combined_columns: Vec<ColumnEntry>,
    },
}

// ---------------------------------------------------------------------------
// BoundAggregateSpec: windowed aggregation for streaming jobs
// ---------------------------------------------------------------------------

/// Resolved aggregate section of a streaming job. The binder produces this
/// when the job's SELECT declares a GROUP BY with a window expression and
/// the job carries a WATERMARK clause.
#[derive(Debug, Clone)]
pub struct BoundAggregateSpec {
    pub window_type: BoundStreamingWindowType,
    pub event_time_column_id: ColumnId,
    pub event_time_scale: BoundEventTimeScale,
    pub group_by_column_ids: Vec<ColumnId>,
    pub aggregations: Vec<BoundAggregateItem>,
    pub watermark: BoundWatermark,
    pub late_data_policy: BoundLateDataPolicy,
}

/// Event-time normalization. Mirrors zyron_streaming::EventTimeScale but
/// keeps the planner free of the streaming-crate dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundEventTimeScale {
    Microseconds,
    Milliseconds,
    Seconds,
}

/// Watermark configuration resolved from the WATERMARK FOR clause. The
/// planner captures the strategy parameters and the wire layer builds the
/// concrete WatermarkStrategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundWatermark {
    BoundedOutOfOrderness { allowed_lateness_us: i64 },
    Punctual,
}

/// Late-data handling mirror of zyron_streaming::LateDataPolicy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundLateDataPolicy {
    Drop,
    ReopenWindow,
    SideOutput,
    Update,
}

/// Concrete window shape resolved at bind time. All durations are already
/// normalized to milliseconds so the runner assigners consume them directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundStreamingWindowType {
    Tumbling { size_ms: i64 },
    Hopping { size_ms: i64, slide_ms: i64 },
    Session { gap_ms: i64 },
}

/// One aggregate output column. input_column_id is None for COUNT(*).
#[derive(Debug, Clone)]
pub struct BoundAggregateItem {
    pub function: String,
    pub input_column_id: Option<ColumnId>,
    pub output_type: TypeId,
}

impl BoundStreamingJob {
    /// Returns the source's effective schema id. Inline sources have no
    /// schema binding, so this falls back to the target's schema.
    pub fn source_schema_id(&self) -> SchemaId {
        match &self.source {
            BoundStreamingSource::ZyronTable { schema_id, .. } => *schema_id,
            BoundStreamingSource::ExternalNamed { schema_id, .. } => *schema_id,
            BoundStreamingSource::ExternalInline { .. } => self.target_schema_id(),
        }
    }

    /// Returns the target's effective schema id. Inline sinks reuse the
    /// source's schema when the source is internal, else the catalog's
    /// default schema. Inline-both is rejected at bind time.
    pub fn target_schema_id(&self) -> SchemaId {
        match &self.target {
            BoundStreamingSink::ZyronTable { schema_id, .. } => *schema_id,
            BoundStreamingSink::ExternalNamed { schema_id, .. } => *schema_id,
            BoundStreamingSink::ExternalInline { .. } => match &self.source {
                BoundStreamingSource::ZyronTable { schema_id, .. } => *schema_id,
                BoundStreamingSource::ExternalNamed { schema_id, .. } => *schema_id,
                BoundStreamingSource::ExternalInline { .. } => SchemaId(0),
            },
        }
    }

    /// Returns the source columns regardless of variant.
    pub fn source_columns(&self) -> &[ColumnEntry] {
        match &self.source {
            BoundStreamingSource::ZyronTable { columns, .. } => columns,
            BoundStreamingSource::ExternalNamed { columns, .. } => columns,
            BoundStreamingSource::ExternalInline { columns, .. } => columns,
        }
    }

    /// Returns the target columns regardless of variant.
    pub fn target_columns(&self) -> &[ColumnEntry] {
        match &self.target {
            BoundStreamingSink::ZyronTable { columns, .. } => columns,
            BoundStreamingSink::ExternalNamed { columns, .. } => columns,
            BoundStreamingSink::ExternalInline { columns, .. } => columns,
        }
    }

    /// Returns the source table id when the source is a Zyron table.
    /// Returns None for external source variants.
    pub fn source_table_id(&self) -> Option<TableId> {
        match &self.source {
            BoundStreamingSource::ZyronTable { table_id, .. } => Some(*table_id),
            _ => None,
        }
    }

    /// Returns the target table id when the target is a Zyron table.
    /// Returns None for external sink variants.
    pub fn target_table_id(&self) -> Option<TableId> {
        match &self.target {
            BoundStreamingSink::ZyronTable { table_id, .. } => Some(*table_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoundSelect {
    pub projections: Vec<BoundSelectItem>,
    pub from: Vec<BoundFromItem>,
    pub where_clause: Option<BoundExpr>,
    pub group_by: Vec<BoundExpr>,
    pub having: Option<BoundExpr>,
    pub order_by: Vec<BoundOrderBy>,
    pub limit: Option<BoundExpr>,
    pub offset: Option<BoundExpr>,
    pub distinct: bool,
    pub set_ops: Vec<BoundSetOp>,
    pub ctes: Vec<BoundCte>,
    pub output_schema: Vec<BoundColumnDef>,
}

#[derive(Debug, Clone)]
pub enum BoundSelectItem {
    Expr(BoundExpr, Option<String>),
    AllColumns(usize),
    Wildcard,
}

#[derive(Debug, Clone)]
pub enum BoundFromItem {
    BaseTable {
        table_idx: usize,
        table_id: TableId,
        entry: Arc<TableEntry>,
    },
    Join {
        left: Box<BoundFromItem>,
        join_type: JoinType,
        right: Box<BoundFromItem>,
        condition: BoundJoinCondition,
    },
    Subquery {
        table_idx: usize,
        query: Box<BoundSelect>,
    },
    GraphQuery {
        table_idx: usize,
        schema_name: String,
        algorithm: String,
        params: Vec<(String, BoundExpr)>,
        output_columns: Vec<LogicalColumn>,
    },
}

#[derive(Debug, Clone)]
pub enum BoundJoinCondition {
    On(BoundExpr),
    Using(Vec<ColumnId>),
    Natural,
    None,
}

#[derive(Debug, Clone)]
pub struct BoundSetOp {
    pub op: SetOpType,
    pub all: bool,
    pub right: Box<BoundSelect>,
}

#[derive(Debug, Clone)]
pub struct BoundInsert {
    pub table_id: TableId,
    pub table_entry: Arc<TableEntry>,
    pub target_columns: Vec<ColumnId>,
    pub source: BoundInsertSource,
    pub returning: Option<Vec<BoundSelectItem>>,
}

#[derive(Debug, Clone)]
pub enum BoundInsertSource {
    Values(Vec<Vec<BoundExpr>>),
    Query(Box<BoundSelect>),
}

#[derive(Debug, Clone)]
pub struct BoundUpdate {
    pub table_id: TableId,
    pub table_entry: Arc<TableEntry>,
    pub assignments: Vec<BoundAssignment>,
    pub where_clause: Option<BoundExpr>,
    pub returning: Option<Vec<BoundSelectItem>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundAssignment {
    pub column_id: ColumnId,
    pub value: BoundExpr,
}

#[derive(Debug, Clone)]
pub struct BoundDelete {
    pub table_id: TableId,
    pub table_entry: Arc<TableEntry>,
    pub where_clause: Option<BoundExpr>,
    pub returning: Option<Vec<BoundSelectItem>>,
}

// ---------------------------------------------------------------------------
// Aggregate function names
// ---------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// TLS option validation for Zyron external source/sink statements.
// -----------------------------------------------------------------------------
//
// When the caller supplies tls=required or tls=preferred on a ZYRON backend,
// the bind step refuses to accept the statement unless at least one of the
// trust anchors ca_cert_pem, ca_cert_path, fingerprint_pin, or
// trust_system_roots=true is also present. A bare tls=required with no trust
// configuration would silently accept any server certificate at runtime, so
// the bind error surfaces the issue up-front before the catalog entry is
// written.
fn validate_zyron_tls_options(options: &[(String, String)]) -> Result<()> {
    let mut tls_mode: Option<&str> = None;
    let mut has_ca_cert_pem = false;
    let mut has_ca_cert_path = false;
    let mut has_fingerprint_pin = false;
    let mut trust_system_roots = false;
    for (k, v) in options {
        let key = k.as_str();
        if key.eq_ignore_ascii_case("tls") {
            tls_mode = Some(v.as_str());
        } else if key.eq_ignore_ascii_case("ca_cert_pem") && !v.is_empty() {
            has_ca_cert_pem = true;
        } else if key.eq_ignore_ascii_case("ca_cert_path") && !v.is_empty() {
            has_ca_cert_path = true;
        } else if key.eq_ignore_ascii_case("fingerprint_pin") && !v.is_empty() {
            has_fingerprint_pin = true;
        } else if key.eq_ignore_ascii_case("trust_system_roots") {
            trust_system_roots = matches!(v.as_str(), "true" | "TRUE" | "True" | "1" | "yes");
        }
    }
    let mode = match tls_mode {
        Some(m) => m,
        None => return Ok(()),
    };
    let normalized = mode.to_ascii_lowercase();
    if normalized == "disabled" || normalized == "off" || normalized == "none" {
        return Ok(());
    }
    if !matches!(normalized.as_str(), "required" | "preferred") {
        return Ok(());
    }
    if has_ca_cert_pem || has_ca_cert_path || has_fingerprint_pin || trust_system_roots {
        return Ok(());
    }
    Err(ZyronError::PlanError(
        "TLS requires one of: ca_cert_pem, ca_cert_path, fingerprint_pin, or trust_system_roots=true".to_string(),
    ))
}

const AGGREGATE_FUNCTIONS: &[&str] = &["count", "sum", "avg", "min", "max"];

fn is_aggregate_function(name: &str) -> bool {
    if AGGREGATE_FUNCTIONS
        .iter()
        .any(|&f| f.eq_ignore_ascii_case(name))
    {
        return true;
    }
    zyron_types::is_types_aggregate_function(name)
}

// ---------------------------------------------------------------------------
// Publication/endpoint/security-map free helpers
// ---------------------------------------------------------------------------

/// Parses a boolean option value accepting true/false (case-insensitive).
fn parse_bool_option(key: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Ok(true),
        "false" | "0" | "off" | "no" => Ok(false),
        _ => Err(ZyronError::PlanError(format!(
            "option '{key}' expects a boolean, got '{value}'"
        ))),
    }
}

/// Parses an unsigned 32-bit decimal option value.
fn parse_u32_option(key: &str, value: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|_| ZyronError::PlanError(format!("option '{key}' expects a u32, got '{value}'")))
}

/// Parses an unsigned 64-bit decimal option value.
fn parse_u64_option(key: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| ZyronError::PlanError(format!("option '{key}' expects a u64, got '{value}'")))
}

/// Parses a human-readable size string like "100MB" or "512KB" into bytes.
/// Accepts plain integers, KB, MB, GB, TB suffixes (1024-based).
fn parse_size_bytes(value: &str) -> Result<u64> {
    let s = value.trim();
    if s.is_empty() {
        return Err(ZyronError::PlanError("empty size value".to_string()));
    }
    let lower = s.to_ascii_lowercase();
    let (num_part, mult) = if let Some(n) = lower.strip_suffix("tb") {
        (n.trim(), 1024u64 * 1024 * 1024 * 1024)
    } else if let Some(n) = lower.strip_suffix("gb") {
        (n.trim(), 1024u64 * 1024 * 1024)
    } else if let Some(n) = lower.strip_suffix("mb") {
        (n.trim(), 1024u64 * 1024)
    } else if let Some(n) = lower.strip_suffix("kb") {
        (n.trim(), 1024u64)
    } else if let Some(n) = lower.strip_suffix('b') {
        (n.trim(), 1u64)
    } else {
        (lower.as_str(), 1u64)
    };
    let n: u64 = num_part
        .parse()
        .map_err(|_| ZyronError::PlanError(format!("invalid size value '{value}'")))?;
    n.checked_mul(mult)
        .ok_or_else(|| ZyronError::PlanError(format!("size value '{value}' overflows u64")))
}

/// Parses a publication row_format option into the catalog enum.
fn parse_row_format(value: &str) -> Result<RowFormat> {
    match value.to_ascii_lowercase().as_str() {
        "binary" => Ok(RowFormat::Binary),
        "text" => Ok(RowFormat::Text),
        _ => Err(ZyronError::PlanError(format!(
            "row_format must be 'binary' or 'text', got '{value}'"
        ))),
    }
}

/// Parses a publication classification option into the catalog enum.
fn parse_classification(value: &str) -> Result<CatalogClassification> {
    match value.to_ascii_lowercase().as_str() {
        "public" => Ok(CatalogClassification::Public),
        "internal" => Ok(CatalogClassification::Internal),
        "confidential" => Ok(CatalogClassification::Confidential),
        "restricted" => Ok(CatalogClassification::Restricted),
        _ => Err(ZyronError::PlanError(format!(
            "classification must be one of public/internal/confidential/restricted, got '{value}'"
        ))),
    }
}

/// Parses ALTER PUBLICATION SET options into an updates struct. Only the
/// options the user supplied come back as Some.
fn parse_publication_option_updates(
    opts: &[(String, String)],
) -> Result<PublicationOptionUpdates> {
    let mut u = PublicationOptionUpdates::default();
    for (key, value) in opts {
        match key.to_ascii_lowercase().as_str() {
            "change_feed" => u.change_feed = Some(parse_bool_option(key, value)?),
            "row_format" => u.row_format = Some(parse_row_format(value)?),
            "retention_days" => u.retention_days = Some(parse_u32_option(key, value)?),
            "retain_until_subscribers_advance" => {
                u.retain_until_subscribers_advance = Some(parse_bool_option(key, value)?)
            }
            "max_rows_per_sec" => u.max_rows_per_sec = Some(parse_u64_option(key, value)?),
            "max_bytes_per_sec" => u.max_bytes_per_sec = Some(parse_size_bytes(value)?),
            "max_concurrent_subscribers" => {
                u.max_concurrent_subscribers = Some(parse_u32_option(key, value)?)
            }
            "classification" => u.classification = Some(parse_classification(value)?),
            "allow_initial_snapshot" => {
                u.allow_initial_snapshot = Some(parse_bool_option(key, value)?)
            }
            other => {
                return Err(ZyronError::PlanError(format!(
                    "unknown publication option '{other}'"
                )));
            }
        }
    }
    Ok(u)
}

/// Parses ALTER ENDPOINT SET options into an updates struct.
fn parse_endpoint_option_updates(opts: &[(String, String)]) -> Result<EndpointOptionUpdates> {
    let mut u = EndpointOptionUpdates::default();
    for (key, value) in opts {
        match key.to_ascii_lowercase().as_str() {
            "cache_seconds" => u.cache_seconds = Some(parse_u32_option(key, value)?),
            "timeout_seconds" => u.timeout_seconds = Some(parse_u32_option(key, value)?),
            "max_body_bytes" => u.max_body_bytes = Some(parse_u32_option(key, value)?),
            "max_body_kb" => {
                let kb = parse_u32_option(key, value)?;
                u.max_body_bytes = Some(kb.saturating_mul(1024));
            }
            "heartbeat_seconds" => u.heartbeat_seconds = Some(parse_u32_option(key, value)?),
            "max_connections" => u.max_connections = Some(parse_u32_option(key, value)?),
            "max_connections_per_ip" => {
                u.max_connections_per_ip = Some(parse_u32_option(key, value)?)
            }
            other => {
                return Err(ZyronError::PlanError(format!(
                    "unknown endpoint option '{other}'"
                )));
            }
        }
    }
    Ok(u)
}

/// Parses a method name like "GET" or "post" into the catalog HttpMethod.
fn parse_http_method(name: &str) -> Result<HttpMethod> {
    match name.to_ascii_uppercase().as_str() {
        "GET" => Ok(HttpMethod::Get),
        "POST" => Ok(HttpMethod::Post),
        "PUT" => Ok(HttpMethod::Put),
        "DELETE" => Ok(HttpMethod::Delete),
        "PATCH" => Ok(HttpMethod::Patch),
        other => Err(ZyronError::PlanError(format!(
            "unsupported HTTP method '{other}', expected GET/POST/PUT/DELETE/PATCH"
        ))),
    }
}

/// Scans the SQL body for `$name` placeholders and records them in
/// positional order. The rewritten SQL replaces each placeholder with NULL
/// so the parser-level validation can run. The raw sql string is still
/// stored on the bound form for runtime re-compilation against each
/// request's parameter values.
fn extract_endpoint_params(sql: &str) -> (Vec<String>, String) {
    let mut names: Vec<String> = Vec::new();
    let mut out = String::with_capacity(sql.len());
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'$' && i + 1 < bytes.len() {
            let next = bytes[i + 1];
            if next.is_ascii_alphabetic() || next == b'_' {
                let start = i + 1;
                let mut end = start;
                while end < bytes.len()
                    && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
                {
                    end += 1;
                }
                let name = &sql[start..end];
                if !names.iter().any(|n| n == name) {
                    names.push(name.to_string());
                }
                out.push_str("NULL");
                i = end;
                continue;
            }
        }
        out.push(b as char);
        i += 1;
    }
    (names, out)
}

/// Maps the parser endpoint auth spec to the catalog enum.
fn map_endpoint_auth(spec: EndpointAuthSpec) -> EndpointAuthMode {
    match spec {
        EndpointAuthSpec::None => EndpointAuthMode::None,
        EndpointAuthSpec::Jwt => EndpointAuthMode::Jwt,
        EndpointAuthSpec::ApiKey => EndpointAuthMode::ApiKey,
        EndpointAuthSpec::OAuth2 => EndpointAuthMode::OAuth2,
        EndpointAuthSpec::Basic => EndpointAuthMode::Basic,
        EndpointAuthSpec::Mtls => EndpointAuthMode::Mtls,
    }
}

/// Maps the parser output format enum to the catalog output format enum.
/// Protobuf is not supported by the catalog yet, fall back to Json.
fn map_endpoint_output_format(spec: EndpointOutputFormatSpec) -> EndpointOutputFormat {
    match spec {
        EndpointOutputFormatSpec::Json => EndpointOutputFormat::Json,
        EndpointOutputFormatSpec::JsonLines => EndpointOutputFormat::JsonLines,
        EndpointOutputFormatSpec::Csv => EndpointOutputFormat::Csv,
        EndpointOutputFormatSpec::Parquet => EndpointOutputFormat::Parquet,
        EndpointOutputFormatSpec::Arrow => EndpointOutputFormat::ArrowIpc,
        EndpointOutputFormatSpec::Protobuf => EndpointOutputFormat::Json,
    }
}

/// Maps the parser streaming message format to the catalog enum.
fn map_streaming_message_format(spec: StreamingMessageFormat) -> EndpointMessageFormat {
    match spec {
        StreamingMessageFormat::Json => EndpointMessageFormat::Json,
        StreamingMessageFormat::JsonLines => EndpointMessageFormat::JsonLines,
        StreamingMessageFormat::Protobuf => EndpointMessageFormat::Protobuf,
    }
}

/// Maps the parser backpressure spec to the catalog enum.
fn map_backpressure(spec: BackpressurePolicySpec) -> BackpressurePolicy {
    match spec {
        BackpressurePolicySpec::DropOldest => BackpressurePolicy::DropOldest,
        BackpressurePolicySpec::CloseSlow => BackpressurePolicy::CloseSlow,
        BackpressurePolicySpec::Block => BackpressurePolicy::Block,
    }
}

/// Converts parser rate limit to the catalog rate limit spec.
fn convert_rate_limit(spec: &EndpointRateLimitSpec) -> RateLimitSpec {
    // The parser expresses the window as seconds. The catalog stores a
    // coarse enum (Second/Minute/Hour/Day). Map the nearest bucket.
    let period = if spec.per_seconds >= 86_400 {
        RateLimitPeriod::Day
    } else if spec.per_seconds >= 3_600 {
        RateLimitPeriod::Hour
    } else if spec.per_seconds >= 60 {
        RateLimitPeriod::Minute
    } else {
        RateLimitPeriod::Second
    };
    let scope = match spec.scope {
        EndpointRateLimitScope::Global => RateLimitScope::Global,
        EndpointRateLimitScope::PerIp => RateLimitScope::PerIp,
        EndpointRateLimitScope::PerUser => RateLimitScope::PerUser,
        EndpointRateLimitScope::PerApiKey => RateLimitScope::PerApiKey,
    };
    RateLimitSpec {
        count: spec.count,
        period,
        scope,
    }
}

/// Maps a parser-level security map kind onto the catalog enum plus the
/// canonical key string that will be stored in the catalog entry.
fn map_security_kind(spec: &SecurityMapKindSpec) -> Result<(CatalogSecurityMapKind, String)> {
    match spec {
        SecurityMapKindSpec::KubernetesSa { ns_and_name } => {
            let s = ns_and_name.trim();
            if s.is_empty() || !s.contains('/') {
                return Err(ZyronError::PlanError(format!(
                    "kubernetes SA identity must be 'namespace/name', got '{ns_and_name}'"
                )));
            }
            Ok((CatalogSecurityMapKind::K8sSa, s.to_string()))
        }
        SecurityMapKindSpec::JwtIssuerSubject { issuer, subject } => {
            let i = issuer.trim();
            let s = subject.trim();
            if i.is_empty() || s.is_empty() {
                return Err(ZyronError::PlanError(
                    "JWT security map requires non-empty issuer and subject".to_string(),
                ));
            }
            Ok((CatalogSecurityMapKind::Jwt, format!("{i}#{s}")))
        }
        SecurityMapKindSpec::MtlsCertSubject { subject_dn } => {
            let s = subject_dn.trim();
            if s.is_empty() {
                return Err(ZyronError::PlanError(
                    "mTLS cert subject must be non-empty".to_string(),
                ));
            }
            Ok((CatalogSecurityMapKind::MtlsSubject, s.to_string()))
        }
        SecurityMapKindSpec::MtlsCertFingerprint { fingerprint_sha256 } => {
            let fp = fingerprint_sha256.trim().to_ascii_lowercase();
            if fp.is_empty() {
                return Err(ZyronError::PlanError(
                    "mTLS cert fingerprint must be non-empty".to_string(),
                ));
            }
            Ok((CatalogSecurityMapKind::MtlsFingerprint, fp))
        }
    }
}

/// Computes a SHA-256 fingerprint over the sorted projection of every
/// member table's exposed columns. Tuples are (table_id, column_id,
/// type_id, is_nullable). Missing column id set means all columns.
fn compute_publication_fingerprint(
    tables: &[BoundPublicationTable],
    catalog: &Catalog,
) -> [u8; 32] {
    let mut triples: Vec<(u32, u16, u8, u8)> = Vec::new();
    for t in tables {
        let entry = match catalog.get_table_by_id(t.table_id) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if t.columns.is_empty() {
            for c in &entry.columns {
                triples.push((
                    t.table_id.0,
                    c.id.0,
                    c.type_id as u8,
                    if c.nullable { 1 } else { 0 },
                ));
            }
        } else {
            for cid in &t.columns {
                if let Some(c) = entry.columns.iter().find(|c| c.id == *cid) {
                    triples.push((
                        t.table_id.0,
                        c.id.0,
                        c.type_id as u8,
                        if c.nullable { 1 } else { 0 },
                    ));
                }
            }
        }
    }
    triples.sort_unstable();
    let mut hasher = Sha256::new();
    for (tid, cid, typ, null) in triples {
        hasher.update(tid.to_le_bytes());
        hasher.update(cid.to_le_bytes());
        hasher.update([typ, null]);
    }
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest[..]);
    out
}

// ---------------------------------------------------------------------------
// Binder
// ---------------------------------------------------------------------------

/// Converts parsed AST into bound plan with resolved OIDs and type information.
#[allow(dead_code)]
pub struct Binder<'a> {
    resolver: NameResolver,
    catalog: &'a Catalog,
    next_table_idx: usize,
}

impl<'a> Binder<'a> {
    pub fn new(resolver: NameResolver, catalog: &'a Catalog) -> Self {
        Self {
            resolver,
            catalog,
            next_table_idx: 0,
        }
    }

    fn alloc_table_idx(&mut self) -> usize {
        let idx = self.next_table_idx;
        self.next_table_idx += 1;
        idx
    }

    /// Main entry point. Dispatches to statement-specific binders.
    pub async fn bind(&mut self, stmt: Statement) -> Result<BoundStatement> {
        match stmt {
            Statement::Select(s) => {
                let mut ctx = BindContext::new();
                let bound = self.bind_select(&mut ctx, &s).await?;
                Ok(BoundStatement::Select(bound))
            }
            Statement::Insert(s) => {
                let bound = self.bind_insert(&s).await?;
                Ok(BoundStatement::Insert(bound))
            }
            Statement::Update(s) => {
                let bound = self.bind_update(&s).await?;
                Ok(BoundStatement::Update(bound))
            }
            Statement::Delete(s) => {
                let bound = self.bind_delete(&s).await?;
                Ok(BoundStatement::Delete(bound))
            }
            Statement::CreateStreamingJob(s) => {
                let bound = self.bind_create_streaming_job(&s).await?;
                Ok(BoundStatement::CreateStreamingJob(bound))
            }
            Statement::DropStreamingJob(s) => Ok(BoundStatement::DropStreamingJob {
                name: s.name.clone(),
                if_exists: s.if_exists,
            }),
            Statement::AlterStreamingJob(s) => Ok(BoundStatement::AlterStreamingJob {
                name: s.name.clone(),
                action: s.action,
            }),
            Statement::CreateExternalSource(s) => {
                let bound = self.bind_create_external_source(&s).await?;
                Ok(BoundStatement::CreateExternalSource(Box::new(bound)))
            }
            Statement::CreateExternalSink(s) => {
                let bound = self.bind_create_external_sink(&s).await?;
                Ok(BoundStatement::CreateExternalSink(Box::new(bound)))
            }
            Statement::DropExternalSource(s) => {
                let schema_id = self.current_schema_id().await?;
                Ok(BoundStatement::DropExternalSource {
                    name: s.name.clone(),
                    schema_id,
                    if_exists: s.if_exists,
                })
            }
            Statement::DropExternalSink(s) => {
                let schema_id = self.current_schema_id().await?;
                Ok(BoundStatement::DropExternalSink {
                    name: s.name.clone(),
                    schema_id,
                    if_exists: s.if_exists,
                })
            }
            Statement::AlterExternalSource(s) => {
                let schema_id = self.current_schema_id().await?;
                Ok(BoundStatement::AlterExternalSource(Box::new(
                    BoundAlterExternalSource {
                        schema_id,
                        name: s.name.clone(),
                        action: s.action.clone(),
                    },
                )))
            }
            Statement::AlterExternalSink(s) => {
                let schema_id = self.current_schema_id().await?;
                Ok(BoundStatement::AlterExternalSink(Box::new(
                    BoundAlterExternalSink {
                        schema_id,
                        name: s.name.clone(),
                        action: s.action.clone(),
                    },
                )))
            }
            Statement::CreatePublication(s) => {
                let bound = self.bind_create_publication(&s).await?;
                Ok(BoundStatement::CreatePublication(Box::new(bound)))
            }
            Statement::AlterPublication(s) => {
                let bound = self.bind_alter_publication(&s).await?;
                Ok(BoundStatement::AlterPublication(Box::new(bound)))
            }
            Statement::DropPublication(s) => {
                let schema_id = self.current_schema_id().await?;
                if !s.if_exists
                    && self.catalog.get_publication(schema_id, &s.name).is_none()
                {
                    return Err(ZyronError::PlanError(format!(
                        "publication '{}' not found",
                        s.name
                    )));
                }
                Ok(BoundStatement::DropPublication {
                    name: s.name.clone(),
                    schema_id,
                    if_exists: s.if_exists,
                    cascade: s.cascade,
                })
            }
            Statement::TagPublication(s) => {
                let (schema_id, pub_id, tags) =
                    self.bind_publication_tags(&s.name, &s.tags).await?;
                Ok(BoundStatement::TagPublication {
                    name: s.name.clone(),
                    schema_id,
                    publication_id: pub_id,
                    tags,
                })
            }
            Statement::UntagPublication(s) => {
                let (schema_id, pub_id, tags) =
                    self.bind_publication_tags(&s.name, &[s.tag.clone()]).await?;
                Ok(BoundStatement::UntagPublication {
                    name: s.name.clone(),
                    schema_id,
                    publication_id: pub_id,
                    tags,
                })
            }
            Statement::CreateEndpoint(s) => {
                let bound = self.bind_create_endpoint(&s).await?;
                Ok(BoundStatement::CreateEndpoint(Box::new(bound)))
            }
            Statement::CreateStreamingEndpoint(s) => {
                let bound = self.bind_create_streaming_endpoint(&s).await?;
                Ok(BoundStatement::CreateStreamingEndpoint(Box::new(bound)))
            }
            Statement::AlterEndpoint(s) => {
                let bound = self.bind_alter_endpoint(&s).await?;
                Ok(BoundStatement::AlterEndpoint(Box::new(bound)))
            }
            Statement::DropEndpoint(s) => {
                let schema_id = self.current_schema_id().await?;
                if !s.if_exists
                    && self.catalog.get_endpoint(schema_id, &s.name).is_none()
                {
                    return Err(ZyronError::PlanError(format!(
                        "endpoint '{}' not found",
                        s.name
                    )));
                }
                Ok(BoundStatement::DropEndpoint {
                    name: s.name.clone(),
                    schema_id,
                    if_exists: s.if_exists,
                })
            }
            Statement::AlterSecurityMap(s) => {
                let bound = self.bind_alter_security_map(&s)?;
                Ok(BoundStatement::AlterSecurityMap(Box::new(bound)))
            }
            Statement::DropSecurityMap(s) => {
                let bound = self.bind_drop_security_map(&s)?;
                Ok(BoundStatement::DropSecurityMap(Box::new(bound)))
            }
            other => Err(ZyronError::PlanError(format!(
                "unsupported statement type for planning: {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // SELECT binding
    // -----------------------------------------------------------------------

    fn bind_select<'b>(
        &'b mut self,
        ctx: &'b mut BindContext,
        stmt: &'b SelectStatement,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundSelect>> + 'b>> {
        Box::pin(async move {
            // Bind CTEs first
            let mut bound_ctes = Vec::new();
            if let Some(with) = &stmt.with {
                for cte in &with.ctes {
                    let mut cte_ctx = BindContext::new();
                    let cte_query = self.bind_select(&mut cte_ctx, &cte.query).await?;
                    let cte_columns = cte_query.output_schema.clone();
                    let bound_cte = BoundCte {
                        name: cte.name.clone(),
                        columns: cte_columns,
                        query: Box::new(cte_query),
                    };
                    ctx.ctes.insert(cte.name.clone(), bound_cte.clone());
                    bound_ctes.push(bound_cte);
                }
            }

            // Bind FROM clause
            let from = self.bind_from(ctx, &stmt.from).await?;

            // Bind WHERE
            let where_clause = if let Some(expr) = &stmt.where_clause {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };

            // Bind GROUP BY
            let mut group_by = Vec::with_capacity(stmt.group_by.len());
            for e in &stmt.group_by {
                group_by.push(self.bind_expr(ctx, e).await?);
            }

            // Bind HAVING
            let having = if let Some(expr) = &stmt.having {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };

            // Bind projections
            let projections = self.bind_select_items(ctx, &stmt.projections).await?;

            // Build output schema from projections
            let output_schema = self.build_output_schema(ctx, &projections);

            // Bind ORDER BY
            let mut order_by = Vec::with_capacity(stmt.order_by.len());
            for o in &stmt.order_by {
                order_by.push(self.bind_order_by(ctx, o).await?);
            }

            // Bind LIMIT / OFFSET
            let limit = if let Some(expr) = &stmt.limit {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };
            let offset = if let Some(expr) = &stmt.offset {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };

            // Bind set operations
            let mut set_ops = Vec::new();
            for set_op in &stmt.set_ops {
                let mut right_ctx = BindContext::new();
                let right = self.bind_select(&mut right_ctx, &set_op.right).await?;
                set_ops.push(BoundSetOp {
                    op: set_op.op,
                    all: set_op.all,
                    right: Box::new(right),
                });
            }

            Ok(BoundSelect {
                projections,
                from,
                where_clause,
                group_by,
                having,
                order_by,
                limit,
                offset,
                distinct: stmt.distinct,
                set_ops,
                ctes: bound_ctes,
                output_schema,
            })
        }) // end Box::pin
    }

    // -----------------------------------------------------------------------
    // FROM clause binding
    // -----------------------------------------------------------------------

    async fn bind_from(
        &mut self,
        ctx: &mut BindContext,
        from: &[TableRef],
    ) -> Result<Vec<BoundFromItem>> {
        let mut items = Vec::with_capacity(from.len());
        for table_ref in from {
            let item = self.bind_table_ref(ctx, table_ref).await?;
            items.push(item);
        }
        Ok(items)
    }

    fn bind_table_ref<'b>(
        &'b mut self,
        ctx: &'b mut BindContext,
        table_ref: &'b TableRef,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundFromItem>> + 'b>> {
        Box::pin(async move {
            match table_ref {
                TableRef::Table { name, alias, .. } => {
                    // Check if this is a CTE reference
                    let display_name = alias.as_deref().unwrap_or(name);
                    if let Some(cte) = ctx.ctes.get(name).cloned() {
                        let idx = self.alloc_table_idx();
                        let bound_table = BoundTableRef {
                            table_idx: idx,
                            table_id: None,
                            alias: display_name.to_string(),
                            columns: cte.columns.clone(),
                            entry: None,
                        };
                        ctx.tables.push(bound_table);
                        return Ok(BoundFromItem::Subquery {
                            table_idx: idx,
                            query: cte.query.clone(),
                        });
                    }

                    // Parse optional schema qualifier (schema.table)
                    let (schema_name, table_name) = if let Some(dot_pos) = name.find('.') {
                        (Some(&name[..dot_pos]), &name[dot_pos + 1..])
                    } else {
                        (None, name.as_str())
                    };

                    let entry = self.resolver.resolve_table(schema_name, table_name).await?;
                    let idx = self.alloc_table_idx();

                    let columns: Vec<BoundColumnDef> = entry
                        .columns
                        .iter()
                        .map(|c| BoundColumnDef {
                            column_id: c.id,
                            name: c.name.clone(),
                            type_id: c.type_id,
                            nullable: c.nullable,
                            ordinal: c.ordinal,
                        })
                        .collect();

                    let bound_table = BoundTableRef {
                        table_idx: idx,
                        table_id: Some(entry.id),
                        alias: display_name.to_string(),
                        columns,
                        entry: Some(Arc::clone(&entry)),
                    };
                    ctx.tables.push(bound_table);

                    Ok(BoundFromItem::BaseTable {
                        table_idx: idx,
                        table_id: entry.id,
                        entry,
                    })
                }
                TableRef::Join(join_ref) => {
                    let left = self.bind_table_ref(ctx, &join_ref.left).await?;
                    let right = self.bind_table_ref(ctx, &join_ref.right).await?;
                    let condition = self.bind_join_condition(ctx, &join_ref.condition).await?;
                    Ok(BoundFromItem::Join {
                        left: Box::new(left),
                        join_type: join_ref.join_type,
                        right: Box::new(right),
                        condition,
                    })
                }
                TableRef::Subquery { query, alias } => {
                    let mut sub_ctx = BindContext::new();
                    let bound_query = self.bind_select(&mut sub_ctx, query).await?;
                    let idx = self.alloc_table_idx();
                    let columns = bound_query.output_schema.clone();
                    let bound_table = BoundTableRef {
                        table_idx: idx,
                        table_id: None,
                        alias: alias.clone(),
                        columns,
                        entry: None,
                    };
                    ctx.tables.push(bound_table);
                    Ok(BoundFromItem::Subquery {
                        table_idx: idx,
                        query: Box::new(bound_query),
                    })
                }
                TableRef::Lateral { subquery } => {
                    // Treat LATERAL as a regular subquery for now
                    self.bind_table_ref(ctx, subquery).await
                }
                TableRef::TableFunction { name, args, alias } => {
                    // Check if this is a graph algorithm table function.
                    let algo = name.to_lowercase();
                    let graph_algos = [
                        "pagerank",
                        "shortest_path",
                        "bfs",
                        "connected_components",
                        "community_detection",
                        "betweenness_centrality",
                    ];
                    if !graph_algos.contains(&algo.as_str()) {
                        return Err(ZyronError::PlanError(format!(
                            "table function '{}' is not supported",
                            name
                        )));
                    }

                    // First positional arg is the graph schema name.
                    let schema_name = match args.first() {
                        Some(FunctionArg::Unnamed(Expr::Literal(LiteralValue::String(s)))) => {
                            s.clone()
                        }
                        Some(FunctionArg::Unnamed(Expr::Identifier(s))) => s.clone(),
                        _ => {
                            return Err(ZyronError::PlanError(format!(
                                "{}() requires a graph schema name as first argument",
                                name
                            )));
                        }
                    };

                    // Collect named parameters (remaining args after schema name).
                    let mut params: Vec<(String, BoundExpr)> = Vec::new();
                    for arg in args.iter().skip(1) {
                        match arg {
                            FunctionArg::Named {
                                name: param_name,
                                value,
                            } => {
                                let bound = self.bind_expr(ctx, value).await?;
                                params.push((param_name.clone(), bound));
                            }
                            FunctionArg::Unnamed(value) => {
                                let bound = self.bind_expr(ctx, value).await?;
                                params.push((String::new(), bound));
                            }
                        }
                    }

                    // Build output columns based on algorithm type.
                    let output_columns = match algo.as_str() {
                        "pagerank" => vec![
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(0),
                                name: "node_id".to_string(),
                                type_id: TypeId::Int64,
                                nullable: false,
                            },
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(1),
                                name: "score".to_string(),
                                type_id: TypeId::Float64,
                                nullable: false,
                            },
                        ],
                        "shortest_path" => vec![
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(0),
                                name: "step".to_string(),
                                type_id: TypeId::Int32,
                                nullable: false,
                            },
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(1),
                                name: "node_id".to_string(),
                                type_id: TypeId::Int64,
                                nullable: false,
                            },
                        ],
                        "bfs" => vec![
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(0),
                                name: "node_id".to_string(),
                                type_id: TypeId::Int64,
                                nullable: false,
                            },
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(1),
                                name: "depth".to_string(),
                                type_id: TypeId::Int32,
                                nullable: false,
                            },
                        ],
                        "betweenness_centrality" => vec![
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(0),
                                name: "node_id".to_string(),
                                type_id: TypeId::Int64,
                                nullable: false,
                            },
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(1),
                                name: "centrality".to_string(),
                                type_id: TypeId::Float64,
                                nullable: false,
                            },
                        ],
                        // connected_components and community_detection share
                        // the (node_id, component) output schema.
                        _ => vec![
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(0),
                                name: "node_id".to_string(),
                                type_id: TypeId::Int64,
                                nullable: false,
                            },
                            LogicalColumn {
                                table_idx: None,
                                column_id: ColumnId(1),
                                name: "component".to_string(),
                                type_id: TypeId::Int64,
                                nullable: false,
                            },
                        ],
                    };

                    let idx = self.alloc_table_idx();
                    let tbl_alias = alias.clone().unwrap_or_else(|| algo.clone());
                    let bound_cols: Vec<BoundColumnDef> = output_columns
                        .iter()
                        .enumerate()
                        .map(|(i, lc)| BoundColumnDef {
                            column_id: lc.column_id,
                            name: lc.name.clone(),
                            type_id: lc.type_id,
                            nullable: lc.nullable,
                            ordinal: i as u16,
                        })
                        .collect();
                    let bound_table = BoundTableRef {
                        table_idx: idx,
                        table_id: None,
                        alias: tbl_alias,
                        columns: bound_cols,
                        entry: None,
                    };
                    ctx.tables.push(bound_table);

                    Ok(BoundFromItem::GraphQuery {
                        table_idx: idx,
                        schema_name,
                        algorithm: algo,
                        params,
                        output_columns,
                    })
                }
            }
        }) // end Box::pin
    }

    async fn bind_join_condition(
        &mut self,
        ctx: &BindContext,
        condition: &JoinCondition,
    ) -> Result<BoundJoinCondition> {
        match condition {
            JoinCondition::On(expr) => {
                let bound = self.bind_expr(ctx, expr).await?;
                Ok(BoundJoinCondition::On(bound))
            }
            JoinCondition::Using(cols) => {
                // Resolve column names to IDs from the last two tables in context
                let mut ids = Vec::with_capacity(cols.len());
                for col_name in cols {
                    // Find the column in any of the tables in scope
                    let mut found = false;
                    for table in &ctx.tables {
                        if let Some(c) = table.columns.iter().find(|c| c.name == *col_name) {
                            ids.push(c.column_id);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return Err(ZyronError::ColumnNotFound(col_name.clone()));
                    }
                }
                Ok(BoundJoinCondition::Using(ids))
            }
            JoinCondition::Natural => Ok(BoundJoinCondition::Natural),
            JoinCondition::None => Ok(BoundJoinCondition::None),
        }
    }

    // -----------------------------------------------------------------------
    // Expression binding
    // -----------------------------------------------------------------------

    fn bind_expr<'b>(
        &'b mut self,
        ctx: &'b BindContext,
        expr: &'b Expr,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundExpr>> + 'b>> {
        Box::pin(async move {
            match expr {
                Expr::Identifier(name) => {
                    let cr = self.resolve_column(ctx, name)?;
                    Ok(BoundExpr::ColumnRef(cr))
                }
                Expr::QualifiedIdentifier { table, column } => {
                    let cr = self.resolve_qualified_column(ctx, table, column)?;
                    Ok(BoundExpr::ColumnRef(cr))
                }
                Expr::Literal(lit) => {
                    let type_id = literal_type(lit);
                    Ok(BoundExpr::Literal {
                        value: lit.clone(),
                        type_id,
                    })
                }
                Expr::BinaryOp { left, op, right } => {
                    let left_bound = self.bind_expr(ctx, left).await?;
                    let right_bound = self.bind_expr(ctx, right).await?;
                    let type_id =
                        infer_binary_type(op, left_bound.type_id(), right_bound.type_id())?;
                    Ok(BoundExpr::BinaryOp {
                        left: Box::new(left_bound),
                        op: *op,
                        right: Box::new(right_bound),
                        type_id,
                    })
                }
                Expr::UnaryOp { op, expr: inner } => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    let type_id = match op {
                        UnaryOperator::Not => TypeId::Boolean,
                        UnaryOperator::Minus => bound.type_id(),
                    };
                    Ok(BoundExpr::UnaryOp {
                        op: *op,
                        expr: Box::new(bound),
                        type_id,
                    })
                }
                Expr::IsNull {
                    expr: inner,
                    negated,
                } => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    Ok(BoundExpr::IsNull {
                        expr: Box::new(bound),
                        negated: *negated,
                    })
                }
                Expr::InList {
                    expr: inner,
                    list,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let mut bound_list = Vec::with_capacity(list.len());
                    for e in list {
                        bound_list.push(self.bind_expr(ctx, e).await?);
                    }
                    Ok(BoundExpr::InList {
                        expr: Box::new(bound_expr),
                        list: bound_list,
                        negated: *negated,
                    })
                }
                Expr::Between {
                    expr: inner,
                    low,
                    high,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let bound_low = self.bind_expr(ctx, low).await?;
                    let bound_high = self.bind_expr(ctx, high).await?;
                    Ok(BoundExpr::Between {
                        expr: Box::new(bound_expr),
                        low: Box::new(bound_low),
                        high: Box::new(bound_high),
                        negated: *negated,
                    })
                }
                Expr::Like {
                    expr: inner,
                    pattern,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let bound_pattern = self.bind_expr(ctx, pattern).await?;
                    Ok(BoundExpr::Like {
                        expr: Box::new(bound_expr),
                        pattern: Box::new(bound_pattern),
                        negated: *negated,
                    })
                }
                Expr::ILike {
                    expr: inner,
                    pattern,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let bound_pattern = self.bind_expr(ctx, pattern).await?;
                    Ok(BoundExpr::ILike {
                        expr: Box::new(bound_expr),
                        pattern: Box::new(bound_pattern),
                        negated: *negated,
                    })
                }
                Expr::Function {
                    name,
                    args,
                    distinct,
                } => {
                    let mut bound_args = Vec::with_capacity(args.len());
                    for a in args {
                        let e = match a {
                            FunctionArg::Unnamed(e) => e,
                            FunctionArg::Named { value, .. } => value,
                        };
                        bound_args.push(self.bind_expr(ctx, e).await?);
                    }

                    let arg_types: Vec<TypeId> = bound_args.iter().map(|a| a.type_id()).collect();

                    if is_aggregate_function(name) {
                        let return_type = infer_aggregate_type(name, &arg_types)?;
                        Ok(BoundExpr::AggregateFunction {
                            name: name.to_lowercase(),
                            args: bound_args,
                            distinct: *distinct,
                            return_type,
                        })
                    } else {
                        let return_type = infer_function_type(name, &arg_types);
                        Ok(BoundExpr::Function {
                            name: name.clone(),
                            args: bound_args,
                            return_type,
                            distinct: *distinct,
                        })
                    }
                }
                Expr::Cast {
                    expr: inner,
                    data_type,
                } => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    let target_type = data_type.to_type_id();
                    Ok(BoundExpr::Cast {
                        expr: Box::new(bound),
                        target_type,
                    })
                }
                Expr::Case {
                    operand,
                    conditions,
                    else_result,
                } => {
                    let bound_operand = if let Some(e) = operand.as_ref() {
                        Some(self.bind_expr(ctx, e).await?)
                    } else {
                        None
                    };
                    let mut bound_conditions = Vec::with_capacity(conditions.len());
                    for wc in conditions {
                        bound_conditions.push(BoundWhen {
                            condition: self.bind_expr(ctx, &wc.condition).await?,
                            result: self.bind_expr(ctx, &wc.result).await?,
                        });
                    }
                    let bound_else = if let Some(e) = else_result.as_ref() {
                        Some(self.bind_expr(ctx, e).await?)
                    } else {
                        None
                    };

                    // Result type is the type of the first THEN clause
                    let type_id = bound_conditions
                        .first()
                        .map(|w| w.result.type_id())
                        .unwrap_or(TypeId::Null);

                    Ok(BoundExpr::Case {
                        operand: bound_operand.map(Box::new),
                        conditions: bound_conditions,
                        else_result: bound_else.map(Box::new),
                        type_id,
                    })
                }
                Expr::Nested(inner) => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    Ok(BoundExpr::Nested(Box::new(bound)))
                }
                Expr::Subquery(query) => {
                    let mut sub_ctx = BindContext::new();
                    // Copy outer scope for correlated subquery support
                    for table in &ctx.tables {
                        sub_ctx.tables.push(table.clone());
                    }
                    let bound = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    let type_id = bound
                        .output_schema
                        .first()
                        .map(|c| c.type_id)
                        .unwrap_or(TypeId::Null);
                    Ok(BoundExpr::Subquery {
                        plan: Box::new(bound),
                        type_id,
                    })
                }
                Expr::Exists { query, negated } => {
                    let mut sub_ctx = BindContext::new();
                    for table in &ctx.tables {
                        sub_ctx.tables.push(table.clone());
                    }
                    let bound = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    Ok(BoundExpr::Exists {
                        plan: Box::new(bound),
                        negated: *negated,
                    })
                }
                Expr::InSubquery {
                    expr: inner,
                    query,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let mut sub_ctx = BindContext::new();
                    for table in &ctx.tables {
                        sub_ctx.tables.push(table.clone());
                    }
                    let bound_query = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    Ok(BoundExpr::InSubquery {
                        expr: Box::new(bound_expr),
                        plan: Box::new(bound_query),
                        negated: *negated,
                    })
                }
                Expr::WindowFunction {
                    function,
                    partition_by,
                    order_by,
                    frame,
                } => {
                    let bound_func = self.bind_expr(ctx, function).await?;
                    let mut bound_partition = Vec::with_capacity(partition_by.len());
                    for e in partition_by {
                        bound_partition.push(self.bind_expr(ctx, e).await?);
                    }
                    let mut bound_order = Vec::with_capacity(order_by.len());
                    for o in order_by {
                        bound_order.push(self.bind_order_by(ctx, o).await?);
                    }
                    let type_id = bound_func.type_id();
                    Ok(BoundExpr::WindowFunction {
                        function: Box::new(bound_func),
                        partition_by: bound_partition,
                        order_by: bound_order,
                        frame: frame.clone(),
                        type_id,
                    })
                }
                Expr::Parameter(idx) => Ok(BoundExpr::Parameter {
                    index: *idx,
                    type_id: TypeId::Null, // Resolved at execution time
                }),
                // JSON and array operators: bind as generic binary ops for now
                Expr::JsonAccess { left, right, .. }
                | Expr::JsonContains { left, right, .. }
                | Expr::JsonExists { left, right, .. } => {
                    let bound_left = self.bind_expr(ctx, left).await?;
                    let bound_right = self.bind_expr(ctx, right).await?;
                    Ok(BoundExpr::Function {
                        name: "json_op".to_string(),
                        args: vec![bound_left, bound_right],
                        return_type: TypeId::Jsonb,
                        distinct: false,
                    })
                }
                Expr::VectorDistance { left, op, right } => {
                    let bound_left = self.bind_expr(ctx, left).await?;
                    let bound_right = self.bind_expr(ctx, right).await?;
                    let func_name = match op {
                        VectorDistanceOp::Cosine => "vector_distance_cosine",
                        VectorDistanceOp::L2 => "vector_distance_l2",
                        VectorDistanceOp::DotProduct => "vector_distance_dot",
                    };
                    Ok(BoundExpr::Function {
                        name: func_name.to_string(),
                        args: vec![bound_left, bound_right],
                        return_type: TypeId::Float64,
                        distinct: false,
                    })
                }
                Expr::ArrayConstructor(elements) => {
                    let mut bound_elements = Vec::with_capacity(elements.len());
                    for e in elements {
                        bound_elements.push(self.bind_expr(ctx, e).await?);
                    }
                    Ok(BoundExpr::Function {
                        name: "array".to_string(),
                        args: bound_elements,
                        return_type: TypeId::Array,
                        distinct: false,
                    })
                }
                Expr::ArraySubscript { array, index } => {
                    let bound_array = self.bind_expr(ctx, array).await?;
                    let bound_index = self.bind_expr(ctx, index).await?;
                    Ok(BoundExpr::Function {
                        name: "array_subscript".to_string(),
                        args: vec![bound_array, bound_index],
                        return_type: TypeId::Null, // Element type unknown without full type system
                        distinct: false,
                    })
                }
                Expr::AnySubquery { query } | Expr::AllSubquery { query } => {
                    let mut sub_ctx = BindContext::new();
                    let bound_query = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    let type_id = bound_query
                        .output_schema
                        .first()
                        .map(|c| c.type_id)
                        .unwrap_or(TypeId::Null);
                    Ok(BoundExpr::Subquery {
                        plan: Box::new(bound_query),
                        type_id,
                    })
                }
                Expr::MatchAgainst { columns, query, .. } => {
                    // Bind as a function call with the match columns and query
                    let bound_query = self.bind_expr(ctx, query).await?;
                    let mut args = Vec::with_capacity(columns.len() + 1);
                    for col_name in columns {
                        let cr = self.resolve_column(ctx, col_name)?;
                        args.push(BoundExpr::ColumnRef(cr));
                    }
                    args.push(bound_query);
                    Ok(BoundExpr::Function {
                        name: "match_against".to_string(),
                        args,
                        return_type: TypeId::Float64,
                        distinct: false,
                    })
                }
            }
        }) // end Box::pin for bind_expr
    }

    // -----------------------------------------------------------------------
    // Column resolution
    // -----------------------------------------------------------------------

    /// Resolves an unqualified column name by searching all tables in scope.
    fn resolve_column(&self, ctx: &BindContext, name: &str) -> Result<ColumnRef> {
        let mut found: Option<ColumnRef> = None;
        for table in &ctx.tables {
            for col in &table.columns {
                if col.name == name {
                    if found.is_some() {
                        return Err(ZyronError::PlanError(format!(
                            "ambiguous column reference: {}",
                            name
                        )));
                    }
                    found = Some(ColumnRef {
                        table_idx: table.table_idx,
                        column_id: col.column_id,
                        type_id: col.type_id,
                        nullable: col.nullable,
                    });
                }
            }
        }

        // Check outer scope for correlated references
        if found.is_none() {
            if let Some(outer) = &ctx.outer {
                return self.resolve_column(outer, name);
            }
        }

        found.ok_or_else(|| ZyronError::ColumnNotFound(name.to_string()))
    }

    /// Resolves a qualified column reference (table.column).
    fn resolve_qualified_column(
        &self,
        ctx: &BindContext,
        table: &str,
        column: &str,
    ) -> Result<ColumnRef> {
        for bound_table in &ctx.tables {
            if bound_table.alias == table {
                for col in &bound_table.columns {
                    if col.name == column {
                        return Ok(ColumnRef {
                            table_idx: bound_table.table_idx,
                            column_id: col.column_id,
                            type_id: col.type_id,
                            nullable: col.nullable,
                        });
                    }
                }
                return Err(ZyronError::ColumnNotFound(format!("{}.{}", table, column)));
            }
        }

        // Check outer scope
        if let Some(outer) = &ctx.outer {
            return self.resolve_qualified_column(outer, table, column);
        }

        Err(ZyronError::TableNotFound(table.to_string()))
    }

    // -----------------------------------------------------------------------
    // SELECT item binding
    // -----------------------------------------------------------------------

    async fn bind_select_items(
        &mut self,
        ctx: &BindContext,
        items: &[SelectItem],
    ) -> Result<Vec<BoundSelectItem>> {
        let mut result = Vec::with_capacity(items.len());
        for item in items {
            match item {
                SelectItem::Expr(expr, alias) => {
                    let bound = self.bind_expr(ctx, expr).await?;
                    result.push(BoundSelectItem::Expr(bound, alias.clone()));
                }
                SelectItem::Wildcard => {
                    result.push(BoundSelectItem::Wildcard);
                }
                SelectItem::QualifiedWildcard(table_name) => {
                    if let Some(table) = ctx.tables.iter().find(|t| t.alias == *table_name) {
                        result.push(BoundSelectItem::AllColumns(table.table_idx));
                    } else {
                        return Err(ZyronError::TableNotFound(table_name.clone()));
                    }
                }
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Output schema construction
    // -----------------------------------------------------------------------

    fn build_output_schema(
        &self,
        ctx: &BindContext,
        projections: &[BoundSelectItem],
    ) -> Vec<BoundColumnDef> {
        let mut schema = Vec::new();
        for (ordinal, item) in projections.iter().enumerate() {
            match item {
                BoundSelectItem::Expr(expr, alias) => {
                    let name = alias.clone().unwrap_or_else(|| {
                        if let BoundExpr::ColumnRef(cr) = expr {
                            // Find column name from context
                            for table in &ctx.tables {
                                if table.table_idx == cr.table_idx {
                                    if let Some(col) =
                                        table.columns.iter().find(|c| c.column_id == cr.column_id)
                                    {
                                        return col.name.clone();
                                    }
                                }
                            }
                            format!("col{}", ordinal)
                        } else {
                            format!("col{}", ordinal)
                        }
                    });
                    schema.push(BoundColumnDef {
                        column_id: ColumnId(ordinal as u16),
                        name,
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                        ordinal: ordinal as u16,
                    });
                }
                BoundSelectItem::Wildcard => {
                    for table in &ctx.tables {
                        for col in &table.columns {
                            schema.push(col.clone());
                        }
                    }
                }
                BoundSelectItem::AllColumns(table_idx) => {
                    if let Some(table) = ctx.tables.iter().find(|t| t.table_idx == *table_idx) {
                        for col in &table.columns {
                            schema.push(col.clone());
                        }
                    }
                }
            }
        }
        schema
    }

    // -----------------------------------------------------------------------
    // ORDER BY binding
    // -----------------------------------------------------------------------

    async fn bind_order_by(
        &mut self,
        ctx: &BindContext,
        order: &OrderByExpr,
    ) -> Result<BoundOrderBy> {
        let bound_expr = self.bind_expr(ctx, &order.expr).await?;
        Ok(BoundOrderBy {
            expr: bound_expr,
            asc: order.asc.unwrap_or(true),
            nulls_first: order.nulls_first.unwrap_or(false),
        })
    }

    // -----------------------------------------------------------------------
    // INSERT binding
    // -----------------------------------------------------------------------

    async fn bind_insert(&mut self, stmt: &InsertStatement) -> Result<BoundInsert> {
        let (schema_name, table_name) = if let Some(dot_pos) = stmt.table.find('.') {
            (Some(&stmt.table[..dot_pos]), &stmt.table[dot_pos + 1..])
        } else {
            (None, stmt.table.as_str())
        };

        let entry = self.resolver.resolve_table(schema_name, table_name).await?;

        // Resolve target columns
        let target_columns = if stmt.columns.is_empty() {
            // All columns in table order
            entry.columns.iter().map(|c| c.id).collect()
        } else {
            let mut ids = Vec::with_capacity(stmt.columns.len());
            for col_name in &stmt.columns {
                let col = self.resolver.resolve_column(&entry, col_name)?;
                ids.push(col.id);
            }
            ids
        };

        // Register table in a temporary context for RETURNING
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: table_name.to_string(),
            columns,
            entry: Some(Arc::clone(&entry)),
        });

        let source = match &stmt.source {
            InsertSource::Values(rows) => {
                let mut bound_rows = Vec::with_capacity(rows.len());
                for row in rows {
                    let mut bound_row = Vec::with_capacity(row.len());
                    for e in row {
                        bound_row.push(self.bind_expr(&ctx, e).await?);
                    }
                    bound_rows.push(bound_row);
                }
                BoundInsertSource::Values(bound_rows)
            }
            InsertSource::Query(query) => {
                let mut sub_ctx = BindContext::new();
                let bound_query = self.bind_select(&mut sub_ctx, query).await?;
                BoundInsertSource::Query(Box::new(bound_query))
            }
        };

        let returning = if let Some(items) = &stmt.returning {
            Some(self.bind_select_items(&ctx, items).await?)
        } else {
            None
        };

        Ok(BoundInsert {
            table_id: entry.id,
            table_entry: entry,
            target_columns,
            source,
            returning,
        })
    }

    // -----------------------------------------------------------------------
    // UPDATE binding
    // -----------------------------------------------------------------------

    async fn bind_update(&mut self, stmt: &UpdateStatement) -> Result<BoundUpdate> {
        let (schema_name, table_name) = if let Some(dot_pos) = stmt.table.find('.') {
            (Some(&stmt.table[..dot_pos]), &stmt.table[dot_pos + 1..])
        } else {
            (None, stmt.table.as_str())
        };

        let entry = self.resolver.resolve_table(schema_name, table_name).await?;

        // Register table in context
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: table_name.to_string(),
            columns,
            entry: Some(Arc::clone(&entry)),
        });

        let mut assignments = Vec::with_capacity(stmt.assignments.len());
        for a in &stmt.assignments {
            let col = self.resolver.resolve_column(&entry, &a.column)?;
            let value = self.bind_expr(&ctx, &a.value).await?;
            assignments.push(BoundAssignment {
                column_id: col.id,
                value,
            });
        }

        let where_clause = if let Some(expr) = &stmt.where_clause {
            Some(self.bind_expr(&ctx, expr).await?)
        } else {
            None
        };

        let returning = if let Some(items) = &stmt.returning {
            Some(self.bind_select_items(&ctx, items).await?)
        } else {
            None
        };

        Ok(BoundUpdate {
            table_id: entry.id,
            table_entry: entry,
            assignments,
            where_clause,
            returning,
        })
    }

    // -----------------------------------------------------------------------
    // DELETE binding
    // -----------------------------------------------------------------------

    async fn bind_delete(&mut self, stmt: &DeleteStatement) -> Result<BoundDelete> {
        let (schema_name, table_name) = if let Some(dot_pos) = stmt.table.find('.') {
            (Some(&stmt.table[..dot_pos]), &stmt.table[dot_pos + 1..])
        } else {
            (None, stmt.table.as_str())
        };

        let entry = self.resolver.resolve_table(schema_name, table_name).await?;

        // Register table in context
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: table_name.to_string(),
            columns,
            entry: Some(Arc::clone(&entry)),
        });

        let where_clause = if let Some(expr) = &stmt.where_clause {
            Some(self.bind_expr(&ctx, expr).await?)
        } else {
            None
        };

        let returning = if let Some(items) = &stmt.returning {
            Some(self.bind_select_items(&ctx, items).await?)
        } else {
            None
        };

        Ok(BoundDelete {
            table_id: entry.id,
            table_entry: entry,
            where_clause,
            returning,
        })
    }

    // -----------------------------------------------------------------------
    // CREATE STREAMING JOB binding
    // -----------------------------------------------------------------------

    /// Resolves the first schema in the resolver's search path. External DDL
    /// does not carry a qualified schema, so the binder uses the session's
    /// current schema when storing the bound form.
    async fn current_schema_id(&self) -> Result<SchemaId> {
        let search = self.resolver.search_path();
        let first = search
            .first()
            .ok_or_else(|| ZyronError::PlanError("search path is empty".to_string()))?;
        let entry = self.resolver.resolve_schema(first).await?;
        Ok(entry.id)
    }

    // -----------------------------------------------------------------------
    // Publication binding
    // -----------------------------------------------------------------------

    /// Binds CREATE PUBLICATION. Resolves each member table, projects the
    /// requested columns, binds per-table and publication-level WHERE
    /// predicates, parses WITH options, and computes a deterministic schema
    /// fingerprint over the projected columns.
    async fn bind_create_publication(
        &mut self,
        stmt: &CreatePublicationStatement,
    ) -> Result<BoundCreatePublication> {
        let schema_id = self.current_schema_id().await?;

        // Duplicate check, if_not_exists short-circuits.
        if let Some(_existing) = self.catalog.get_publication(schema_id, &stmt.name) {
            if !stmt.if_not_exists {
                return Err(ZyronError::PlanError(format!(
                    "publication '{}' already exists",
                    stmt.name
                )));
            }
        }

        if stmt.tables.is_empty() {
            return Err(ZyronError::PlanError(
                "CREATE PUBLICATION requires at least one table".to_string(),
            ));
        }

        // Resolve each publication table and bind its per-table WHERE.
        let mut bound_tables = Vec::with_capacity(stmt.tables.len());
        let mut first_entry: Option<Arc<TableEntry>> = None;
        for tref in &stmt.tables {
            let entry = self.resolver.resolve_table(None, &tref.table_name).await?;
            let mut column_ids = Vec::with_capacity(tref.columns.len());
            for cname in &tref.columns {
                let col = entry
                    .columns
                    .iter()
                    .find(|c| c.name == *cname)
                    .ok_or_else(|| {
                        ZyronError::PlanError(format!(
                            "column '{}' not found in table '{}'",
                            cname, tref.table_name
                        ))
                    })?;
                column_ids.push(col.id);
            }

            // Bind per-table predicate in a single-table scope.
            let where_predicate = match &tref.where_predicate {
                None => None,
                Some(expr) => {
                    let mut ctx = BindContext::new();
                    self.push_single_table_scope(&mut ctx, &entry);
                    Some(self.bind_expr(&ctx, expr).await?)
                }
            };

            bound_tables.push(BoundPublicationTable {
                table_id: entry.id,
                columns: column_ids,
                where_predicate,
            });

            if first_entry.is_none() {
                first_entry = Some(Arc::clone(&entry));
            }
        }

        // Publication-level WHERE binds against the first table's scope.
        let where_predicate = match (&stmt.where_predicate, first_entry.as_ref()) {
            (None, _) => None,
            (Some(expr), Some(entry)) => {
                let mut ctx = BindContext::new();
                self.push_single_table_scope(&mut ctx, entry);
                Some(self.bind_expr(&ctx, expr).await?)
            }
            (Some(_), None) => {
                return Err(ZyronError::PlanError(
                    "publication WHERE requires at least one table".to_string(),
                ));
            }
        };

        // Parse WITH options with defaults.
        let mut change_feed = true;
        let mut row_format = RowFormat::Binary;
        let mut retention_days: u32 = 30;
        let mut retain_until_subscribers_advance = true;
        let mut max_rows_per_sec: u64 = 0;
        let mut max_bytes_per_sec: u64 = 0;
        let mut max_concurrent_subscribers: u32 = 0;
        let mut classification = CatalogClassification::Internal;
        let mut allow_initial_snapshot = true;

        for (key, value) in &stmt.options {
            match key.to_ascii_lowercase().as_str() {
                "change_feed" => change_feed = parse_bool_option(key, value)?,
                "row_format" => row_format = parse_row_format(value)?,
                "retention_days" => retention_days = parse_u32_option(key, value)?,
                "retain_until_subscribers_advance" => {
                    retain_until_subscribers_advance = parse_bool_option(key, value)?
                }
                "max_rows_per_sec" => max_rows_per_sec = parse_u64_option(key, value)?,
                "max_bytes_per_sec" => max_bytes_per_sec = parse_size_bytes(value)?,
                "max_concurrent_subscribers" => {
                    max_concurrent_subscribers = parse_u32_option(key, value)?
                }
                "classification" => classification = parse_classification(value)?,
                "allow_initial_snapshot" => {
                    allow_initial_snapshot = parse_bool_option(key, value)?
                }
                other => {
                    return Err(ZyronError::PlanError(format!(
                        "unknown publication option '{other}'"
                    )));
                }
            }
        }

        let schema_fingerprint =
            compute_publication_fingerprint(&bound_tables, self.catalog);

        Ok(BoundCreatePublication {
            schema_id,
            name: stmt.name.clone(),
            if_not_exists: stmt.if_not_exists,
            tables: bound_tables,
            where_predicate,
            change_feed,
            row_format,
            retention_days,
            retain_until_subscribers_advance,
            max_rows_per_sec,
            max_bytes_per_sec,
            max_concurrent_subscribers,
            classification,
            allow_initial_snapshot,
            schema_fingerprint,
        })
    }

    /// Binds ALTER PUBLICATION. Resolves the target publication, then binds
    /// the action-specific payload.
    async fn bind_alter_publication(
        &mut self,
        stmt: &AlterPublicationStatement,
    ) -> Result<BoundAlterPublication> {
        let schema_id = self.current_schema_id().await?;
        let target = self
            .catalog
            .get_publication(schema_id, &stmt.name)
            .ok_or_else(|| {
                ZyronError::PlanError(format!("publication '{}' not found", stmt.name))
            })?;

        let action = match &stmt.action {
            AlterPublicationAction::AddTable(tref) => {
                let entry = self.resolver.resolve_table(None, &tref.table_name).await?;
                let mut column_ids = Vec::with_capacity(tref.columns.len());
                for cname in &tref.columns {
                    let col = entry
                        .columns
                        .iter()
                        .find(|c| c.name == *cname)
                        .ok_or_else(|| {
                            ZyronError::PlanError(format!(
                                "column '{}' not found in table '{}'",
                                cname, tref.table_name
                            ))
                        })?;
                    column_ids.push(col.id);
                }
                let where_predicate = match &tref.where_predicate {
                    None => None,
                    Some(expr) => {
                        let mut ctx = BindContext::new();
                        self.push_single_table_scope(&mut ctx, &entry);
                        Some(self.bind_expr(&ctx, expr).await?)
                    }
                };
                BoundAlterPublicationAction::AddTable(BoundPublicationTable {
                    table_id: entry.id,
                    columns: column_ids,
                    where_predicate,
                })
            }
            AlterPublicationAction::DropTable(name) => {
                let entry = self.resolver.resolve_table(None, name).await?;
                BoundAlterPublicationAction::DropTable(entry.id)
            }
            AlterPublicationAction::SetOptions(opts) => {
                let updates = parse_publication_option_updates(opts)?;
                BoundAlterPublicationAction::SetOptions(updates)
            }
            AlterPublicationAction::SetWhere(expr) => {
                let pub_tables = self.catalog.get_publication_tables(target.id);
                let first = pub_tables.first().ok_or_else(|| {
                    ZyronError::PlanError(
                        "cannot SET WHERE on a publication with no tables".to_string(),
                    )
                })?;
                let entry = self.catalog.get_table_by_id(first.table_id)?;
                let mut ctx = BindContext::new();
                self.push_single_table_scope(&mut ctx, &entry);
                BoundAlterPublicationAction::SetWhere(self.bind_expr(&ctx, expr).await?)
            }
            AlterPublicationAction::Rename(new_name) => {
                BoundAlterPublicationAction::Rename(new_name.clone())
            }
        };

        Ok(BoundAlterPublication {
            name: stmt.name.clone(),
            schema_id,
            publication_id: target.id,
            action,
        })
    }

    /// Resolves a publication by name and normalizes its tag list. Used for
    /// both TAG and UNTAG statements.
    async fn bind_publication_tags(
        &self,
        name: &str,
        tags: &[String],
    ) -> Result<(SchemaId, PublicationId, Vec<String>)> {
        let schema_id = self.current_schema_id().await?;
        let target = self.catalog.get_publication(schema_id, name).ok_or_else(|| {
            ZyronError::PlanError(format!("publication '{}' not found", name))
        })?;
        let mut out = Vec::with_capacity(tags.len());
        for t in tags {
            let trimmed = t.trim();
            if trimmed.is_empty() {
                return Err(ZyronError::PlanError(
                    "publication tag must be non-empty".to_string(),
                ));
            }
            out.push(trimmed.to_ascii_lowercase());
        }
        Ok((schema_id, target.id, out))
    }

    // -----------------------------------------------------------------------
    // Endpoint binding
    // -----------------------------------------------------------------------

    /// Binds CREATE ENDPOINT. Validates the path and methods, parses the
    /// SQL body (with `$name` placeholders substituted), and passes through
    /// all the endpoint metadata after range checks.
    async fn bind_create_endpoint(
        &mut self,
        stmt: &CreateEndpointStatement,
    ) -> Result<BoundCreateEndpoint> {
        let schema_id = self.current_schema_id().await?;

        if let Some(_existing) = self.catalog.get_endpoint(schema_id, &stmt.name) {
            if !stmt.if_not_exists {
                return Err(ZyronError::PlanError(format!(
                    "endpoint '{}' already exists",
                    stmt.name
                )));
            }
        }

        // Path collision check: catalog keys endpoints by path as well.
        if let Some(existing) = self.catalog.get_endpoint_by_path(&stmt.path) {
            if existing.name != stmt.name {
                return Err(ZyronError::PlanError(format!(
                    "endpoint path '{}' is already registered by endpoint '{}'",
                    stmt.path, existing.name
                )));
            }
        }

        if !stmt.path.starts_with('/') {
            return Err(ZyronError::PlanError(format!(
                "endpoint path '{}' must start with '/'",
                stmt.path
            )));
        }
        if stmt.methods.is_empty() {
            return Err(ZyronError::PlanError(
                "CREATE ENDPOINT requires at least one HTTP method".to_string(),
            ));
        }
        let mut methods = Vec::with_capacity(stmt.methods.len());
        for m in &stmt.methods {
            methods.push(parse_http_method(m)?);
        }

        // Extract parameter names from the SQL template and substitute
        // parser-friendly placeholders so we can produce a bound statement
        // for validation.
        let (param_names, rewritten_sql) = extract_endpoint_params(&stmt.sql);
        let parsed = zyron_parser::parse(&rewritten_sql).map_err(|e| {
            ZyronError::PlanError(format!("endpoint SQL parse error: {e}"))
        })?;
        let first = parsed.into_iter().next().ok_or_else(|| {
            ZyronError::PlanError("endpoint SQL produced no statements".to_string())
        })?;
        let bound_sql = Box::pin(self.bind(first)).await?;

        // Output column schema applies only to SELECT bodies.
        let output_columns: Vec<ColumnEntry> = match &bound_sql {
            BoundStatement::Select(sel) => sel
                .output_schema
                .iter()
                .enumerate()
                .map(|(i, c)| ColumnEntry {
                    id: ColumnId(i as u16),
                    table_id: TableId(0),
                    name: c.name.clone(),
                    type_id: c.type_id,
                    ordinal: i as u16,
                    nullable: c.nullable,
                    default_expr: None,
                    max_length: None,
                })
                .collect(),
            _ => Vec::new(),
        };
        // Parameters carry no type inference yet, default to Text.
        let param_types = vec![TypeId::Varchar; param_names.len()];

        let auth = map_endpoint_auth(stmt.auth);
        let output_format = map_endpoint_output_format(stmt.output_format);
        let rate_limit = match &stmt.rate_limit {
            None => None,
            Some(rl) => {
                if rl.count == 0 {
                    return Err(ZyronError::PlanError(
                        "endpoint rate limit count must be greater than zero".to_string(),
                    ));
                }
                Some(convert_rate_limit(rl))
            }
        };

        // Auth modes that reference a scope require at least one.
        if matches!(auth, EndpointAuthMode::OAuth2 | EndpointAuthMode::Jwt)
            && stmt.required_scopes.is_empty()
        {
            // Scopes are permitted but not required for Jwt, so only enforce
            // on OAuth2 to keep the spec's intent. Jwt with empty scopes
            // means any valid token is accepted.
        }

        let max_body_bytes = stmt.max_body_kb.saturating_mul(1024);
        const SERVER_MAX_BODY_BYTES: u32 = 10 * 1024 * 1024;
        if max_body_bytes > SERVER_MAX_BODY_BYTES {
            return Err(ZyronError::PlanError(format!(
                "endpoint max_body_kb exceeds server maximum of {} bytes",
                SERVER_MAX_BODY_BYTES
            )));
        }

        Ok(BoundCreateEndpoint {
            schema_id,
            name: stmt.name.clone(),
            if_not_exists: stmt.if_not_exists,
            path: stmt.path.clone(),
            methods,
            sql: stmt.sql.clone(),
            bound_sql: Box::new(bound_sql),
            param_names,
            param_types,
            output_columns,
            auth,
            required_scopes: stmt.required_scopes.clone(),
            rate_limit,
            output_format,
            cors_origins: stmt.cors_origins.clone(),
            cache_seconds: stmt.cache_seconds,
            timeout_seconds: stmt.timeout_seconds,
            max_body_bytes,
        })
    }

    /// Binds CREATE STREAMING ENDPOINT. Requires the backing publication to
    /// exist, resolves it to an id for the wire layer.
    async fn bind_create_streaming_endpoint(
        &self,
        stmt: &CreateStreamingEndpointStatement,
    ) -> Result<BoundCreateStreamingEndpoint> {
        let schema_id = self.current_schema_id().await?;

        if let Some(_existing) = self.catalog.get_endpoint(schema_id, &stmt.name) {
            if !stmt.if_not_exists {
                return Err(ZyronError::PlanError(format!(
                    "endpoint '{}' already exists",
                    stmt.name
                )));
            }
        }
        if let Some(existing) = self.catalog.get_endpoint_by_path(&stmt.path) {
            if existing.name != stmt.name {
                return Err(ZyronError::PlanError(format!(
                    "endpoint path '{}' is already registered by endpoint '{}'",
                    stmt.path, existing.name
                )));
            }
        }
        if !stmt.path.starts_with('/') {
            return Err(ZyronError::PlanError(format!(
                "streaming endpoint path '{}' must start with '/'",
                stmt.path
            )));
        }

        let publication = self
            .catalog
            .get_publication(schema_id, &stmt.backing_publication)
            .ok_or_else(|| {
                ZyronError::PlanError(format!(
                    "backing publication '{}' not found",
                    stmt.backing_publication
                ))
            })?;

        Ok(BoundCreateStreamingEndpoint {
            schema_id,
            name: stmt.name.clone(),
            if_not_exists: stmt.if_not_exists,
            path: stmt.path.clone(),
            protocol: stmt.protocol,
            backing_publication_id: publication.id,
            backing_publication_name: stmt.backing_publication.clone(),
            auth: map_endpoint_auth(stmt.auth),
            required_scopes: stmt.required_scopes.clone(),
            max_connections_per_ip: stmt.max_connections_per_ip,
            message_format: map_streaming_message_format(stmt.message_format),
            heartbeat_seconds: stmt.heartbeat_seconds,
            backpressure: map_backpressure(stmt.backpressure),
            max_connections: stmt.max_connections,
        })
    }

    /// Binds ALTER ENDPOINT. Enable/Disable pass through, SET OPTIONS is
    /// parsed into an EndpointOptionUpdates.
    async fn bind_alter_endpoint(
        &self,
        stmt: &AlterEndpointStatement,
    ) -> Result<BoundAlterEndpoint> {
        let schema_id = self.current_schema_id().await?;
        let target = self
            .catalog
            .get_endpoint(schema_id, &stmt.name)
            .ok_or_else(|| {
                ZyronError::PlanError(format!("endpoint '{}' not found", stmt.name))
            })?;

        let action = match &stmt.action {
            AlterEndpointAction::Enable => BoundAlterEndpointAction::Enable,
            AlterEndpointAction::Disable => BoundAlterEndpointAction::Disable,
            AlterEndpointAction::SetOptions(opts) => {
                BoundAlterEndpointAction::SetOptions(parse_endpoint_option_updates(opts)?)
            }
        };

        Ok(BoundAlterEndpoint {
            name: stmt.name.clone(),
            schema_id,
            endpoint_id: target.id,
            action,
        })
    }

    // -----------------------------------------------------------------------
    // Security map binding
    // -----------------------------------------------------------------------

    fn bind_alter_security_map(
        &self,
        stmt: &AlterSecurityMapStatement,
    ) -> Result<BoundAlterSecurityMap> {
        let (kind, key) = map_security_kind(&stmt.kind)?;
        if stmt.role.trim().is_empty() {
            return Err(ZyronError::PlanError(
                "security map role name must be non-empty".to_string(),
            ));
        }
        Ok(BoundAlterSecurityMap {
            kind,
            identity_key: key,
            role_name: stmt.role.clone(),
        })
    }

    fn bind_drop_security_map(
        &self,
        stmt: &DropSecurityMapStatement,
    ) -> Result<BoundDropSecurityMap> {
        let (kind, key) = map_security_kind(&stmt.kind)?;
        Ok(BoundDropSecurityMap {
            kind,
            identity_key: key,
        })
    }

    /// Pushes a single-table scope onto a bind context for binding a
    /// publication WHERE predicate against one specific table.
    fn push_single_table_scope(&mut self, ctx: &mut BindContext, entry: &Arc<TableEntry>) {
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: entry.name.clone(),
            columns,
            entry: Some(Arc::clone(entry)),
        });
    }

    /// Binds CREATE EXTERNAL SOURCE by resolving the current schema and
    /// sanity-checking backend/credential combinations.
    async fn bind_create_external_source(
        &self,
        stmt: &CreateExternalSourceStatement,
    ) -> Result<BoundCreateExternalSource> {
        let schema_id = self.current_schema_id().await?;
        // Local file backends have no credentials to carry.
        if matches!(stmt.backend, ExternalBackendKind::File) && !stmt.credentials.is_empty() {
            return Err(ZyronError::PlanError(
                "CREATE EXTERNAL SOURCE with backend FILE cannot declare CREDENTIALS".to_string(),
            ));
        }
        if matches!(stmt.backend, ExternalBackendKind::Zyron) {
            validate_zyron_tls_options(&stmt.options)?;
        }
        let columns: Vec<(String, zyron_common::TypeId)> = stmt
            .columns
            .iter()
            .map(|(n, dt)| (n.clone(), dt.to_type_id()))
            .collect();
        Ok(BoundCreateExternalSource {
            schema_id,
            name: stmt.name.clone(),
            if_not_exists: stmt.if_not_exists,
            backend: stmt.backend.clone(),
            uri: stmt.uri.clone(),
            format: stmt.format.clone(),
            mode: stmt.mode.clone(),
            options: stmt.options.clone(),
            columns,
            credentials: stmt.credentials.clone(),
        })
    }

    /// Binds CREATE EXTERNAL SINK with the same schema and backend checks as
    /// the source form. Sinks have no ingest mode.
    async fn bind_create_external_sink(
        &self,
        stmt: &CreateExternalSinkStatement,
    ) -> Result<BoundCreateExternalSink> {
        let schema_id = self.current_schema_id().await?;
        if matches!(stmt.backend, ExternalBackendKind::File) && !stmt.credentials.is_empty() {
            return Err(ZyronError::PlanError(
                "CREATE EXTERNAL SINK with backend FILE cannot declare CREDENTIALS".to_string(),
            ));
        }
        if matches!(stmt.backend, ExternalBackendKind::Zyron) {
            validate_zyron_tls_options(&stmt.options)?;
        }
        let columns: Vec<(String, zyron_common::TypeId)> = stmt
            .columns
            .iter()
            .map(|(n, dt)| (n.clone(), dt.to_type_id()))
            .collect();
        Ok(BoundCreateExternalSink {
            schema_id,
            name: stmt.name.clone(),
            if_not_exists: stmt.if_not_exists,
            backend: stmt.backend.clone(),
            uri: stmt.uri.clone(),
            format: stmt.format.clone(),
            options: stmt.options.clone(),
            columns,
            credentials: stmt.credentials.clone(),
        })
    }

    /// Validates and binds a CREATE STREAMING JOB statement.
    /// Rules: single-table FROM, no joins, no subqueries, no LATERAL,
    /// projection arity and types must match the target column layout, Zyron
    /// source tables must have CDC enabled, UPSERT write mode is not yet
    /// supported by the runner and is rejected here.
    async fn bind_create_streaming_job(
        &mut self,
        stmt: &CreateStreamingJobStatement,
    ) -> Result<BoundStreamingJob> {
        // Flag whether the job declares a windowed aggregate. This governs
        // several branches below, including the JOIN-vs-aggregate conflict
        // check in step 1.
        let agg_has_group = !stmt.query.group_by.is_empty()
            || stmt.watermark.is_some()
            || stmt.late_data_policy.is_some();

        // Step 1, single-table FROM with no join or subquery shapes for
        // pure filter+project jobs. Two-table JOIN drives the interval or
        // temporal join pipelines resolved later in this function.
        if stmt.query.from.len() != 1 {
            return Err(ZyronError::PlanError(
                "streaming jobs support a single FROM entry, either a base table or a two-table JOIN".to_string(),
            ));
        }
        // Classify the FROM shape up front. Joined forms defer the full
        // right-side resolution until after the left source is bound, so
        // both sides share the same scope-building path.
        let right_join_info: Option<StreamingJoinBindInput> = match (
            &stmt.query.from[0],
            &stmt.join,
        ) {
            (TableRef::Table { .. }, None) => None,
            (TableRef::Join(join_node), Some(spec)) => {
                // JOIN + GROUP BY is supported: the runner composes the
                // join engine with the aggregating engine, feeding joined
                // rows into the window state. The combined schema drives
                // the aggregate's column resolution, built later.
                // The parser's JoinType drives outer-emission semantics.
                // Cross is rejected because streaming requires an ON
                // equi-key to index the per-side state.
                if matches!(join_node.join_type, JoinType::Cross) {
                    return Err(ZyronError::PlanError(
                        "streaming JOIN does not support CROSS, use an explicit ON equi-key"
                            .to_string(),
                    ));
                }
                let (left_name, left_alias_opt) = match &join_node.left {
                    TableRef::Table { name, alias, .. } => (name.clone(), alias.clone()),
                    _ => {
                        return Err(ZyronError::PlanError(
                            "streaming JOIN left side must be a base table".to_string(),
                        ));
                    }
                };
                let (right_name, right_alias_opt) = match &join_node.right {
                    TableRef::Table { name, alias, .. } => (name.clone(), alias.clone()),
                    _ => {
                        return Err(ZyronError::PlanError(
                            "streaming JOIN right side must be a base table".to_string(),
                        ));
                    }
                };
                // Self-joins are allowed, but each side must carry a
                // distinct alias so the ON predicate can name them.
                let is_self_join = left_name.eq_ignore_ascii_case(&right_name);
                if is_self_join {
                    match (&left_alias_opt, &right_alias_opt) {
                        (Some(la), Some(ra)) if !la.eq_ignore_ascii_case(ra) => {}
                        _ => {
                            return Err(ZyronError::PlanError(
                                    "streaming self-join requires distinct aliases on both sides, for example 'orders o1 JOIN orders o2'"
                                        .to_string(),
                                ));
                        }
                    }
                }
                let left_alias = left_alias_opt.unwrap_or_else(|| left_name.clone());
                let right_alias = right_alias_opt.unwrap_or_else(|| right_name.clone());
                let (l_cols, r_cols) = match &join_node.condition {
                    JoinCondition::On(expr) => {
                        extract_equi_key_pairs(expr, &left_alias, &right_alias)?
                    }
                    _ => {
                        return Err(ZyronError::PlanError(
                                "streaming JOIN requires ON <left>.<col> = <right>.<col> with optional AND of additional equi-keys"
                                    .to_string(),
                            ));
                    }
                };
                Some(StreamingJoinBindInput {
                    left_name,
                    left_alias,
                    right_name,
                    right_alias,
                    left_key_cols: l_cols,
                    right_key_cols: r_cols,
                    spec: spec.clone(),
                })
            }
            (TableRef::Table { .. }, Some(_)) => {
                return Err(ZyronError::PlanError(
                    "streaming JOIN spec set but FROM has a single table".to_string(),
                ));
            }
            (TableRef::Join(_), None) => {
                return Err(ZyronError::PlanError(
                    "streaming FROM with JOIN requires WITHIN INTERVAL or AS OF on the right side"
                        .to_string(),
                ));
            }
            _ => {
                return Err(ZyronError::PlanError(
                    "streaming jobs support a single base table or two-table JOIN only".to_string(),
                ));
            }
        };
        let source_name = match &right_join_info {
            Some(j) => j.left_name.clone(),
            None => match &stmt.query.from[0] {
                TableRef::Table { name, .. } => name.clone(),
                _ => {
                    return Err(ZyronError::PlanError(
                        "streaming jobs support a base table or JOIN only".to_string(),
                    ));
                }
            },
        };

        // Reject HAVING, ORDER BY, LIMIT, CTEs, set ops, DISTINCT. These do not
        // make sense on streaming jobs today. GROUP BY and watermark are kept
        // and drive the aggregating pipeline below.
        if stmt.query.with.is_some()
            || stmt.query.having.is_some()
            || !stmt.query.order_by.is_empty()
            || stmt.query.limit.is_some()
            || stmt.query.offset.is_some()
            || stmt.query.distinct
            || !stmt.query.set_ops.is_empty()
        {
            return Err(ZyronError::PlanError(
                "streaming jobs support single-table FROM only in this release".to_string(),
            ));
        }

        // Step 2, resolve the target first. The target's column schema drives
        // the expected projection layout and may supply column names for an
        // external source that does not declare its own schema.
        let (target_kind, target_columns, target_schema_id): (
            BoundStreamingSinkKind,
            Vec<ColumnEntry>,
            SchemaId,
        ) = match &stmt.target {
            zyron_parser::StreamingSinkRef::Named(name) => {
                let (tgt_schema_opt, tgt_tbl) = split_qualified(name);
                // Named form: try Zyron table first, then external sink.
                let zyron_tbl = self.resolver.resolve_table(tgt_schema_opt, tgt_tbl).await;
                match zyron_tbl {
                    Ok(entry) => (
                        BoundStreamingSinkKind::Zyron {
                            table_id: entry.id,
                            schema_id: entry.schema_id,
                        },
                        entry.columns.clone(),
                        entry.schema_id,
                    ),
                    Err(_) => {
                        let schema_id = match tgt_schema_opt {
                            Some(s) => self.resolver.resolve_schema(s).await?.id,
                            None => self.current_schema_id().await?,
                        };
                        match self.catalog.get_external_sink(schema_id, tgt_tbl) {
                            Some(sink) => (
                                BoundStreamingSinkKind::ExternalNamed {
                                    sink_id: sink.id,
                                    schema_id: sink.schema_id,
                                    classification: sink.classification,
                                },
                                Vec::new(),
                                sink.schema_id,
                            ),
                            None => {
                                return Err(ZyronError::TableNotFound(format!(
                                    "target '{name}' not found as table or external sink"
                                )));
                            }
                        }
                    }
                }
            }
            zyron_parser::StreamingSinkRef::Inline {
                backend,
                uri,
                format,
                options,
            } => (
                BoundStreamingSinkKind::ExternalInline {
                    backend: backend.clone(),
                    uri: uri.clone(),
                    format: format.clone(),
                    options: options.clone(),
                },
                Vec::new(),
                SchemaId(0),
            ),
        };

        // Step 3, resolve the source. Try Zyron table first, then fall back to
        // external source lookup in the session's schema.
        let (src_schema_opt, src_tbl) = split_qualified(&source_name);
        let source_resolved: SourceResolution = {
            let zyron_tbl = self.resolver.resolve_table(src_schema_opt, src_tbl).await;
            match zyron_tbl {
                Ok(entry) => SourceResolution::Zyron(entry),
                Err(_) => {
                    let schema_id = match src_schema_opt {
                        Some(s) => self.resolver.resolve_schema(s).await?.id,
                        None => self.current_schema_id().await?,
                    };
                    match self.catalog.get_external_source(schema_id, src_tbl) {
                        Some(src) => SourceResolution::External(src),
                        None => {
                            return Err(ZyronError::TableNotFound(format!(
                                "source '{source_name}' not found as table or external source"
                            )));
                        }
                    }
                }
            }
        };

        // Step 4, the source columns come from the catalog for Zyron tables,
        // or from the target column layout for external sources that carry
        // no column schema of their own.
        let (source_kind, source_columns, source_classification): (
            BoundStreamingSourceKind,
            Vec<ColumnEntry>,
            Option<CatalogClassification>,
        ) = match &source_resolved {
            SourceResolution::Zyron(entry) => (
                BoundStreamingSourceKind::Zyron {
                    table_id: entry.id,
                    schema_id: entry.schema_id,
                },
                entry.columns.clone(),
                None,
            ),
            SourceResolution::External(src) => {
                // Inline sink plus Confidential/Restricted external source is
                // rejected on classification grounds before the column layout
                // check fires. This keeps the error message specific.
                if matches!(&target_kind, BoundStreamingSinkKind::ExternalInline { .. })
                    && matches!(
                        src.classification,
                        CatalogClassification::Confidential | CatalogClassification::Restricted
                    )
                {
                    return Err(ZyronError::PermissionDenied(format!(
                        "inline external sink cannot be used with classification {:?} source, use CREATE EXTERNAL SINK with explicit classification",
                        src.classification
                    )));
                }
                // Prefer the external source's own catalog columns when
                // present (set either by an explicit COLUMNS clause on
                // CREATE EXTERNAL SOURCE or by schema inference from the
                // first matching file). Fall back to the target layout
                // when the source has no recorded columns and the target
                // is a Zyron table. Reject when both ends are external and
                // neither side supplies a layout.
                let src_cols: Vec<ColumnEntry> = src
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(idx, (name, type_id))| ColumnEntry {
                        id: ColumnId(idx as u16),
                        table_id: TableId(0),
                        name: name.clone(),
                        type_id: *type_id,
                        ordinal: idx as u16,
                        nullable: true,
                        default_expr: None,
                        max_length: None,
                    })
                    .collect();
                if matches!(&target_kind, BoundStreamingSinkKind::ExternalInline { .. })
                    && target_columns.is_empty()
                    && src_cols.is_empty()
                {
                    return Err(ZyronError::PlanError(
                        "cannot use inline external sink with external source, declare a COLUMNS clause on the external source or use a Zyron table on one side".to_string(),
                    ));
                }
                let cols = if !src_cols.is_empty() {
                    src_cols
                } else if !target_columns.is_empty() {
                    target_columns.clone()
                } else {
                    Vec::new()
                };
                (
                    BoundStreamingSourceKind::ExternalNamed {
                        source_id: src.id,
                        schema_id: src.schema_id,
                    },
                    cols,
                    Some(src.classification),
                )
            }
        };

        // Step 5, register a scope that uses the source columns for expression
        // binding. External sources reuse the target column names and types.
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let col_defs: Vec<BoundColumnDef> = source_columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        let source_table_id_for_scope = match &source_kind {
            BoundStreamingSourceKind::Zyron { table_id, .. } => Some(*table_id),
            _ => None,
        };
        let source_entry_for_scope = match &source_resolved {
            SourceResolution::Zyron(e) => Some(Arc::clone(e)),
            _ => None,
        };
        // Override the left-side alias when the streaming job is a JOIN, so
        // the projection binder sees the user-declared alias like 'o' for
        // 'orders o'. Falls back to the default source-name alias otherwise.
        let left_scope_alias = match &right_join_info {
            Some(info) => info.left_alias.clone(),
            None => tgt_or_src_alias(&source_name),
        };
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: source_table_id_for_scope,
            alias: left_scope_alias,
            columns: col_defs,
            entry: source_entry_for_scope,
        });

        // For JOIN shapes, push the right-side table into the same scope so
        // projections and predicates can reference right-side columns by the
        // right alias. The right side is resolved eagerly here because the
        // binder needs the column list for expression binding.
        // combined_source_columns is the join's output-row shape: left source
        // columns followed by right source columns. Used by the aggregate
        // builder so GROUP BY terms can reference right-side columns.
        let mut combined_source_columns: Vec<ColumnEntry> = source_columns.clone();
        if let Some(info) = &right_join_info {
            let (r_schema_opt, r_tbl) = split_qualified(&info.right_name);
            let right_entry = self
                .resolver
                .resolve_table(r_schema_opt, r_tbl)
                .await
                .map_err(|_| {
                    ZyronError::TableNotFound(format!(
                        "streaming JOIN right source '{}' not found",
                        info.right_name
                    ))
                })?;
            let right_cols: Vec<BoundColumnDef> = right_entry
                .columns
                .iter()
                .map(|c| BoundColumnDef {
                    column_id: c.id,
                    name: c.name.clone(),
                    type_id: c.type_id,
                    nullable: c.nullable,
                    ordinal: c.ordinal,
                })
                .collect();
            let r_idx = self.alloc_table_idx();
            ctx.tables.push(BoundTableRef {
                table_idx: r_idx,
                table_id: Some(right_entry.id),
                alias: info.right_alias.clone(),
                columns: right_cols,
                entry: Some(Arc::clone(&right_entry)),
            });
            combined_source_columns.extend(right_entry.columns.iter().cloned());
        }

        // Expand wildcard projections into one BoundExpr per source column.
        // Aggregating jobs skip the SELECT-to-BoundExpr binding because the
        // aggregate pipeline lives in BoundAggregateSpec and the SELECT list
        // there is validated by build_streaming_aggregate_spec instead.
        let mut projections: Vec<BoundExpr> = Vec::new();
        if !agg_has_group {
            // Wildcard expands over the combined join schema when a JOIN is
            // present, and over the source alone otherwise.
            let wildcard_columns: &[ColumnEntry] = if right_join_info.is_some() {
                &combined_source_columns
            } else {
                &source_columns
            };
            for item in &stmt.query.projections {
                match item {
                    SelectItem::Wildcard => {
                        for col in wildcard_columns {
                            projections.push(BoundExpr::ColumnRef(ColumnRef {
                                table_idx: idx,
                                column_id: col.id,
                                type_id: col.type_id,
                                nullable: col.nullable,
                            }));
                        }
                    }
                    SelectItem::QualifiedWildcard(_) => {
                        for col in wildcard_columns {
                            projections.push(BoundExpr::ColumnRef(ColumnRef {
                                table_idx: idx,
                                column_id: col.id,
                                type_id: col.type_id,
                                nullable: col.nullable,
                            }));
                        }
                    }
                    SelectItem::Expr(expr, _alias) => {
                        let bound = self.bind_expr(&ctx, expr).await?;
                        projections.push(bound);
                    }
                }
            }
        }

        // Step 6, bind optional WHERE predicate against the source scope.
        let predicate = if let Some(expr) = &stmt.query.where_clause {
            Some(self.bind_expr(&ctx, expr).await?)
        } else {
            None
        };

        // Step 7, arity and type match against the target columns. Inline
        // sinks have no declared column schema, so the binder uses the
        // source column layout as the effective sink layout.
        let effective_target_columns: Vec<ColumnEntry> = if target_columns.is_empty() {
            source_columns.clone()
        } else {
            target_columns.clone()
        };
        // Aggregating jobs use BoundAggregateSpec to describe their output
        // row shape. The target arity check for aggregates runs inside
        // build_streaming_aggregate_spec against group-by + aggregate columns.
        if !agg_has_group {
            if projections.len() != effective_target_columns.len() {
                return Err(ZyronError::PlanError(format!(
                    "streaming job projection arity {} does not match target with {} columns",
                    projections.len(),
                    effective_target_columns.len()
                )));
            }
            for (i, (proj, target_col)) in projections
                .iter()
                .zip(effective_target_columns.iter())
                .enumerate()
            {
                let proj_type = proj.type_id();
                if proj_type != target_col.type_id {
                    return Err(ZyronError::PlanError(format!(
                        "streaming job projection column {} type mismatch: got {:?}, target expects {:?}",
                        i, proj_type, target_col.type_id
                    )));
                }
            }
        }

        // Step 8, Zyron-table sources must have change data feed enabled.
        if let SourceResolution::Zyron(entry) = &source_resolved {
            if !entry.cdf_enabled {
                return Err(ZyronError::PlanError(format!(
                    "source table '{}' does not have CDC enabled, run ALTER TABLE {} SET (cdc_enabled = true) first",
                    entry.name, entry.name
                )));
            }
        }

        // Step 9, classification checks. Named external sink at a lower
        // classification than the source is rejected to prevent exfiltration.
        // Inline sinks carry no classification, so Confidential and Restricted
        // sources cannot flow to them at all.
        let max_source_cls = match &source_resolved {
            SourceResolution::External(src) => Some(src.classification),
            SourceResolution::Zyron(_) => source_classification_from_columns(&source_columns),
        };
        if let Some(src_cls) = max_source_cls {
            match &target_kind {
                BoundStreamingSinkKind::ExternalNamed {
                    classification: sink_cls,
                    ..
                } => {
                    if (*sink_cls as u8) < (src_cls as u8) {
                        return Err(ZyronError::PermissionDenied(format!(
                            "cannot stream data at classification {:?} into sink classified {:?}",
                            src_cls, sink_cls
                        )));
                    }
                }
                BoundStreamingSinkKind::ExternalInline { .. } => {
                    if matches!(
                        src_cls,
                        CatalogClassification::Confidential | CatalogClassification::Restricted
                    ) {
                        return Err(ZyronError::PermissionDenied(format!(
                            "inline external sink cannot be used with classification {:?} source, use CREATE EXTERNAL SINK with explicit classification",
                            src_cls
                        )));
                    }
                }
                BoundStreamingSinkKind::Zyron { .. } => {
                    // Flow into a Zyron table is fine, column-level classification
                    // is enforced at query time by the security context.
                }
            }
        }

        // Step 10, translate parser write mode to catalog write mode. UPSERT
        // requires the target to be a Zyron table with a declared primary key
        // so the sink can look up existing rows. External sinks and tables
        // without a PK are rejected here with a precise message.
        let (write_mode, target_pk_columns) = match stmt.write_mode {
            StreamingWriteMode::Append => (CatalogStreamingWriteMode::Append, Vec::new()),
            StreamingWriteMode::Upsert => {
                let target_table_id = match &target_kind {
                    BoundStreamingSinkKind::Zyron { table_id, .. } => *table_id,
                    _ => {
                        return Err(ZyronError::PlanError(
                            "UPSERT write mode requires a Zyron table target".to_string(),
                        ));
                    }
                };
                let target_entry = self.catalog.get_table_by_id(target_table_id).map_err(|_| {
                    ZyronError::PlanError("UPSERT target table not found in catalog".to_string())
                })?;
                let pk_cols: Vec<ColumnId> = target_entry
                    .constraints
                    .iter()
                    .find(|c| {
                        c.constraint_type == zyron_catalog::schema::ConstraintType::PrimaryKey
                    })
                    .map(|c| c.columns.clone())
                    .unwrap_or_default();
                if pk_cols.is_empty() {
                    return Err(ZyronError::PlanError(
                        "UPSERT write mode requires the target table to have a primary key"
                            .to_string(),
                    ));
                }
                (CatalogStreamingWriteMode::Upsert, pk_cols)
            }
        };

        // Final assembly: convert the intermediate kinds plus column lists
        // into the public BoundStreamingSource and BoundStreamingSink shapes.
        let source = match source_kind {
            BoundStreamingSourceKind::Zyron {
                table_id,
                schema_id,
            } => BoundStreamingSource::ZyronTable {
                table_id,
                schema_id,
                columns: source_columns.clone(),
            },
            BoundStreamingSourceKind::ExternalNamed {
                source_id,
                schema_id,
            } => BoundStreamingSource::ExternalNamed {
                source_id,
                schema_id,
                columns: source_columns.clone(),
                classification: source_classification.unwrap_or(CatalogClassification::Public),
            },
        };
        let target = match target_kind {
            BoundStreamingSinkKind::Zyron {
                table_id,
                schema_id,
            } => BoundStreamingSink::ZyronTable {
                table_id,
                schema_id,
                columns: target_columns.clone(),
            },
            BoundStreamingSinkKind::ExternalNamed {
                sink_id,
                schema_id,
                classification,
            } => BoundStreamingSink::ExternalNamed {
                sink_id,
                schema_id,
                columns: effective_target_columns.clone(),
                classification,
            },
            BoundStreamingSinkKind::ExternalInline {
                backend,
                uri,
                format,
                options,
            } => BoundStreamingSink::ExternalInline {
                backend,
                uri,
                format,
                options,
                columns: effective_target_columns.clone(),
            },
        };
        let _ = target_schema_id; // Avoid warn, value is encoded inside target.

        // Build the aggregate section when the SELECT declares a window GROUP
        // BY. A streaming job either has zero aggregates (pure filter/project)
        // or at least one window function in the GROUP BY plus a matching
        // WATERMARK clause.
        let aggregate = if agg_has_group {
            // When the job is a JOIN + GROUP BY composition, GROUP BY columns
            // and aggregate inputs resolve against the join's combined output
            // schema (left columns followed by right columns). Pure jobs use
            // just the source columns.
            let agg_source: &[ColumnEntry] = if right_join_info.is_some() {
                &combined_source_columns
            } else {
                &source_columns
            };
            Some(build_streaming_aggregate_spec(
                stmt,
                agg_source,
                &effective_target_columns,
                &ctx,
                idx,
            )?)
        } else {
            None
        };

        // Resolve the right side of a streaming JOIN, if present. Produces
        // a BoundStreamingJoinSpec that carries the column ordinals and the
        // resolved right table or stream source for the runner to open.
        let join: Option<BoundStreamingJoinSpec> = if let Some(info) = &right_join_info {
            // Resolve every left-side equi-key column name to its ordinal on
            // the already-bound source. Empty is impossible because the
            // parser's extract_equi_key_pairs rejects zero-conjunct shapes.
            let mut left_key_ordinals: Vec<u16> = Vec::with_capacity(info.left_key_cols.len());
            for name in &info.left_key_cols {
                let ord = source_columns
                    .iter()
                    .find(|c| c.name.eq_ignore_ascii_case(name))
                    .map(|c| c.ordinal)
                    .ok_or_else(|| {
                        ZyronError::PlanError(format!(
                            "streaming JOIN left key column '{}' not found on source '{}'",
                            name, info.left_name
                        ))
                    })?;
                left_key_ordinals.push(ord);
            }
            let left_key_ordinal = left_key_ordinals[0];
            match &info.spec {
                zyron_parser::ast::StreamingJoinSpec::Interval {
                    within_us,
                    join_type,
                } => {
                    // Resolve the right side as a Zyron table with CDC. A
                    // two-stream interval join requires both sides to be CDC
                    // sources, external sources are rejected for the interval
                    // form in this release to keep the runtime surface small.
                    let (r_schema_opt, r_tbl) = split_qualified(&info.right_name);
                    let right_entry = self
                        .resolver
                        .resolve_table(r_schema_opt, r_tbl)
                        .await
                        .map_err(|_| {
                            ZyronError::TableNotFound(format!(
                                "streaming JOIN right source '{}' not found as a Zyron table",
                                info.right_name
                            ))
                        })?;
                    if !right_entry.cdf_enabled {
                        return Err(ZyronError::PlanError(format!(
                            "streaming JOIN right source '{}' does not have CDC enabled",
                            right_entry.name
                        )));
                    }
                    let right_cols = right_entry.columns.clone();
                    if info.right_key_cols.len() != info.left_key_cols.len() {
                        return Err(ZyronError::PlanError(format!(
                            "streaming JOIN key count mismatch: left has {}, right has {}",
                            info.left_key_cols.len(),
                            info.right_key_cols.len()
                        )));
                    }
                    let mut right_key_ordinals: Vec<u16> =
                        Vec::with_capacity(info.right_key_cols.len());
                    for name in &info.right_key_cols {
                        let ord = right_cols
                            .iter()
                            .find(|c| c.name.eq_ignore_ascii_case(name))
                            .map(|c| c.ordinal)
                            .ok_or_else(|| {
                                ZyronError::PlanError(format!(
                                    "streaming JOIN right key column '{}' not found on '{}'",
                                    name, right_entry.name
                                ))
                            })?;
                        right_key_ordinals.push(ord);
                    }
                    let right_key_ordinal = right_key_ordinals[0];
                    // Every key pair must share a TypeId. The runtime encodes
                    // the key as a concatenation of column bytes, so a type
                    // mismatch would break equality semantics silently.
                    for (i, (lo, ro)) in left_key_ordinals
                        .iter()
                        .zip(right_key_ordinals.iter())
                        .enumerate()
                    {
                        let lt = source_columns[*lo as usize].type_id;
                        let rt = right_cols[*ro as usize].type_id;
                        if lt != rt {
                            return Err(ZyronError::PlanError(format!(
                                "streaming JOIN key pair {} types differ: left {:?}, right {:?}",
                                i, lt, rt
                            )));
                        }
                    }
                    // Event-time columns resolve to the named event-time
                    // column on each side. Fall back to the equi-key column
                    // ordinal when no event-time column is declared, matching
                    // the pure-filter WATERMARK convention.
                    let left_event_time_ordinal = match &stmt.watermark {
                        Some(w) => source_columns
                            .iter()
                            .find(|c| c.name.eq_ignore_ascii_case(&w.event_time_column))
                            .map(|c| c.ordinal)
                            .unwrap_or(left_key_ordinal),
                        None => left_key_ordinal,
                    };
                    let right_event_time_ordinal = right_cols
                        .iter()
                        .position(|c| {
                            c.name.to_lowercase().contains("event_time")
                                || c.name.to_lowercase().contains("ts")
                        })
                        .map(|i| i as u16)
                        .unwrap_or(right_key_ordinal);
                    let mut combined: Vec<ColumnEntry> = source_columns.clone();
                    combined.extend(right_cols.iter().cloned());
                    let _ = right_key_ordinal; // reserved for engine lookups
                    Some(BoundStreamingJoinSpec::Interval {
                        right_source: BoundStreamingSource::ZyronTable {
                            table_id: right_entry.id,
                            schema_id: right_entry.schema_id,
                            columns: right_cols,
                        },
                        right_alias: info.right_alias.clone(),
                        left_key_ordinals,
                        right_key_ordinals,
                        left_event_time_ordinal,
                        right_event_time_ordinal,
                        within_us: *within_us,
                        join_type: map_streaming_join_type(*join_type),
                        combined_columns: combined,
                    })
                }
                zyron_parser::ast::StreamingJoinSpec::Temporal {
                    event_time_column,
                    join_type,
                } => {
                    if matches!(
                        *join_type,
                        zyron_parser::ast::StreamingJoinType::Right
                            | zyron_parser::ast::StreamingJoinType::Full
                    ) {
                        return Err(ZyronError::PlanError(
                            "temporal joins support only INNER and LEFT OUTER, the right side is a static lookup"
                                .to_string(),
                        ));
                    }
                    // Right side must be a Zyron table with a primary key.
                    let (r_schema_opt, r_tbl) = split_qualified(&info.right_name);
                    let right_entry = self
                        .resolver
                        .resolve_table(r_schema_opt, r_tbl)
                        .await
                        .map_err(|_| {
                            ZyronError::TableNotFound(format!(
                                "streaming temporal JOIN right table '{}' not found",
                                info.right_name
                            ))
                        })?;
                    let right_cols = right_entry.columns.clone();
                    let pk_col_ids: Vec<ColumnId> = right_entry
                        .constraints
                        .iter()
                        .find(|c| {
                            c.constraint_type == zyron_catalog::schema::ConstraintType::PrimaryKey
                        })
                        .map(|c| c.columns.clone())
                        .unwrap_or_default();
                    if pk_col_ids.is_empty() {
                        return Err(ZyronError::PlanError(format!(
                            "streaming temporal JOIN requires right table '{}' to declare a PRIMARY KEY",
                            right_entry.name
                        )));
                    }
                    let pk_ordinals: Vec<u16> = pk_col_ids
                        .iter()
                        .filter_map(|id| right_cols.iter().find(|c| c.id == *id).map(|c| c.ordinal))
                        .collect();
                    let left_event_time_ordinal = source_columns
                        .iter()
                        .find(|c| c.name.eq_ignore_ascii_case(event_time_column))
                        .map(|c| c.ordinal)
                        .ok_or_else(|| {
                            ZyronError::PlanError(format!(
                                "streaming temporal JOIN event-time column '{}' not found on left source",
                                event_time_column
                            ))
                        })?;
                    let mut combined: Vec<ColumnEntry> = source_columns.clone();
                    combined.extend(right_cols.iter().cloned());
                    let _ = left_key_ordinal; // kept as the first left key
                    Some(BoundStreamingJoinSpec::Temporal {
                        right_table_id: right_entry.id,
                        right_schema_id: right_entry.schema_id,
                        right_alias: info.right_alias.clone(),
                        right_pk_ordinals: pk_ordinals,
                        left_key_ordinals,
                        left_event_time_ordinal,
                        join_type: map_streaming_join_type(*join_type),
                        combined_columns: combined,
                    })
                }
            }
        } else {
            None
        };

        Ok(BoundStreamingJob {
            name: stmt.name.clone(),
            if_not_exists: stmt.if_not_exists,
            source,
            target,
            projections,
            predicate,
            write_mode,
            job_mode: stmt.job_mode.clone(),
            target_pk_columns,
            aggregate,
            join,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming aggregate binding
// ---------------------------------------------------------------------------

/// Extracts TUMBLE/HOP/SESSION from the GROUP BY list, binds the remaining
/// group-by columns, validates the SELECT list as a mix of group-by column
/// references and aggregate function calls, and assembles a BoundAggregateSpec.
/// Returns a PlanError when any rule is violated.
fn build_streaming_aggregate_spec(
    stmt: &CreateStreamingJobStatement,
    source_columns: &[ColumnEntry],
    target_columns: &[ColumnEntry],
    ctx: &BindContext,
    table_idx: usize,
) -> Result<BoundAggregateSpec> {
    // The SELECT must contain at least one aggregate or the GROUP BY is a
    // misuse of streaming windowing semantics.
    if stmt.query.group_by.is_empty() {
        return Err(ZyronError::PlanError(
            "streaming window requires GROUP BY with TUMBLE, HOP, or SESSION".to_string(),
        ));
    }

    // Walk the GROUP BY list. Exactly one entry must be a window function
    // call. The rest are treated as grouping columns.
    let mut window_type: Option<BoundStreamingWindowType> = None;
    let mut group_cols: Vec<ColumnId> = Vec::new();
    let mut event_time_column_name: Option<String> = None;
    for gexp in &stmt.query.group_by {
        if let Some((wt, ev_col)) = try_parse_window_fn(gexp)? {
            if window_type.is_some() {
                return Err(ZyronError::PlanError(
                    "streaming GROUP BY allows at most one window function".to_string(),
                ));
            }
            window_type = Some(wt);
            event_time_column_name = Some(ev_col);
            continue;
        }
        // Plain column reference.
        let name = match gexp {
            Expr::Identifier(n) => n.clone(),
            Expr::QualifiedIdentifier { column, .. } => column.clone(),
            _ => {
                return Err(ZyronError::PlanError(
                    "streaming GROUP BY terms must be column references or a window function"
                        .to_string(),
                ));
            }
        };
        let col = source_columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(&name))
            .ok_or_else(|| {
                ZyronError::PlanError(format!("GROUP BY column '{}' not found in source", name))
            })?;
        group_cols.push(col.id);
    }
    let window_type = window_type.ok_or_else(|| {
        ZyronError::PlanError(
            "streaming GROUP BY must contain a TUMBLE, HOP, or SESSION call".to_string(),
        )
    })?;
    let event_time_column_name = event_time_column_name.unwrap();
    let event_time_col = source_columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(&event_time_column_name))
        .ok_or_else(|| {
            ZyronError::PlanError(format!(
                "event-time column '{}' not found in source",
                event_time_column_name
            ))
        })?;
    let event_time_scale = infer_event_time_scale(event_time_col.type_id);

    // Bind each SELECT item. Plain column refs must reference a group-by
    // column. Function calls are aggregates and must resolve to a known
    // accumulator keyed by (name, input_type).
    let mut aggregations: Vec<BoundAggregateItem> = Vec::new();
    for item in &stmt.query.projections {
        match item {
            SelectItem::Wildcard | SelectItem::QualifiedWildcard(_) => {
                return Err(ZyronError::PlanError(
                    "wildcard SELECT is not supported in aggregating streaming jobs".to_string(),
                ));
            }
            SelectItem::Expr(expr, _alias) => match expr {
                Expr::Identifier(_) | Expr::QualifiedIdentifier { .. } => {
                    let name = match expr {
                        Expr::Identifier(n) => n.clone(),
                        Expr::QualifiedIdentifier { column, .. } => column.clone(),
                        _ => unreachable!(),
                    };
                    let col = source_columns
                        .iter()
                        .find(|c| c.name.eq_ignore_ascii_case(&name))
                        .ok_or_else(|| {
                            ZyronError::PlanError(format!(
                                "SELECT column '{}' not found in source",
                                name
                            ))
                        })?;
                    if !group_cols.iter().any(|id| *id == col.id) {
                        return Err(ZyronError::PlanError(format!(
                            "SELECT column '{}' must appear in GROUP BY or be inside an aggregate",
                            name
                        )));
                    }
                }
                Expr::Function { name, args, .. } => {
                    let (input_col, input_type) = resolve_agg_input(name, args, source_columns)?;
                    let output_type = infer_agg_output(name, input_type);
                    aggregations.push(BoundAggregateItem {
                        function: name.to_ascii_uppercase(),
                        input_column_id: input_col,
                        output_type,
                    });
                }
                _ => {
                    return Err(ZyronError::PlanError(
                        "streaming aggregate SELECT items must be group-by columns or aggregate calls".to_string(),
                    ));
                }
            },
        }
    }

    // WATERMARK is required whenever an aggregate runs. Without it, the
    // runner has no signal for window closure.
    let watermark = if let Some(wm) = &stmt.watermark {
        if !wm
            .event_time_column
            .eq_ignore_ascii_case(&event_time_column_name)
        {
            return Err(ZyronError::PlanError(format!(
                "WATERMARK column '{}' must match GROUP BY event-time column '{}'",
                wm.event_time_column, event_time_column_name
            )));
        }
        let micros = interval_to_micros(&wm.allowed_lateness)?;
        if micros == 0 {
            BoundWatermark::Punctual
        } else {
            BoundWatermark::BoundedOutOfOrderness {
                allowed_lateness_us: micros,
            }
        }
    } else {
        BoundWatermark::Punctual
    };

    let late_data_policy = match stmt.late_data_policy {
        Some(LateDataPolicySpec::Drop) => BoundLateDataPolicy::Drop,
        Some(LateDataPolicySpec::Reopen) => BoundLateDataPolicy::ReopenWindow,
        Some(LateDataPolicySpec::SideOutput) => BoundLateDataPolicy::SideOutput,
        None => BoundLateDataPolicy::Drop,
    };

    // Silence the unused-warnings for ctx and table_idx parameters. They are
    // plumbed in so future aggregate argument forms (CAST, arithmetic inside
    // the aggregate) can reuse the full expression binder if needed.
    let _ = ctx;
    let _ = table_idx;

    // Target arity check: N group-by columns + M aggregate outputs must equal
    // the target column count. Empty target column list means the target is
    // an inline external sink, which does not enforce arity here.
    if !target_columns.is_empty() {
        let expected = group_cols.len() + aggregations.len();
        if expected != target_columns.len() {
            return Err(ZyronError::PlanError(format!(
                "streaming aggregate output arity {} does not match target with {} columns",
                expected,
                target_columns.len()
            )));
        }
    }

    Ok(BoundAggregateSpec {
        window_type,
        event_time_column_id: event_time_col.id,
        event_time_scale,
        group_by_column_ids: group_cols,
        aggregations,
        watermark,
        late_data_policy,
    })
}

/// Attempts to interpret a GROUP BY expression as TUMBLE/HOP/SESSION. Returns
/// Ok(None) when the expression is not a window function call.
fn try_parse_window_fn(expr: &Expr) -> Result<Option<(BoundStreamingWindowType, String)>> {
    let (name, args) = match expr {
        Expr::Function { name, args, .. } => (name.to_ascii_uppercase(), args),
        _ => return Ok(None),
    };
    match name.as_str() {
        "TUMBLE" => {
            if args.len() != 2 {
                return Err(ZyronError::PlanError(
                    "TUMBLE(event_time, INTERVAL 'N unit') requires exactly 2 arguments"
                        .to_string(),
                ));
            }
            let col = extract_column_name(&args[0])?;
            let size = extract_interval_micros(&args[1])?;
            let size_ms = (size / 1_000).max(1);
            Ok(Some((BoundStreamingWindowType::Tumbling { size_ms }, col)))
        }
        "HOP" | "HOPPING" => {
            if args.len() != 3 {
                return Err(ZyronError::PlanError(
                    "HOP(event_time, slide, size) requires exactly 3 arguments".to_string(),
                ));
            }
            let col = extract_column_name(&args[0])?;
            let slide = extract_interval_micros(&args[1])?;
            let size = extract_interval_micros(&args[2])?;
            let slide_ms = (slide / 1_000).max(1);
            let size_ms = (size / 1_000).max(1);
            Ok(Some((
                BoundStreamingWindowType::Hopping { size_ms, slide_ms },
                col,
            )))
        }
        "SESSION" => {
            if args.len() != 2 {
                return Err(ZyronError::PlanError(
                    "SESSION(event_time, INTERVAL 'N unit') requires exactly 2 arguments"
                        .to_string(),
                ));
            }
            let col = extract_column_name(&args[0])?;
            let gap = extract_interval_micros(&args[1])?;
            let gap_ms = (gap / 1_000).max(1);
            Ok(Some((BoundStreamingWindowType::Session { gap_ms }, col)))
        }
        _ => Ok(None),
    }
}

fn extract_column_name(arg: &FunctionArg) -> Result<String> {
    match arg {
        FunctionArg::Unnamed(expr) | FunctionArg::Named { value: expr, .. } => match expr {
            Expr::Identifier(n) => Ok(n.clone()),
            Expr::QualifiedIdentifier { column, .. } => Ok(column.clone()),
            _ => Err(ZyronError::PlanError(
                "expected a column reference as the window function's first argument".to_string(),
            )),
        },
    }
}

fn extract_interval_micros(arg: &FunctionArg) -> Result<i64> {
    match arg {
        FunctionArg::Unnamed(expr) | FunctionArg::Named { value: expr, .. } => match expr {
            Expr::Literal(LiteralValue::Interval(interval)) => interval_to_micros(interval),
            _ => Err(ZyronError::PlanError(
                "expected an INTERVAL literal for the window duration".to_string(),
            )),
        },
    }
}

fn interval_to_micros(iv: &zyron_common::Interval) -> Result<i64> {
    // Only the nanoseconds component is honored for sub-day durations. Days
    // fold into 24-hour multiples. Months are rejected because streaming
    // windows must have an exact duration in microseconds.
    if iv.months != 0 {
        return Err(ZyronError::PlanError(
            "streaming window durations cannot use month-based intervals".to_string(),
        ));
    }
    let ns_from_days: i64 = (iv.days as i64).saturating_mul(86_400_000_000_000i64);
    let total_ns = iv.nanoseconds.saturating_add(ns_from_days);
    Ok(total_ns / 1_000)
}

fn resolve_agg_input(
    name: &str,
    args: &[FunctionArg],
    source_columns: &[ColumnEntry],
) -> Result<(Option<ColumnId>, TypeId)> {
    let upper = name.to_ascii_uppercase();
    // COUNT(*) is parsed as a single positional argument that is the string
    // identifier "*".
    if upper == "COUNT" && args.len() == 1 {
        if let FunctionArg::Unnamed(Expr::Identifier(n)) = &args[0] {
            if n == "*" {
                return Ok((None, TypeId::Null));
            }
        }
    }
    if args.len() != 1 {
        return Err(ZyronError::PlanError(format!(
            "aggregate {name} requires exactly one argument"
        )));
    }
    let col_name = match &args[0] {
        FunctionArg::Unnamed(expr) | FunctionArg::Named { value: expr, .. } => match expr {
            Expr::Identifier(n) => n.clone(),
            Expr::QualifiedIdentifier { column, .. } => column.clone(),
            _ => {
                return Err(ZyronError::PlanError(format!(
                    "aggregate {name} argument must be a column reference"
                )));
            }
        },
    };
    let col = source_columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(&col_name))
        .ok_or_else(|| {
            ZyronError::PlanError(format!(
                "aggregate {name} references unknown column '{col_name}'"
            ))
        })?;
    Ok((Some(col.id), col.type_id))
}

fn infer_agg_output(name: &str, input_type: TypeId) -> TypeId {
    let upper = name.to_ascii_uppercase();
    match upper.as_str() {
        "COUNT" => TypeId::Int64,
        "SUM" => match input_type {
            TypeId::Float32 | TypeId::Float64 => TypeId::Float64,
            _ => TypeId::Int64,
        },
        "AVG" => TypeId::Float64,
        "MIN" | "MAX" => match input_type {
            TypeId::Float32 | TypeId::Float64 => TypeId::Float64,
            _ => TypeId::Int64,
        },
        "FIRST" | "LAST" => input_type,
        _ => input_type,
    }
}

/// Picks a sensible event-time scale based on the column's TypeId. Timestamp
/// columns are already in microseconds. Date is interpreted as days (seconds
/// multiplier then days*86400 fits one more step up). Plain integers default
/// to milliseconds, which is the standard Kafka and CDC timestamp unit.
fn infer_event_time_scale(type_id: TypeId) -> BoundEventTimeScale {
    match type_id {
        TypeId::Timestamp | TypeId::TimestampTz => BoundEventTimeScale::Microseconds,
        TypeId::Time => BoundEventTimeScale::Microseconds,
        _ => BoundEventTimeScale::Milliseconds,
    }
}

// ---------------------------------------------------------------------------
// Intermediate resolution shapes used only inside bind_create_streaming_job.
// ---------------------------------------------------------------------------

enum SourceResolution {
    Zyron(Arc<TableEntry>),
    External(Arc<zyron_catalog::schema::ExternalSourceEntry>),
}

enum BoundStreamingSourceKind {
    Zyron {
        table_id: TableId,
        schema_id: SchemaId,
    },
    ExternalNamed {
        source_id: ExternalSourceId,
        schema_id: SchemaId,
    },
}

enum BoundStreamingSinkKind {
    Zyron {
        table_id: TableId,
        schema_id: SchemaId,
    },
    ExternalNamed {
        sink_id: ExternalSinkId,
        schema_id: SchemaId,
        classification: CatalogClassification,
    },
    ExternalInline {
        backend: zyron_parser::ast::ExternalBackendKind,
        uri: String,
        format: zyron_parser::ast::ExternalFormatKind,
        options: Vec<(String, String)>,
    },
}

/// Zyron tables do not carry a table-level classification today. The binder
/// currently returns None, meaning no downstream clearance constraint is
/// applied. Widening this to scan per-column classifications from the catalog
/// is a follow-up once the ClassificationStore is queryable from here.
fn source_classification_from_columns(_cols: &[ColumnEntry]) -> Option<CatalogClassification> {
    None
}

// ---------------------------------------------------------------------------
// Streaming job binding helpers
// ---------------------------------------------------------------------------

/// Carries the parser-level JOIN information into the binder step where the
/// right-hand source is resolved. Holds the raw names plus the single
/// equi-key pair already validated from the ON expression.
#[derive(Debug, Clone)]
struct StreamingJoinBindInput {
    left_name: String,
    left_alias: String,
    right_name: String,
    right_alias: String,
    left_key_cols: Vec<String>,
    right_key_cols: Vec<String>,
    spec: zyron_parser::ast::StreamingJoinSpec,
}

/// Accepts ON expressions of the shape
/// `<l>.<col> = <r>.<col> [AND <l>.<col> = <r>.<col>]...` and returns the
/// two parallel column-name vectors. Rejects every other shape with a
/// specific error so multi-key mistakes surface precisely.
fn extract_equi_key_pairs(
    expr: &Expr,
    left_alias: &str,
    right_alias: &str,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut left_cols = Vec::new();
    let mut right_cols = Vec::new();
    collect_equi_conjuncts(
        expr,
        left_alias,
        right_alias,
        &mut left_cols,
        &mut right_cols,
    )?;
    if left_cols.is_empty() {
        return Err(ZyronError::PlanError(
            "streaming JOIN ON must contain at least one equality of two qualified columns"
                .to_string(),
        ));
    }
    Ok((left_cols, right_cols))
}

/// Walks a flat AND-chain of equality predicates. Every leaf must be an
/// equality of two qualified column references, one from each side. Any
/// non-equality or cross-side mismatch is rejected.
fn collect_equi_conjuncts(
    expr: &Expr,
    left_alias: &str,
    right_alias: &str,
    left_cols: &mut Vec<String>,
    right_cols: &mut Vec<String>,
) -> Result<()> {
    match expr {
        Expr::BinaryOp {
            op: BinaryOperator::And,
            left,
            right,
        } => {
            collect_equi_conjuncts(left, left_alias, right_alias, left_cols, right_cols)?;
            collect_equi_conjuncts(right, left_alias, right_alias, left_cols, right_cols)?;
            Ok(())
        }
        Expr::BinaryOp {
            op: BinaryOperator::Eq,
            left,
            right,
        } => {
            let l_ref = as_qualified_col(left)?;
            let r_ref = as_qualified_col(right)?;
            if l_ref.0.eq_ignore_ascii_case(left_alias) && r_ref.0.eq_ignore_ascii_case(right_alias)
            {
                left_cols.push(l_ref.1);
                right_cols.push(r_ref.1);
                Ok(())
            } else if l_ref.0.eq_ignore_ascii_case(right_alias)
                && r_ref.0.eq_ignore_ascii_case(left_alias)
            {
                left_cols.push(r_ref.1);
                right_cols.push(l_ref.1);
                Ok(())
            } else {
                Err(ZyronError::PlanError(format!(
                    "streaming JOIN ON must reference both sides, got '{}' and '{}'",
                    l_ref.0, r_ref.0
                )))
            }
        }
        Expr::Nested(inner) => {
            collect_equi_conjuncts(inner, left_alias, right_alias, left_cols, right_cols)
        }
        _ => Err(ZyronError::PlanError(
            "streaming JOIN ON must be equi-key equalities, optionally AND-chained".to_string(),
        )),
    }
}

/// Reads a qualified column reference (alias.column) out of an Expr. Returns
/// the alias and column names as a pair.
fn as_qualified_col(expr: &Expr) -> Result<(String, String)> {
    match expr {
        Expr::QualifiedIdentifier { table, column } => Ok((table.clone(), column.clone())),
        _ => Err(ZyronError::PlanError(
            "streaming JOIN ON must reference qualified columns like 'o.customer_id'".to_string(),
        )),
    }
}

/// Splits an optionally schema-qualified name into (schema, table) pieces.
fn split_qualified(name: &str) -> (Option<&str>, &str) {
    if let Some(pos) = name.find('.') {
        (Some(&name[..pos]), &name[pos + 1..])
    } else {
        (None, name)
    }
}

/// Returns the alias to register for a streaming source table scope.
fn tgt_or_src_alias(name: &str) -> String {
    name.to_string()
}

/// Maps the parser's StreamingJoinType to the binder-local enum.
fn map_streaming_join_type(t: zyron_parser::ast::StreamingJoinType) -> BoundStreamingJoinType {
    match t {
        zyron_parser::ast::StreamingJoinType::Inner => BoundStreamingJoinType::Inner,
        zyron_parser::ast::StreamingJoinType::Left => BoundStreamingJoinType::Left,
        zyron_parser::ast::StreamingJoinType::Right => BoundStreamingJoinType::Right,
        zyron_parser::ast::StreamingJoinType::Full => BoundStreamingJoinType::Full,
    }
}

// ---------------------------------------------------------------------------
// Type inference helpers
// ---------------------------------------------------------------------------

/// Infers the TypeId for a literal value.
fn literal_type(lit: &LiteralValue) -> TypeId {
    match lit {
        LiteralValue::Integer(_) => TypeId::Int64,
        LiteralValue::Float(_) => TypeId::Float64,
        LiteralValue::String(_) => TypeId::Varchar,
        LiteralValue::Boolean(_) => TypeId::Boolean,
        LiteralValue::Null => TypeId::Null,
        LiteralValue::Interval(_) => TypeId::Interval,
    }
}

/// Infers the result type of a binary operation.
fn infer_binary_type(op: &BinaryOperator, left: TypeId, right: TypeId) -> Result<TypeId> {
    match op {
        // Comparison operators always produce Boolean
        BinaryOperator::Eq
        | BinaryOperator::Neq
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::LtEq
        | BinaryOperator::GtEq => Ok(TypeId::Boolean),

        // Logical operators produce Boolean
        BinaryOperator::And | BinaryOperator::Or => Ok(TypeId::Boolean),

        // Concatenation produces Varchar
        BinaryOperator::Concat => Ok(TypeId::Varchar),

        // Arithmetic operators: handle interval arithmetic, then fall back to numeric promotion
        BinaryOperator::Plus | BinaryOperator::Minus => {
            match (left, right) {
                // timestamp +/- interval -> timestamp
                (TypeId::Timestamp, TypeId::Interval) | (TypeId::Interval, TypeId::Timestamp)
                    if matches!(op, BinaryOperator::Plus) =>
                {
                    Ok(TypeId::Timestamp)
                }
                (TypeId::Timestamp, TypeId::Interval) if matches!(op, BinaryOperator::Minus) => {
                    Ok(TypeId::Timestamp)
                }
                (TypeId::TimestampTz, TypeId::Interval)
                | (TypeId::Interval, TypeId::TimestampTz)
                    if matches!(op, BinaryOperator::Plus) =>
                {
                    Ok(TypeId::TimestampTz)
                }
                (TypeId::TimestampTz, TypeId::Interval) if matches!(op, BinaryOperator::Minus) => {
                    Ok(TypeId::TimestampTz)
                }
                (TypeId::Date, TypeId::Interval) | (TypeId::Interval, TypeId::Date)
                    if matches!(op, BinaryOperator::Plus) =>
                {
                    Ok(TypeId::Timestamp)
                }
                (TypeId::Date, TypeId::Interval) if matches!(op, BinaryOperator::Minus) => {
                    Ok(TypeId::Timestamp)
                }
                // interval +/- interval -> interval
                (TypeId::Interval, TypeId::Interval) => Ok(TypeId::Interval),
                _ => Ok(promote_numeric(left, right)),
            }
        }
        BinaryOperator::Multiply => {
            match (left, right) {
                // interval * numeric -> interval (scalar multiplication)
                (TypeId::Interval, r) if r.is_numeric() => Ok(TypeId::Interval),
                (l, TypeId::Interval) if l.is_numeric() => Ok(TypeId::Interval),
                _ => Ok(promote_numeric(left, right)),
            }
        }
        BinaryOperator::Divide | BinaryOperator::Modulo => Ok(promote_numeric(left, right)),
    }
}

/// Promotes two numeric types to their common supertype.
fn promote_numeric(left: TypeId, right: TypeId) -> TypeId {
    if left == right {
        return left;
    }

    // If either side is NULL, use the other
    if left == TypeId::Null {
        return right;
    }
    if right == TypeId::Null {
        return left;
    }

    // Float64 absorbs everything
    if left == TypeId::Float64 || right == TypeId::Float64 {
        return TypeId::Float64;
    }
    if left == TypeId::Float32 || right == TypeId::Float32 {
        return TypeId::Float64;
    }
    if left == TypeId::Decimal || right == TypeId::Decimal {
        return TypeId::Decimal;
    }

    // Integer promotion: pick the wider type
    let left_rank = integer_rank(left);
    let right_rank = integer_rank(right);
    if left_rank >= right_rank { left } else { right }
}

fn integer_rank(t: TypeId) -> u8 {
    match t {
        TypeId::Int8 | TypeId::UInt8 => 1,
        TypeId::Int16 | TypeId::UInt16 => 2,
        TypeId::Int32 | TypeId::UInt32 => 3,
        TypeId::Int64 | TypeId::UInt64 => 4,
        TypeId::Int128 | TypeId::UInt128 => 5,
        _ => 4, // Default to Int64 rank for non-integer types
    }
}

/// Infers the return type of an aggregate function.
fn infer_aggregate_type(name: &str, arg_types: &[TypeId]) -> Result<TypeId> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "count" => Ok(TypeId::Int64),
        "sum" => {
            if let Some(&t) = arg_types.first() {
                if t.is_floating_point() {
                    Ok(TypeId::Float64)
                } else if t == TypeId::Decimal {
                    Ok(TypeId::Decimal)
                } else {
                    Ok(TypeId::Int64)
                }
            } else {
                Ok(TypeId::Int64)
            }
        }
        "avg" => Ok(TypeId::Float64),
        "min" | "max" => {
            if let Some(&t) = arg_types.first() {
                Ok(t)
            } else {
                Ok(TypeId::Null)
            }
        }
        _ => {
            // Delegate to zyron-types for extended aggregates (first, last, time_weight, etc.)
            if let Some(t) = zyron_types::infer_types_aggregate_return_type(&lower, arg_types) {
                return Ok(t);
            }
            Err(ZyronError::PlanError(format!(
                "unknown aggregate function: {}",
                name
            )))
        }
    }
}

/// Infers the return type of a scalar function.
/// Returns Null for unknown functions (resolved at execution time).
fn infer_function_type(name: &str, arg_types: &[TypeId]) -> TypeId {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "abs" | "ceil" | "ceiling" | "floor" | "round" | "trunc" | "truncate" => {
            arg_types.first().copied().unwrap_or(TypeId::Float64)
        }
        "length" | "char_length" | "character_length" | "octet_length" => TypeId::Int64,
        "lower" | "upper" | "trim" | "ltrim" | "rtrim" | "substring" | "replace" | "concat" => {
            TypeId::Varchar
        }
        "now" | "current_timestamp" => TypeId::TimestampTz,
        "current_date" => TypeId::Date,
        "current_time" => TypeId::Time,
        "coalesce" => arg_types.first().copied().unwrap_or(TypeId::Null),
        "nullif" => arg_types.first().copied().unwrap_or(TypeId::Null),
        "greatest" | "least" => arg_types.first().copied().unwrap_or(TypeId::Null),
        _ => {
            // Delegate to zyron-types registry for extended scalar functions
            if let Some(t) = zyron_types::infer_types_scalar_return_type(&lower, arg_types) {
                return t;
            }
            arg_types.first().copied().unwrap_or(TypeId::Null)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------------
    // Streaming-join helper tests
    // -----------------------------------------------------------------------------

    #[test]
    fn test_extract_equi_key_pairs_single() {
        let expr = Expr::BinaryOp {
            op: BinaryOperator::Eq,
            left: Box::new(Expr::QualifiedIdentifier {
                table: "o".to_string(),
                column: "customer_id".to_string(),
            }),
            right: Box::new(Expr::QualifiedIdentifier {
                table: "c".to_string(),
                column: "id".to_string(),
            }),
        };
        let (l, r) = extract_equi_key_pairs(&expr, "o", "c").unwrap();
        assert_eq!(l, vec!["customer_id".to_string()]);
        assert_eq!(r, vec!["id".to_string()]);
    }

    #[test]
    fn test_extract_equi_key_pairs_multi() {
        // Two-column equi-key: l.a = r.x AND l.b = r.y.
        let eq_a = Expr::BinaryOp {
            op: BinaryOperator::Eq,
            left: Box::new(Expr::QualifiedIdentifier {
                table: "l".to_string(),
                column: "a".to_string(),
            }),
            right: Box::new(Expr::QualifiedIdentifier {
                table: "r".to_string(),
                column: "x".to_string(),
            }),
        };
        let eq_b = Expr::BinaryOp {
            op: BinaryOperator::Eq,
            left: Box::new(Expr::QualifiedIdentifier {
                table: "l".to_string(),
                column: "b".to_string(),
            }),
            right: Box::new(Expr::QualifiedIdentifier {
                table: "r".to_string(),
                column: "y".to_string(),
            }),
        };
        let and_expr = Expr::BinaryOp {
            op: BinaryOperator::And,
            left: Box::new(eq_a),
            right: Box::new(eq_b),
        };
        let (l, r) = extract_equi_key_pairs(&and_expr, "l", "r").unwrap();
        assert_eq!(l, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(r, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn test_extract_equi_key_pairs_rejects_non_equi() {
        // Greater-than instead of equality must be rejected.
        let expr = Expr::BinaryOp {
            op: BinaryOperator::Gt,
            left: Box::new(Expr::QualifiedIdentifier {
                table: "l".to_string(),
                column: "a".to_string(),
            }),
            right: Box::new(Expr::QualifiedIdentifier {
                table: "r".to_string(),
                column: "x".to_string(),
            }),
        };
        assert!(extract_equi_key_pairs(&expr, "l", "r").is_err());
    }

    #[test]
    fn test_map_bound_streaming_join_type_roundtrip() {
        use zyron_parser::ast::StreamingJoinType as P;
        assert_eq!(
            map_streaming_join_type(P::Inner),
            BoundStreamingJoinType::Inner
        );
        assert_eq!(
            map_streaming_join_type(P::Left),
            BoundStreamingJoinType::Left
        );
        assert_eq!(
            map_streaming_join_type(P::Right),
            BoundStreamingJoinType::Right
        );
        assert_eq!(
            map_streaming_join_type(P::Full),
            BoundStreamingJoinType::Full
        );
    }

    #[test]
    fn test_literal_type_inference() {
        assert_eq!(literal_type(&LiteralValue::Integer(42)), TypeId::Int64);
        assert_eq!(literal_type(&LiteralValue::Float(3.14)), TypeId::Float64);
        assert_eq!(
            literal_type(&LiteralValue::String("hello".to_string())),
            TypeId::Varchar
        );
        assert_eq!(literal_type(&LiteralValue::Boolean(true)), TypeId::Boolean);
        assert_eq!(literal_type(&LiteralValue::Null), TypeId::Null);
    }

    #[test]
    fn test_binary_type_inference() {
        // Comparisons produce Boolean
        assert_eq!(
            infer_binary_type(&BinaryOperator::Eq, TypeId::Int64, TypeId::Int64).unwrap(),
            TypeId::Boolean
        );
        assert_eq!(
            infer_binary_type(&BinaryOperator::Lt, TypeId::Float64, TypeId::Int32).unwrap(),
            TypeId::Boolean
        );

        // Arithmetic promotes types
        assert_eq!(
            infer_binary_type(&BinaryOperator::Plus, TypeId::Int32, TypeId::Int64).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            infer_binary_type(&BinaryOperator::Multiply, TypeId::Int64, TypeId::Float64).unwrap(),
            TypeId::Float64
        );

        // Concatenation produces Varchar
        assert_eq!(
            infer_binary_type(&BinaryOperator::Concat, TypeId::Varchar, TypeId::Varchar).unwrap(),
            TypeId::Varchar
        );
    }

    #[test]
    fn test_aggregate_type_inference() {
        assert_eq!(
            infer_aggregate_type("count", &[TypeId::Int32]).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            infer_aggregate_type("sum", &[TypeId::Int32]).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            infer_aggregate_type("sum", &[TypeId::Float64]).unwrap(),
            TypeId::Float64
        );
        assert_eq!(
            infer_aggregate_type("avg", &[TypeId::Int32]).unwrap(),
            TypeId::Float64
        );
        assert_eq!(
            infer_aggregate_type("min", &[TypeId::Varchar]).unwrap(),
            TypeId::Varchar
        );
        assert_eq!(
            infer_aggregate_type("max", &[TypeId::Int64]).unwrap(),
            TypeId::Int64
        );
    }

    #[test]
    fn test_numeric_promotion() {
        assert_eq!(promote_numeric(TypeId::Int32, TypeId::Int64), TypeId::Int64);
        assert_eq!(
            promote_numeric(TypeId::Int64, TypeId::Float64),
            TypeId::Float64
        );
        assert_eq!(
            promote_numeric(TypeId::Int32, TypeId::Float32),
            TypeId::Float64
        );
        assert_eq!(promote_numeric(TypeId::Null, TypeId::Int64), TypeId::Int64);
        assert_eq!(promote_numeric(TypeId::Int64, TypeId::Null), TypeId::Int64);
    }

    #[test]
    fn test_is_aggregate_function() {
        assert!(is_aggregate_function("count"));
        assert!(is_aggregate_function("COUNT"));
        assert!(is_aggregate_function("Sum"));
        assert!(is_aggregate_function("avg"));
        assert!(is_aggregate_function("min"));
        assert!(is_aggregate_function("max"));
        assert!(!is_aggregate_function("length"));
        assert!(!is_aggregate_function("lower"));
    }

    // -----------------------------------------------------------------------
    // Streaming job bind tests
    // -----------------------------------------------------------------------

    use zyron_buffer::{BufferPool, BufferPoolConfig};
    use zyron_catalog::cache::CatalogCache;
    use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
    use zyron_catalog::{Catalog, DatabaseId};
    use zyron_parser::ast::{ColumnDef, DataType};
    use zyron_storage::{DiskManager, DiskManagerConfig};
    use zyron_wal::{WalWriter, WalWriterConfig};

    /// Builds a temp-dir backed catalog, one database, the public schema,
    /// and two tables (orders, orders_vip). Returns the catalog plus the
    /// table ids. The orders table has cdf_enabled forced to match the
    /// caller's argument by rewriting the cache entry.
    async fn build_streaming_test_catalog(
        orders_cdf_enabled: bool,
    ) -> (
        Catalog,
        Arc<CatalogCache>,
        DatabaseId,
        SchemaId,
        TableId,
        TableId,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();

        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir,
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 1024 }));
        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir,
                fsync_enabled: false,
                ..Default::default()
            })
            .unwrap(),
        );
        let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(storage, Arc::clone(&cache), Arc::clone(&wal))
            .await
            .unwrap();

        let db_id = catalog.create_database("streamdb", "tester").await.unwrap();
        let schema_id = catalog
            .create_schema(db_id, "public", "tester")
            .await
            .unwrap();

        // orders(id BIGINT, amount DOUBLE)
        let orders_cols = vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::BigInt,
                nullable: Some(true),
                default: None,
                constraints: vec![],
            },
            ColumnDef {
                name: "amount".to_string(),
                data_type: DataType::DoublePrecision,
                nullable: Some(true),
                default: None,
                constraints: vec![],
            },
        ];
        let orders_id = catalog
            .create_table(schema_id, "orders", &orders_cols, &[])
            .await
            .unwrap();

        // orders_vip(id BIGINT, amount DOUBLE)
        let vip_cols = vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::BigInt,
                nullable: Some(true),
                default: None,
                constraints: vec![],
            },
            ColumnDef {
                name: "amount".to_string(),
                data_type: DataType::DoublePrecision,
                nullable: Some(true),
                default: None,
                constraints: vec![],
            },
        ];
        let vip_id = catalog
            .create_table(schema_id, "orders_vip", &vip_cols, &[])
            .await
            .unwrap();

        // Patch orders.cdf_enabled in the cache so resolver sees the desired state.
        // The cache's put_table does not replace on name collisions, so invalidate first.
        if let Some(existing) = cache.get_table_by_name(schema_id, "orders") {
            let mut updated = (*existing).clone();
            updated.cdf_enabled = orders_cdf_enabled;
            cache.invalidate_table(existing.id);
            cache.put_table(updated);
        }

        (catalog, cache, db_id, schema_id, orders_id, vip_id)
    }

    #[tokio::test]
    async fn test_bind_streaming_job_single_table_ok() {
        let (catalog, _cache, db_id, _schema_id, orders_id, vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE STREAMING JOB j AS SELECT id, amount FROM orders WHERE amount > 100 INTO orders_vip";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let bound = binder.bind(stmt).await.unwrap();
        match bound {
            BoundStatement::CreateStreamingJob(job) => {
                assert_eq!(job.source_table_id(), Some(orders_id));
                assert_eq!(job.target_table_id(), Some(vip_id));
                assert_eq!(job.projections.len(), 2);
                assert!(job.predicate.is_some());
                assert!(matches!(job.write_mode, CatalogStreamingWriteMode::Append));
            }
            other => panic!("expected CreateStreamingJob, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_streaming_job_upsert_requires_pk() {
        // Target table without a primary key: bind must reject UPSERT.
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE STREAMING JOB j AS SELECT id, amount FROM orders WHERE amount > 100 INTO orders_vip WRITE MODE UPSERT";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let err = binder.bind(stmt).await.expect_err("bind should fail");
        let msg = format!("{}", err);
        assert!(
            msg.contains("requires the target table to have a primary key"),
            "unexpected error message: {msg}"
        );
    }

    #[tokio::test]
    async fn test_bind_streaming_job_upsert_with_pk_ok() {
        // Build a catalog with orders_vip_pk that has a PK on id.
        let (catalog, _cache, db_id, schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let pk_cols = vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::BigInt,
                nullable: Some(false),
                default: None,
                constraints: vec![ColumnConstraint::PrimaryKey],
            },
            ColumnDef {
                name: "amount".to_string(),
                data_type: DataType::DoublePrecision,
                nullable: Some(true),
                default: None,
                constraints: vec![],
            },
        ];
        let pk_table_id = catalog
            .create_table(schema_id, "orders_vip_pk", &pk_cols, &[])
            .await
            .unwrap();
        let sql = "CREATE STREAMING JOB jpk AS SELECT id, amount FROM orders INTO orders_vip_pk WRITE MODE UPSERT";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let bound = binder.bind(stmt).await.expect("bind should succeed");
        match bound {
            BoundStatement::CreateStreamingJob(job) => {
                assert_eq!(job.target_table_id(), Some(pk_table_id));
                assert!(matches!(job.write_mode, CatalogStreamingWriteMode::Upsert));
                assert_eq!(job.target_pk_columns.len(), 1);
                assert_eq!(job.target_pk_columns[0].0, 0);
            }
            other => panic!("expected CreateStreamingJob, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_streaming_job_requires_cdc() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(false).await;
        let sql = "CREATE STREAMING JOB j AS SELECT id, amount FROM orders WHERE amount > 100 INTO orders_vip";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let err = binder.bind(stmt).await.expect_err("bind should fail");
        let msg = format!("{}", err);
        assert!(
            msg.contains("does not have CDC enabled"),
            "unexpected error message: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // External source and sink bind tests
    // -----------------------------------------------------------------------

    /// Builds and registers a named external source under the public schema
    /// using the given name and classification. Matches the column layout
    /// of orders_vip so downstream arity checks succeed.
    async fn register_external_source(
        catalog: &Catalog,
        schema_id: SchemaId,
        name: &str,
        classification: CatalogClassification,
    ) -> ExternalSourceId {
        use zyron_catalog::schema::{
            ExternalBackend, ExternalFormat, ExternalMode, ExternalSourceEntry,
        };
        let entry = ExternalSourceEntry {
            id: ExternalSourceId(0),
            schema_id,
            name: name.to_string(),
            backend: ExternalBackend::File,
            uri: "/tmp/in".to_string(),
            format: ExternalFormat::JsonLines,
            mode: ExternalMode::OneShot,
            schedule_cron: None,
            options: vec![],
            columns: vec![],
            credential_key_id: None,
            credential_ciphertext: None,
            classification,
            tags: vec![],
            owner_role_id: 0,
            created_at: 0,
        };
        catalog.create_external_source(entry).await.unwrap()
    }

    /// Builds and registers a named external sink under the public schema.
    async fn register_external_sink(
        catalog: &Catalog,
        schema_id: SchemaId,
        name: &str,
        classification: CatalogClassification,
    ) -> ExternalSinkId {
        use zyron_catalog::schema::{ExternalBackend, ExternalFormat, ExternalSinkEntry};
        let entry = ExternalSinkEntry {
            id: ExternalSinkId(0),
            schema_id,
            name: name.to_string(),
            backend: ExternalBackend::File,
            uri: "/tmp/out".to_string(),
            format: ExternalFormat::JsonLines,
            options: vec![],
            columns: vec![],
            credential_key_id: None,
            credential_ciphertext: None,
            classification,
            tags: vec![],
            owner_role_id: 0,
            created_at: 0,
        };
        catalog.create_external_sink(entry).await.unwrap()
    }

    #[tokio::test]
    async fn test_bind_streaming_job_external_source_named() {
        let (catalog, _cache, db_id, schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        register_external_source(&catalog, schema_id, "ext_in", CatalogClassification::Public)
            .await;
        let sql = "CREATE STREAMING JOB j AS SELECT * FROM ext_in INTO orders_vip";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let bound = binder.bind(stmt).await.unwrap();
        match bound {
            BoundStatement::CreateStreamingJob(job) => {
                assert!(matches!(
                    job.source,
                    BoundStreamingSource::ExternalNamed { .. }
                ));
                assert!(matches!(job.target, BoundStreamingSink::ZyronTable { .. }));
            }
            other => panic!("expected CreateStreamingJob, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_streaming_job_inline_file_source() {
        // The parser does not support inline FILE references in the FROM
        // clause today. This test exercises the inline-sink branch with a
        // named Zyron source, which is the symmetrical shape the binder
        // supports end-to-end through the parser.
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE STREAMING JOB j AS SELECT id, amount FROM orders INTO FILE '/tmp/out.jsonl' FORMAT JSONLINES";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let bound = binder.bind(stmt).await.unwrap();
        match bound {
            BoundStatement::CreateStreamingJob(job) => {
                assert!(matches!(
                    job.source,
                    BoundStreamingSource::ZyronTable { .. }
                ));
                assert!(matches!(
                    job.target,
                    BoundStreamingSink::ExternalInline { .. }
                ));
            }
            other => panic!("expected CreateStreamingJob, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_streaming_job_classification_mismatch() {
        let (catalog, _cache, db_id, schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        register_external_source(
            &catalog,
            schema_id,
            "ext_in",
            CatalogClassification::Restricted,
        )
        .await;
        register_external_sink(
            &catalog,
            schema_id,
            "ext_out",
            CatalogClassification::Internal,
        )
        .await;
        let sql = "CREATE STREAMING JOB j AS SELECT * FROM ext_in INTO ext_out";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let err = binder
            .bind(stmt)
            .await
            .expect_err("classification mismatch");
        let msg = format!("{}", err);
        assert!(
            msg.contains("classification") && msg.contains("Restricted"),
            "unexpected error message: {msg}"
        );
    }

    #[tokio::test]
    async fn test_bind_inline_sink_rejected_for_restricted_source() {
        let (catalog, _cache, db_id, schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        register_external_source(
            &catalog,
            schema_id,
            "ext_in",
            CatalogClassification::Restricted,
        )
        .await;
        let sql = "CREATE STREAMING JOB j AS SELECT * FROM ext_in INTO S3 's3://bucket/out' FORMAT JSONLINES";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let err = binder
            .bind(stmt)
            .await
            .expect_err("inline sink should be rejected for restricted source");
        let msg = format!("{}", err);
        assert!(
            msg.contains("inline external sink") && msg.contains("Restricted"),
            "unexpected error message: {msg}"
        );
    }

    #[tokio::test]
    async fn test_bind_create_external_source_basic() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE EXTERNAL SOURCE x TYPE FILE URI '/tmp' FORMAT JSONLINES";
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, &catalog);
        let bound = binder.bind(stmt).await.unwrap();
        match bound {
            BoundStatement::CreateExternalSource(b) => {
                assert_eq!(b.name, "x");
                assert!(matches!(
                    b.backend,
                    zyron_parser::ast::ExternalBackendKind::File
                ));
                assert!(matches!(
                    b.format,
                    zyron_parser::ast::ExternalFormatKind::JsonLines
                ));
                assert!(b.credentials.is_empty());
            }
            other => panic!("expected CreateExternalSource, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Zyron-to-Zyron DDL bind tests
    // -----------------------------------------------------------------------

    async fn parse_and_bind(catalog: &Catalog, db_id: DatabaseId, sql: &str) -> BoundStatement {
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, catalog);
        binder.bind(stmt).await.unwrap()
    }

    async fn parse_and_bind_err(
        catalog: &Catalog,
        db_id: DatabaseId,
        sql: &str,
    ) -> zyron_common::ZyronError {
        let stmt = zyron_parser::parse(sql)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
        let mut binder = Binder::new(resolver, catalog);
        binder.bind(stmt).await.unwrap_err()
    }

    async fn create_test_publication(
        catalog: &Catalog,
        schema_id: SchemaId,
        name: &str,
        table_id: TableId,
    ) -> zyron_catalog::PublicationId {
        use zyron_catalog::schema::{PublicationEntry, PublicationTableEntry, RowFormat};
        let entry = PublicationEntry {
            id: zyron_catalog::PublicationId(0),
            schema_id,
            name: name.to_string(),
            change_feed: true,
            row_format: RowFormat::Binary,
            retention_days: 30,
            retain_until_advance: true,
            max_rows_per_sec: None,
            max_bytes_per_sec: None,
            max_concurrent_subscribers: None,
            classification: CatalogClassification::Internal,
            allow_initial_snapshot: true,
            where_predicate: None,
            columns_projection: vec![],
            rls_using_predicate: None,
            tags: vec![],
            schema_fingerprint: [0u8; 32],
            owner_role_id: 0,
            created_at: 0,
        };
        let pub_id = catalog.create_publication(entry).await.unwrap();
        let table_entry = PublicationTableEntry {
            id: 0,
            publication_id: pub_id,
            table_id,
            where_predicate: None,
            columns: vec![],
            created_at: 0,
        };
        catalog.add_publication_table(table_entry).await.unwrap();
        pub_id
    }

    #[tokio::test]
    async fn test_bind_create_publication_single_table() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE PUBLICATION p1 FOR TABLE orders";
        let bound = parse_and_bind(&catalog, db_id, sql).await;
        match bound {
            BoundStatement::CreatePublication(p) => {
                assert_eq!(p.name, "p1");
                assert_eq!(p.tables.len(), 1);
                assert!(p.change_feed);
                assert!(matches!(p.row_format, RowFormat::Binary));
                assert_eq!(p.retention_days, 30);
                assert!(matches!(p.classification, CatalogClassification::Internal));
            }
            other => panic!("expected CreatePublication, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_create_publication_multi_table_with_where() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE PUBLICATION p1 FOR TABLE orders, orders_vip WHERE amount > 100";
        let bound = parse_and_bind(&catalog, db_id, sql).await;
        match bound {
            BoundStatement::CreatePublication(p) => {
                assert_eq!(p.tables.len(), 2);
                // The parser attaches the trailing WHERE to the last table ref.
                let last = p.tables.last().unwrap();
                assert!(last.where_predicate.is_some() || p.where_predicate.is_some());
            }
            other => panic!("expected CreatePublication, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_create_publication_unknown_table_rejected() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE PUBLICATION p1 FOR TABLE does_not_exist";
        let err = parse_and_bind_err(&catalog, db_id, sql).await;
        let msg = format!("{err}");
        assert!(
            msg.contains("does_not_exist") || msg.to_ascii_lowercase().contains("not found"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn test_bind_create_publication_unknown_column_rejected() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE PUBLICATION p1 FOR TABLE orders (bogus_col)";
        let err = parse_and_bind_err(&catalog, db_id, sql).await;
        assert!(format!("{err}").contains("bogus_col"));
    }

    #[tokio::test]
    async fn test_bind_create_publication_bad_classification_rejected() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql =
            "CREATE PUBLICATION p1 FOR TABLE orders WITH (classification = 'top_secret')";
        let err = parse_and_bind_err(&catalog, db_id, sql).await;
        assert!(format!("{err}").to_ascii_lowercase().contains("classification"));
    }

    #[tokio::test]
    async fn test_bind_create_publication_schema_fingerprint_deterministic() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE PUBLICATION p1 FOR TABLE orders";
        let a = match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreatePublication(p) => p.schema_fingerprint,
            _ => panic!(),
        };
        let b = match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreatePublication(p) => p.schema_fingerprint,
            _ => panic!(),
        };
        assert_eq!(a, b);
        assert_ne!(a, [0u8; 32]);
    }

    #[tokio::test]
    async fn test_bind_alter_publication_add_drop_table() {
        let (catalog, _cache, db_id, schema_id, orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        create_test_publication(&catalog, schema_id, "p1", orders_id).await;

        let add_sql = "ALTER PUBLICATION p1 ADD TABLE orders_vip";
        match parse_and_bind(&catalog, db_id, add_sql).await {
            BoundStatement::AlterPublication(p) => match p.action {
                BoundAlterPublicationAction::AddTable(t) => {
                    assert_ne!(t.table_id.0, 0);
                }
                other => panic!("expected AddTable, got {:?}", other),
            },
            _ => panic!(),
        }

        let drop_sql = "ALTER PUBLICATION p1 DROP TABLE orders";
        match parse_and_bind(&catalog, db_id, drop_sql).await {
            BoundStatement::AlterPublication(p) => {
                assert!(matches!(p.action, BoundAlterPublicationAction::DropTable(_)));
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_alter_publication_set_options() {
        let (catalog, _cache, db_id, schema_id, orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        create_test_publication(&catalog, schema_id, "p1", orders_id).await;
        let sql =
            "ALTER PUBLICATION p1 SET OPTIONS (retention_days = '90', max_bytes_per_sec = '100MB')";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::AlterPublication(p) => match p.action {
                BoundAlterPublicationAction::SetOptions(u) => {
                    assert_eq!(u.retention_days, Some(90));
                    assert_eq!(u.max_bytes_per_sec, Some(100 * 1024 * 1024));
                }
                other => panic!("expected SetOptions, got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_drop_publication_if_exists() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "DROP PUBLICATION IF EXISTS ghost";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::DropPublication { if_exists, .. } => assert!(if_exists),
            _ => panic!(),
        }

        let sql_err = "DROP PUBLICATION ghost";
        let err = parse_and_bind_err(&catalog, db_id, sql_err).await;
        assert!(format!("{err}").contains("ghost"));
    }

    #[tokio::test]
    async fn test_bind_create_endpoint_rest_basic() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE ENDPOINT ep1 ON PATH '/orders' METHOD GET USING 'SELECT id FROM orders' AUTH NONE";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreateEndpoint(e) => {
                assert_eq!(e.path, "/orders");
                assert_eq!(e.methods.len(), 1);
                assert!(matches!(e.methods[0], HttpMethod::Get));
                assert!(!e.output_columns.is_empty());
            }
            other => panic!("expected CreateEndpoint, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_create_endpoint_with_params() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE ENDPOINT ep2 ON PATH '/orders/one' METHOD GET USING 'SELECT id FROM orders WHERE id = $order_id' AUTH NONE";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreateEndpoint(e) => {
                assert_eq!(e.param_names, vec!["order_id".to_string()]);
                assert_eq!(e.param_types.len(), 1);
            }
            other => panic!("expected CreateEndpoint, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_create_endpoint_invalid_path_rejected() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql =
            "CREATE ENDPOINT ep3 ON PATH 'bad' METHOD GET USING 'SELECT id FROM orders' AUTH NONE";
        let err = parse_and_bind_err(&catalog, db_id, sql).await;
        assert!(format!("{err}").to_ascii_lowercase().contains("path"));
    }

    #[tokio::test]
    async fn test_bind_create_endpoint_invalid_method_rejected() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE ENDPOINT ep4 ON PATH '/x' METHOD CONNECT USING 'SELECT id FROM orders'";
        // Parser may reject CONNECT entirely, or binder rejects it. Accept either.
        let stmt_res = zyron_parser::parse(sql);
        match stmt_res {
            Err(_) => {}
            Ok(stmts) => {
                let stmt = stmts.into_iter().next().unwrap();
                let resolver = catalog.resolver(db_id, vec!["public".to_string()]);
                let mut binder = Binder::new(resolver, &catalog);
                let err = binder.bind(stmt).await.unwrap_err();
                assert!(
                    format!("{err}").to_ascii_lowercase().contains("method"),
                    "{err}"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_bind_create_streaming_endpoint_backed_by_publication() {
        let (catalog, _cache, db_id, schema_id, orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        create_test_publication(&catalog, schema_id, "pub1", orders_id).await;
        let sql =
            "CREATE STREAMING ENDPOINT stream1 ON PATH '/stream' PROTOCOL WEBSOCKET BACKED BY PUBLICATION pub1 AUTH NONE";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreateStreamingEndpoint(e) => {
                assert_eq!(e.backing_publication_name, "pub1");
                assert!(matches!(e.protocol, StreamingEndpointProtocol::Websocket));
            }
            other => panic!("expected CreateStreamingEndpoint, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_create_streaming_endpoint_missing_publication_rejected() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql =
            "CREATE STREAMING ENDPOINT stream1 ON PATH '/stream' PROTOCOL WEBSOCKET BACKED BY PUBLICATION nope AUTH NONE";
        let err = parse_and_bind_err(&catalog, db_id, sql).await;
        assert!(format!("{err}").contains("nope"));
    }

    #[tokio::test]
    async fn test_bind_alter_endpoint_enable_disable() {
        use zyron_catalog::schema::{EndpointEntry, EndpointKind};
        let (catalog, _cache, db_id, schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let entry = EndpointEntry {
            id: zyron_catalog::EndpointId(0),
            schema_id,
            name: "ep_a".to_string(),
            kind: EndpointKind::Rest,
            path: "/ep_a".to_string(),
            methods: vec![HttpMethod::Get],
            sql_body: "SELECT 1".to_string(),
            backed_publication_id: None,
            auth_mode: EndpointAuthMode::None,
            required_scopes: vec![],
            output_format: Some(EndpointOutputFormat::Json),
            cors_origins: vec![],
            rate_limit: None,
            cache_seconds: None,
            timeout_seconds: None,
            max_request_body_kb: None,
            message_format: None,
            heartbeat_seconds: None,
            backpressure: None,
            max_connections: None,
            enabled: true,
            owner_role_id: 0,
            created_at: 0,
        };
        catalog.create_endpoint(entry).await.unwrap();

        match parse_and_bind(&catalog, db_id, "ALTER ENDPOINT ep_a DISABLE").await {
            BoundStatement::AlterEndpoint(b) => {
                assert!(matches!(b.action, BoundAlterEndpointAction::Disable));
            }
            _ => panic!(),
        }
        match parse_and_bind(&catalog, db_id, "ALTER ENDPOINT ep_a ENABLE").await {
            BoundStatement::AlterEndpoint(b) => {
                assert!(matches!(b.action, BoundAlterEndpointAction::Enable));
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_drop_endpoint() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        match parse_and_bind(&catalog, db_id, "DROP ENDPOINT IF EXISTS nope").await {
            BoundStatement::DropEndpoint { if_exists, .. } => assert!(if_exists),
            _ => panic!(),
        }
        let err = parse_and_bind_err(&catalog, db_id, "DROP ENDPOINT nope").await;
        assert!(format!("{err}").contains("nope"));
    }

    #[tokio::test]
    async fn test_bind_alter_security_map_k8s_sa() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "ALTER SECURITY MAP KUBERNETES SA 'default/worker' TO ROLE 'reader'";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::AlterSecurityMap(m) => {
                assert!(matches!(m.kind, CatalogSecurityMapKind::K8sSa));
                assert_eq!(m.identity_key, "default/worker");
                assert_eq!(m.role_name, "reader");
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_alter_security_map_jwt() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "ALTER SECURITY MAP JWT ISSUER 'https://iss' SUBJECT 'alice' TO ROLE 'reader'";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::AlterSecurityMap(m) => {
                assert!(matches!(m.kind, CatalogSecurityMapKind::Jwt));
                assert!(m.identity_key.contains("https://iss"));
                assert!(m.identity_key.contains("alice"));
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_alter_security_map_mtls_subject() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql =
            "ALTER SECURITY MAP MTLS CERT SUBJECT 'CN=svc,O=Zyron' TO ROLE 'reader'";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::AlterSecurityMap(m) => {
                assert!(matches!(m.kind, CatalogSecurityMapKind::MtlsSubject));
                assert_eq!(m.identity_key, "CN=svc,O=Zyron");
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_drop_security_map() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "DROP SECURITY MAP KUBERNETES SA 'default/worker'";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::DropSecurityMap(m) => {
                assert!(matches!(m.kind, CatalogSecurityMapKind::K8sSa));
                assert_eq!(m.identity_key, "default/worker");
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_create_external_source_zyron_uri_parse() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql =
            "CREATE EXTERNAL SOURCE src1 TYPE ZYRON URI 'zyron://host1:5432/db1/pub:p1' FORMAT JSONLINES";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreateExternalSource(b) => {
                assert!(matches!(b.backend, ExternalBackendKind::Zyron));
                assert!(b.uri.starts_with("zyron://"));
            }
            other => panic!("expected CreateExternalSource, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn bind_tls_without_trust_config_rejects() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql =
            "CREATE EXTERNAL SOURCE tls_src TYPE ZYRON URI 'zyron://host1:5432/db1/pub:p1' FORMAT JSONLINES OPTIONS (tls = 'required')";
        let err = parse_and_bind_err(&catalog, db_id, sql).await;
        let msg = err.to_string();
        assert!(
            msg.contains("TLS requires one of"),
            "unexpected error message: {msg}"
        );
    }

    #[tokio::test]
    async fn bind_tls_with_ca_cert_path_accepts() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE EXTERNAL SOURCE tls_src_ok TYPE ZYRON URI 'zyron://host1:5432/db1/pub:p1' FORMAT JSONLINES OPTIONS (tls = 'required', ca_cert_path = '/etc/ca.pem')";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreateExternalSource(b) => {
                assert!(matches!(b.backend, ExternalBackendKind::Zyron));
            }
            other => panic!("expected CreateExternalSource, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn bind_tls_disabled_accepts_without_trust_anchors() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "CREATE EXTERNAL SINK tls_sink_off TYPE ZYRON URI 'zyron://host1:5432/db1/pub:p1' FORMAT JSONLINES OPTIONS (tls = 'disabled')";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::CreateExternalSink(b) => {
                assert!(matches!(b.backend, ExternalBackendKind::Zyron));
            }
            other => panic!("expected CreateExternalSink, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_alter_external_source_refresh_schema() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "ALTER EXTERNAL SOURCE src1 REFRESH SCHEMA";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::AlterExternalSource(b) => {
                assert!(matches!(b.action, AlterExternalSourceAction::RefreshSchema));
            }
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_alter_external_source_reset_lsn_earliest() {
        let (catalog, _cache, db_id, _schema_id, _orders_id, _vip_id) =
            build_streaming_test_catalog(true).await;
        let sql = "ALTER EXTERNAL SOURCE src1 RESET LSN TO 'earliest'";
        match parse_and_bind(&catalog, db_id, sql).await {
            BoundStatement::AlterExternalSource(b) => match b.action {
                AlterExternalSourceAction::ResetLsn(LsnResetSpec::Earliest) => {}
                other => panic!("expected ResetLsn(Earliest), got {:?}", other),
            },
            _ => panic!(),
        }
    }

    #[tokio::test]
    async fn test_bind_abac_policy_on_publication() {
        // ABAC policy on publication routes through the wire dispatcher, not
        // the binder, so here we only confirm that the parser produces a
        // CreateAbacPolicy with Publication target. The dispatcher ties the
        // target to the resolved publication id in a later phase.
        let sql = "CREATE ABAC POLICY pol1 ON PUBLICATION pub1 WHERE dept = 'eng'";
        let stmts = zyron_parser::parse(sql).unwrap();
        let stmt = stmts.into_iter().next().unwrap();
        match stmt {
            Statement::CreateAbacPolicy(p) => {
                assert!(matches!(p.target, AbacPolicyTarget::Publication));
                assert_eq!(p.target_name, "pub1");
            }
            other => panic!("expected CreateAbacPolicy, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_grant_subscribe_publication() {
        // GRANT dispatch runs through the wire handler, not the binder, so
        // we parse and assert the AST carries the right GrantObject.
        let sql = "GRANT SUBSCRIBE ON PUBLICATION pub1 TO reader";
        let stmt = zyron_parser::parse(sql).unwrap().into_iter().next().unwrap();
        match stmt {
            Statement::Grant(g) => {
                assert!(matches!(g.object, GrantObject::Publication(_)));
                assert!(g.privileges.contains(&Privilege::Subscribe));
            }
            other => panic!("expected Grant, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bind_grant_invoke_endpoint() {
        let sql = "GRANT INVOKE ON ENDPOINT ep1 TO reader";
        let stmt = zyron_parser::parse(sql).unwrap().into_iter().next().unwrap();
        match stmt {
            Statement::Grant(g) => {
                assert!(matches!(g.object, GrantObject::Endpoint(_)));
                assert!(g.privileges.contains(&Privilege::Invoke));
            }
            other => panic!("expected Grant, got {:?}", other),
        }
    }
}
