//! Abstract syntax tree (AST) node definitions for parsed SQL statements.

use zyron_common::TypeId;

// ---------------------------------------------------------------------------
// Top-level statement
// ---------------------------------------------------------------------------

/// A parsed SQL statement.
/// All variants are boxed to keep the enum size at 16 bytes (discriminant + pointer)
/// instead of 384 bytes (driven by SelectStatement at 304 bytes inline).
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Select(Box<SelectStatement>),
    Insert(Box<InsertStatement>),
    Update(Box<UpdateStatement>),
    Delete(Box<DeleteStatement>),
    CreateTable(Box<CreateTableStatement>),
    DropTable(Box<DropTableStatement>),
    AlterTable(Box<AlterTableStatement>),
    CreateIndex(Box<CreateIndexStatement>),
    DropIndex(Box<DropIndexStatement>),
    CreateView(Box<CreateViewStatement>),
    DropView(Box<DropViewStatement>),
    CreateSchema(Box<CreateSchemaStatement>),
    DropSchema(Box<DropSchemaStatement>),
    CreateSequence(Box<CreateSequenceStatement>),
    DropSequence(Box<DropSequenceStatement>),
    Truncate(Box<TruncateStatement>),
    Begin(Box<BeginStatement>),
    Commit(Box<CommitStatement>),
    Rollback(Box<RollbackStatement>),
    Savepoint(Box<SavepointStatement>),
    ReleaseSavepoint(Box<ReleaseSavepointStatement>),
    Explain(Box<ExplainStatement>),
    Grant(Box<GrantStatement>),
    Revoke(Box<RevokeStatement>),
    Vacuum(Box<VacuumStatement>),
    Reindex(Box<ReindexStatement>),
    SetVariable(Box<SetVariableStatement>),
    Show(Box<ShowStatement>),
    Copy(Box<CopyStatement>),
    Merge(Box<MergeStatement>),
    Prepare(Box<PrepareStatement>),
    Execute(Box<ExecuteStatement>),
    Deallocate(Box<DeallocateStatement>),
    Listen(Box<ListenStatement>),
    Notify(Box<NotifyStatement>),
    DeclareCursor(Box<DeclareCursorStatement>),
    FetchCursor(Box<FetchCursorStatement>),
    CloseCursor(Box<CloseCursorStatement>),
    CommentOn(Box<CommentOnStatement>),
    AlterIndex(Box<AlterIndexStatement>),
    AlterSequence(Box<AlterSequenceStatement>),
    AlterView(Box<AlterViewStatement>),
    CreateMaterializedView(Box<CreateMaterializedViewStatement>),
    DropMaterializedView(Box<DropMaterializedViewStatement>),
    RefreshMaterializedView(Box<RefreshMaterializedViewStatement>),
    DoBlock(Box<DoBlockStatement>),
    Checkpoint(Box<CheckpointStatement>),
    /// Standalone VALUES query: VALUES (row1), (row2), ...
    ValuesQuery(Box<ValuesQueryStatement>),
    /// ALTER TABLE ... SET TTL / DROP TTL
    AlterTableTtl(Box<AlterTableTtlStatement>),
    /// CREATE SCHEDULE name EVERY interval DO statement
    CreateSchedule(Box<CreateScheduleStatement>),
    /// DROP SCHEDULE name
    DropSchedule(Box<DropScheduleStatement>),
    /// PAUSE SCHEDULE name
    PauseSchedule(Box<PauseScheduleStatement>),
    /// RESUME SCHEDULE name
    ResumeSchedule(Box<ResumeScheduleStatement>),
    /// OPTIMIZE TABLE name
    OptimizeTable(Box<OptimizeTableStatement>),
    /// CREATE USER name WITH PASSWORD 'pw' [options]
    CreateUser(Box<CreateUserStatement>),
    /// ALTER USER name [SET PASSWORD 'pw'] [RENAME TO new_name] [options]
    AlterUser(Box<AlterUserStatement>),
    /// DROP USER [IF EXISTS] name
    DropUser(Box<DropUserStatement>),
    /// CREATE ROLE name [WITH options]
    CreateRole(Box<CreateRoleStatement>),
    /// ALTER ROLE name [options]
    AlterRole(Box<AlterRoleStatement>),
    /// DROP ROLE [IF EXISTS] name
    DropRole(Box<DropRoleStatement>),
    /// CREATE PIPELINE name AS (STAGE ...)
    CreatePipeline(Box<CreatePipelineStatement>),
    /// RUN PIPELINE name [STAGE stage_name]
    RunPipeline(Box<RunPipelineStatement>),
    /// DROP PIPELINE [IF EXISTS] name
    DropPipeline(Box<DropPipelineStatement>),
    /// ARCHIVE TABLE name WHERE expr TO 'path'
    ArchiveTable(Box<ArchiveTableStatement>),
    /// RESTORE TABLE name FROM 'path' [INTO target]
    RestoreTable(Box<RestoreTableStatement>),
    /// ALTER TABLE name SET (key = value, ...)
    AlterTableOptions(Box<AlterTableOptionsStatement>),
    /// ALTER TABLE name ADD EXPECTATION name EXPECT expr ON VIOLATION action
    AddExpectation(Box<AddExpectationStatement>),
    /// ALTER TABLE name DROP EXPECTATION name
    DropExpectation(Box<DropExpectationStatement>),
    /// ALTER TABLE name ENABLE feature
    EnableFeature(Box<EnableFeatureStatement>),
    /// ALTER TABLE name DISABLE feature
    DisableFeature(Box<DisableFeatureStatement>),
    /// CREATE FULLTEXT INDEX name ON table(columns) [WITH (options)]
    CreateFulltextIndex(Box<CreateFulltextIndexStatement>),
    /// CREATE VECTOR INDEX name ON table(column) [WITH (options)]
    CreateVectorIndex(Box<CreateVectorIndexStatement>),
    CreateSpatialIndex(Box<CreateSpatialIndexStatement>),
    /// ALTER SYSTEM SET name = value
    AlterSystemSet(Box<AlterSystemSetStatement>),
    /// ANALYZE [table_name]
    Analyze(Box<AnalyzeStatement>),
    /// CREATE BRANCH name [FROM name] [AT VERSION expr]
    CreateBranch(Box<CreateBranchStatement>),
    /// MERGE BRANCH name INTO name
    MergeBranch(Box<MergeBranchStatement>),
    /// DROP BRANCH [IF EXISTS] name
    DropBranch(Box<DropBranchStatement>),
    /// USE BRANCH name
    UseBranch(Box<UseBranchStatement>),
    /// CREATE VERSION name ON table [AS OF VERSION expr]
    CreateVersion(Box<CreateVersionStatement>),
    /// CREATE REPLICATION SLOT name PLUGIN 'plugin_name'
    CreateReplicationSlot(Box<CreateReplicationSlotStatement>),
    /// DROP REPLICATION SLOT name
    DropReplicationSlot(Box<DropReplicationSlotStatement>),
    /// CREATE CDC STREAM name ON table TO sink WITH (...)
    CreateCdcStream(Box<CreateCdcStreamStatement>),
    /// DROP CDC STREAM name
    DropCdcStream(Box<DropCdcStreamStatement>),
    /// CREATE CDC INGEST name FROM source INTO table WITH (...)
    CreateCdcIngest(Box<CreateCdcIngestStatement>),
    /// DROP CDC INGEST name
    DropCdcIngest(Box<DropCdcIngestStatement>),
    /// CREATE STREAMING JOB name AS <select> INTO target [WRITE MODE APPEND|UPSERT]
    CreateStreamingJob(Box<CreateStreamingJobStatement>),
    /// DROP STREAMING JOB [IF EXISTS] name
    DropStreamingJob(Box<DropStreamingJobStatement>),
    /// ALTER STREAMING JOB name PAUSE | RESUME
    AlterStreamingJob(Box<AlterStreamingJobStatement>),
    /// CREATE PUBLICATION name FOR TABLE t1, t2, ...
    CreatePublication(Box<CreatePublicationStatement>),
    /// ALTER PUBLICATION name ADD/DROP TABLE t
    AlterPublication(Box<AlterPublicationStatement>),
    /// DROP PUBLICATION name
    DropPublication(Box<DropPublicationStatement>),
    /// CREATE TRIGGER name BEFORE/AFTER/INSTEAD OF event ON table ...
    CreateTrigger(Box<CreateTriggerStatement>),
    /// DROP TRIGGER [IF EXISTS] name ON table [CASCADE|RESTRICT]
    DropTrigger(Box<DropTriggerStatement>),
    /// CREATE [OR REPLACE] FUNCTION name(...) RETURNS type ...
    CreateFunction(Box<CreateFunctionStatement>),
    /// DROP FUNCTION [IF EXISTS] name [CASCADE|RESTRICT]
    DropFunction(Box<DropFunctionStatement>),
    /// CREATE AGGREGATE name(...) (SFUNC = ..., STYPE = ..., ...)
    CreateAggregate(Box<CreateAggregateStatement>),
    /// DROP AGGREGATE [IF EXISTS] name [CASCADE|RESTRICT]
    DropAggregate(Box<DropAggregateStatement>),
    /// CREATE [OR REPLACE] PROCEDURE name(...) AS $$ ... $$
    CreateProcedure(Box<CreateProcedureStatement>),
    /// DROP PROCEDURE [IF EXISTS] name [CASCADE|RESTRICT]
    DropProcedure(Box<DropProcedureStatement>),
    /// CALL procedure_name(args...)
    Call(Box<CallStatement>),
    /// CREATE EVENT HANDLER name WHEN event EXECUTE FUNCTION func
    CreateEventHandler(Box<CreateEventHandlerStatement>),
    /// DROP EVENT HANDLER [IF EXISTS] name
    DropEventHandler(Box<DropEventHandlerStatement>),
    /// CREATE GRAPH SCHEMA name (NODE Label(...), EDGE Label FROM X TO Y (...))
    CreateGraphSchema(Box<CreateGraphSchemaStatement>),
    /// DROP GRAPH SCHEMA [IF EXISTS] name
    DropGraphSchema(Box<DropGraphSchemaStatement>),
    /// CREATE EXTERNAL SOURCE name TYPE backend URI uri FORMAT fmt ...
    CreateExternalSource(Box<CreateExternalSourceStatement>),
    /// CREATE EXTERNAL SINK name TYPE backend URI uri FORMAT fmt ...
    CreateExternalSink(Box<CreateExternalSinkStatement>),
    /// DROP EXTERNAL SOURCE [IF EXISTS] name
    DropExternalSource(Box<DropExternalSourceStatement>),
    /// DROP EXTERNAL SINK [IF EXISTS] name
    DropExternalSink(Box<DropExternalSinkStatement>),
    /// ALTER EXTERNAL SOURCE name ...
    AlterExternalSource(Box<AlterExternalSourceStatement>),
    /// ALTER EXTERNAL SINK name ...
    AlterExternalSink(Box<AlterExternalSinkStatement>),

    // Zyron-to-Zyron data plane DDL.
    /// CREATE ENDPOINT name ON PATH '/...' METHOD ... USING <sql> ...
    CreateEndpoint(Box<CreateEndpointStatement>),
    /// CREATE STREAMING ENDPOINT name ON PATH '/...' PROTOCOL ws|sse ...
    CreateStreamingEndpoint(Box<CreateStreamingEndpointStatement>),
    /// ALTER ENDPOINT name (ENABLE | DISABLE | SET OPTIONS (...))
    AlterEndpoint(Box<AlterEndpointStatement>),
    /// DROP ENDPOINT [IF EXISTS] name
    DropEndpoint(Box<DropEndpointStatement>),
    /// ALTER SECURITY MAP (KUBERNETES SA 'ns/sa' | JWT ISSUER '..' SUBJECT '..' |
    ///   MTLS CERT SUBJECT '..' | MTLS CERT FINGERPRINT '..') TO ROLE '..'
    AlterSecurityMap(Box<AlterSecurityMapStatement>),
    /// DROP SECURITY MAP ...
    DropSecurityMap(Box<DropSecurityMapStatement>),
    /// TAG PUBLICATION name WITH 'tag1', 'tag2'
    TagPublication(Box<TagPublicationStatement>),
    /// UNTAG PUBLICATION name 'tag'
    UntagPublication(Box<UntagPublicationStatement>),
    /// CREATE ABAC POLICY name ON <object-type> <name> WHERE <expr>
    CreateAbacPolicy(Box<CreateAbacPolicyStatement>),
}

// ---------------------------------------------------------------------------
// DML statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    pub with: Option<WithClause>,
    pub distinct: bool,
    pub distinct_on: Vec<Expr>,
    pub projections: Vec<SelectItem>,
    pub from: Vec<TableRef>,
    pub where_clause: Option<Box<Expr>>,
    pub group_by: Vec<Expr>,
    pub group_by_sets: Option<GroupBySets>,
    pub having: Option<Box<Expr>>,
    pub qualify: Option<Box<Expr>>,
    pub set_ops: Vec<SetOpItem>,
    pub order_by: Vec<OrderByExpr>,
    pub limit: Option<Box<Expr>>,
    pub offset: Option<Box<Expr>>,
    pub fetch: Option<FetchFirst>,
    pub for_clause: Option<ForClause>,
}

/// WITH [RECURSIVE] cte_name [(columns)] AS (select), ...
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    pub recursive: bool,
    pub ctes: Vec<Cte>,
}

/// A single common table expression: name [(columns)] AS (select)
#[derive(Debug, Clone, PartialEq)]
pub struct Cte {
    pub name: String,
    pub columns: Vec<String>,
    pub query: Box<SelectStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// An expression with an optional alias: `expr [AS alias]`.
    Expr(Expr, Option<String>),
    /// Unqualified wildcard: `*`.
    Wildcard,
    /// Qualified wildcard: `table.*`.
    QualifiedWildcard(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct InsertStatement {
    pub table: String,
    pub columns: Vec<String>,
    pub source: InsertSource,
    pub on_conflict: Option<OnConflict>,
    pub returning: Option<Vec<SelectItem>>,
}

/// Source of rows for an INSERT statement.
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// INSERT INTO ... VALUES (row1), (row2), ...
    Values(Vec<Vec<Expr>>),
    /// INSERT INTO ... SELECT ...
    Query(Box<SelectStatement>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStatement {
    pub table: String,
    pub assignments: Vec<Assignment>,
    pub where_clause: Option<Box<Expr>>,
    pub returning: Option<Vec<SelectItem>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub column: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStatement {
    pub table: String,
    pub where_clause: Option<Box<Expr>>,
    pub returning: Option<Vec<SelectItem>>,
}

// ---------------------------------------------------------------------------
// DDL statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub columns: Vec<ColumnDef>,
    pub constraints: Vec<TableConstraint>,
    pub options: Vec<TableOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropTableStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStatement {
    pub name: String,
    pub operation: AlterTableOperation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterTableOperation {
    /// ALTER TABLE ... ADD [COLUMN] <column_def>
    AddColumn(ColumnDef),
    /// ALTER TABLE ... DROP [COLUMN] name [IF EXISTS]
    DropColumn { name: String, if_exists: bool },
    /// ALTER TABLE ... RENAME [COLUMN] old_name TO new_name
    RenameColumn { old_name: String, new_name: String },
    /// ALTER TABLE ... ALTER [COLUMN] name SET DEFAULT <expr>
    AlterColumnSetDefault { column: String, default: Expr },
    /// ALTER TABLE ... ALTER [COLUMN] name DROP DEFAULT
    AlterColumnDropDefault { column: String },
    /// ALTER TABLE ... ALTER [COLUMN] name SET NOT NULL
    AlterColumnSetNotNull { column: String },
    /// ALTER TABLE ... ALTER [COLUMN] name DROP NOT NULL
    AlterColumnDropNotNull { column: String },
    /// ALTER TABLE ... ALTER [COLUMN] name TYPE <data_type>
    AlterColumnSetType { column: String, data_type: DataType },
    /// ALTER TABLE ... ADD CONSTRAINT ...
    AddConstraint(TableConstraint),
    /// ALTER TABLE ... DROP CONSTRAINT name [IF EXISTS]
    DropConstraint { name: String, if_exists: bool },
    /// ALTER TABLE ... RENAME TO new_name
    RenameTable { new_name: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStatement {
    pub name: String,
    pub table: String,
    pub columns: Vec<OrderByExpr>,
    pub unique: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TruncateStatement {
    pub table: String,
}

// ---------------------------------------------------------------------------
// Transaction control statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct BeginStatement {}

#[derive(Debug, Clone, PartialEq)]
pub struct CommitStatement {}

/// ROLLBACK or ROLLBACK TO [SAVEPOINT] name.
#[derive(Debug, Clone, PartialEq)]
pub struct RollbackStatement {
    pub savepoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavepointStatement {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReleaseSavepointStatement {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Query analysis
// ---------------------------------------------------------------------------

/// EXPLAIN [(option_list)] <statement>
/// Supports: EXPLAIN ANALYZE stmt, EXPLAIN (ANALYZE, COSTS, BUFFERS, TIMING, FORMAT TEXT|JSON|YAML) stmt
#[derive(Debug, Clone, PartialEq)]
pub struct ExplainStatement {
    pub analyze: bool,
    pub costs: bool,
    pub buffers: bool,
    pub timing: bool,
    pub format: Option<String>,
    pub statement: Box<Statement>,
}

// ---------------------------------------------------------------------------
// Set operations
// ---------------------------------------------------------------------------

/// A single set operation in a chain: UNION/INTERSECT/EXCEPT [ALL] SELECT ...
#[derive(Debug, Clone, PartialEq)]
pub struct SetOpItem {
    pub op: SetOpType,
    pub all: bool,
    pub right: Box<SelectStatement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    Union,
    Intersect,
    Except,
}

// ---------------------------------------------------------------------------
// ON CONFLICT (UPSERT)
// ---------------------------------------------------------------------------

/// ON CONFLICT [(columns)] action
#[derive(Debug, Clone, PartialEq)]
pub struct OnConflict {
    pub columns: Vec<String>,
    pub action: ConflictAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConflictAction {
    DoNothing,
    DoUpdate(Vec<Assignment>),
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateViewStatement {
    pub name: String,
    pub columns: Vec<String>,
    pub query: Box<SelectStatement>,
    pub or_replace: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropViewStatement {
    pub name: String,
    pub if_exists: bool,
}

// ---------------------------------------------------------------------------
// DCL (permissions)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct GrantStatement {
    pub privileges: Vec<Privilege>,
    /// Target table name when granting on a table.
    /// Empty when the object is not a table.
    pub on_table: String,
    /// Structured object reference. Tables still populate on_table for
    /// backward compatibility and object == GrantObject::Table(name).
    pub object: GrantObject,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RevokeStatement {
    pub privileges: Vec<Privilege>,
    pub on_table: String,
    pub object: GrantObject,
    pub from: String,
}

/// Target of a GRANT or REVOKE. Supports individual tables, specific
/// publications, pattern-matched publications, tag-matched publications,
/// and individual endpoints.
#[derive(Debug, Clone, PartialEq)]
pub enum GrantObject {
    Table(String),
    Publication(String),
    PublicationsLike(String),
    PublicationsTagged(String),
    Endpoint(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Privilege {
    Select,
    Insert,
    Update,
    Delete,
    CreateIndex,
    DropIndex,
    Reindex,
    AlterIndex,
    Subscribe,
    Invoke,
    All,
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateSchemaStatement {
    pub name: String,
    pub if_not_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropSchemaStatement {
    pub name: String,
    pub if_exists: bool,
    pub cascade: bool,
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateSequenceStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub increment: Option<i64>,
    pub min_value: Option<i64>,
    pub max_value: Option<i64>,
    pub start: Option<i64>,
    pub cache: Option<i64>,
    pub cycle: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropSequenceStatement {
    pub name: String,
    pub if_exists: bool,
}

// ---------------------------------------------------------------------------
// Maintenance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct VacuumStatement {
    pub table: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReindexStatement {
    pub target: ReindexTarget,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReindexTarget {
    Table(String),
    Index(String),
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// SET name = value or SET name TO value
#[derive(Debug, Clone, PartialEq)]
pub struct SetVariableStatement {
    pub name: String,
    pub value: Expr,
}

/// SHOW name
#[derive(Debug, Clone, PartialEq)]
pub struct ShowStatement {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Bulk copy
// ---------------------------------------------------------------------------

/// COPY statement in one of three shapes. STDIN / STDOUT / local-file
/// targets are represented as `CopyExternal::Stdio` or `CopyExternal::LocalFile`
/// inside `CopyKind::IntoTable` or `CopyKind::FromTable`, so the wire layer
/// recognises the PostgreSQL simple-query forms while the binder and
/// executor distinguish external endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct CopyStatement {
    pub kind: CopyKind,
    pub options: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CopyKind {
    /// External endpoint (or STDIN, or a local file path) into a Zyron table.
    IntoTable {
        table: String,
        columns: Vec<String>,
        source: CopyExternal,
    },
    /// Zyron table into an external endpoint (or STDOUT, or a local file path).
    FromTable {
        table: String,
        columns: Vec<String>,
        sink: CopyExternal,
    },
    /// External endpoint into another external endpoint with no Zyron table
    /// on either side. Runs as a one-shot streaming copy in the session.
    ExternalToExternal {
        source: CopyExternal,
        sink: CopyExternal,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum CopyExternal {
    /// Fully inline external endpoint: BACKEND 'uri' FORMAT fmt
    /// [CREDENTIALS (k=v, ...)]. The credentials list is consumed unsealed.
    Inline {
        backend: ExternalBackendKind,
        uri: String,
        format: ExternalFormatKind,
        credentials: Vec<(String, String)>,
    },
    /// Catalog-registered external source or sink referenced by name.
    Named(String),
    /// PostgreSQL STDIN or STDOUT, selected by surrounding direction.
    Stdio,
    /// Bare local file path with no backend or format keywords. The wire
    /// layer rejects this for transport but the parser retains it so
    /// `COPY t FROM '/tmp/data.csv'` where the path alone implies local file
    /// continues to bind.
    LocalFile(String),
}

// ---------------------------------------------------------------------------
// FOR UPDATE/SHARE (row locking)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ForClause {
    pub lock_type: ForLockType,
    pub tables: Vec<String>,
    pub wait: ForWait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForLockType {
    Update,
    Share,
    NoKeyUpdate,
    KeyShare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForWait {
    Wait,
    Nowait,
    SkipLocked,
}

// ---------------------------------------------------------------------------
// FETCH FIRST/NEXT
// ---------------------------------------------------------------------------

/// FETCH FIRST/NEXT n ROWS ONLY (SQL standard alternative to LIMIT)
#[derive(Debug, Clone, PartialEq)]
pub struct FetchFirst {
    pub count: Box<Expr>,
    pub percent: bool,
    pub with_ties: bool,
}

// ---------------------------------------------------------------------------
// MERGE
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct MergeStatement {
    pub target: String,
    pub source: TableRef,
    pub on: Expr,
    pub clauses: Vec<MergeClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MergeClause {
    WhenMatched {
        condition: Option<Expr>,
        action: MergeAction,
    },
    WhenNotMatched {
        condition: Option<Expr>,
        action: MergeAction,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MergeAction {
    Update(Vec<Assignment>),
    Insert {
        columns: Vec<String>,
        values: Vec<Expr>,
    },
    Delete,
    DoNothing,
}

// ---------------------------------------------------------------------------
// Prepared statements
// ---------------------------------------------------------------------------

/// PREPARE name [(param_types)] AS statement
#[derive(Debug, Clone, PartialEq)]
pub struct PrepareStatement {
    pub name: String,
    pub param_types: Vec<DataType>,
    pub statement: Box<Statement>,
}

/// EXECUTE name [(params)]
#[derive(Debug, Clone, PartialEq)]
pub struct ExecuteStatement {
    pub name: String,
    pub params: Vec<Expr>,
}

/// DEALLOCATE [PREPARE] name | ALL
#[derive(Debug, Clone, PartialEq)]
pub struct DeallocateStatement {
    pub name: Option<String>,
    pub all: bool,
}

// ---------------------------------------------------------------------------
// LISTEN / NOTIFY
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ListenStatement {
    pub channel: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NotifyStatement {
    pub channel: String,
    pub payload: Option<String>,
}

// ---------------------------------------------------------------------------
// Cursors
// ---------------------------------------------------------------------------

/// DECLARE name [SCROLL | NO SCROLL] CURSOR [WITH HOLD | WITHOUT HOLD] FOR select
#[derive(Debug, Clone, PartialEq)]
pub struct DeclareCursorStatement {
    pub name: String,
    pub scroll: Option<bool>,
    pub hold: Option<bool>,
    pub query: Box<SelectStatement>,
}

/// FETCH [direction] [FROM | IN] cursor_name
#[derive(Debug, Clone, PartialEq)]
pub struct FetchCursorStatement {
    pub direction: FetchDirection,
    pub cursor: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FetchDirection {
    Next,
    Prior,
    First,
    Last,
    Absolute(i64),
    Relative(i64),
    Forward(Option<i64>),
    Backward(Option<i64>),
    All,
}

/// CLOSE cursor_name | ALL
#[derive(Debug, Clone, PartialEq)]
pub struct CloseCursorStatement {
    pub name: Option<String>,
    pub all: bool,
}

// ---------------------------------------------------------------------------
// COMMENT ON
// ---------------------------------------------------------------------------

/// COMMENT ON object_type name IS 'comment' | NULL
#[derive(Debug, Clone, PartialEq)]
pub struct CommentOnStatement {
    pub object_type: CommentObjectType,
    pub name: String,
    pub column: Option<String>,
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentObjectType {
    Table,
    Column,
    Index,
    Schema,
    Sequence,
    View,
}

// ---------------------------------------------------------------------------
// ALTER INDEX / SEQUENCE / VIEW
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct AlterIndexStatement {
    pub name: String,
    pub operation: AlterIndexOperation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterIndexOperation {
    Rename { new_name: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterSequenceStatement {
    pub name: String,
    pub increment: Option<i64>,
    pub min_value: Option<Option<i64>>,
    pub max_value: Option<Option<i64>>,
    pub start: Option<i64>,
    pub restart: Option<Option<i64>>,
    pub cache: Option<i64>,
    pub cycle: Option<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterViewStatement {
    pub name: String,
    pub operation: AlterViewOperation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterViewOperation {
    Rename { new_name: String },
}

// ---------------------------------------------------------------------------
// Materialized views
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateMaterializedViewStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub query: Box<SelectStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropMaterializedViewStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefreshMaterializedViewStatement {
    pub name: String,
    pub concurrently: bool,
}

// ---------------------------------------------------------------------------
// DO block
// ---------------------------------------------------------------------------

/// DO $$ body $$ [LANGUAGE lang]
#[derive(Debug, Clone, PartialEq)]
pub struct DoBlockStatement {
    pub body: String,
    pub language: Option<String>,
}

// ---------------------------------------------------------------------------
// TABLESAMPLE
// ---------------------------------------------------------------------------

/// Sampling method for TABLESAMPLE clause
#[derive(Debug, Clone, PartialEq)]
pub struct TableSample {
    pub method: String,
    pub argument: Expr,
    pub seed: Option<Expr>,
}

// ---------------------------------------------------------------------------
// ZyronDB custom statements
// ---------------------------------------------------------------------------

/// CHECKPOINT
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointStatement {}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterSystemSetStatement {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalyzeStatement {
    pub table: Option<String>,
}

// ---------------------------------------------------------------------------
// TTL / data retention
// ---------------------------------------------------------------------------

/// ALTER TABLE name SET TTL duration ON column
/// ALTER TABLE name SET TTL ARCHIVE duration ON column
/// ALTER TABLE name DROP TTL
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableTtlStatement {
    pub table: String,
    pub operation: TtlOperation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TtlOperation {
    Set {
        duration: TtlDuration,
        column: String,
        action: TtlAction,
    },
    Drop,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtlDuration {
    pub value: i64,
    pub unit: TtlUnit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtlUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtlAction {
    Delete,
    Archive,
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

/// CREATE SCHEDULE name EVERY interval DO statement
/// CREATE SCHEDULE name CRON 'expr' DO statement
#[derive(Debug, Clone, PartialEq)]
pub struct CreateScheduleStatement {
    pub name: String,
    pub interval: ScheduleInterval,
    pub body: Box<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleInterval {
    Every(TtlDuration),
    Cron(String),
}

/// DROP SCHEDULE [IF EXISTS] name
#[derive(Debug, Clone, PartialEq)]
pub struct DropScheduleStatement {
    pub name: String,
    pub if_exists: bool,
}

/// PAUSE SCHEDULE name
#[derive(Debug, Clone, PartialEq)]
pub struct PauseScheduleStatement {
    pub name: String,
}

/// RESUME SCHEDULE name
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeScheduleStatement {
    pub name: String,
}

// ---------------------------------------------------------------------------
// OPTIMIZE
// ---------------------------------------------------------------------------

/// OPTIMIZE TABLE name (vacuum + reindex + consolidate free space)
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeTableStatement {
    pub table: String,
}

// ---------------------------------------------------------------------------
// Security: Users and Roles
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateUserStatement {
    pub name: String,
    pub password: Option<String>,
    pub options: Vec<UserOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UserOption {
    Superuser(bool),
    Login(bool),
    ValidUntil(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterUserStatement {
    pub name: String,
    pub operation: AlterUserOperation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterUserOperation {
    SetPassword(String),
    Rename { new_name: String },
    SetOption(UserOption),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropUserStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateRoleStatement {
    pub name: String,
    pub options: Vec<UserOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterRoleStatement {
    pub name: String,
    pub operation: AlterUserOperation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropRoleStatement {
    pub name: String,
    pub if_exists: bool,
}

// ---------------------------------------------------------------------------
// Pipeline / Medallion architecture
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreatePipelineStatement {
    pub name: String,
    pub stages: Vec<PipelineStage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStage {
    pub name: String,
    pub source: String,
    pub target: String,
    pub mode: Option<String>,
    pub transform: Option<Box<SelectStatement>>,
    pub expectations: Vec<PipelineExpectation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineExpectation {
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunPipelineStatement {
    pub name: String,
    pub stage: Option<String>,
    pub preview_limit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropPipelineStatement {
    pub name: String,
    pub if_exists: bool,
}

// ---------------------------------------------------------------------------
// Archive / Restore
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ArchiveTableStatement {
    pub table: String,
    pub where_clause: Option<Box<Expr>>,
    pub destination: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RestoreTableStatement {
    pub table: String,
    pub source: String,
    pub into_table: Option<String>,
    pub at_version: Option<Expr>,
    pub at_timestamp: Option<Expr>,
}

// ---------------------------------------------------------------------------
// Branching
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateBranchStatement {
    pub name: String,
    pub from_branch: Option<String>,
    pub at_version: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MergeBranchStatement {
    pub source: String,
    pub into_target: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropBranchStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UseBranchStatement {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateVersionStatement {
    pub name: String,
    pub table: String,
    pub at_version: Option<Expr>,
}

// ---------------------------------------------------------------------------
// CDC Statement Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateReplicationSlotStatement {
    pub name: String,
    pub plugin: String,
    pub table_filter: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropReplicationSlotStatement {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateCdcStreamStatement {
    pub name: String,
    pub table_name: String,
    pub sink_type: String,
    pub options: Vec<TableOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropCdcStreamStatement {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateCdcIngestStatement {
    pub name: String,
    pub source_type: String,
    pub target_table: String,
    pub options: Vec<TableOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropCdcIngestStatement {
    pub name: String,
}

/// CREATE STREAMING JOB <name> AS <select> INTO <sink-expr>
///                      [WRITE MODE APPEND | UPSERT]
///                      [MODE ONESHOT | SCHEDULED EVERY '...' | SCHEDULED CRON '...' | WATCH]
///
/// A streaming job tails the FROM-clause table's change-data-feed and
/// applies the SELECT's filter+project to every incoming row, writing
/// the result into the sink. Supports multi-row pipelines by chaining
/// source/target pairs through successive jobs. Single-table FROM runs
/// the pure filter+project loop. Two-table FROM with a JOIN runs the
/// interval-join or temporal-join pipeline, selected by the trailing
/// WITHIN INTERVAL clause or an AS OF expression on the right side.
/// Inline FROM file or cloud URI support is not yet wired into the grammar.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateStreamingJobStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub query: SelectStatement,
    pub target: StreamingSinkRef,
    pub write_mode: StreamingWriteMode,
    pub job_mode: ExternalModeSpec,
    /// Streaming window spec extracted from the GROUP BY clause, if any.
    pub window_spec: Option<StreamingWindowSpec>,
    /// Watermark specification attached to the job body.
    pub watermark: Option<WatermarkSpec>,
    /// Late-data policy attached to the job body.
    pub late_data_policy: Option<LateDataPolicySpec>,
    /// Join shape when the FROM clause joins two sources. None for
    /// single-table FROM. The parser fills this from the SELECT's FROM
    /// join node plus the trailing WITHIN INTERVAL or AS OF clause.
    pub join: Option<StreamingJoinSpec>,
}

/// Streaming-join row-match semantics. Mirrors the outer-form enumeration
/// used on base-table joins but kept local to streaming so the runner does
/// not depend on planner or parser join-type enums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingJoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Shape of a streaming-job JOIN. Carries everything the binder and
/// runner need beyond the bare JOIN already represented in the SELECT's
/// FROM clause.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingJoinSpec {
    /// Stream-stream interval join. Both sides are live feeds. Rows match
    /// when the equi-key matches and the event times fall within the
    /// symmetric window bound.
    Interval {
        /// Symmetric time window in microseconds. A row on one side matches
        /// any opposite-side row whose event time is within +/- within_us.
        within_us: i64,
        /// Inner, Left, Right, or Full outer. Outer forms flush unmatched
        /// rows on watermark advance past event_time + within_us.
        join_type: StreamingJoinType,
    },
    /// Stream-table temporal join. The left side is a live feed, the
    /// right side is a Zyron table looked up per incoming row.
    Temporal {
        /// Column on the left side carrying the event time used for
        /// AS OF semantics. Parsed from the FOR SYSTEM_TIME AS OF clause
        /// on the right table reference.
        event_time_column: String,
        /// Only Inner and Left are meaningful for temporal joins. Right
        /// and Full are rejected at bind time because the right side is
        /// a static lookup.
        join_type: StreamingJoinType,
    },
}

/// Streaming windowing spec parsed from GROUP BY TUMBLE/HOP/SESSION(...).
/// Durations are parsed as Interval literals and normalized to microseconds
/// at bind time by the planner.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingWindowSpec {
    /// TUMBLE(<event_time_col>, INTERVAL 'N' unit)
    Tumbling {
        event_time_column: String,
        size: zyron_common::Interval,
    },
    /// HOP(<event_time_col>, INTERVAL 'N' unit, INTERVAL 'N' unit)
    Hopping {
        event_time_column: String,
        size: zyron_common::Interval,
        slide: zyron_common::Interval,
    },
    /// SESSION(<event_time_col>, INTERVAL 'N' unit)
    Session {
        event_time_column: String,
        gap: zyron_common::Interval,
    },
}

/// Watermark spec: WATERMARK FOR <col> AS <col> - INTERVAL 'N' unit.
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkSpec {
    pub event_time_column: String,
    pub allowed_lateness: zyron_common::Interval,
}

/// Late-data handling policy declared on a streaming job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LateDataPolicySpec {
    Drop,
    Reopen,
    SideOutput,
}

/// Streaming job source reference. The parser currently only produces
/// Named; Inline is reserved for a future FROM-clause grammar extension
/// that accepts file or cloud URIs directly in the FROM position.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingSourceRef {
    /// A table or named external source, resolved at bind time.
    Named(String),
    /// Inline file or cloud URI. No credentials allowed inline. The
    /// binder rejects inline use for backends that require credentials.
    Inline {
        backend: ExternalBackendKind,
        uri: String,
        format: ExternalFormatKind,
        options: Vec<(String, String)>,
        mode: ExternalModeSpec,
    },
}

/// Streaming job sink reference. A sink is either a named catalog object,
/// a named table, or an inline file or cloud URI.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingSinkRef {
    Named(String),
    Inline {
        backend: ExternalBackendKind,
        uri: String,
        format: ExternalFormatKind,
        options: Vec<(String, String)>,
    },
}

/// Target-side write semantics. Append always INSERTs; Upsert merges on
/// primary key when the target has one, falling back to INSERT otherwise
/// (runtime error if no PK exists).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingWriteMode {
    Append,
    Upsert,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropStreamingJobStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterStreamingJobStatement {
    pub name: String,
    pub action: AlterStreamingJobAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlterStreamingJobAction {
    Pause,
    Resume,
}

// -----------------------------------------------------------------------------
// Publications
// -----------------------------------------------------------------------------

/// CREATE PUBLICATION name FOR TABLE <t1> [(cols) [WHERE <e>]], ...
///   [WHERE <e>] [WITH (key = value, ...)]
#[derive(Debug, Clone, PartialEq)]
pub struct CreatePublicationStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub tables: Vec<PublicationTableRef>,
    pub where_predicate: Option<Expr>,
    pub options: Vec<(String, String)>,
}

/// One table in a publication with optional projection and per-table WHERE.
#[derive(Debug, Clone, PartialEq)]
pub struct PublicationTableRef {
    pub table_name: String,
    pub columns: Vec<String>,
    pub where_predicate: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterPublicationStatement {
    pub name: String,
    pub action: AlterPublicationAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterPublicationAction {
    AddTable(PublicationTableRef),
    DropTable(String),
    SetOptions(Vec<(String, String)>),
    SetWhere(Expr),
    Rename(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropPublicationStatement {
    pub name: String,
    pub if_exists: bool,
    pub cascade: bool,
}

/// TAG PUBLICATION name WITH 'tag1', 'tag2'
#[derive(Debug, Clone, PartialEq)]
pub struct TagPublicationStatement {
    pub name: String,
    pub tags: Vec<String>,
}

/// UNTAG PUBLICATION name 'tag'
#[derive(Debug, Clone, PartialEq)]
pub struct UntagPublicationStatement {
    pub name: String,
    pub tag: String,
}

// -----------------------------------------------------------------------------
// Endpoints
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateEndpointStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub path: String,
    pub methods: Vec<String>,
    pub sql: String,
    pub auth: EndpointAuthSpec,
    pub required_scopes: Vec<String>,
    pub rate_limit: Option<EndpointRateLimitSpec>,
    pub output_format: EndpointOutputFormatSpec,
    pub cors_origins: Vec<String>,
    pub cache_seconds: u32,
    pub timeout_seconds: u32,
    pub max_body_kb: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointAuthSpec {
    None,
    Jwt,
    ApiKey,
    OAuth2,
    Basic,
    Mtls,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointRateLimitSpec {
    pub count: u64,
    pub per_seconds: u32,
    pub scope: EndpointRateLimitScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointRateLimitScope {
    Global,
    PerIp,
    PerUser,
    PerApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointOutputFormatSpec {
    Json,
    JsonLines,
    Csv,
    Parquet,
    Arrow,
    Protobuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateStreamingEndpointStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub path: String,
    pub protocol: StreamingEndpointProtocol,
    pub backing_publication: String,
    pub auth: EndpointAuthSpec,
    pub required_scopes: Vec<String>,
    pub max_connections_per_ip: Option<u32>,
    pub message_format: StreamingMessageFormat,
    pub heartbeat_seconds: u32,
    pub backpressure: BackpressurePolicySpec,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingEndpointProtocol {
    Websocket,
    Sse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingMessageFormat {
    Json,
    JsonLines,
    Protobuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressurePolicySpec {
    DropOldest,
    CloseSlow,
    Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterEndpointStatement {
    pub name: String,
    pub action: AlterEndpointAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterEndpointAction {
    Enable,
    Disable,
    SetOptions(Vec<(String, String)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropEndpointStatement {
    pub name: String,
    pub if_exists: bool,
}

// -----------------------------------------------------------------------------
// Security map
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct AlterSecurityMapStatement {
    pub kind: SecurityMapKindSpec,
    pub identity: String,
    pub role: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropSecurityMapStatement {
    pub kind: SecurityMapKindSpec,
    pub identity: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SecurityMapKindSpec {
    KubernetesSa { ns_and_name: String },
    JwtIssuerSubject { issuer: String, subject: String },
    MtlsCertSubject { subject_dn: String },
    MtlsCertFingerprint { fingerprint_sha256: String },
}

// -----------------------------------------------------------------------------
// Credential provider clause (extends CREATE EXTERNAL SOURCE/SINK)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CredentialProviderSpec {
    pub provider_type: CredentialProviderType,
    pub options: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialProviderType {
    Vault,
    AwsSecretsManager,
    GcpSecretManager,
    AzureKeyVault,
    OAuth2ClientCredentials,
    AwsIamAssumeRole,
    K8sSaToken,
}

// -----------------------------------------------------------------------------
// ABAC policy DDL
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateAbacPolicyStatement {
    pub name: String,
    pub target: AbacPolicyTarget,
    pub target_name: String,
    pub predicate: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbacPolicyTarget {
    Table,
    Publication,
}

// ---------------------------------------------------------------------------
// Table options: ALTER TABLE ... SET (key = value, ...)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableOptionsStatement {
    pub table: String,
    pub options: Vec<TableOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TableOption {
    pub key: String,
    pub value: TableOptionValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableOptionValue {
    String(String),
    Integer(i64),
    Boolean(bool),
    Identifier(String),
    StringList(Vec<String>),
}

// ---------------------------------------------------------------------------
// Data quality expectations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct AddExpectationStatement {
    pub table: String,
    pub name: String,
    pub expr: Expr,
    pub on_violation: ViolationAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationAction {
    Fail,
    Warn,
    Drop,
    Quarantine,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropExpectationStatement {
    pub table: String,
    pub name: String,
}

// ---------------------------------------------------------------------------
// Feature toggles (ENABLE/DISABLE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct EnableFeatureStatement {
    pub table: String,
    pub feature: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DisableFeatureStatement {
    pub table: String,
    pub feature: String,
}

// ---------------------------------------------------------------------------
// Search indexes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateFulltextIndexStatement {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub options: Vec<TableOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateVectorIndexStatement {
    pub name: String,
    pub table: String,
    pub column: String,
    pub options: Vec<TableOption>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateSpatialIndexStatement {
    pub name: String,
    pub table: String,
    pub column: String,
    pub if_not_exists: bool,
    /// Optional WITH (key = value, ...) tuning, e.g. WITH (dims = 3, srid = 4326).
    pub options: Vec<TableOption>,
}

// ---------------------------------------------------------------------------
// Graph schema
// ---------------------------------------------------------------------------

/// CREATE GRAPH SCHEMA name (NODE Label(...), EDGE Label FROM X TO Y (...))
#[derive(Debug, Clone, PartialEq)]
pub struct CreateGraphSchemaStatement {
    pub name: String,
    pub elements: Vec<GraphSchemaElement>,
    pub if_not_exists: bool,
}

/// Element within a CREATE GRAPH SCHEMA definition.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphSchemaElement {
    Node {
        label: String,
        properties: Vec<ColumnDef>,
    },
    Edge {
        label: String,
        from_label: String,
        to_label: String,
        properties: Vec<ColumnDef>,
    },
}

/// DROP GRAPH SCHEMA [IF EXISTS] name
#[derive(Debug, Clone, PartialEq)]
pub struct DropGraphSchemaStatement {
    pub name: String,
    pub if_exists: bool,
}

// ---------------------------------------------------------------------------
// GROUP BY ROLLUP / CUBE / GROUPING SETS
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum GroupBySets {
    Rollup(Vec<Expr>),
    Cube(Vec<Expr>),
    GroupingSets(Vec<Vec<Expr>>),
}

// ---------------------------------------------------------------------------
// Time travel: AS OF
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum AsOf {
    Timestamp(Expr),
    SystemTime {
        start: Expr,
        end: Expr,
    },
    Version(Expr),
    ApplicationTime {
        start: Expr,
        end: Expr,
    },
    ForPortionOf {
        period: String,
        start: Expr,
        end: Expr,
    },
}

// ---------------------------------------------------------------------------
// Function arguments (positional or named)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionArg {
    /// Positional argument
    Unnamed(Expr),
    /// Named argument: name => value
    Named { name: String, value: Expr },
}

// ---------------------------------------------------------------------------
// Vector distance operators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorDistanceOp {
    /// <=> (cosine distance)
    Cosine,
    /// <-> (L2 / Euclidean distance)
    L2,
    /// <#> (negative dot product)
    DotProduct,
}

// ---------------------------------------------------------------------------
// VALUES query
// ---------------------------------------------------------------------------

/// VALUES (row1), (row2), ... as a standalone query
#[derive(Debug, Clone, PartialEq)]
pub struct ValuesQueryStatement {
    pub rows: Vec<Vec<Expr>>,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Unqualified column or variable reference.
    Identifier(String),
    /// Qualified column reference: table.column.
    QualifiedIdentifier {
        table: String,
        column: String,
    },
    /// Literal value.
    Literal(LiteralValue),
    /// Binary operation: left op right.
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    /// Unary operation: op expr.
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expr>,
    },
    /// IS [NOT] NULL check.
    IsNull {
        expr: Box<Expr>,
        negated: bool,
    },
    /// [NOT] IN (list).
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    /// [NOT] BETWEEN low AND high.
    Between {
        expr: Box<Expr>,
        low: Box<Expr>,
        high: Box<Expr>,
        negated: bool,
    },
    /// [NOT] LIKE pattern.
    Like {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    /// [NOT] ILIKE pattern (case-insensitive LIKE).
    ILike {
        expr: Box<Expr>,
        pattern: Box<Expr>,
        negated: bool,
    },
    /// Function call: name([DISTINCT] args...).
    Function {
        name: String,
        args: Vec<FunctionArg>,
        distinct: bool,
    },
    /// CAST(expr AS data_type) or expr::data_type.
    Cast {
        expr: Box<Expr>,
        data_type: DataType,
    },
    /// CASE [operand] WHEN condition THEN result ... [ELSE result] END.
    Case {
        operand: Option<Box<Expr>>,
        conditions: Vec<WhenClause>,
        else_result: Option<Box<Expr>>,
    },
    /// Parenthesized expression.
    Nested(Box<Expr>),
    /// Scalar subquery: (SELECT ...)
    Subquery(Box<SelectStatement>),
    /// [NOT] IN (SELECT ...)
    InSubquery {
        expr: Box<Expr>,
        query: Box<SelectStatement>,
        negated: bool,
    },
    /// EXISTS (SELECT ...)
    Exists {
        query: Box<SelectStatement>,
        negated: bool,
    },
    /// Window function: expr OVER (window_spec)
    WindowFunction {
        function: Box<Expr>,
        partition_by: Vec<Expr>,
        order_by: Vec<OrderByExpr>,
        frame: Option<WindowFrame>,
    },
    /// Array constructor: ARRAY[expr, ...]
    ArrayConstructor(Vec<Expr>),
    /// Array subscript: expr[index]
    ArraySubscript {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    /// JSON access: expr -> key, expr ->> key, expr #> path, expr #>> path
    JsonAccess {
        left: Box<Expr>,
        op: JsonOperator,
        right: Box<Expr>,
    },
    /// JSON containment: expr @> expr or expr <@ expr
    JsonContains {
        left: Box<Expr>,
        op: JsonContainsOp,
        right: Box<Expr>,
    },
    /// JSON exists: expr ? key, expr ?| keys, expr ?& keys
    JsonExists {
        left: Box<Expr>,
        op: JsonExistsOp,
        right: Box<Expr>,
    },
    /// ANY(array_expr) or SOME(array_expr) for comparisons
    AnySubquery {
        query: Box<SelectStatement>,
    },
    AllSubquery {
        query: Box<SelectStatement>,
    },
    /// Parameter placeholder: $1, $2, etc. (for prepared statements)
    Parameter(usize),
    /// Vector distance: expr <=> expr, expr <-> expr, expr <#> expr
    VectorDistance {
        left: Box<Expr>,
        op: VectorDistanceOp,
        right: Box<Expr>,
    },
    /// MATCH (columns) AGAINST (query [IN mode])
    MatchAgainst {
        columns: Vec<String>,
        query: Box<Expr>,
        mode: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    /// SQL INTERVAL literal - composite (months, days, nanoseconds) value.
    Interval(zyron_common::Interval),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOperator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Eq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
    Concat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    Not,
    Minus,
}

/// JSON access operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonOperator {
    /// -> (get JSON object field or array element)
    Arrow,
    /// ->> (get JSON object field or array element as text)
    DoubleArrow,
    /// #> (get JSON object at path)
    HashArrow,
    /// #>> (get JSON object at path as text)
    HashDoubleArrow,
}

/// JSON containment operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonContainsOp {
    /// @> (left contains right)
    Contains,
    /// <@ (left is contained by right)
    ContainedBy,
}

/// JSON existence operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonExistsOp {
    /// ? (key exists)
    Exists,
    /// ?| (any key exists)
    ExistsAny,
    /// ?& (all keys exist)
    ExistsAll,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhenClause {
    pub condition: Expr,
    pub result: Expr,
}

// ---------------------------------------------------------------------------
// Window frame
// ---------------------------------------------------------------------------

/// Window frame specification: ROWS/RANGE BETWEEN start AND end
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    pub mode: WindowFrameMode,
    pub start: WindowFrameBound,
    pub end: Option<WindowFrameBound>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameMode {
    Rows,
    Range,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameBound {
    /// CURRENT ROW
    CurrentRow,
    /// UNBOUNDED PRECEDING / UNBOUNDED FOLLOWING
    Unbounded(WindowFrameDirection),
    /// N PRECEDING / N FOLLOWING (row count in ROWS mode, numeric offset in RANGE mode)
    Offset(u64, WindowFrameDirection),
    /// INTERVAL 'N unit' PRECEDING / FOLLOWING.
    /// Only valid in RANGE mode. The comparison uses calendar-aware arithmetic
    /// against the ORDER BY column's timestamp values.
    IntervalBound(zyron_common::Interval, WindowFrameDirection),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameDirection {
    Preceding,
    Following,
}

// ---------------------------------------------------------------------------
// Table references
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    /// Simple table reference with optional alias and time travel.
    Table {
        name: String,
        alias: Option<String>,
        as_of: Option<Box<AsOf>>,
    },
    /// JOIN between two table references. Boxed to keep TableRef small.
    Join(Box<JoinTableRef>),
    /// Subquery in FROM clause: (SELECT ...) AS alias
    Subquery {
        query: Box<SelectStatement>,
        alias: String,
    },
    /// LATERAL subquery or function call in FROM
    Lateral { subquery: Box<TableRef> },
    /// Table-valued function call in FROM clause: FUNC_NAME(args...)
    TableFunction {
        name: String,
        args: Vec<FunctionArg>,
        alias: Option<String>,
    },
}

/// Join data extracted to a separate struct for boxing inside TableRef.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinTableRef {
    pub left: TableRef,
    pub join_type: JoinType,
    pub right: TableRef,
    pub condition: JoinCondition,
}

impl TableRef {
    /// Optional TABLESAMPLE attached to a table reference
    pub fn name(&self) -> Option<&str> {
        match self {
            TableRef::Table { name, .. } => Some(name),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinCondition {
    On(Box<Expr>),
    Using(Vec<String>),
    Natural,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

// ---------------------------------------------------------------------------
// ORDER BY
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpr {
    pub expr: Expr,
    /// None = default ordering, Some(true) = ASC, Some(false) = DESC.
    pub asc: Option<bool>,
    /// None = default null ordering, Some(true) = NULLS FIRST, Some(false) = NULLS LAST.
    pub nulls_first: Option<bool>,
}

// ---------------------------------------------------------------------------
// Column and type definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: Option<bool>,
    pub default: Option<Expr>,
    pub constraints: Vec<ColumnConstraint>,
}

/// SQL data type as written in a query. Maps to TypeId for storage.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Boolean,
    TinyInt,
    SmallInt,
    Int,
    BigInt,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Real,
    DoublePrecision,
    Float(Option<u32>),
    Decimal(Option<u8>, Option<u8>),
    Numeric(Option<u8>, Option<u8>),
    Char(Option<usize>),
    Varchar(Option<usize>),
    Text,
    Binary(Option<usize>),
    Varbinary(Option<usize>),
    Bytea,
    Date,
    Time,
    Timestamp,
    TimestampTz,
    Interval,
    Uuid,
    Json,
    Jsonb,
    /// Array type: element_type[]
    Array(Box<DataType>),
    /// Vector type: VECTOR(dimensions)
    Vector(Option<usize>),
    /// Geospatial geometry type (WKB storage)
    Geometry,
    /// Dense matrix type (row-major f64 array)
    Matrix,
    /// RGBA color type (packed u32)
    Color,
    /// Semantic versioning type
    SemVer,
    /// IPv4/IPv6 host address with optional prefix
    Inet,
    /// IPv4/IPv6 network address (CIDR notation)
    Cidr,
    /// MAC address (6 bytes)
    MacAddr,
    /// Currency-aware money type (fixed-point with currency code)
    Money,
    /// Range type parameterized by element type
    Range(Box<DataType>),
    /// HyperLogLog approximate cardinality sketch
    HyperLogLog,
    /// Bloom filter for probabilistic set membership
    BloomFilter,
    /// T-Digest for approximate quantile estimation
    TDigest,
    /// Count-Min Sketch for approximate frequency counting
    CountMinSketch,
    /// Named bitfield (u64 with named bit positions)
    Bitfield,
    /// Unit-aware quantity (value + unit)
    Quantity,
}

impl DataType {
    /// Converts this SQL data type to the corresponding internal TypeId.
    pub fn to_type_id(&self) -> TypeId {
        match self {
            DataType::Boolean => TypeId::Boolean,
            DataType::TinyInt => TypeId::Int8,
            DataType::SmallInt => TypeId::Int16,
            DataType::Int => TypeId::Int32,
            DataType::BigInt => TypeId::Int64,
            DataType::Int128 => TypeId::Int128,
            DataType::UInt8 => TypeId::UInt8,
            DataType::UInt16 => TypeId::UInt16,
            DataType::UInt32 => TypeId::UInt32,
            DataType::UInt64 => TypeId::UInt64,
            DataType::UInt128 => TypeId::UInt128,
            DataType::Real => TypeId::Float32,
            DataType::DoublePrecision => TypeId::Float64,
            DataType::Float(prec) => {
                // FLOAT with precision <= 24 maps to Float32, otherwise Float64.
                match prec {
                    Some(p) if *p <= 24 => TypeId::Float32,
                    _ => TypeId::Float64,
                }
            }
            DataType::Decimal(_, _) => TypeId::Decimal,
            DataType::Numeric(_, _) => TypeId::Decimal,
            DataType::Char(_) => TypeId::Char,
            DataType::Varchar(_) => TypeId::Varchar,
            DataType::Text => TypeId::Text,
            DataType::Binary(_) => TypeId::Binary,
            DataType::Varbinary(_) => TypeId::Varbinary,
            DataType::Bytea => TypeId::Bytea,
            DataType::Date => TypeId::Date,
            DataType::Time => TypeId::Time,
            DataType::Timestamp => TypeId::Timestamp,
            DataType::TimestampTz => TypeId::TimestampTz,
            DataType::Interval => TypeId::Interval,
            DataType::Uuid => TypeId::Uuid,
            DataType::Json => TypeId::Json,
            DataType::Jsonb => TypeId::Jsonb,
            DataType::Array(_) => TypeId::Array,
            DataType::Vector(_) => TypeId::Vector,
            DataType::Geometry => TypeId::Geometry,
            DataType::Matrix => TypeId::Matrix,
            DataType::Color => TypeId::Color,
            DataType::SemVer => TypeId::SemVer,
            DataType::Inet => TypeId::Inet,
            DataType::Cidr => TypeId::Cidr,
            DataType::MacAddr => TypeId::MacAddr,
            DataType::Money => TypeId::Money,
            DataType::Range(_) => TypeId::Range,
            DataType::HyperLogLog => TypeId::HyperLogLog,
            DataType::BloomFilter => TypeId::BloomFilter,
            DataType::TDigest => TypeId::TDigest,
            DataType::CountMinSketch => TypeId::CountMinSketch,
            DataType::Bitfield => TypeId::Bitfield,
            DataType::Quantity => TypeId::Quantity,
        }
    }
}

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnConstraint {
    PrimaryKey,
    Unique,
    NotNull,
    Default(Expr),
    Check(Expr),
    References { table: String, column: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    PrimaryKey(Vec<String>),
    Unique(Vec<String>),
    Check(Expr),
    ForeignKey {
        columns: Vec<String>,
        ref_table: String,
        ref_columns: Vec<String>,
    },
}

// ---------------------------------------------------------------------------
// Triggers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
    InsteadOf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
    Truncate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerGranularity {
    Row,
    Statement,
}

/// REFERENCING OLD TABLE AS name NEW TABLE AS name (for statement-level triggers)
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionTables {
    pub old_table: Option<String>,
    pub new_table: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateTriggerStatement {
    pub name: String,
    pub timing: TriggerTiming,
    pub events: Vec<TriggerEvent>,
    pub table: String,
    pub for_each: TriggerGranularity,
    pub when_condition: Option<Box<Expr>>,
    pub referencing: Option<TransitionTables>,
    pub execute_function: String,
    pub args: Vec<Expr>,
    pub priority: Option<u32>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropTriggerStatement {
    pub name: String,
    pub table: String,
    pub if_exists: bool,
    pub drop_behavior: Option<DropBehavior>,
}

/// CASCADE or RESTRICT behavior for DROP statements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropBehavior {
    Cascade,
    Restrict,
}

// ---------------------------------------------------------------------------
// User-defined functions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Volatility {
    Immutable,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionLanguage {
    Sql,
    Rust,
    RustVectorized,
}

/// Return type for a function: scalar, table, or set of scalars
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionReturnType {
    Scalar(DataType),
    Table(Vec<FunctionParam>),
    SetOf(DataType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParam {
    pub name: String,
    pub data_type: DataType,
    pub default_value: Option<Box<Expr>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateFunctionStatement {
    pub name: String,
    pub or_replace: bool,
    pub params: Vec<FunctionParam>,
    pub return_type: FunctionReturnType,
    pub language: FunctionLanguage,
    pub body: String,
    pub volatility: Volatility,
    pub rust_library: Option<String>,
    pub rust_symbol: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropFunctionStatement {
    pub name: String,
    pub if_exists: bool,
    pub drop_behavior: Option<DropBehavior>,
}

// ---------------------------------------------------------------------------
// User-defined aggregates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateAggregateStatement {
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub sfunc: String,
    pub stype: DataType,
    pub finalfunc: Option<String>,
    pub combinefunc: Option<String>,
    pub initcond: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropAggregateStatement {
    pub name: String,
    pub if_exists: bool,
    pub drop_behavior: Option<DropBehavior>,
}

// ---------------------------------------------------------------------------
// Stored procedures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityMode {
    Definer,
    Invoker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcedureLanguage {
    Sql,
    PlSql,
    Rust,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CreateProcedureStatement {
    pub name: String,
    pub or_replace: bool,
    pub params: Vec<FunctionParam>,
    pub language: ProcedureLanguage,
    pub body: String,
    pub security: SecurityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropProcedureStatement {
    pub name: String,
    pub if_exists: bool,
    pub drop_behavior: Option<DropBehavior>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallStatement {
    pub name: String,
    pub args: Vec<Expr>,
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CreateEventHandlerStatement {
    pub name: String,
    pub event_type: String,
    pub condition: Option<Box<Expr>>,
    pub execute_function: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropEventHandlerStatement {
    pub name: String,
    pub if_exists: bool,
}

// -----------------------------------------------------------------------------
// External source and sink DDL
// -----------------------------------------------------------------------------

/// Storage backend for an external source or sink.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalBackendKind {
    File,
    S3,
    Gcs,
    Azure,
    Http,
    Zyron,
}

/// On-disk row format for an external source or sink.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalFormatKind {
    Json,
    JsonLines,
    Csv,
    Parquet,
    ArrowIpc,
    Avro,
}

/// Ingest or emit cadence for an external source or streaming job.
/// OneShot runs a single pass, Scheduled fires on a cron or interval,
/// Watch tails the backend for new objects as they arrive.
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalModeSpec {
    OneShot,
    Scheduled {
        cron: Option<String>,
        every: Option<String>,
    },
    Watch,
}

/// CREATE EXTERNAL SOURCE name TYPE backend URI uri FORMAT fmt
///   [MODE modespec] [OPTIONS (k=v, ...)] [CREDENTIALS (k=v, ...)]
#[derive(Debug, Clone, PartialEq)]
pub struct CreateExternalSourceStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub backend: ExternalBackendKind,
    pub uri: String,
    pub format: ExternalFormatKind,
    pub mode: ExternalModeSpec,
    pub options: Vec<(String, String)>,
    pub credentials: Vec<(String, String)>,
    /// Optional dynamic credential provider. Mutually exclusive with credentials.
    pub credential_provider: Option<CredentialProviderSpec>,
    // Optional explicit column layout. Empty when the user omits the clause.
    pub columns: Vec<(String, DataType)>,
}

/// CREATE EXTERNAL SINK name TYPE backend URI uri FORMAT fmt
///   [OPTIONS (k=v, ...)] [CREDENTIALS (k=v, ...)]
#[derive(Debug, Clone, PartialEq)]
pub struct CreateExternalSinkStatement {
    pub name: String,
    pub if_not_exists: bool,
    pub backend: ExternalBackendKind,
    pub uri: String,
    pub format: ExternalFormatKind,
    pub options: Vec<(String, String)>,
    pub credentials: Vec<(String, String)>,
    /// Optional dynamic credential provider. Mutually exclusive with credentials.
    pub credential_provider: Option<CredentialProviderSpec>,
    // Optional explicit column layout. Empty when the user omits the clause.
    pub columns: Vec<(String, DataType)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropExternalSourceStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropExternalSinkStatement {
    pub name: String,
    pub if_exists: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterExternalSourceAction {
    SetOptions(Vec<(String, String)>),
    SetCredentials(Vec<(String, String)>),
    SetCredentialProvider(CredentialProviderSpec),
    SetMode(ExternalModeSpec),
    SetColumns(Vec<(String, DataType)>),
    Rename(String),
    RefreshSchema,
    ResetLsn(LsnResetSpec),
    Pause,
    Resume,
}

/// Specification for ALTER EXTERNAL SOURCE ... RESET LSN TO ...
#[derive(Debug, Clone, PartialEq)]
pub enum LsnResetSpec {
    Earliest,
    Latest,
    Explicit(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterExternalSourceStatement {
    pub name: String,
    pub action: AlterExternalSourceAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlterExternalSinkAction {
    SetOptions(Vec<(String, String)>),
    SetCredentials(Vec<(String, String)>),
    Rename(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlterExternalSinkStatement {
    pub name: String,
    pub action: AlterExternalSinkAction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_to_type_id() {
        assert_eq!(DataType::Boolean.to_type_id(), TypeId::Boolean);
        assert_eq!(DataType::SmallInt.to_type_id(), TypeId::Int16);
        assert_eq!(DataType::Int.to_type_id(), TypeId::Int32);
        assert_eq!(DataType::BigInt.to_type_id(), TypeId::Int64);
        assert_eq!(DataType::Real.to_type_id(), TypeId::Float32);
        assert_eq!(DataType::DoublePrecision.to_type_id(), TypeId::Float64);
        assert_eq!(DataType::Float(Some(24)).to_type_id(), TypeId::Float32);
        assert_eq!(DataType::Float(Some(53)).to_type_id(), TypeId::Float64);
        assert_eq!(DataType::Float(None).to_type_id(), TypeId::Float64);
        assert_eq!(
            DataType::Decimal(Some(10), Some(2)).to_type_id(),
            TypeId::Decimal
        );
        assert_eq!(DataType::Numeric(None, None).to_type_id(), TypeId::Decimal);
        assert_eq!(DataType::Varchar(Some(255)).to_type_id(), TypeId::Varchar);
        assert_eq!(DataType::Char(Some(10)).to_type_id(), TypeId::Char);
        assert_eq!(DataType::Text.to_type_id(), TypeId::Text);
        assert_eq!(DataType::Date.to_type_id(), TypeId::Date);
        assert_eq!(DataType::Time.to_type_id(), TypeId::Time);
        assert_eq!(DataType::Timestamp.to_type_id(), TypeId::Timestamp);
        assert_eq!(DataType::TimestampTz.to_type_id(), TypeId::TimestampTz);
        assert_eq!(DataType::Interval.to_type_id(), TypeId::Interval);
        assert_eq!(DataType::Uuid.to_type_id(), TypeId::Uuid);
        assert_eq!(DataType::Json.to_type_id(), TypeId::Json);
        assert_eq!(DataType::Jsonb.to_type_id(), TypeId::Jsonb);
        assert_eq!(DataType::Binary(Some(16)).to_type_id(), TypeId::Binary);
        assert_eq!(
            DataType::Varbinary(Some(256)).to_type_id(),
            TypeId::Varbinary
        );
        assert_eq!(DataType::Bytea.to_type_id(), TypeId::Bytea);
    }

    #[test]
    fn test_statement_variants() {
        // Verify all statement variants can be constructed
        let select = Statement::Select(Box::new(SelectStatement {
            with: None,
            distinct: false,
            distinct_on: vec![],
            projections: vec![SelectItem::Wildcard],
            from: vec![TableRef::Table {
                name: "users".to_string(),
                alias: None,
                as_of: None,
            }],
            where_clause: None,
            group_by: vec![],
            group_by_sets: None,
            having: None,
            qualify: None,
            set_ops: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            fetch: None,
            for_clause: None,
        }));
        assert!(matches!(select, Statement::Select(_)));

        let truncate = Statement::Truncate(Box::new(TruncateStatement {
            table: "users".to_string(),
        }));
        assert!(matches!(truncate, Statement::Truncate(_)));

        let begin = Statement::Begin(Box::new(BeginStatement {}));
        assert!(matches!(begin, Statement::Begin(_)));

        let explain = Statement::Explain(Box::new(ExplainStatement {
            analyze: true,
            costs: true,
            buffers: false,
            timing: true,
            format: None,
            statement: Box::new(select.clone()),
        }));
        assert!(matches!(explain, Statement::Explain(_)));
    }

    #[test]
    fn test_alter_table_operations() {
        let add_col = AlterTableOperation::AddColumn(ColumnDef {
            name: "email".to_string(),
            data_type: DataType::Varchar(Some(255)),
            nullable: Some(true),
            default: None,
            constraints: vec![],
        });
        assert!(matches!(add_col, AlterTableOperation::AddColumn(_)));

        let drop_col = AlterTableOperation::DropColumn {
            name: "old_field".to_string(),
            if_exists: true,
        };
        assert!(matches!(drop_col, AlterTableOperation::DropColumn { .. }));

        let rename_col = AlterTableOperation::RenameColumn {
            old_name: "fname".to_string(),
            new_name: "first_name".to_string(),
        };
        assert!(matches!(
            rename_col,
            AlterTableOperation::RenameColumn { .. }
        ));

        let rename_table = AlterTableOperation::RenameTable {
            new_name: "customers".to_string(),
        };
        assert!(matches!(
            rename_table,
            AlterTableOperation::RenameTable { .. }
        ));
    }

    #[test]
    fn test_expr_variants() {
        let lit = Expr::Literal(LiteralValue::Integer(42));
        assert!(matches!(lit, Expr::Literal(LiteralValue::Integer(42))));

        let binop = Expr::BinaryOp {
            left: Box::new(Expr::Literal(LiteralValue::Integer(1))),
            op: BinaryOperator::Plus,
            right: Box::new(Expr::Literal(LiteralValue::Integer(2))),
        };
        assert!(matches!(binop, Expr::BinaryOp { .. }));

        let is_null = Expr::IsNull {
            expr: Box::new(Expr::Identifier("x".to_string())),
            negated: false,
        };
        assert!(matches!(is_null, Expr::IsNull { negated: false, .. }));
    }

    #[test]
    fn test_join_types() {
        let join = TableRef::Join(Box::new(JoinTableRef {
            left: TableRef::Table {
                name: "a".to_string(),
                alias: None,
                as_of: None,
            },
            join_type: JoinType::Left,
            right: TableRef::Table {
                name: "b".to_string(),
                alias: None,
                as_of: None,
            },
            condition: JoinCondition::On(Box::new(Expr::BinaryOp {
                left: Box::new(Expr::QualifiedIdentifier {
                    table: "a".to_string(),
                    column: "id".to_string(),
                }),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::QualifiedIdentifier {
                    table: "b".to_string(),
                    column: "a_id".to_string(),
                }),
            })),
        }));
        assert!(matches!(
            join,
            TableRef::Join(ref j) if j.join_type == JoinType::Left
        ));
    }
}
