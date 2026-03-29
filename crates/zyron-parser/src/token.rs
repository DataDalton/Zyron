//! Token types, spans, and keyword lookup for the SQL lexer.

/// Source location of a token within the input string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Byte offset from the start of the input.
    pub offset: usize,
    /// Length in bytes.
    pub length: usize,
}

impl Span {
    pub fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }
}

/// A token paired with its source location.
#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

impl SpannedToken {
    pub fn new(token: Token, span: Span) -> Self {
        Self { token, span }
    }
}

/// SQL keyword recognized by the lexer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Keyword {
    // DML
    Select,
    From,
    Where,
    Insert,
    Update,
    Delete,
    Into,
    Values,
    Set,

    // DDL
    Create,
    Drop,
    Alter,
    Table,
    Index,
    Column,
    Rename,
    To,
    Add,
    Truncate,

    // Clauses
    Order,
    By,
    Asc,
    Desc,
    Limit,
    Offset,
    Group,
    Having,
    Distinct,
    All,
    As,
    On,

    // Joins
    Join,
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Outer,

    // Logical operators
    And,
    Or,
    Not,

    // Predicates
    Is,
    In,
    Like,
    Ilike,
    Between,
    Exists,

    // Window functions
    Over,
    Partition,
    Rows,
    Range,
    Unbounded,
    Preceding,
    Following,
    Current,
    Row,

    // CTEs
    Recursive,

    // Literals
    Null,
    True,
    False,

    // Constraints
    Primary,
    Key,
    Unique,
    Check,
    Default,
    References,
    Foreign,
    Constraint,
    Cascade,
    Restrict,

    // Conditional
    If,
    Nulls,
    First,
    Last,

    // Expressions
    Cast,
    Case,
    When,
    Then,
    Else,
    End,

    // Transaction control
    Begin,
    Commit,
    Rollback,
    Savepoint,
    Release,
    Transaction,

    // Query analysis
    Explain,
    Analyze,

    // Set operations
    Union,
    Intersect,
    Except,

    // Views
    View,
    Replace,

    // DCL (permissions)
    Grant,
    Revoke,
    Privileges,
    Public,

    // RETURNING clause
    Returning,

    // UPSERT (ON CONFLICT)
    Conflict,
    Do,
    Nothing,

    // Join variants
    Natural,
    Using,

    // Schema
    Schema,

    // Sequences
    Sequence,
    Increment,
    Minvalue,
    Maxvalue,
    Start,
    Cache,
    Cycle,
    No,

    // Maintenance
    Vacuum,
    Reindex,

    // Session
    Show,

    // Bulk copy
    Copy,
    Stdin,
    Stdout,

    // FOR UPDATE/SHARE (row locking)
    For,
    Share,
    Lock,
    Nowait,
    Skip,
    Locked,

    // MERGE
    Merge,
    Matched,

    // Prepared statements
    Prepare,
    Execute,
    Deallocate,

    // FETCH FIRST/NEXT
    Fetch,
    Next,
    Only,
    Percent,

    // LATERAL
    Lateral,

    // Array
    Array,
    Any,
    Some,

    // LISTEN/NOTIFY
    Listen,
    Notify,
    Payload,

    // TABLESAMPLE
    Tablesample,
    Bernoulli,
    System,

    // Cursors
    Declare,
    Cursor,
    Close,
    Scroll,
    Hold,
    Without,
    Absolute,
    Relative,
    Forward,
    Backward,

    // COMMENT ON
    Comment,

    // Materialized views
    Refresh,
    Materialized,

    // Checkpoint
    Checkpoint,

    // DO block
    Language,

    // ZyronDB custom
    Segments,
    Status,
    Buffer,
    Pool,
    Transactions,
    Format,
    Storage,
    Columnar,
    Heap,

    // TTL / data retention
    Ttl,
    Days,
    Hours,
    Minutes,
    Seconds,
    Archive,
    Retain,
    Expire,

    // Scheduling
    Schedule,
    Every,
    Pause,
    Resume,
    Cron,

    // Optimize
    Optimize,

    // Extended integer types
    Tinyint,
    Int128,

    // Unsigned integer types
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,

    // Vector type
    Vector,

    // Analytics (ROLLUP, CUBE, GROUPING SETS, QUALIFY)
    Rollup,
    Cube,
    Grouping,
    Sets,
    Qualify,

    // Time travel
    Versioning,
    Period,
    Of,

    // Pipeline / Medallion
    Pipeline,
    Stage,
    Source,
    Target,
    Mode,
    Transform,
    Expect,
    Expectation,
    Violation,
    Fail,
    Quarantine,
    Run,

    // Archive / Restore
    Restore,

    // Branching / Versioning
    Branch,
    Version,
    Portion,
    Use,
    At,

    // Security
    User,
    Role,
    Password,
    Login,
    Superuser,
    Valid,
    Until,
    Nologin,

    // Search
    Fulltext,
    Match,
    Against,

    // CDC / Feature toggles
    Enable,
    Disable,
    Feed,
    Change,
    Replication,
    Slot,
    Plugin,
    Cdc,
    Stream,
    Ingest,
    Publication,
    Include,
    Ddl,

    // Triggers, UDFs, procedures, aggregates
    Trigger,
    Before,
    After,
    Instead,
    Each,
    Referencing,
    Old,
    New,
    Priority,
    Function,
    Returns,
    Immutable,
    Volatile,
    Stable,
    Setof,
    Procedure,
    Call,
    Aggregate,
    Sfunc,
    Stype,
    Finalfunc,
    Combinefunc,
    Initcond,
    Definer,
    Security,
    Invoker,
    Handler,
    Event,
    Preview,
    Plsql,
    Sql,
    Symbol,
    Rust,
    RustVectorized,

    // Data type keywords
    Type,
    Int,
    Integer,
    Smallint,
    Bigint,
    Real,
    Double,
    Precision,
    Float,
    Boolean,
    Char,
    Varchar,
    Text,
    Decimal,
    Numeric,
    Date,
    Time,
    Timestamp,
    Timestamptz,
    Interval,
    Uuid,
    Json,
    Jsonb,
    Binary,
    Varbinary,
    Bytea,
    With,
    Zone,
}

impl std::fmt::Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", format!("{self:?}").to_uppercase())
    }
}

/// Lexical token produced by the SQL lexer.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// SQL keyword.
    Keyword(Keyword),

    /// Integer literal.
    Integer(i64),

    /// Floating-point literal.
    Float(f64),

    /// String literal (single-quoted).
    String(String),

    /// Identifier (unquoted or double-quoted).
    Ident(String),

    // Arithmetic operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Comparison operators
    Eq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,

    // String concatenation
    Concat,

    // Punctuation
    Comma,
    Semicolon,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Dot,
    DoubleColon,

    // JSON operators
    Arrow,           // ->
    DoubleArrow,     // ->>
    HashArrow,       // #>
    HashDoubleArrow, // #>>
    AtArrow,         // @>
    ArrowAt,         // <@
    Question,        // ?
    QuestionPipe,    // ?|
    QuestionAmp,     // ?&

    // Named parameter / fat arrow
    FatArrow, // =>

    // Vector distance operators
    CosineDistance, // <=>
    L2Distance,     // <->
    DotDistance,    // <#>

    /// End of input.
    Eof,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Keyword(kw) => write!(f, "{kw}"),
            Token::Integer(n) => write!(f, "{n}"),
            Token::Float(n) => write!(f, "{n}"),
            Token::String(s) => write!(f, "'{s}'"),
            Token::Ident(s) => write!(f, "{s}"),
            Token::Plus => f.write_str("+"),
            Token::Minus => f.write_str("-"),
            Token::Star => f.write_str("*"),
            Token::Slash => f.write_str("/"),
            Token::Percent => f.write_str("%"),
            Token::Eq => f.write_str("="),
            Token::Neq => f.write_str("!="),
            Token::Lt => f.write_str("<"),
            Token::Gt => f.write_str(">"),
            Token::LtEq => f.write_str("<="),
            Token::GtEq => f.write_str(">="),
            Token::Concat => f.write_str("||"),
            Token::Comma => f.write_str(","),
            Token::Semicolon => f.write_str(";"),
            Token::LParen => f.write_str("("),
            Token::RParen => f.write_str(")"),
            Token::LBracket => f.write_str("["),
            Token::RBracket => f.write_str("]"),
            Token::Dot => f.write_str("."),
            Token::DoubleColon => f.write_str("::"),
            Token::Arrow => f.write_str("->"),
            Token::DoubleArrow => f.write_str("->>"),
            Token::HashArrow => f.write_str("#>"),
            Token::HashDoubleArrow => f.write_str("#>>"),
            Token::AtArrow => f.write_str("@>"),
            Token::ArrowAt => f.write_str("<@"),
            Token::Question => f.write_str("?"),
            Token::QuestionPipe => f.write_str("?|"),
            Token::QuestionAmp => f.write_str("?&"),
            Token::FatArrow => f.write_str("=>"),
            Token::CosineDistance => f.write_str("<=>"),
            Token::L2Distance => f.write_str("<->"),
            Token::DotDistance => f.write_str("<#>"),
            Token::Eof => f.write_str("end of input"),
        }
    }
}

/// Looks up a keyword from a word string. Case-insensitive.
/// Returns None if the word is not a recognized SQL keyword.
/// Uses a stack-allocated buffer to avoid heap allocation per call.
pub fn lookup_keyword(word: &str) -> Option<Keyword> {
    const MAX_KW_LEN: usize = 32;
    let len = word.len();
    if len > MAX_KW_LEN {
        return None;
    }
    let mut buf = [0u8; MAX_KW_LEN];
    for (i, &b) in word.as_bytes().iter().enumerate() {
        buf[i] = b.to_ascii_uppercase();
    }
    // Safety: input was valid UTF-8, uppercasing ASCII bytes preserves validity.
    let upper = unsafe { std::str::from_utf8_unchecked(&buf[..len]) };
    match upper {
        // DML
        "SELECT" => Some(Keyword::Select),
        "FROM" => Some(Keyword::From),
        "WHERE" => Some(Keyword::Where),
        "INSERT" => Some(Keyword::Insert),
        "UPDATE" => Some(Keyword::Update),
        "DELETE" => Some(Keyword::Delete),
        "INTO" => Some(Keyword::Into),
        "VALUES" => Some(Keyword::Values),
        "SET" => Some(Keyword::Set),

        // DDL
        "CREATE" => Some(Keyword::Create),
        "DROP" => Some(Keyword::Drop),
        "ALTER" => Some(Keyword::Alter),
        "TABLE" => Some(Keyword::Table),
        "INDEX" => Some(Keyword::Index),
        "COLUMN" => Some(Keyword::Column),
        "RENAME" => Some(Keyword::Rename),
        "TO" => Some(Keyword::To),
        "ADD" => Some(Keyword::Add),
        "TRUNCATE" => Some(Keyword::Truncate),

        // Clauses
        "ORDER" => Some(Keyword::Order),
        "BY" => Some(Keyword::By),
        "ASC" => Some(Keyword::Asc),
        "DESC" => Some(Keyword::Desc),
        "LIMIT" => Some(Keyword::Limit),
        "OFFSET" => Some(Keyword::Offset),
        "GROUP" => Some(Keyword::Group),
        "HAVING" => Some(Keyword::Having),
        "DISTINCT" => Some(Keyword::Distinct),
        "ALL" => Some(Keyword::All),
        "AS" => Some(Keyword::As),
        "ON" => Some(Keyword::On),

        // Joins
        "JOIN" => Some(Keyword::Join),
        "INNER" => Some(Keyword::Inner),
        "LEFT" => Some(Keyword::Left),
        "RIGHT" => Some(Keyword::Right),
        "FULL" => Some(Keyword::Full),
        "CROSS" => Some(Keyword::Cross),
        "OUTER" => Some(Keyword::Outer),

        // Logical operators
        "AND" => Some(Keyword::And),
        "OR" => Some(Keyword::Or),
        "NOT" => Some(Keyword::Not),

        // Predicates
        "IS" => Some(Keyword::Is),
        "IN" => Some(Keyword::In),
        "LIKE" => Some(Keyword::Like),
        "ILIKE" => Some(Keyword::Ilike),
        "BETWEEN" => Some(Keyword::Between),
        "EXISTS" => Some(Keyword::Exists),

        // Window functions
        "OVER" => Some(Keyword::Over),
        "PARTITION" => Some(Keyword::Partition),
        "ROWS" => Some(Keyword::Rows),
        "RANGE" => Some(Keyword::Range),
        "UNBOUNDED" => Some(Keyword::Unbounded),
        "PRECEDING" => Some(Keyword::Preceding),
        "FOLLOWING" => Some(Keyword::Following),
        "CURRENT" => Some(Keyword::Current),
        "ROW" => Some(Keyword::Row),

        // CTEs
        "RECURSIVE" => Some(Keyword::Recursive),

        // Literals
        "NULL" => Some(Keyword::Null),
        "TRUE" => Some(Keyword::True),
        "FALSE" => Some(Keyword::False),

        // Constraints
        "PRIMARY" => Some(Keyword::Primary),
        "KEY" => Some(Keyword::Key),
        "UNIQUE" => Some(Keyword::Unique),
        "CHECK" => Some(Keyword::Check),
        "DEFAULT" => Some(Keyword::Default),
        "REFERENCES" => Some(Keyword::References),
        "FOREIGN" => Some(Keyword::Foreign),
        "CONSTRAINT" => Some(Keyword::Constraint),
        "CASCADE" => Some(Keyword::Cascade),
        "RESTRICT" => Some(Keyword::Restrict),

        // Conditional
        "IF" => Some(Keyword::If),
        "NULLS" => Some(Keyword::Nulls),
        "FIRST" => Some(Keyword::First),
        "LAST" => Some(Keyword::Last),

        // Expressions
        "CAST" => Some(Keyword::Cast),
        "CASE" => Some(Keyword::Case),
        "WHEN" => Some(Keyword::When),
        "THEN" => Some(Keyword::Then),
        "ELSE" => Some(Keyword::Else),
        "END" => Some(Keyword::End),

        // Transaction control
        "BEGIN" => Some(Keyword::Begin),
        "COMMIT" => Some(Keyword::Commit),
        "ROLLBACK" => Some(Keyword::Rollback),
        "SAVEPOINT" => Some(Keyword::Savepoint),
        "RELEASE" => Some(Keyword::Release),
        "TRANSACTION" => Some(Keyword::Transaction),

        // Query analysis
        "EXPLAIN" => Some(Keyword::Explain),
        "ANALYZE" => Some(Keyword::Analyze),

        // Set operations
        "UNION" => Some(Keyword::Union),
        "INTERSECT" => Some(Keyword::Intersect),
        "EXCEPT" => Some(Keyword::Except),

        // Views
        "VIEW" => Some(Keyword::View),
        "REPLACE" => Some(Keyword::Replace),

        // DCL
        "GRANT" => Some(Keyword::Grant),
        "REVOKE" => Some(Keyword::Revoke),
        "PRIVILEGES" => Some(Keyword::Privileges),
        "PUBLIC" => Some(Keyword::Public),

        // RETURNING
        "RETURNING" => Some(Keyword::Returning),

        // UPSERT
        "CONFLICT" => Some(Keyword::Conflict),
        "DO" => Some(Keyword::Do),
        "NOTHING" => Some(Keyword::Nothing),

        // Join variants
        "NATURAL" => Some(Keyword::Natural),
        "USING" => Some(Keyword::Using),

        // Schema
        "SCHEMA" => Some(Keyword::Schema),

        // Sequences
        "SEQUENCE" => Some(Keyword::Sequence),
        "INCREMENT" => Some(Keyword::Increment),
        "MINVALUE" => Some(Keyword::Minvalue),
        "MAXVALUE" => Some(Keyword::Maxvalue),
        "START" => Some(Keyword::Start),
        "CACHE" => Some(Keyword::Cache),
        "CYCLE" => Some(Keyword::Cycle),
        "NO" => Some(Keyword::No),

        // Maintenance
        "VACUUM" => Some(Keyword::Vacuum),
        "REINDEX" => Some(Keyword::Reindex),

        // Session
        "SHOW" => Some(Keyword::Show),

        // Bulk copy
        "COPY" => Some(Keyword::Copy),
        "STDIN" => Some(Keyword::Stdin),
        "STDOUT" => Some(Keyword::Stdout),

        // FOR UPDATE/SHARE
        "FOR" => Some(Keyword::For),
        "SHARE" => Some(Keyword::Share),
        "LOCK" => Some(Keyword::Lock),
        "NOWAIT" => Some(Keyword::Nowait),
        "SKIP" => Some(Keyword::Skip),
        "LOCKED" => Some(Keyword::Locked),

        // MERGE
        "MERGE" => Some(Keyword::Merge),
        "MATCHED" => Some(Keyword::Matched),

        // Prepared statements
        "PREPARE" => Some(Keyword::Prepare),
        "EXECUTE" => Some(Keyword::Execute),
        "DEALLOCATE" => Some(Keyword::Deallocate),

        // FETCH FIRST/NEXT
        "FETCH" => Some(Keyword::Fetch),
        "NEXT" => Some(Keyword::Next),
        "ONLY" => Some(Keyword::Only),

        // LATERAL
        "LATERAL" => Some(Keyword::Lateral),

        // Array
        "ARRAY" => Some(Keyword::Array),
        "ANY" => Some(Keyword::Any),
        "SOME" => Some(Keyword::Some),

        // LISTEN/NOTIFY
        "LISTEN" => Some(Keyword::Listen),
        "NOTIFY" => Some(Keyword::Notify),
        "PAYLOAD" => Some(Keyword::Payload),

        // TABLESAMPLE
        "TABLESAMPLE" => Some(Keyword::Tablesample),
        "BERNOULLI" => Some(Keyword::Bernoulli),
        "SYSTEM" => Some(Keyword::System),

        // Cursors
        "DECLARE" => Some(Keyword::Declare),
        "CURSOR" => Some(Keyword::Cursor),
        "CLOSE" => Some(Keyword::Close),
        "SCROLL" => Some(Keyword::Scroll),
        "HOLD" => Some(Keyword::Hold),
        "WITHOUT" => Some(Keyword::Without),
        "ABSOLUTE" => Some(Keyword::Absolute),
        "RELATIVE" => Some(Keyword::Relative),
        "FORWARD" => Some(Keyword::Forward),
        "BACKWARD" => Some(Keyword::Backward),
        "PERCENT" => Some(Keyword::Percent),

        // COMMENT ON
        "COMMENT" => Some(Keyword::Comment),

        // Materialized views
        "REFRESH" => Some(Keyword::Refresh),
        "MATERIALIZED" => Some(Keyword::Materialized),

        // Checkpoint
        "CHECKPOINT" => Some(Keyword::Checkpoint),

        // DO block
        "LANGUAGE" => Some(Keyword::Language),

        // ZyronDB custom
        "SEGMENTS" => Some(Keyword::Segments),
        "STATUS" => Some(Keyword::Status),
        "BUFFER" => Some(Keyword::Buffer),
        "POOL" => Some(Keyword::Pool),
        "TRANSACTIONS" => Some(Keyword::Transactions),
        "FORMAT" => Some(Keyword::Format),
        "STORAGE" => Some(Keyword::Storage),
        "COLUMNAR" => Some(Keyword::Columnar),
        "HEAP" => Some(Keyword::Heap),

        // TTL / data retention
        "TTL" => Some(Keyword::Ttl),
        "DAYS" => Some(Keyword::Days),
        "HOURS" => Some(Keyword::Hours),
        "MINUTES" => Some(Keyword::Minutes),
        "SECONDS" => Some(Keyword::Seconds),
        "ARCHIVE" => Some(Keyword::Archive),
        "RETAIN" => Some(Keyword::Retain),
        "EXPIRE" => Some(Keyword::Expire),

        // Scheduling
        "SCHEDULE" => Some(Keyword::Schedule),
        "EVERY" => Some(Keyword::Every),
        "PAUSE" => Some(Keyword::Pause),
        "RESUME" => Some(Keyword::Resume),
        "CRON" => Some(Keyword::Cron),

        // Optimize
        "OPTIMIZE" => Some(Keyword::Optimize),

        // Extended integer types
        "TINYINT" => Some(Keyword::Tinyint),
        "INT128" => Some(Keyword::Int128),
        "UINT8" => Some(Keyword::Uint8),
        "UINT16" => Some(Keyword::Uint16),
        "UINT32" => Some(Keyword::Uint32),
        "UINT64" => Some(Keyword::Uint64),
        "UINT128" => Some(Keyword::Uint128),

        // Vector
        "VECTOR" => Some(Keyword::Vector),

        // Analytics
        "ROLLUP" => Some(Keyword::Rollup),
        "CUBE" => Some(Keyword::Cube),
        "GROUPING" => Some(Keyword::Grouping),
        "SETS" => Some(Keyword::Sets),
        "QUALIFY" => Some(Keyword::Qualify),

        // Time travel
        "VERSIONING" => Some(Keyword::Versioning),
        "PERIOD" => Some(Keyword::Period),
        "OF" => Some(Keyword::Of),

        // Pipeline
        "PIPELINE" => Some(Keyword::Pipeline),
        "STAGE" => Some(Keyword::Stage),
        "SOURCE" => Some(Keyword::Source),
        "TARGET" => Some(Keyword::Target),
        "MODE" => Some(Keyword::Mode),
        "TRANSFORM" => Some(Keyword::Transform),
        "EXPECT" => Some(Keyword::Expect),
        "EXPECTATION" => Some(Keyword::Expectation),
        "VIOLATION" => Some(Keyword::Violation),
        "FAIL" => Some(Keyword::Fail),
        "QUARANTINE" => Some(Keyword::Quarantine),
        "RUN" => Some(Keyword::Run),

        // Archive / Restore
        "RESTORE" => Some(Keyword::Restore),

        // Branching / Versioning
        "BRANCH" => Some(Keyword::Branch),
        "VERSION" => Some(Keyword::Version),
        "PORTION" => Some(Keyword::Portion),
        "USE" => Some(Keyword::Use),
        "AT" => Some(Keyword::At),

        // Security
        "USER" => Some(Keyword::User),
        "ROLE" => Some(Keyword::Role),
        "PASSWORD" => Some(Keyword::Password),
        "LOGIN" => Some(Keyword::Login),
        "SUPERUSER" => Some(Keyword::Superuser),
        "VALID" => Some(Keyword::Valid),
        "UNTIL" => Some(Keyword::Until),
        "NOLOGIN" => Some(Keyword::Nologin),

        // Search
        "FULLTEXT" => Some(Keyword::Fulltext),
        "MATCH" => Some(Keyword::Match),
        "AGAINST" => Some(Keyword::Against),

        // CDC
        "ENABLE" => Some(Keyword::Enable),
        "DISABLE" => Some(Keyword::Disable),
        "FEED" => Some(Keyword::Feed),
        "CHANGE" => Some(Keyword::Change),
        "REPLICATION" => Some(Keyword::Replication),
        "SLOT" => Some(Keyword::Slot),
        "PLUGIN" => Some(Keyword::Plugin),
        "CDC" => Some(Keyword::Cdc),
        "STREAM" => Some(Keyword::Stream),
        "INGEST" => Some(Keyword::Ingest),
        "PUBLICATION" => Some(Keyword::Publication),
        "INCLUDE" => Some(Keyword::Include),
        "DDL" => Some(Keyword::Ddl),

        // Triggers, UDFs, procedures, aggregates
        "TRIGGER" => Some(Keyword::Trigger),
        "BEFORE" => Some(Keyword::Before),
        "AFTER" => Some(Keyword::After),
        "INSTEAD" => Some(Keyword::Instead),
        "EACH" => Some(Keyword::Each),
        "REFERENCING" => Some(Keyword::Referencing),
        "OLD" => Some(Keyword::Old),
        "NEW" => Some(Keyword::New),
        "PRIORITY" => Some(Keyword::Priority),
        "FUNCTION" => Some(Keyword::Function),
        "RETURNS" => Some(Keyword::Returns),
        "IMMUTABLE" => Some(Keyword::Immutable),
        "VOLATILE" => Some(Keyword::Volatile),
        "STABLE" => Some(Keyword::Stable),
        "SETOF" => Some(Keyword::Setof),
        "PROCEDURE" => Some(Keyword::Procedure),
        "CALL" => Some(Keyword::Call),
        "AGGREGATE" => Some(Keyword::Aggregate),
        "SFUNC" => Some(Keyword::Sfunc),
        "STYPE" => Some(Keyword::Stype),
        "FINALFUNC" => Some(Keyword::Finalfunc),
        "COMBINEFUNC" => Some(Keyword::Combinefunc),
        "INITCOND" => Some(Keyword::Initcond),
        "DEFINER" => Some(Keyword::Definer),
        "SECURITY" => Some(Keyword::Security),
        "INVOKER" => Some(Keyword::Invoker),
        "HANDLER" => Some(Keyword::Handler),
        "EVENT" => Some(Keyword::Event),
        "PREVIEW" => Some(Keyword::Preview),
        "PLSQL" => Some(Keyword::Plsql),
        "SQL" => Some(Keyword::Sql),
        "SYMBOL" => Some(Keyword::Symbol),
        "RUST" => Some(Keyword::Rust),
        "RUST_VECTORIZED" => Some(Keyword::RustVectorized),

        // Data type keywords
        "TYPE" => Some(Keyword::Type),
        "INT" => Some(Keyword::Int),
        "INTEGER" => Some(Keyword::Integer),
        "SMALLINT" => Some(Keyword::Smallint),
        "BIGINT" => Some(Keyword::Bigint),
        "REAL" => Some(Keyword::Real),
        "DOUBLE" => Some(Keyword::Double),
        "PRECISION" => Some(Keyword::Precision),
        "FLOAT" => Some(Keyword::Float),
        "BOOLEAN" | "BOOL" => Some(Keyword::Boolean),
        "CHAR" | "CHARACTER" => Some(Keyword::Char),
        "VARCHAR" => Some(Keyword::Varchar),
        "TEXT" => Some(Keyword::Text),
        "DECIMAL" => Some(Keyword::Decimal),
        "NUMERIC" => Some(Keyword::Numeric),
        "DATE" => Some(Keyword::Date),
        "TIME" => Some(Keyword::Time),
        "TIMESTAMP" => Some(Keyword::Timestamp),
        "TIMESTAMPTZ" => Some(Keyword::Timestamptz),
        "INTERVAL" => Some(Keyword::Interval),
        "UUID" => Some(Keyword::Uuid),
        "JSON" => Some(Keyword::Json),
        "JSONB" => Some(Keyword::Jsonb),
        "BINARY" => Some(Keyword::Binary),
        "VARBINARY" => Some(Keyword::Varbinary),
        "BYTEA" => Some(Keyword::Bytea),
        "WITH" => Some(Keyword::With),
        "ZONE" => Some(Keyword::Zone),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_keyword_known() {
        assert_eq!(lookup_keyword("SELECT"), Some(Keyword::Select));
        assert_eq!(lookup_keyword("from"), Some(Keyword::From));
        assert_eq!(lookup_keyword("WhErE"), Some(Keyword::Where));
        assert_eq!(lookup_keyword("INSERT"), Some(Keyword::Insert));
        assert_eq!(lookup_keyword("create"), Some(Keyword::Create));
        assert_eq!(lookup_keyword("ALTER"), Some(Keyword::Alter));
        assert_eq!(lookup_keyword("TRUNCATE"), Some(Keyword::Truncate));
        assert_eq!(lookup_keyword("BEGIN"), Some(Keyword::Begin));
        assert_eq!(lookup_keyword("COMMIT"), Some(Keyword::Commit));
        assert_eq!(lookup_keyword("ROLLBACK"), Some(Keyword::Rollback));
        assert_eq!(lookup_keyword("EXPLAIN"), Some(Keyword::Explain));
        assert_eq!(lookup_keyword("ANALYZE"), Some(Keyword::Analyze));
    }

    #[test]
    fn test_lookup_keyword_unknown() {
        assert_eq!(lookup_keyword("users"), None);
        assert_eq!(lookup_keyword("my_column"), None);
        assert_eq!(lookup_keyword("foo"), None);
    }

    #[test]
    fn test_lookup_keyword_case_insensitive() {
        assert_eq!(lookup_keyword("select"), Some(Keyword::Select));
        assert_eq!(lookup_keyword("SELECT"), Some(Keyword::Select));
        assert_eq!(lookup_keyword("SeLeCt"), Some(Keyword::Select));
    }

    #[test]
    fn test_lookup_keyword_type_aliases() {
        assert_eq!(lookup_keyword("BOOL"), Some(Keyword::Boolean));
        assert_eq!(lookup_keyword("BOOLEAN"), Some(Keyword::Boolean));
        assert_eq!(lookup_keyword("CHARACTER"), Some(Keyword::Char));
        assert_eq!(lookup_keyword("CHAR"), Some(Keyword::Char));
    }

    #[test]
    fn test_lookup_keyword_data_types() {
        assert_eq!(lookup_keyword("INT"), Some(Keyword::Int));
        assert_eq!(lookup_keyword("INTEGER"), Some(Keyword::Integer));
        assert_eq!(lookup_keyword("BIGINT"), Some(Keyword::Bigint));
        assert_eq!(lookup_keyword("VARCHAR"), Some(Keyword::Varchar));
        assert_eq!(lookup_keyword("TIMESTAMP"), Some(Keyword::Timestamp));
        assert_eq!(lookup_keyword("TIMESTAMPTZ"), Some(Keyword::Timestamptz));
        assert_eq!(lookup_keyword("UUID"), Some(Keyword::Uuid));
        assert_eq!(lookup_keyword("JSONB"), Some(Keyword::Jsonb));
    }

    #[test]
    fn test_span_new() {
        let span = Span::new(10, 5);
        assert_eq!(span.offset, 10);
        assert_eq!(span.length, 5);
    }

    #[test]
    fn test_spanned_token() {
        let st = SpannedToken::new(Token::Keyword(Keyword::Select), Span::new(0, 6));
        assert_eq!(st.token, Token::Keyword(Keyword::Select));
        assert_eq!(st.span, Span::new(0, 6));
    }

    #[test]
    fn test_token_equality() {
        assert_eq!(Token::Plus, Token::Plus);
        assert_eq!(Token::Integer(42), Token::Integer(42));
        assert_ne!(Token::Integer(42), Token::Integer(43));
        assert_eq!(
            Token::String("hello".to_string()),
            Token::String("hello".to_string())
        );
        assert_eq!(Token::Eof, Token::Eof);
    }
}
