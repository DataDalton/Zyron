//! Renders a parsed statement back to SQL text.
//!
//! The inverse of the parser for the statements a user object can hold: a
//! query, DML, and the CREATE statements the catalog stores as text (view,
//! materialized view, function, procedure, schedule, pipeline, streaming
//! job, endpoint). The upgrade rewriter changes a statement tree and this is
//! how the changed tree gets back into the catalog, so the property that
//! matters is that `parse(render(parse(sql)))` is the tree `parse(sql)`
//! produced. Every construct is rendered in a spelling the parser reads
//! back to the same node, and a construct with no such spelling is refused
//! rather than approximated.
//!
//! Parentheses are never added. The parser keeps the ones a statement was
//! written with as `Expr::Nested`, and its operator precedence is what
//! decided the shape of the tree, so rendering the tree flat in that same
//! precedence reads back to the same tree. Identifiers keep their case and
//! are quoted only when the lexer could not read them bare: a name with a
//! character outside the identifier alphabet, a leading digit, or a
//! reserved word that cannot stand as an identifier

use std::fmt::Write as _;

use crate::ast::*;
use crate::expr_sql::data_type_to_sql;
use crate::parser::keyword_to_ident_str;
use crate::token::lookup_keyword;

/// Why a tree could not be rendered
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnparseError {
    /// The tree holds something the grammar has no spelling for
    Unsupported { what: String },
}

impl std::fmt::Display for UnparseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnparseError::Unsupported { what } => {
                write!(f, "{what} has no SQL spelling this release renders")
            }
        }
    }
}

impl std::error::Error for UnparseError {}

type Out = Result<(), UnparseError>;

fn unsupported<T>(what: impl Into<String>) -> Result<T, UnparseError> {
    Err(UnparseError::Unsupported { what: what.into() })
}

/// Renders a statement as one line of SQL
pub fn statement_to_sql(statement: &Statement) -> Result<String, UnparseError> {
    let mut out = String::new();
    write_statement(&mut out, statement)?;
    Ok(out)
}

/// Renders a query as SQL
pub fn select_to_sql(query: &SelectStatement) -> Result<String, UnparseError> {
    let mut out = String::new();
    write_select(&mut out, query)?;
    Ok(out)
}

/// Renders an expression as SQL
/// Writes one FROM item back as SQL.
///
/// A caller keying a per-bind memo on a relation as written uses this: two
/// items that unparse alike name the same relation.
pub fn table_ref_to_sql(table: &TableRef) -> Result<String, UnparseError> {
    let mut out = String::new();
    write_table_ref(&mut out, table)?;
    Ok(out)
}

pub fn expr_to_sql(expr: &Expr) -> Result<String, UnparseError> {
    let mut out = String::new();
    write_expr(&mut out, expr)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Names and literals
// ---------------------------------------------------------------------------

/// Whether the lexer reads this name back as the same identifier bare
fn bare_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return false;
    }
    match lookup_keyword(name) {
        Some(keyword) => keyword_to_ident_str(keyword).is_some(),
        None => true,
    }
}

fn write_ident(out: &mut String, name: &str) {
    if bare_identifier(name) {
        out.push_str(name);
    } else {
        out.push('"');
        out.push_str(&name.replace('"', "\"\""));
        out.push('"');
    }
}

/// A name the parser joined from dotted parts, rendered part by part so a
/// part that needs quoting gets it and the dots stay dots
fn write_qualified(out: &mut String, name: &str) {
    for (i, part) in name.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        write_ident(out, part);
    }
}

fn write_string(out: &mut String, text: &str) {
    out.push('\'');
    out.push_str(&text.replace('\'', "''"));
    out.push('\'');
}

/// A float in a spelling the lexer reads as a float rather than an integer
fn write_float(out: &mut String, value: f64) -> Out {
    if !value.is_finite() {
        return unsupported("a float literal that is not a finite number");
    }
    let text = format!("{value:?}");
    out.push_str(&text);
    Ok(())
}

/// An interval as the units its parts are exact in. The nanosecond field
/// is spelled in hours, minutes, seconds, and nanoseconds so every count
/// is small enough for the parser's floating point arithmetic to read back
/// exactly, and days stay days because the parser keeps them apart
fn interval_text(interval: &zyron_common::Interval) -> String {
    let mut parts = Vec::new();
    if interval.months != 0 {
        parts.push(format!("{} month", interval.months));
    }
    if interval.days != 0 {
        parts.push(format!("{} day", interval.days));
    }
    let nanos = interval.nanoseconds;
    let hours = nanos / 3_600_000_000_000;
    let minutes = (nanos % 3_600_000_000_000) / 60_000_000_000;
    let seconds = (nanos % 60_000_000_000) / 1_000_000_000;
    let rest = nanos % 1_000_000_000;
    if hours != 0 {
        parts.push(format!("{hours} hour"));
    }
    if minutes != 0 {
        parts.push(format!("{minutes} minute"));
    }
    if seconds != 0 {
        parts.push(format!("{seconds} second"));
    }
    if rest != 0 {
        parts.push(format!("{rest} nanosecond"));
    }
    if parts.is_empty() {
        parts.push("0 second".to_string());
    }
    parts.join(" ")
}

fn write_interval(out: &mut String, interval: &zyron_common::Interval) {
    out.push_str("INTERVAL ");
    write_string(out, &interval_text(interval));
}

fn write_literal(out: &mut String, literal: &LiteralValue) -> Out {
    match literal {
        LiteralValue::Integer(v) => write!(out, "{v}").ok(),
        LiteralValue::Int128(v) => write!(out, "{v}").ok(),
        LiteralValue::Decimal { digits, scale } => {
            out.push_str(&zyron_common::format_decimal(*digits, *scale));
            Some(())
        }
        LiteralValue::Float(v) => return write_float(out, *v),
        LiteralValue::String(s) => {
            write_string(out, s);
            Some(())
        }
        LiteralValue::Boolean(true) => {
            out.push_str("TRUE");
            Some(())
        }
        LiteralValue::Boolean(false) => {
            out.push_str("FALSE");
            Some(())
        }
        LiteralValue::Null => {
            out.push_str("NULL");
            Some(())
        }
        LiteralValue::Interval(interval) => {
            write_interval(out, interval);
            Some(())
        }
        LiteralValue::Bytes(_) => {
            return unsupported("a literal already in its stored binary form");
        }
    };
    Ok(())
}

fn write_list<T>(
    out: &mut String,
    items: &[T],
    mut each: impl FnMut(&mut String, &T) -> Out,
) -> Out {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        each(out, item)?;
    }
    Ok(())
}

fn write_ident_list(out: &mut String, names: &[String]) {
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_ident(out, name);
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn binary_operator(op: BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Plus => "+",
        BinaryOperator::Minus => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Modulo => "%",
        BinaryOperator::Eq => "=",
        BinaryOperator::Neq => "<>",
        BinaryOperator::Lt => "<",
        BinaryOperator::Gt => ">",
        BinaryOperator::LtEq => "<=",
        BinaryOperator::GtEq => ">=",
        BinaryOperator::And => "AND",
        BinaryOperator::Or => "OR",
        BinaryOperator::Concat => "||",
    }
}

fn negation(negated: bool) -> &'static str {
    if negated { "NOT " } else { "" }
}

pub(crate) fn write_expr(out: &mut String, expr: &Expr) -> Out {
    match expr {
        Expr::Identifier(name) => {
            if name == "*" {
                out.push('*');
            } else {
                write_ident(out, name);
            }
        }
        Expr::QualifiedIdentifier { table, column } => {
            write_qualified(out, table);
            out.push('.');
            write_ident(out, column);
        }
        Expr::Literal(literal) => write_literal(out, literal)?,
        Expr::Lambda { parameter, body } => {
            write_ident(out, parameter);
            out.push_str(" -> ");
            write_expr(out, body)?;
        }
        Expr::Collate { expr, collation } => {
            write_expr(out, expr)?;
            out.push_str(" COLLATE ");
            write_string(out, collation);
        }
        Expr::BinaryOp { left, op, right } => {
            write_expr(out, left)?;
            out.push(' ');
            out.push_str(binary_operator(*op));
            out.push(' ');
            write_expr(out, right)?;
        }
        Expr::UnaryOp { op, expr } => {
            match op {
                UnaryOperator::Not => out.push_str("NOT "),
                UnaryOperator::Minus => out.push('-'),
            }
            write_expr(out, expr)?;
        }
        Expr::IsNull { expr, negated } => {
            write_expr(out, expr)?;
            out.push_str(" IS ");
            out.push_str(negation(*negated));
            out.push_str("NULL");
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            write_expr(out, expr)?;
            out.push(' ');
            out.push_str(negation(*negated));
            out.push_str("IN (");
            write_list(out, list, write_expr)?;
            out.push(')');
        }
        Expr::Between {
            expr,
            low,
            high,
            negated,
        } => {
            write_expr(out, expr)?;
            out.push(' ');
            out.push_str(negation(*negated));
            out.push_str("BETWEEN ");
            write_expr(out, low)?;
            out.push_str(" AND ");
            write_expr(out, high)?;
        }
        Expr::Like {
            expr,
            pattern,
            negated,
        } => {
            write_expr(out, expr)?;
            out.push(' ');
            out.push_str(negation(*negated));
            out.push_str("LIKE ");
            write_expr(out, pattern)?;
        }
        Expr::ILike {
            expr,
            pattern,
            negated,
        } => {
            write_expr(out, expr)?;
            out.push(' ');
            out.push_str(negation(*negated));
            out.push_str("ILIKE ");
            write_expr(out, pattern)?;
        }
        Expr::Function {
            name,
            args,
            distinct,
        } => write_function(out, name, args, *distinct)?,
        Expr::Cast { expr, data_type } => {
            out.push_str("CAST(");
            write_expr(out, expr)?;
            out.push_str(" AS ");
            out.push_str(&data_type_to_sql(data_type));
            out.push(')');
        }
        Expr::Case {
            operand,
            conditions,
            else_result,
        } => {
            out.push_str("CASE");
            if let Some(operand) = operand {
                out.push(' ');
                write_expr(out, operand)?;
            }
            for when in conditions {
                out.push_str(" WHEN ");
                write_expr(out, &when.condition)?;
                out.push_str(" THEN ");
                write_expr(out, &when.result)?;
            }
            if let Some(else_result) = else_result {
                out.push_str(" ELSE ");
                write_expr(out, else_result)?;
            }
            out.push_str(" END");
        }
        Expr::Nested(inner) => {
            out.push('(');
            write_expr(out, inner)?;
            out.push(')');
        }
        Expr::Subquery(query) => {
            out.push('(');
            write_select(out, query)?;
            out.push(')');
        }
        Expr::InSubquery {
            expr,
            query,
            negated,
        } => {
            write_expr(out, expr)?;
            out.push(' ');
            out.push_str(negation(*negated));
            out.push_str("IN (");
            write_select(out, query)?;
            out.push(')');
        }
        Expr::Exists { query, negated } => {
            out.push_str(negation(*negated));
            out.push_str("EXISTS (");
            write_select(out, query)?;
            out.push(')');
        }
        Expr::WindowFunction {
            function,
            partition_by,
            order_by,
            frame,
        } => {
            write_expr(out, function)?;
            out.push_str(" OVER (");
            let mut wrote = false;
            if !partition_by.is_empty() {
                out.push_str("PARTITION BY ");
                write_list(out, partition_by, write_expr)?;
                wrote = true;
            }
            if !order_by.is_empty() {
                if wrote {
                    out.push(' ');
                }
                out.push_str("ORDER BY ");
                write_list(out, order_by, write_order_by)?;
                wrote = true;
            }
            if let Some(frame) = frame {
                if wrote {
                    out.push(' ');
                }
                write_window_frame(out, frame);
            }
            out.push(')');
        }
        Expr::ArrayConstructor(items) => {
            out.push_str("ARRAY[");
            write_list(out, items, write_expr)?;
            out.push(']');
        }
        Expr::ArraySubscript { array, index } => {
            write_expr(out, array)?;
            out.push('[');
            write_expr(out, index)?;
            out.push(']');
        }
        Expr::JsonAccess { left, op, right } => match op {
            JsonOperator::Dot => {
                // A dotted chain into a variant or struct, which the parser
                // builds from bare identifiers, so the field has to be one
                let Expr::Literal(LiteralValue::String(field)) = right.as_ref() else {
                    return unsupported("a dotted field access whose field is not a name");
                };
                write_expr(out, left)?;
                out.push('.');
                write_ident(out, field);
            }
            other => {
                write_expr(out, left)?;
                out.push(' ');
                out.push_str(match other {
                    JsonOperator::Arrow => "->",
                    JsonOperator::DoubleArrow => "->>",
                    JsonOperator::HashArrow => "#>",
                    JsonOperator::HashDoubleArrow => "#>>",
                    JsonOperator::Dot => ".",
                });
                out.push(' ');
                write_expr(out, right)?;
            }
        },
        Expr::JsonContains { left, op, right } => {
            write_expr(out, left)?;
            out.push_str(match op {
                JsonContainsOp::Contains => " @> ",
                JsonContainsOp::ContainedBy => " <@ ",
            });
            write_expr(out, right)?;
        }
        Expr::JsonExists { left, op, right } => {
            write_expr(out, left)?;
            out.push_str(match op {
                JsonExistsOp::Exists => " ? ",
                JsonExistsOp::ExistsAny => " ?| ",
                JsonExistsOp::ExistsAll => " ?& ",
            });
            write_expr(out, right)?;
        }
        Expr::AnySubquery { query } => {
            out.push_str("ANY (");
            write_select(out, query)?;
            out.push(')');
        }
        Expr::AllSubquery { query } => {
            out.push_str("ALL (");
            write_select(out, query)?;
            out.push(')');
        }
        Expr::Parameter(index) => {
            let _ = write!(out, "${index}");
        }
        Expr::VectorDistance { left, op, right } => {
            write_expr(out, left)?;
            out.push_str(match op {
                VectorDistanceOp::Cosine => " <=> ",
                VectorDistanceOp::L2 => " <-> ",
                VectorDistanceOp::DotProduct => " <#> ",
            });
            write_expr(out, right)?;
        }
        Expr::MatchAgainst {
            columns,
            query,
            mode,
        } => {
            out.push_str("MATCH (");
            write_ident_list(out, columns);
            out.push_str(") AGAINST (");
            write_expr(out, query)?;
            if let Some(mode) = mode {
                out.push_str(" IN ");
                out.push_str(mode);
            }
            out.push(')');
        }
        Expr::TemporalRef { inner, temporal } => {
            write_expr(out, inner)?;
            match temporal.as_ref() {
                AsOf::Timestamp(when) => {
                    out.push_str(" AS OF ");
                    write_expr(out, when)?;
                }
                AsOf::Version(version) => {
                    out.push_str(" VERSION AS OF ");
                    write_expr(out, version)?;
                }
                AsOf::Branch(branch) => {
                    out.push_str(" IN BRANCH ");
                    write_expr(out, branch)?;
                }
                _ => return unsupported("a period clause in expression position"),
            }
        }
    }
    Ok(())
}

fn write_function(out: &mut String, name: &str, args: &[FunctionArg], distinct: bool) -> Out {
    // EXTRACT is parsed from its keyword spelling only, so the lowered call
    // goes back to that spelling
    if name.eq_ignore_ascii_case("extract") && !distinct && args.len() == 2 {
        if let (
            FunctionArg::Unnamed(Expr::Literal(LiteralValue::String(field))),
            FunctionArg::Unnamed(source),
        ) = (&args[0], &args[1])
        {
            out.push_str("EXTRACT(");
            write_string(out, field);
            out.push_str(" FROM ");
            write_expr(out, source)?;
            out.push(')');
            return Ok(());
        }
    }
    write_qualified(out, name);
    out.push('(');
    if distinct {
        out.push_str("DISTINCT ");
    }
    write_list(out, args, |out, arg| match arg {
        FunctionArg::Unnamed(expr) => write_expr(out, expr),
        FunctionArg::Named { name, value } => {
            write_ident(out, name);
            out.push_str(" => ");
            write_expr(out, value)
        }
        FunctionArg::Wildcard => {
            out.push('*');
            Ok(())
        }
    })?;
    out.push(')');
    Ok(())
}

fn write_order_by(out: &mut String, item: &OrderByExpr) -> Out {
    write_expr(out, &item.expr)?;
    match item.asc {
        Some(true) => out.push_str(" ASC"),
        Some(false) => out.push_str(" DESC"),
        None => {}
    }
    match item.nulls_first {
        Some(true) => out.push_str(" NULLS FIRST"),
        Some(false) => out.push_str(" NULLS LAST"),
        None => {}
    }
    Ok(())
}

fn write_window_frame(out: &mut String, frame: &WindowFrame) {
    out.push_str(match frame.mode {
        WindowFrameMode::Rows => "ROWS ",
        WindowFrameMode::Range => "RANGE ",
    });
    match &frame.end {
        Some(end) => {
            out.push_str("BETWEEN ");
            write_frame_bound(out, &frame.start);
            out.push_str(" AND ");
            write_frame_bound(out, end);
        }
        None => write_frame_bound(out, &frame.start),
    }
}

fn write_frame_bound(out: &mut String, bound: &WindowFrameBound) {
    let direction = |d: &WindowFrameDirection| match d {
        WindowFrameDirection::Preceding => "PRECEDING",
        WindowFrameDirection::Following => "FOLLOWING",
    };
    match bound {
        WindowFrameBound::CurrentRow => out.push_str("CURRENT ROW"),
        WindowFrameBound::Unbounded(d) => {
            out.push_str("UNBOUNDED ");
            out.push_str(direction(d));
        }
        WindowFrameBound::Offset(n, d) => {
            let _ = write!(out, "{n} {}", direction(d));
        }
        WindowFrameBound::IntervalBound(interval, d) => {
            write_interval(out, interval);
            out.push(' ');
            out.push_str(direction(d));
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

pub(crate) fn write_select(out: &mut String, query: &SelectStatement) -> Out {
    write_select_core(out, query)?;
    for item in &query.set_ops {
        out.push_str(match item.op {
            SetOpType::Union => " UNION ",
            SetOpType::Intersect => " INTERSECT ",
            SetOpType::Except => " EXCEPT ",
        });
        if item.all {
            out.push_str("ALL ");
        }
        write_select_core(out, &item.right)?;
    }
    if !query.order_by.is_empty() {
        out.push_str(" ORDER BY ");
        write_list(out, &query.order_by, write_order_by)?;
    }
    if let Some(limit) = &query.limit {
        out.push_str(" LIMIT ");
        write_expr(out, limit)?;
    }
    if let Some(offset) = &query.offset {
        out.push_str(" OFFSET ");
        write_expr(out, offset)?;
    }
    if let Some(fetch) = &query.fetch {
        out.push_str(" FETCH FIRST ");
        write_expr(out, &fetch.count)?;
        if fetch.percent {
            out.push_str(" PERCENT");
        }
        out.push_str(" ROWS ");
        out.push_str(if fetch.with_ties { "WITH TIES" } else { "ONLY" });
    }
    if let Some(for_clause) = &query.for_clause {
        out.push_str(match for_clause.lock_type {
            ForLockType::Update => " FOR UPDATE",
            ForLockType::Share => " FOR SHARE",
            ForLockType::NoKeyUpdate => " FOR NO KEY UPDATE",
            ForLockType::KeyShare => " FOR KEY SHARE",
        });
        if !for_clause.tables.is_empty() {
            out.push_str(" OF ");
            write_ident_list(out, &for_clause.tables);
        }
        match for_clause.wait {
            ForWait::Wait => {}
            ForWait::Nowait => out.push_str(" NOWAIT"),
            ForWait::SkipLocked => out.push_str(" SKIP LOCKED"),
        }
    }
    Ok(())
}

/// One SELECT without its set operations and trailing clauses, which is
/// what each side of a UNION is
fn write_select_core(out: &mut String, query: &SelectStatement) -> Out {
    if let Some(with) = &query.with {
        out.push_str("WITH ");
        if with.recursive {
            out.push_str("RECURSIVE ");
        }
        write_list(out, &with.ctes, |out, cte| {
            write_ident(out, &cte.name);
            if !cte.columns.is_empty() {
                out.push_str(" (");
                write_ident_list(out, &cte.columns);
                out.push(')');
            }
            out.push_str(" AS (");
            write_select(out, &cte.query)?;
            out.push(')');
            Ok(())
        })?;
        out.push(' ');
    }
    out.push_str("SELECT ");
    if query.distinct {
        out.push_str("DISTINCT ");
        if !query.distinct_on.is_empty() {
            out.push_str("ON (");
            write_list(out, &query.distinct_on, write_expr)?;
            out.push_str(") ");
        }
    }
    write_list(out, &query.projections, write_select_item)?;
    if !query.from.is_empty() {
        out.push_str(" FROM ");
        write_list(out, &query.from, write_table_ref)?;
    }
    if let Some(filter) = &query.where_clause {
        out.push_str(" WHERE ");
        write_expr(out, filter)?;
    }
    if let Some(sets) = &query.group_by_sets {
        out.push_str(" GROUP BY ");
        match sets {
            GroupBySets::Rollup(exprs) => {
                out.push_str("ROLLUP (");
                write_list(out, exprs, write_expr)?;
                out.push(')');
            }
            GroupBySets::Cube(exprs) => {
                out.push_str("CUBE (");
                write_list(out, exprs, write_expr)?;
                out.push(')');
            }
            GroupBySets::GroupingSets(sets) => {
                out.push_str("GROUPING SETS (");
                write_list(out, sets, |out, set| {
                    out.push('(');
                    write_list(out, set, write_expr)?;
                    out.push(')');
                    Ok(())
                })?;
                out.push(')');
            }
        }
    } else if !query.group_by.is_empty() {
        out.push_str(" GROUP BY ");
        write_list(out, &query.group_by, write_expr)?;
    }
    if let Some(having) = &query.having {
        out.push_str(" HAVING ");
        write_expr(out, having)?;
    }
    if let Some(qualify) = &query.qualify {
        out.push_str(" QUALIFY ");
        write_expr(out, qualify)?;
    }
    match query.soft_delete_mode {
        SoftDeleteSelectMode::Default => {}
        SoftDeleteSelectMode::IncludingDeleted => out.push_str(" INCLUDING DELETED"),
        SoftDeleteSelectMode::OnlyDeleted => out.push_str(" ONLY DELETED"),
    }
    Ok(())
}

fn write_select_item(out: &mut String, item: &SelectItem) -> Out {
    match item {
        SelectItem::Wildcard => out.push('*'),
        SelectItem::QualifiedWildcard(table) => {
            write_qualified(out, table);
            out.push_str(".*");
        }
        SelectItem::Expr(expr, alias) => {
            write_expr(out, expr)?;
            if let Some(alias) = alias {
                out.push_str(" AS ");
                write_ident(out, alias);
            }
        }
    }
    Ok(())
}

fn write_as_of(out: &mut String, as_of: &AsOf) -> Out {
    match as_of {
        AsOf::Timestamp(when) => {
            out.push_str(" AS OF ");
            write_expr(out, when)?;
        }
        AsOf::Version(version) => {
            out.push_str(" VERSION AS OF ");
            write_expr(out, version)?;
        }
        AsOf::Branch(branch) => {
            out.push_str(" IN BRANCH ");
            write_expr(out, branch)?;
        }
        AsOf::SystemTime { start, end } => {
            out.push_str(" FOR SYSTEM BETWEEN ");
            write_expr(out, start)?;
            out.push_str(" AND ");
            write_expr(out, end)?;
        }
        AsOf::ApplicationTime { start, end } => {
            out.push_str(" FOR application_time BETWEEN ");
            write_expr(out, start)?;
            out.push_str(" AND ");
            write_expr(out, end)?;
        }
        AsOf::ForPortionOf { period, start, end } => {
            out.push_str(" FOR PORTION OF ");
            write_ident(out, period);
            out.push_str(" FROM ");
            write_expr(out, start)?;
            out.push_str(" TO ");
            write_expr(out, end)?;
        }
    }
    Ok(())
}

fn write_alias(out: &mut String, alias: &Option<String>) {
    if let Some(alias) = alias {
        out.push_str(" AS ");
        write_ident(out, alias);
    }
}

fn backend_word(backend: &ExternalBackendKind) -> &'static str {
    match backend {
        ExternalBackendKind::File => "FILE",
        ExternalBackendKind::S3 => "S3",
        ExternalBackendKind::Gcs => "GCS",
        ExternalBackendKind::Azure => "AZURE",
        ExternalBackendKind::Http => "HTTP",
        ExternalBackendKind::Zyron => "ZYRON",
    }
}

fn format_word(format: &ExternalFormatKind) -> &'static str {
    match format {
        ExternalFormatKind::Json => "JSON",
        ExternalFormatKind::JsonLines => "JSONLINES",
        ExternalFormatKind::Csv => "CSV",
        ExternalFormatKind::Parquet => "PARQUET",
        ExternalFormatKind::ArrowIpc => "ARROW",
        ExternalFormatKind::Avro => "AVRO",
    }
}

fn write_options(out: &mut String, options: &[(String, String)]) {
    out.push_str(" OPTIONS (");
    for (i, (key, value)) in options.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_ident(out, key);
        out.push_str(" = ");
        write_string(out, value);
    }
    out.push(')');
}

fn write_table_ref(out: &mut String, table: &TableRef) -> Out {
    match table {
        TableRef::Table { name, alias, as_of } => {
            write_qualified(out, name);
            if let Some(as_of) = as_of {
                write_as_of(out, as_of)?;
            }
            write_alias(out, alias);
        }
        TableRef::Join(join) => {
            write_table_ref(out, &join.left)?;
            if join.condition == JoinCondition::Natural {
                out.push_str(" NATURAL");
            }
            // An ASOF join is written back in its own spelling, so stored SQL
            // round-trips as the statement that was typed
            if let Some(asof) = &join.asof {
                out.push_str(match join.join_type {
                    JoinType::Left => " ASOF LEFT JOIN ",
                    _ => " ASOF JOIN ",
                });
                write_table_ref(out, &join.right)?;
                out.push_str(" MATCH_CONDITION (");
                write_expr(out, &asof.condition)?;
                out.push(')');
                if let JoinCondition::On(expr) = &join.condition {
                    out.push_str(" ON ");
                    write_expr(out, expr)?;
                }
                return Ok(());
            }
            out.push_str(match join.join_type {
                JoinType::Inner => " INNER JOIN ",
                JoinType::Left => " LEFT JOIN ",
                JoinType::Right => " RIGHT JOIN ",
                JoinType::Full => " FULL JOIN ",
                JoinType::Cross => " CROSS JOIN ",
            });
            write_table_ref(out, &join.right)?;
            match &join.condition {
                JoinCondition::On(expr) => {
                    out.push_str(" ON ");
                    write_expr(out, expr)?;
                }
                JoinCondition::Using(columns) => {
                    out.push_str(" USING (");
                    write_ident_list(out, columns);
                    out.push(')');
                }
                JoinCondition::Natural | JoinCondition::None => {}
            }
        }
        TableRef::Subquery { query, alias } => {
            out.push('(');
            write_select(out, query)?;
            out.push_str(") AS ");
            write_ident(out, alias);
        }
        TableRef::Lateral { subquery } => {
            out.push_str("LATERAL ");
            write_table_ref(out, subquery)?;
        }
        TableRef::TableFunction(function) => {
            write_qualified(out, &function.name);
            out.push('(');
            write_list(out, &function.args, |out, arg| match arg {
                FunctionArg::Unnamed(expr) => write_expr(out, expr),
                FunctionArg::Named { name, value } => {
                    write_ident(out, name);
                    out.push_str(" => ");
                    write_expr(out, value)
                }
                FunctionArg::Wildcard => {
                    out.push('*');
                    Ok(())
                }
            })?;
            out.push(')');
            write_alias(out, &function.alias);
        }
        TableRef::ExternalInline(external) => {
            out.push_str(backend_word(&external.backend));
            out.push(' ');
            write_string(out, &external.uri);
            out.push_str(" FORMAT ");
            out.push_str(format_word(&external.format));
            if !external.options.is_empty() {
                write_options(out, &external.options);
            }
            if !external.columns.is_empty() {
                out.push_str(" COLUMNS (");
                for (i, (name, data_type)) in external.columns.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_ident(out, name);
                    out.push(' ');
                    out.push_str(&data_type_to_sql(data_type));
                }
                out.push(')');
            }
            write_alias(out, &external.alias);
        }
        TableRef::Unnest(unnest) => {
            out.push_str("UNNEST(");
            write_list(out, &unnest.arrays, write_expr)?;
            out.push(')');
            if unnest.with_ordinality {
                out.push_str(" WITH ORDINALITY");
            }
            write_rows_function_alias(out, &unnest.alias, &unnest.column_aliases);
        }
        TableRef::Flatten(flatten) => {
            out.push_str("FLATTEN(");
            write_expr(out, &flatten.input)?;
            if let Some(path) = &flatten.path {
                out.push_str(", path => ");
                write_string(out, path);
            }
            if flatten.outer {
                out.push_str(", outer => TRUE");
            }
            if flatten.recursive {
                out.push_str(", recursive => TRUE");
            }
            out.push(')');
            write_rows_function_alias(out, &flatten.alias, &flatten.column_aliases);
        }
        TableRef::Pivot(pivot) => {
            write_table_ref(out, &pivot.input)?;
            out.push_str(" PIVOT (");
            write_list(out, &pivot.aggregates, |out, agg| {
                out.push_str(&agg.function);
                out.push('(');
                write_expr(out, &agg.argument)?;
                out.push(')');
                if let Some(alias) = &agg.alias {
                    out.push_str(" AS ");
                    write_ident(out, alias);
                }
                Ok(())
            })?;
            out.push_str(" FOR ");
            write_expr(out, &pivot.pivot_column)?;
            out.push_str(" IN (");
            write_list(out, &pivot.values, |out, value| {
                write_literal(out, &value.value)?;
                if let Some(alias) = &value.alias {
                    out.push_str(" AS ");
                    write_ident(out, alias);
                }
                Ok(())
            })?;
            out.push_str("))");
            write_alias(out, &pivot.alias);
        }
        TableRef::Unpivot(unpivot) => {
            write_table_ref(out, &unpivot.input)?;
            out.push_str(if unpivot.include_nulls {
                " UNPIVOT INCLUDE NULLS ("
            } else {
                " UNPIVOT EXCLUDE NULLS ("
            });
            if unpivot.value_columns.len() == 1 {
                write_ident(out, &unpivot.value_columns[0]);
            } else {
                out.push('(');
                write_ident_list(out, &unpivot.value_columns);
                out.push(')');
            }
            out.push_str(" FOR ");
            write_ident(out, &unpivot.name_column);
            out.push_str(" IN (");
            write_list(out, &unpivot.items, |out, item| {
                if item.columns.len() == 1 {
                    write_ident(out, &item.columns[0]);
                } else {
                    out.push('(');
                    write_ident_list(out, &item.columns);
                    out.push(')');
                }
                if let Some(label) = &item.label {
                    out.push_str(" AS ");
                    write_literal(out, label)?;
                }
                Ok(())
            })?;
            out.push_str("))");
            write_alias(out, &unpivot.alias);
        }
    }
    Ok(())
}

/// `[AS alias [(col, ...)]]` on UNNEST or FLATTEN.
fn write_rows_function_alias(out: &mut String, alias: &Option<String>, columns: &[String]) {
    let Some(alias) = alias else {
        return;
    };
    out.push_str(" AS ");
    write_ident(out, alias);
    if !columns.is_empty() {
        out.push_str(" (");
        write_ident_list(out, columns);
        out.push(')');
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

fn write_assignments(out: &mut String, assignments: &[Assignment]) -> Out {
    write_list(out, assignments, |out, assignment| {
        write_ident(out, &assignment.column);
        out.push_str(" = ");
        write_expr(out, &assignment.value)
    })
}

fn write_returning(out: &mut String, returning: &Option<Vec<SelectItem>>) -> Out {
    if let Some(items) = returning {
        out.push_str(" RETURNING ");
        write_list(out, items, write_select_item)?;
    }
    Ok(())
}

fn write_rows(out: &mut String, rows: &[Vec<Expr>]) -> Out {
    write_list(out, rows, |out, row| {
        out.push('(');
        write_list(out, row, write_expr)?;
        out.push(')');
        Ok(())
    })
}

fn write_params(out: &mut String, params: &[FunctionParam]) -> Out {
    write_list(out, params, |out, param| {
        write_ident(out, &param.name);
        out.push(' ');
        out.push_str(&data_type_to_sql(&param.data_type));
        if let Some(default) = &param.default_value {
            out.push_str(" DEFAULT ");
            write_expr(out, default)?;
        }
        Ok(())
    })
}

fn write_ttl_duration(out: &mut String, duration: &TtlDuration) {
    let _ = write!(out, "{} ", duration.value);
    out.push_str(match duration.unit {
        TtlUnit::Seconds => "SECONDS",
        TtlUnit::Minutes => "MINUTES",
        TtlUnit::Hours => "HOURS",
        TtlUnit::Days => "DAYS",
    });
}

fn write_mode_spec(out: &mut String, mode: &ExternalModeSpec) -> Out {
    match mode {
        ExternalModeSpec::OneShot => out.push_str("ONESHOT"),
        ExternalModeSpec::Watch => out.push_str("WATCH"),
        ExternalModeSpec::Scheduled { cron, every } => match (cron, every) {
            (Some(cron), _) => {
                out.push_str("SCHEDULED CRON ");
                write_string(out, cron);
            }
            (None, Some(every)) => {
                out.push_str("SCHEDULED EVERY ");
                write_string(out, every);
            }
            (None, None) => return unsupported("a schedule with neither a cron nor an interval"),
        },
    }
    Ok(())
}

pub(crate) fn write_statement(out: &mut String, statement: &Statement) -> Out {
    match statement {
        Statement::Select(query) => write_select(out, query)?,
        Statement::ValuesQuery(values) => {
            out.push_str("VALUES ");
            write_rows(out, &values.rows)?;
        }
        Statement::Insert(insert) => {
            out.push_str("INSERT INTO ");
            write_qualified(out, &insert.table);
            if !insert.columns.is_empty() {
                out.push_str(" (");
                write_ident_list(out, &insert.columns);
                out.push(')');
            }
            out.push(' ');
            match &insert.source {
                InsertSource::Values(rows) => {
                    out.push_str("VALUES ");
                    write_rows(out, rows)?;
                }
                InsertSource::Query(query) => write_select(out, query)?,
            }
            if let Some(conflict) = &insert.on_conflict {
                out.push_str(" ON CONFLICT");
                if !conflict.columns.is_empty() {
                    out.push_str(" (");
                    write_ident_list(out, &conflict.columns);
                    out.push(')');
                }
                match &conflict.action {
                    ConflictAction::DoNothing => out.push_str(" DO NOTHING"),
                    ConflictAction::DoUpdate(assignments) => {
                        out.push_str(" DO UPDATE SET ");
                        write_assignments(out, assignments)?;
                    }
                }
            }
            write_returning(out, &insert.returning)?;
        }
        Statement::Update(update) => {
            out.push_str("UPDATE ");
            write_qualified(out, &update.table);
            out.push_str(" SET ");
            write_assignments(out, &update.assignments)?;
            if let Some(filter) = &update.where_clause {
                out.push_str(" WHERE ");
                write_expr(out, filter)?;
            }
            write_returning(out, &update.returning)?;
        }
        Statement::Delete(delete) => {
            out.push_str("DELETE FROM ");
            write_qualified(out, &delete.table);
            if let Some(filter) = &delete.where_clause {
                out.push_str(" WHERE ");
                write_expr(out, filter)?;
            }
            write_returning(out, &delete.returning)?;
            if delete.hard {
                out.push_str(" HARD");
            }
        }
        Statement::Merge(merge) => {
            out.push_str("MERGE INTO ");
            write_ident(out, &merge.target);
            out.push_str(" USING ");
            write_table_ref(out, &merge.source)?;
            out.push_str(" ON ");
            write_expr(out, &merge.on)?;
            for clause in &merge.clauses {
                let (matched, condition, action) = match clause {
                    MergeClause::WhenMatched { condition, action } => (true, condition, action),
                    MergeClause::WhenNotMatched { condition, action } => (false, condition, action),
                };
                out.push_str(if matched {
                    " WHEN MATCHED"
                } else {
                    " WHEN NOT MATCHED"
                });
                if let Some(condition) = condition {
                    out.push_str(" AND ");
                    write_expr(out, condition)?;
                }
                out.push_str(" THEN ");
                match action {
                    MergeAction::Update(assignments) => {
                        out.push_str("UPDATE SET ");
                        write_assignments(out, assignments)?;
                    }
                    MergeAction::Delete => out.push_str("DELETE"),
                    MergeAction::Insert { columns, values } => {
                        out.push_str("INSERT");
                        if !columns.is_empty() {
                            out.push_str(" (");
                            write_ident_list(out, columns);
                            out.push(')');
                        }
                        out.push_str(" VALUES (");
                        write_list(out, values, write_expr)?;
                        out.push(')');
                    }
                    MergeAction::DoNothing => {
                        return unsupported("a MERGE clause that does nothing");
                    }
                }
            }
        }
        Statement::CreateView(view) => {
            out.push_str("CREATE ");
            if view.or_replace {
                out.push_str("OR REPLACE ");
            }
            out.push_str("VIEW ");
            write_qualified(out, &view.name);
            if !view.columns.is_empty() {
                out.push_str(" (");
                write_ident_list(out, &view.columns);
                out.push(')');
            }
            out.push_str(" AS ");
            write_select(out, &view.query)?;
        }
        Statement::CreateMaterializedView(view) => {
            out.push_str("CREATE MATERIALIZED VIEW ");
            if view.if_not_exists {
                out.push_str("IF NOT EXISTS ");
            }
            write_qualified(out, &view.name);
            out.push_str(" AS ");
            write_select(out, &view.query)?;
        }
        Statement::CreateFunction(function) => {
            out.push_str("CREATE ");
            if function.or_replace {
                out.push_str("OR REPLACE ");
            }
            out.push_str("FUNCTION ");
            write_qualified(out, &function.name);
            out.push('(');
            write_params(out, &function.params)?;
            out.push_str(") RETURNS ");
            match &function.return_type {
                FunctionReturnType::Scalar(data_type) => out.push_str(&data_type_to_sql(data_type)),
                FunctionReturnType::SetOf(data_type) => {
                    out.push_str("SETOF ");
                    out.push_str(&data_type_to_sql(data_type));
                }
                FunctionReturnType::Table(columns) => {
                    out.push_str("TABLE(");
                    write_params(out, columns)?;
                    out.push(')');
                }
            }
            out.push_str(" AS ");
            write_string(out, &function.body);
            out.push_str(" LANGUAGE ");
            out.push_str(match function.language {
                FunctionLanguage::Sql => "SQL",
                FunctionLanguage::Rust => "RUST",
                FunctionLanguage::RustVectorized => "RUST_VECTORIZED",
            });
            out.push_str(match function.volatility {
                Volatility::Immutable => " IMMUTABLE",
                Volatility::Stable => " STABLE",
                Volatility::Volatile => " VOLATILE",
            });
            if let (Some(library), Some(symbol)) = (&function.rust_library, &function.rust_symbol) {
                out.push_str(" LIBRARY ");
                write_string(out, library);
                out.push_str(" SYMBOL ");
                write_string(out, symbol);
            }
        }
        Statement::CreateProcedure(procedure) => {
            out.push_str("CREATE ");
            if procedure.or_replace {
                out.push_str("OR REPLACE ");
            }
            out.push_str("PROCEDURE ");
            write_qualified(out, &procedure.name);
            out.push('(');
            write_params(out, &procedure.params)?;
            out.push_str(") AS ");
            write_string(out, &procedure.body);
            out.push_str(" LANGUAGE ");
            out.push_str(match procedure.language {
                ProcedureLanguage::Sql => "SQL",
                ProcedureLanguage::PlSql => "PLSQL",
                ProcedureLanguage::Rust => "RUST",
            });
            out.push_str(match procedure.security {
                SecurityMode::Definer => " SECURITY DEFINER",
                SecurityMode::Invoker => " SECURITY INVOKER",
            });
        }
        Statement::Call(call) => {
            out.push_str("CALL ");
            write_qualified(out, &call.name);
            out.push('(');
            write_list(out, &call.args, write_expr)?;
            out.push(')');
        }
        Statement::CreateSchedule(schedule) => {
            out.push_str("CREATE SCHEDULE ");
            write_ident(out, &schedule.name);
            match &schedule.interval {
                ScheduleInterval::Every(duration) => {
                    out.push_str(" EVERY ");
                    write_ttl_duration(out, duration);
                }
                ScheduleInterval::Cron(cron) => {
                    out.push_str(" CRON ");
                    write_string(out, cron);
                }
            }
            out.push_str(" DO ");
            write_statement(out, &schedule.body)?;
        }
        Statement::CreatePipeline(pipeline) => {
            out.push_str("CREATE PIPELINE ");
            write_ident(out, &pipeline.name);
            out.push_str(" AS (");
            write_list(out, &pipeline.stages, |out, stage| {
                out.push_str("STAGE ");
                write_ident(out, &stage.name);
                out.push_str(" (");
                let mut clauses: Vec<String> = Vec::new();
                if !stage.source.is_empty() {
                    let mut clause = String::from("SOURCE ");
                    write_ident(&mut clause, &stage.source);
                    clauses.push(clause);
                }
                if !stage.target.is_empty() {
                    let mut clause = String::from("TARGET ");
                    write_ident(&mut clause, &stage.target);
                    clauses.push(clause);
                }
                if let Some(mode) = &stage.mode {
                    clauses.push(format!("MODE {mode}"));
                }
                if let Some(transform) = &stage.transform {
                    let mut clause = String::from("TRANSFORM AS (");
                    write_select(&mut clause, transform)?;
                    clause.push(')');
                    clauses.push(clause);
                }
                for expectation in &stage.expectations {
                    let mut clause = String::from("EXPECT ");
                    write_expr(&mut clause, &expectation.expr)?;
                    clauses.push(clause);
                }
                out.push_str(&clauses.join(", "));
                out.push(')');
                Ok(())
            })?;
            out.push(')');
        }
        Statement::CreateStreamingJob(job) => {
            out.push_str("CREATE STREAMING JOB ");
            if job.if_not_exists {
                out.push_str("IF NOT EXISTS ");
            }
            write_ident(out, &job.name);
            out.push_str(" AS ");
            write_select(out, &job.query)?;
            if let Some(StreamingJoinSpec::Interval { within_us, .. }) = &job.join {
                out.push_str(" WITHIN INTERVAL ");
                write_string(out, &within_us.to_string());
                out.push_str(" microsecond");
            }
            out.push_str(" INTO ");
            match &job.target {
                StreamingSinkRef::Named(name) => write_ident(out, name),
                StreamingSinkRef::Inline {
                    backend,
                    uri,
                    format,
                    options,
                } => {
                    out.push_str(backend_word(backend));
                    out.push(' ');
                    write_string(out, uri);
                    out.push_str(" FORMAT ");
                    out.push_str(format_word(format));
                    if !options.is_empty() {
                        write_options(out, options);
                    }
                }
            }
            out.push_str(match job.write_mode {
                StreamingWriteMode::Append => " WRITE MODE APPEND",
                StreamingWriteMode::Upsert => " WRITE MODE UPSERT",
            });
            out.push_str(" MODE ");
            write_mode_spec(out, &job.job_mode)?;
            if let Some(watermark) = &job.watermark {
                out.push_str(" WATERMARK FOR ");
                write_ident(out, &watermark.event_time_column);
                out.push_str(" AS ");
                write_ident(out, &watermark.event_time_column);
                out.push_str(" - INTERVAL ");
                write_string(out, &interval_text(&watermark.allowed_lateness));
            }
            if let Some(policy) = &job.late_data_policy {
                out.push_str(match policy {
                    LateDataPolicySpec::Drop => " WITH LATE DATA POLICY DROP",
                    LateDataPolicySpec::Reopen => " WITH LATE DATA POLICY REOPEN",
                    LateDataPolicySpec::SideOutput => " WITH LATE DATA POLICY SIDE OUTPUT",
                });
            }
        }
        Statement::CreateEndpoint(endpoint) => {
            out.push_str("CREATE ENDPOINT ");
            if endpoint.if_not_exists {
                out.push_str("IF NOT EXISTS ");
            }
            write_ident(out, &endpoint.name);
            out.push_str(" ON PATH ");
            write_string(out, &endpoint.path);
            out.push_str(" METHOD ");
            out.push_str(&endpoint.methods.join(", "));
            out.push_str(" USING ");
            write_string(out, &endpoint.sql);
            out.push_str(" AUTH ");
            out.push_str(match endpoint.auth {
                EndpointAuthSpec::None => "NONE",
                EndpointAuthSpec::Jwt => "JWT",
                EndpointAuthSpec::ApiKey => "API_KEY",
                EndpointAuthSpec::OAuth2 => "OAUTH2",
                EndpointAuthSpec::Basic => "BASIC",
                EndpointAuthSpec::Mtls => "MTLS",
            });
            if !endpoint.required_scopes.is_empty() {
                out.push_str(" REQUIRE SCOPE ");
                write_list(out, &endpoint.required_scopes, |out, scope| {
                    write_string(out, scope);
                    Ok(())
                })?;
            }
            if let Some(rate) = &endpoint.rate_limit {
                let unit = match rate.per_seconds {
                    1 => "second",
                    60 => "minute",
                    3600 => "hour",
                    86_400 => "day",
                    _ => return unsupported("a rate limit period that is not a unit"),
                };
                let _ = write!(out, " RATE LIMIT {} / {unit}", rate.count);
                out.push_str(match rate.scope {
                    EndpointRateLimitScope::Global => "",
                    EndpointRateLimitScope::PerIp => " PER IP",
                    EndpointRateLimitScope::PerUser => " PER USER",
                    EndpointRateLimitScope::PerApiKey => " PER API_KEY",
                });
            }
            out.push_str(" OUTPUT FORMAT ");
            out.push_str(match endpoint.output_format {
                EndpointOutputFormatSpec::Json => "JSON",
                EndpointOutputFormatSpec::JsonLines => "JSONLINES",
                EndpointOutputFormatSpec::Csv => "CSV",
                EndpointOutputFormatSpec::Parquet => "PARQUET",
                EndpointOutputFormatSpec::Arrow => "ARROW",
                EndpointOutputFormatSpec::Protobuf => "PROTOBUF",
            });
            if !endpoint.cors_origins.is_empty() {
                out.push_str(" CORS ORIGINS ");
                write_list(out, &endpoint.cors_origins, |out, origin| {
                    write_string(out, origin);
                    Ok(())
                })?;
            }
            if endpoint.cache_seconds > 0 {
                let _ = write!(out, " CACHE {} SECONDS", endpoint.cache_seconds);
            }
            if endpoint.timeout_seconds > 0 {
                let _ = write!(out, " TIMEOUT {} SECONDS", endpoint.timeout_seconds);
            }
            if endpoint.max_body_kb > 0 {
                let _ = write!(out, " MAX REQUEST BODY {} KB", endpoint.max_body_kb);
            }
        }
        Statement::CreateExternalSource(source) => {
            out.push_str("CREATE EXTERNAL SOURCE ");
            if source.if_not_exists {
                out.push_str("IF NOT EXISTS ");
            }
            write_ident(out, &source.name);
            out.push_str(" TYPE ");
            out.push_str(external_backend_sql(&source.backend));
            out.push_str(" URI ");
            write_string(out, &source.uri);
            out.push_str(" FORMAT ");
            out.push_str(external_format_sql(&source.format));
            write_external_mode(out, &source.mode)?;
            if !source.options.is_empty() {
                out.push_str(" OPTIONS ");
                write_kv_options(out, &source.options);
            }
            if !source.credentials.is_empty() {
                out.push_str(" CREDENTIALS ");
                write_kv_options(out, &source.credentials);
            }
            if let Some(provider) = &source.credential_provider {
                out.push_str(" CREDENTIAL_PROVIDER ");
                write_credential_provider(out, provider);
            }
            if !source.columns.is_empty() {
                out.push_str(" COLUMNS (");
                for (i, (name, data_type)) in source.columns.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_ident(out, name);
                    out.push(' ');
                    out.push_str(&data_type_to_sql(data_type));
                }
                out.push(')');
            }
        }
        Statement::AlterExternalSource(alter) => {
            out.push_str("ALTER EXTERNAL SOURCE ");
            write_ident(out, &alter.name);
            out.push(' ');
            write_alter_external_source_action(out, &alter.action)?;
        }
        Statement::AlterExternalSink(alter) => {
            out.push_str("ALTER EXTERNAL SINK ");
            write_ident(out, &alter.name);
            out.push(' ');
            match &alter.action {
                AlterExternalSinkAction::Rename(new_name) => {
                    out.push_str("RENAME TO ");
                    write_ident(out, new_name);
                }
                AlterExternalSinkAction::SetOptions(options) => {
                    out.push_str("SET OPTIONS ");
                    write_kv_options(out, options);
                }
                AlterExternalSinkAction::SetCredentials(credentials) => {
                    out.push_str("SET CREDENTIALS ");
                    write_kv_options(out, credentials);
                }
                AlterExternalSinkAction::SetCredentialProvider(provider) => {
                    out.push_str("SET CREDENTIAL_PROVIDER ");
                    write_credential_provider(out, provider);
                }
            }
        }
        other => {
            let debug = format!("{other:?}");
            let kind = debug
                .split(|c: char| c == '(' || c == ' ' || c == '{')
                .next()
                .unwrap_or("this statement")
                .to_string();
            return unsupported(format!("a {kind} statement"));
        }
    }
    Ok(())
}

const fn external_backend_sql(backend: &ExternalBackendKind) -> &'static str {
    match backend {
        ExternalBackendKind::File => "FILE",
        ExternalBackendKind::S3 => "S3",
        ExternalBackendKind::Gcs => "GCS",
        ExternalBackendKind::Azure => "AZURE",
        ExternalBackendKind::Http => "HTTP",
        ExternalBackendKind::Zyron => "ZYRON",
    }
}

const fn external_format_sql(format: &ExternalFormatKind) -> &'static str {
    match format {
        ExternalFormatKind::Json => "JSON",
        ExternalFormatKind::JsonLines => "JSONLINES",
        ExternalFormatKind::Csv => "CSV",
        ExternalFormatKind::Parquet => "PARQUET",
        ExternalFormatKind::ArrowIpc => "ARROW",
        ExternalFormatKind::Avro => "AVRO",
    }
}

const fn credential_provider_type_sql(kind: &CredentialProviderType) -> &'static str {
    match kind {
        CredentialProviderType::Vault => "vault",
        CredentialProviderType::AwsSecretsManager => "aws_secrets_manager",
        CredentialProviderType::GcpSecretManager => "gcp_secret_manager",
        CredentialProviderType::AzureKeyVault => "azure_key_vault",
        CredentialProviderType::OAuth2ClientCredentials => "oauth2_client_credentials",
        CredentialProviderType::AwsIamAssumeRole => "aws_iam_assume_role",
        CredentialProviderType::K8sSaToken => "k8s_sa_token",
    }
}

/// ONESHOT is what the parser leaves behind when no mode was written, so it
/// renders as nothing and the statement reads back to the same tree either way
fn write_external_mode(out: &mut String, mode: &ExternalModeSpec) -> Out {
    match mode {
        ExternalModeSpec::OneShot => {}
        ExternalModeSpec::Watch => out.push_str(" MODE WATCH"),
        ExternalModeSpec::Scheduled { cron, every } => match (cron, every) {
            (Some(cron), _) => {
                out.push_str(" MODE SCHEDULED CRON ");
                write_string(out, cron);
            }
            (None, Some(every)) => {
                out.push_str(" MODE SCHEDULED EVERY ");
                write_string(out, every);
            }
            (None, None) => {
                return unsupported("a scheduled mode with neither a cron nor an interval");
            }
        },
    }
    Ok(())
}

/// Writes one ALTER EXTERNAL SOURCE action back in the grammar that parses
/// it. RESET LSN takes a string literal, so the position is spelled the way
/// `parse_lsn_reset_spec` reads it back
fn write_alter_external_source_action(out: &mut String, action: &AlterExternalSourceAction) -> Out {
    match action {
        AlterExternalSourceAction::Rename(new_name) => {
            out.push_str("RENAME TO ");
            write_ident(out, new_name);
        }
        AlterExternalSourceAction::RefreshSchema => out.push_str("REFRESH SCHEMA"),
        AlterExternalSourceAction::ResetLsn(reset) => {
            out.push_str("RESET LSN TO ");
            match reset {
                LsnResetSpec::Earliest => write_string(out, "earliest"),
                LsnResetSpec::Latest => write_string(out, "latest"),
                LsnResetSpec::Explicit(n) => write_string(out, &format!("lsn:{n}")),
            }
        }
        AlterExternalSourceAction::Pause => out.push_str("PAUSE"),
        AlterExternalSourceAction::Resume => out.push_str("RESUME"),
        AlterExternalSourceAction::SetOptions(options) => {
            out.push_str("SET OPTIONS ");
            write_kv_options(out, options);
        }
        AlterExternalSourceAction::SetCredentials(credentials) => {
            out.push_str("SET CREDENTIALS ");
            write_kv_options(out, credentials);
        }
        AlterExternalSourceAction::SetCredentialProvider(provider) => {
            out.push_str("SET CREDENTIAL_PROVIDER ");
            write_credential_provider(out, provider);
        }
        // SET MODE takes the spec on its own, where the CREATE form writes
        // the MODE keyword with it and omits the whole clause for one shot
        AlterExternalSourceAction::SetMode(mode) => {
            out.push_str("SET MODE ");
            match mode {
                ExternalModeSpec::OneShot => out.push_str("ONESHOT"),
                ExternalModeSpec::Watch => out.push_str("WATCH"),
                ExternalModeSpec::Scheduled { cron, every } => match (cron, every) {
                    (Some(cron), _) => {
                        out.push_str("SCHEDULED CRON ");
                        write_string(out, cron);
                    }
                    (None, Some(every)) => {
                        out.push_str("SCHEDULED EVERY ");
                        write_string(out, every);
                    }
                    (None, None) => {
                        return unsupported("a scheduled mode with neither a cron nor an interval");
                    }
                },
            }
        }
        AlterExternalSourceAction::SetColumns(columns) => {
            out.push_str("SET COLUMNS (");
            for (i, (name, data_type)) in columns.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_ident(out, name);
                out.push(' ');
                out.push_str(&data_type_to_sql(data_type));
            }
            out.push(')');
        }
    }
    Ok(())
}

/// `(type = 'kind', key = 'value', ...)`, the spelling
/// `parse_credential_provider_clause` reads back. The provider kind travels
/// as the `type` entry rather than as a keyword, which is what that clause
/// pulls out of the option list
fn write_credential_provider(out: &mut String, provider: &CredentialProviderSpec) {
    out.push_str("(type = ");
    write_string(out, credential_provider_type_sql(&provider.provider_type));
    for (key, value) in &provider.options {
        out.push_str(", ");
        write_ident(out, key);
        out.push_str(" = ");
        write_string(out, value);
    }
    out.push(')');
}

/// `(key = 'value', ...)`. Every value renders as a string literal, which is a
/// spelling `parse_kv_options` reads for any of the forms it accepts, and what
/// it stores is the text either way
fn write_kv_options(out: &mut String, options: &[(String, String)]) {
    out.push('(');
    for (i, (key, value)) in options.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_ident(out, key);
        out.push_str(" = ");
        write_string(out, value);
    }
    out.push(')');
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rendering a tree and parsing it back gives the tree it came from.
    ///
    /// This is what lets a node resolve a statement and hand the resolved form
    /// to the group: what the other members parse has to mean exactly what
    /// this node settled on
    fn round_trips(sql: &str) {
        let parsed = crate::Parser::new(sql)
            .expect("lexes")
            .parse_statement()
            .expect("parses");
        let rendered = statement_to_sql(&parsed).expect("renders");
        let reparsed = crate::Parser::new(&rendered)
            .unwrap_or_else(|e| panic!("`{rendered}` does not lex: {e}"))
            .parse_statement()
            .unwrap_or_else(|e| panic!("`{rendered}` does not parse: {e}"));
        assert_eq!(parsed, reparsed, "`{sql}` rendered to `{rendered}`");
    }

    /// Every ALTER EXTERNAL SOURCE action survives being rendered.
    ///
    /// A node that resolves REFRESH SCHEMA into a column list replicates the
    /// resolved statement rather than the one that was typed, so an action
    /// this cannot render is an action the other members never hear about
    #[test]
    fn every_alter_external_source_action_round_trips() {
        round_trips("ALTER EXTERNAL SOURCE s RENAME TO t");
        round_trips("ALTER EXTERNAL SOURCE s REFRESH SCHEMA");
        round_trips("ALTER EXTERNAL SOURCE s RESET LSN TO 'earliest'");
        round_trips("ALTER EXTERNAL SOURCE s RESET LSN TO 'latest'");
        round_trips("ALTER EXTERNAL SOURCE s RESET LSN TO 'lsn:4096'");
        round_trips("ALTER EXTERNAL SOURCE s PAUSE");
        round_trips("ALTER EXTERNAL SOURCE s RESUME");
        round_trips("ALTER EXTERNAL SOURCE s SET OPTIONS (region = 'us-east-1')");
        round_trips("ALTER EXTERNAL SOURCE s SET CREDENTIALS (key = 'secret')");
        round_trips(
            "ALTER EXTERNAL SOURCE s SET CREDENTIAL_PROVIDER (type = 'vault',              url = 'https://v:8200')",
        );
        round_trips("ALTER EXTERNAL SOURCE s SET MODE ONESHOT");
        round_trips("ALTER EXTERNAL SOURCE s SET MODE WATCH");
        round_trips("ALTER EXTERNAL SOURCE s SET MODE SCHEDULED CRON '0 */5 * * *'");
        round_trips("ALTER EXTERNAL SOURCE s SET MODE SCHEDULED EVERY '60s'");
        round_trips("ALTER EXTERNAL SOURCE s SET COLUMNS (id BIGINT, name VARCHAR)");
    }

    /// Every ALTER EXTERNAL SINK action survives being rendered
    #[test]
    fn every_alter_external_sink_action_round_trips() {
        round_trips("ALTER EXTERNAL SINK s RENAME TO t");
        round_trips("ALTER EXTERNAL SINK s SET OPTIONS (compression = 'gzip')");
        round_trips("ALTER EXTERNAL SINK s SET CREDENTIALS (key = 'secret')");
        round_trips(
            "ALTER EXTERNAL SINK s SET CREDENTIAL_PROVIDER (type = 'aws_secrets_manager',              region = 'us-west-2', secret_id = 'prod/etl')",
        );
    }

    /// Every clause of CREATE EXTERNAL SOURCE survives being rendered.
    ///
    /// A node that infers a Parquet layout replicates the resolved statement
    /// rather than the one that was typed, so a clause this drops is a clause
    /// the other members never hear about
    #[test]
    fn an_external_source_round_trips_through_every_clause() {
        round_trips("CREATE EXTERNAL SOURCE s TYPE FILE URI '/tmp/x' FORMAT PARQUET");
        round_trips(
            "CREATE EXTERNAL SOURCE IF NOT EXISTS s TYPE S3 URI 's3://b/k' FORMAT AVRO              COLUMNS (id BIGINT, name VARCHAR)",
        );
        round_trips(
            "CREATE EXTERNAL SOURCE s TYPE GCS URI 'gs://b/k' FORMAT ARROW MODE WATCH              OPTIONS (region = 'us-east-1') CREDENTIALS (key = 'secret')",
        );
        round_trips(
            "CREATE EXTERNAL SOURCE s TYPE AZURE URI 'az://c/p' FORMAT JSONLINES              MODE SCHEDULED EVERY '5m' COLUMNS (a INT)",
        );
        round_trips(
            "CREATE EXTERNAL SOURCE s TYPE FILE URI '/tmp/y' FORMAT CSV              MODE SCHEDULED CRON '0 * * * *'",
        );
        round_trips(
            "CREATE EXTERNAL SOURCE s TYPE ZYRON URI 'zyron://u@h/db/pub:p'              CREDENTIAL_PROVIDER (type = 'vault', path = 'secret/data/x')",
        );
    }

    #[test]
    fn identifiers_are_quoted_only_when_the_lexer_needs_it() {
        let mut out = String::new();
        write_ident(&mut out, "plain_name");
        write_ident(&mut out, "MixedCase");
        assert_eq!(out, "plain_nameMixedCase");
        let mut out = String::new();
        write_ident(&mut out, "select");
        assert_eq!(out, "\"select\"");
        let mut out = String::new();
        write_ident(&mut out, "with space");
        assert_eq!(out, "\"with space\"");
        let mut out = String::new();
        write_ident(&mut out, "1st");
        assert_eq!(out, "\"1st\"");
        let mut out = String::new();
        write_ident(&mut out, "say \"hi\"");
        assert_eq!(out, "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn an_interval_is_spelled_in_units_its_parts_are_exact_in() {
        let interval = zyron_common::Interval {
            months: 14,
            days: -3,
            nanoseconds: 3_600_000_000_000 + 90_000_000_000 + 7,
        };
        assert_eq!(
            interval_text(&interval),
            "14 month -3 day 1 hour 1 minute 30 second 7 nanosecond"
        );
        assert_eq!(interval_text(&zyron_common::Interval::ZERO), "0 second");
        let parsed = zyron_common::parse_interval_string(&interval_text(&interval))
            .expect("the parser reads it back");
        assert_eq!(parsed, interval);
    }

    #[test]
    fn a_statement_with_no_spelling_names_its_kind() {
        let statement = crate::parse("CREATE SCHEMA s").expect("parses").remove(0);
        let err = statement_to_sql(&statement).expect_err("no spelling");
        assert!(err.to_string().contains("CreateSchema"), "{err}");
    }
}
