//! Renders a bound predicate back to SQL text.
//!
//! Two callers need the same thing for different reasons. A CREATE/ALTER
//! PUBLICATION WHERE clause is bound against its member tables and persisted
//! as SQL, so the streaming read path can re-parse and bind it per table. A
//! foreign scan pushes its filter to a peer, and the peer is another database
//! that accepts SQL, so the predicate has to arrive as text.
//!
//! Each ColumnRef resolves to its column name through the table schema. The
//! resolver covers the grammar valid in a row filter: literals, column refs,
//! operators, IN/BETWEEN/LIKE/IS NULL, CAST, and scalar functions. Forms with
//! no faithful text rendering (subqueries, window functions, aggregates,
//! parameters) yield None, so a caller either rejects the statement or keeps
//! the predicate as a residual it evaluates itself. Approximating one would
//! change the answer, which is the one thing neither caller can accept.

use crate::binder::BoundExpr;
use zyron_catalog::ColumnEntry;
use zyron_parser::ast::{BinaryOperator as B, LiteralValue as L, UnaryOperator as U};

/// Renders a bound predicate to SQL, resolving column refs against `columns`.
/// Returns None for a form that cannot be re-parsed as a row filter.
pub fn bound_predicate_to_sql(expr: &BoundExpr, columns: &[ColumnEntry]) -> Option<String> {
    match expr {
        BoundExpr::ColumnRef(c) => {
            let name = columns
                .iter()
                .find(|e| e.id == c.column_id)
                .map(|e| e.name.clone())?;
            Some(format!("\"{}\"", name.replace('"', "\"\"")))
        }
        BoundExpr::Literal { value, .. } => Some(literal_to_sql(value)),
        BoundExpr::BinaryOp {
            left, op, right, ..
        } => {
            let o = binary_op_str(*op);
            Some(format!(
                "({} {} {})",
                bound_predicate_to_sql(left, columns)?,
                o,
                bound_predicate_to_sql(right, columns)?
            ))
        }
        BoundExpr::UnaryOp { op, expr, .. } => {
            let o = match op {
                U::Not => "NOT ",
                U::Minus => "-",
            };
            Some(format!("{}{}", o, bound_predicate_to_sql(expr, columns)?))
        }
        BoundExpr::IsNull { expr, negated } => Some(format!(
            "{} IS {}NULL",
            bound_predicate_to_sql(expr, columns)?,
            if *negated { "NOT " } else { "" }
        )),
        BoundExpr::InList {
            expr,
            list,
            negated,
        } => {
            let items: Vec<String> = list
                .iter()
                .map(|e| bound_predicate_to_sql(e, columns))
                .collect::<Option<Vec<_>>>()?;
            Some(format!(
                "{} {}IN ({})",
                bound_predicate_to_sql(expr, columns)?,
                if *negated { "NOT " } else { "" },
                items.join(", ")
            ))
        }
        BoundExpr::Between {
            expr,
            low,
            high,
            negated,
        } => Some(format!(
            "{} {}BETWEEN {} AND {}",
            bound_predicate_to_sql(expr, columns)?,
            if *negated { "NOT " } else { "" },
            bound_predicate_to_sql(low, columns)?,
            bound_predicate_to_sql(high, columns)?
        )),
        BoundExpr::Like {
            expr,
            pattern,
            negated,
        } => Some(format!(
            "{} {}LIKE {}",
            bound_predicate_to_sql(expr, columns)?,
            if *negated { "NOT " } else { "" },
            bound_predicate_to_sql(pattern, columns)?
        )),
        BoundExpr::ILike {
            expr,
            pattern,
            negated,
        } => Some(format!(
            "{} {}ILIKE {}",
            bound_predicate_to_sql(expr, columns)?,
            if *negated { "NOT " } else { "" },
            bound_predicate_to_sql(pattern, columns)?
        )),
        BoundExpr::Function { name, args, .. } => {
            let rendered: Vec<String> = args
                .iter()
                .map(|a| bound_predicate_to_sql(a, columns))
                .collect::<Option<Vec<_>>>()?;
            Some(format!("{}({})", name, rendered.join(", ")))
        }
        BoundExpr::Cast { expr, .. } => bound_predicate_to_sql(expr, columns),
        BoundExpr::Nested(inner) => Some(format!("({})", bound_predicate_to_sql(inner, columns)?)),
        // Subqueries, window functions, aggregates, parameters, and temporal
        // refs cannot appear in a row filter, so they have no round-tripping
        // form.
        _ => None,
    }
}

fn binary_op_str(op: B) -> &'static str {
    match op {
        B::Plus => "+",
        B::Minus => "-",
        B::Multiply => "*",
        B::Divide => "/",
        B::Modulo => "%",
        B::Eq => "=",
        B::Neq => "<>",
        B::Lt => "<",
        B::Gt => ">",
        B::LtEq => "<=",
        B::GtEq => ">=",
        B::And => "AND",
        B::Or => "OR",
        B::Concat => "||",
    }
}

fn literal_to_sql(value: &L) -> String {
    match value {
        L::Integer(i) => i.to_string(),
        L::Int128(i) => i.to_string(),
        L::Float(f) => f.to_string(),
        L::String(s) => format!("'{}'", s.replace('\'', "''")),
        L::Boolean(b) => b.to_string(),
        L::Null => "NULL".to_string(),
        L::Interval(_) => "INTERVAL".to_string(),
    }
}
