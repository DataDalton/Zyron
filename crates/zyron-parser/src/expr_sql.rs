//! Renders an expression AST back to SQL text.
//!
//! Column defaults and CHECK constraints are persisted in the catalog as SQL
//! strings so they can be re-parsed and evaluated (e.g. to fill an omitted
//! column on INSERT). This is the inverse of `parse_expr` for the subset of the
//! grammar valid in those positions: literals, identifiers, operators,
//! functions, BETWEEN/LIKE/IN/IS NULL, CASE, and CAST. Expression forms that
//! cannot appear in a default or constraint (subqueries, window functions)
//! fall through to a debug rendering that does not round-trip.

use crate::ast::{BinaryOperator as B, Expr, FunctionArg, LiteralValue as L, UnaryOperator as U};

/// Renders an expression as SQL. The result re-parses via `parse_expr` for the
/// default/constraint grammar.
pub fn expr_to_sql(e: &Expr) -> String {
    match e {
        Expr::Identifier(n) => n.clone(),
        Expr::QualifiedIdentifier { table, column } => format!("{table}.{column}"),
        Expr::Literal(L::Integer(i)) => i.to_string(),
        Expr::Literal(L::Int128(i)) => i.to_string(),
        Expr::Literal(L::Decimal { digits, scale }) => {
            zyron_common::format_decimal(*digits, *scale)
        }
        Expr::Literal(L::Float(f)) => f.to_string(),
        Expr::Literal(L::String(s)) => format!("'{}'", s.replace('\'', "''")),
        Expr::Literal(L::Boolean(b)) => b.to_string(),
        Expr::Literal(L::Null) => "NULL".to_string(),
        Expr::Literal(L::Interval(_)) => "INTERVAL".to_string(),
        Expr::BinaryOp { left, op, right } => {
            let o = match op {
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
            };
            format!("({} {} {})", expr_to_sql(left), o, expr_to_sql(right))
        }
        Expr::UnaryOp { op, expr } => {
            let o = match op {
                U::Not => "NOT ",
                U::Minus => "-",
            };
            format!("{}{}", o, expr_to_sql(expr))
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            let items: Vec<String> = list.iter().map(expr_to_sql).collect();
            format!(
                "{} {}IN ({})",
                expr_to_sql(expr),
                if *negated { "NOT " } else { "" },
                items.join(", ")
            )
        }
        Expr::Between {
            expr,
            low,
            high,
            negated,
        } => format!(
            "{} {}BETWEEN {} AND {}",
            expr_to_sql(expr),
            if *negated { "NOT " } else { "" },
            expr_to_sql(low),
            expr_to_sql(high)
        ),
        Expr::Like {
            expr,
            pattern,
            negated,
        } => format!(
            "{} {}LIKE {}",
            expr_to_sql(expr),
            if *negated { "NOT " } else { "" },
            expr_to_sql(pattern)
        ),
        Expr::ILike {
            expr,
            pattern,
            negated,
        } => format!(
            "{} {}ILIKE {}",
            expr_to_sql(expr),
            if *negated { "NOT " } else { "" },
            expr_to_sql(pattern)
        ),
        Expr::IsNull { expr, negated } => format!(
            "{} IS {}NULL",
            expr_to_sql(expr),
            if *negated { "NOT " } else { "" }
        ),
        Expr::Function {
            name,
            args,
            distinct,
        } => {
            let rendered: Vec<String> = args
                .iter()
                .map(|a| match a {
                    FunctionArg::Unnamed(e) => expr_to_sql(e),
                    FunctionArg::Named { name, value } => {
                        format!("{name} => {}", expr_to_sql(value))
                    }
                    FunctionArg::Wildcard => "*".to_string(),
                })
                .collect();
            format!(
                "{}({}{})",
                name,
                if *distinct { "DISTINCT " } else { "" },
                rendered.join(", ")
            )
        }
        Expr::Case {
            operand,
            conditions,
            else_result,
        } => {
            let mut s = String::from("CASE");
            if let Some(op) = operand {
                s.push(' ');
                s.push_str(&expr_to_sql(op));
            }
            for w in conditions {
                s.push_str(&format!(
                    " WHEN {} THEN {}",
                    expr_to_sql(&w.condition),
                    expr_to_sql(&w.result)
                ));
            }
            if let Some(els) = else_result {
                s.push_str(&format!(" ELSE {}", expr_to_sql(els)));
            }
            s.push_str(" END");
            s
        }
        Expr::Cast { expr, data_type } => {
            format!(
                "CAST({} AS {})",
                expr_to_sql(expr),
                data_type_to_sql(data_type)
            )
        }
        Expr::Nested(inner) => format!("({})", expr_to_sql(inner)),
        // Subqueries, window functions, and the like cannot appear in a default
        // or CHECK constraint, so a non-round-tripping rendering is acceptable.
        other => format!("({other:?})"),
    }
}

/// Renders a data type as the SQL text parse_data_type accepts, so a CAST inside
/// a persisted default or CHECK constraint round-trips. The match is exhaustive:
/// adding a DataType variant is a compile error here rather than a silent
/// debug-form rendering that fails to re-parse.
fn data_type_to_sql(dt: &crate::ast::DataType) -> String {
    use crate::ast::DataType as D;
    match dt {
        D::Boolean => "BOOLEAN".to_string(),
        D::TinyInt => "TINYINT".to_string(),
        D::SmallInt => "SMALLINT".to_string(),
        D::Int => "INT".to_string(),
        D::BigInt => "BIGINT".to_string(),
        D::Int128 => "INT128".to_string(),
        D::UInt8 => "UINT8".to_string(),
        D::UInt16 => "UINT16".to_string(),
        D::UInt32 => "UINT32".to_string(),
        D::UInt64 => "UINT64".to_string(),
        D::UInt128 => "UINT128".to_string(),
        D::Real => "REAL".to_string(),
        D::DoublePrecision => "DOUBLE PRECISION".to_string(),
        D::Float(p) => format!("FLOAT{}", type_param(p)),
        D::Decimal(p, s) => format!("DECIMAL{}", precision_scale(p, s)),
        D::Numeric(p, s) => format!("NUMERIC{}", precision_scale(p, s)),
        D::Char(n) => format!("CHAR{}", type_param(n)),
        D::Varchar(n) => format!("VARCHAR{}", type_param(n)),
        D::Text => "TEXT".to_string(),
        D::Binary(n) => format!("BINARY{}", type_param(n)),
        D::Varbinary(n) => format!("VARBINARY{}", type_param(n)),
        D::Bytea => "BYTEA".to_string(),
        D::Date => "DATE".to_string(),
        D::Time => "TIME".to_string(),
        D::Timestamp(p) => format!("TIMESTAMP{}", type_param(p)),
        D::TimestampTz(p) => format!("TIMESTAMPTZ{}", type_param(p)),
        D::Hlc => "HLC".to_string(),
        D::Interval => "INTERVAL".to_string(),
        D::Uuid => "UUID".to_string(),
        D::Json => "JSON".to_string(),
        D::Jsonb => "JSONB".to_string(),
        D::Array(inner) => format!("{}[]", data_type_to_sql(inner)),
        D::Vector(n) => format!("VECTOR{}", type_param(n)),
        D::Geometry => "GEOMETRY".to_string(),
        D::Matrix => "MATRIX".to_string(),
        D::Color => "COLOR".to_string(),
        D::SemVer => "SEMVER".to_string(),
        D::Inet => "INET".to_string(),
        D::Cidr => "CIDR".to_string(),
        D::MacAddr => "MACADDR".to_string(),
        D::Money => "MONEY".to_string(),
        D::Range(inner) => format!("RANGE({})", data_type_to_sql(inner)),
        D::HyperLogLog => "HYPERLOGLOG".to_string(),
        D::BloomFilter => "BLOOMFILTER".to_string(),
        D::TDigest => "TDIGEST".to_string(),
        D::CountMinSketch => "COUNTMINSKETCH".to_string(),
        D::Bitfield => "BITFIELD".to_string(),
        D::Quantity => "QUANTITY".to_string(),
    }
}

/// Formats an optional single length or precision parameter as "(n)" or "".
fn type_param<T: std::fmt::Display>(n: &Option<T>) -> String {
    match n {
        Some(v) => format!("({v})"),
        None => String::new(),
    }
}

/// Formats optional precision and scale for DECIMAL/NUMERIC as "(p, s)", "(p)",
/// or "".
fn precision_scale(p: &Option<u8>, s: &Option<u8>) -> String {
    match (p, s) {
        (Some(p), Some(s)) => format!("({p}, {s})"),
        (Some(p), None) => format!("({p})"),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_expr;

    /// The rendered SQL must re-parse: that is the property storage relies on.
    fn reparses(sql: &str) {
        let e = parse_expr(sql).unwrap_or_else(|err| panic!("parse {sql:?}: {err}"));
        let rendered = expr_to_sql(&e);
        parse_expr(&rendered)
            .unwrap_or_else(|err| panic!("reparse {rendered:?} (from {sql:?}): {err}"));
    }

    #[test]
    fn default_expression_forms_round_trip() {
        for sql in [
            "0",
            "-1",
            "'active'",
            "'it''s'",
            "TRUE",
            "NULL",
            "now()",
            "current_timestamp()",
            "1 + 2",
            "(1 + 2) * 3",
            "x IN (1, 2, 3)",
            "x IS NOT NULL",
            "a BETWEEN 1 AND 10",
            "name LIKE 'a%'",
            "CASE WHEN a THEN 1 ELSE 0 END",
            "coalesce(x, 0)",
        ] {
            reparses(sql);
        }
    }

    /// A CAST to any data type (including parameterized and domain types) must
    /// round-trip, since a CHECK/DEFAULT can contain one and the catalog persists
    /// it as re-parsed SQL. A debug-form rendering here would brick writes to the
    /// table (the binder re-parse of check_expr would fail).
    #[test]
    fn cast_data_types_round_trip() {
        for ty in [
            "BOOLEAN",
            "TINYINT",
            "SMALLINT",
            "INT",
            "BIGINT",
            "INT128",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "UINT128",
            "REAL",
            "DOUBLE PRECISION",
            "FLOAT",
            "FLOAT(24)",
            "DECIMAL",
            "DECIMAL(10)",
            "DECIMAL(10, 2)",
            "NUMERIC(12, 4)",
            "CHAR",
            "CHAR(2)",
            "VARCHAR",
            "VARCHAR(255)",
            "TEXT",
            "BINARY(16)",
            "VARBINARY(32)",
            "BYTEA",
            "DATE",
            "TIME",
            "TIMESTAMP",
            "TIMESTAMP(6)",
            "TIMESTAMPTZ",
            "TIMESTAMPTZ(9)",
            "HLC",
            "INTERVAL",
            "UUID",
            "JSON",
            "JSONB",
            "VECTOR(128)",
            "GEOMETRY",
            "MATRIX",
            "COLOR",
            "SEMVER",
            "INET",
            "CIDR",
            "MACADDR",
            "MONEY",
            "RANGE(INT)",
            "HYPERLOGLOG",
            "BLOOMFILTER",
            "TDIGEST",
            "COUNTMINSKETCH",
            "BITFIELD",
            "QUANTITY",
        ] {
            reparses(&format!("CAST(x AS {ty})"));
        }
    }
}
