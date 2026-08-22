//! Canonical identity for an expression a table is clustered by.
//!
//! Clustering targets an expression by giving it a column: the values are
//! stored, so file statistics cover them and every pruning path works on
//! the expression exactly as it works on a stored column. Matching a
//! query's expression to that column is what this module answers.
//!
//! The identity is a hash of a canonical rendering of the **bound tree**,
//! not of statement text. Rendering from the tree is what makes whitespace,
//! redundant parentheses and column aliases free: none of them exist by the
//! time binding is done, so none of them can change an identity.
//!
//! What is normalized, and what deliberately is not:
//!
//! * Column references render as their id, so renaming a column does not
//!   change what the expression is.
//! * Function names lower case. Keyword arguments lower case **only** at
//!   the positions where the SQL specification makes them case insensitive,
//!   which is the unit of `date_trunc` and `date_part` and the field of
//!   `extract`. Lower casing every string argument would fold
//!   `upper(name) = 'X'` into `upper(name) = 'x'`, which is a different
//!   predicate.
//! * String literals are preserved byte for byte. They are user data.
//! * Numeric literals keep their type, so `1` and `1.0` are distinct, as
//!   SQL treats them.
//! * Commutative operands keep their written order. `a + b` and `b + a`
//!   stay distinct: floating point addition is not associative, so the two
//!   can produce different values, and folding them would make the stored
//!   column disagree with the expression a query wrote.
//!
//! The hash is persisted in the lake manifest, so it uses
//! `zyron_common::checksum::hash64`, whose output is bit identical across
//! the tiers it dispatches over. A `DefaultHasher` is not stable across
//! toolchains and would make a manifest unreadable after an upgrade.

use zyron_catalog::schema::ColumnEntry;
use zyron_common::checksum::hash64;
use zyron_parser::ast::{BinaryOperator, LiteralValue, UnaryOperator};

use crate::binder::{BoundExpr, ColumnRef};

/// The identity of one expression a table may be clustered by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalExpr {
    /// Hash of the canonical rendering, the persisted identity
    pub canonical_hash: u64,
    /// The expression rendered back with column names, for display and for
    /// recomputing the column
    pub sql: String,
    /// Column ids the expression reads, ascending and deduplicated
    pub source_columns: Vec<u32>,
}

/// Functions whose value depends on something other than their arguments.
///
/// Clustering stores an expression's value, so a function that answers
/// differently on two calls would leave the stored column disagreeing with
/// the expression a query writes, and the pruning built on it would drop
/// rows that match
const VOLATILE_FUNCTIONS: &[&str] = &[
    "now",
    "current_timestamp",
    "current_date",
    "current_time",
    "localtime",
    "localtimestamp",
    "random",
    "uuid",
    "gen_random_uuid",
    "nextval",
    "currval",
    "lastval",
    "clock_timestamp",
    "statement_timestamp",
    "transaction_timestamp",
];

/// The operator a function node was desugared from, for the binder's
/// internal names that have no call syntax.
///
/// `embedding <-> ARRAY[...]` binds to `vector_distance_l2`, and nothing
/// parses that name back, so rendering it as a call would store text the
/// write path could not read
fn infix_syntax(function: &str) -> Option<&'static str> {
    match function {
        "vector_distance_cosine" => Some("<=>"),
        "vector_distance_l2" => Some("<->"),
        "vector_distance_dot" => Some("<#>"),
        _ => None,
    }
}

/// Argument positions a function reads as a case-insensitive keyword.
///
/// These are the positions the SQL specification defines as keywords
/// rather than as data, so `date_trunc('DAY', ts)` and
/// `date_trunc('day', ts)` are the same expression
fn keyword_argument_positions(function: &str) -> &'static [usize] {
    match function {
        "date_trunc" | "date_part" | "datepart" | "extract" => &[0],
        _ => &[],
    }
}

/// Canonicalizes a bound expression into a cluster-target identity.
///
/// None when the expression is not one clustering can target: an
/// aggregate, a subquery, a window function, anything volatile, or a
/// reference to a column the table does not have. A bare column reference
/// is also None, because a column is already clusterable as itself and
/// giving it a second identity would split its statistics
pub fn canonicalize(expr: &BoundExpr, columns: &[ColumnEntry]) -> Option<CanonicalExpr> {
    let expr = unwrap_nested(expr);
    // A bare column is clusterable without any of this
    if matches!(expr, BoundExpr::ColumnRef(_)) {
        return None;
    }
    let mut canonical = String::new();
    let mut sql = String::new();
    let mut sources = Vec::new();
    render(expr, columns, &mut canonical, &mut sql, &mut sources)?;
    if sources.is_empty() {
        // An expression over no column is a constant, the same value in
        // every row, so ordering by it orders nothing
        return None;
    }
    sources.sort_unstable();
    sources.dedup();
    Some(CanonicalExpr {
        canonical_hash: hash64(canonical.as_bytes()),
        sql,
        source_columns: sources,
    })
}

/// Strips the parenthesis nodes binding left in place. Redundant
/// parentheses are grouping the parser already applied, so they carry no
/// identity
fn unwrap_nested(expr: &BoundExpr) -> &BoundExpr {
    match expr {
        BoundExpr::Nested(inner) => unwrap_nested(inner),
        other => other,
    }
}

/// Writes one node's canonical form and its display form, collecting the
/// columns it reads. Returns None the moment it meets a node clustering
/// cannot target, so a rejected expression costs a partial walk and no
/// allocation beyond it
fn render(
    expr: &BoundExpr,
    columns: &[ColumnEntry],
    canonical: &mut String,
    sql: &mut String,
    sources: &mut Vec<u32>,
) -> Option<()> {
    use std::fmt::Write;

    match unwrap_nested(expr) {
        BoundExpr::ColumnRef(ColumnRef { column_id, .. }) => {
            let entry = columns.iter().find(|c| c.id == *column_id)?;
            let id = entry.id.0 as u32;
            // The id, not the name: a rename must not change identity
            let _ = write!(canonical, "#{}", id);
            sql.push_str(&entry.name);
            sources.push(id);
        }
        BoundExpr::Literal { value, .. } => {
            render_literal(value, canonical, sql);
        }
        BoundExpr::BinaryOp {
            left, op, right, ..
        } => {
            let symbol = binary_symbol(*op)?;
            // Operands keep their written order, commutative or not
            canonical.push('(');
            sql.push('(');
            render(left, columns, canonical, sql, sources)?;
            let _ = write!(canonical, "{}", symbol);
            let _ = write!(sql, " {} ", symbol);
            render(right, columns, canonical, sql, sources)?;
            canonical.push(')');
            sql.push(')');
        }
        BoundExpr::UnaryOp { op, expr, .. } => {
            let symbol = match op {
                UnaryOperator::Minus => "-",
                // A negation produces a boolean, which has nothing to order
                UnaryOperator::Not => return None,
            };
            canonical.push('(');
            canonical.push_str(symbol);
            sql.push('(');
            sql.push_str(symbol);
            render(expr, columns, canonical, sql, sources)?;
            canonical.push(')');
            sql.push(')');
        }
        BoundExpr::Function {
            name,
            args,
            distinct,
            ..
        } => {
            if *distinct {
                return None;
            }
            let lowered = name.to_ascii_lowercase();
            if VOLATILE_FUNCTIONS.contains(&lowered.as_str()) {
                return None;
            }
            let keywords = keyword_argument_positions(&lowered);
            // The binder desugars some surface syntax into a function node,
            // and the display form has to go back to the syntax the parser
            // reads, because the write path re-parses it to recompute the
            // column on every insert. A node whose name has no call syntax
            // and is not handled here renders as a call, reads back as an
            // unknown function, and is refused where it is declared
            if let Some(symbol) = infix_syntax(&lowered) {
                if args.len() != 2 {
                    return None;
                }
                canonical.push_str(&lowered);
                canonical.push('(');
                sql.push('(');
                render(&args[0], columns, canonical, sql, sources)?;
                canonical.push(',');
                let _ = write!(sql, " {} ", symbol);
                render(&args[1], columns, canonical, sql, sources)?;
                canonical.push(')');
                sql.push(')');
                return Some(());
            }
            if lowered == "array_subscript" {
                if args.len() != 2 {
                    return None;
                }
                canonical.push_str(&lowered);
                canonical.push('(');
                render(&args[0], columns, canonical, sql, sources)?;
                canonical.push(',');
                sql.push('[');
                render(&args[1], columns, canonical, sql, sources)?;
                canonical.push(')');
                sql.push(']');
                return Some(());
            }
            let close = match lowered.as_str() {
                "array" => {
                    sql.push_str("ARRAY[");
                    ']'
                }
                _ => {
                    sql.push_str(&lowered);
                    sql.push('(');
                    ')'
                }
            };
            canonical.push_str(&lowered);
            canonical.push('(');
            for (i, arg) in args.iter().enumerate() {
                if i > 0 {
                    canonical.push(',');
                    sql.push_str(", ");
                }
                // A keyword argument is a spelling of the same thing, so it
                // folds. Every other argument is data and is preserved
                if keywords.contains(&i) {
                    if let BoundExpr::Literal {
                        value: LiteralValue::String(text),
                        ..
                    } = unwrap_nested(arg)
                    {
                        let folded = text.to_ascii_lowercase();
                        let _ = write!(canonical, "k{}:{}", folded.len(), folded);
                        let _ = write!(sql, "'{}'", folded);
                        continue;
                    }
                }
                render(arg, columns, canonical, sql, sources)?;
            }
            canonical.push(')');
            sql.push(close);
        }
        BoundExpr::Cast {
            expr,
            target_type,
            fractional_digits,
        } => {
            let _ = write!(canonical, "cast{}", *target_type as u8);
            if let Some(digits) = fractional_digits {
                let _ = write!(canonical, ".{}", digits);
            }
            canonical.push('(');
            sql.push_str("CAST(");
            render(expr, columns, canonical, sql, sources)?;
            canonical.push(')');
            let _ = write!(sql, " AS {})", target_type);
        }
        // Aggregates fold rows together, subqueries and window functions
        // read beyond the row, and none of them has a per-row value a
        // column could hold
        _ => return None,
    }
    Some(())
}

/// Renders a literal.
///
/// The kind tag comes from the literal the statement wrote, not from the
/// type the binder inferred for it. Inference is allowed to pick Varchar in
/// one path and Text in another for the same constant, and folding that
/// into the identity would let a declared expression and the query that
/// matches it hash differently, which fails silently as a lost match. The
/// tag still separates `1` from `1.0`, because those are different
/// literals rather than differently inferred ones.
///
/// A string carries its length so no value is confused with a longer one
/// that starts the same way
fn render_literal(value: &LiteralValue, canonical: &mut String, sql: &mut String) {
    use std::fmt::Write;

    canonical.push('l');
    match value {
        LiteralValue::Integer(v) => {
            let _ = write!(canonical, "i{}", v);
            let _ = write!(sql, "{}", v);
        }
        LiteralValue::Int128(v) => {
            let _ = write!(canonical, "I{}", v);
            let _ = write!(sql, "{}", v);
        }
        LiteralValue::Decimal { digits, scale } => {
            let _ = write!(canonical, "d{}e{}", digits, scale);
            // Written out with its point rather than as a mantissa and an
            // exponent, because the exponent form re-parses as a float and
            // binds at a different type than the decimal that was written
            sql.push_str(&zyron_common::format_decimal(*digits, *scale));
        }
        LiteralValue::Float(v) => {
            // The bits, so two spellings of one value agree and two values
            // that print alike stay apart
            let _ = write!(canonical, "f{:016x}", v.to_bits());
            // The debug form, which keeps the point on an integral value.
            // `{}` renders 0.0 as `0`, which re-parses as an integer and
            // binds at a different type than the float that was written
            let _ = write!(sql, "{:?}", v);
        }
        LiteralValue::String(text) => {
            // Byte preserved, length prefixed
            let _ = write!(canonical, "s{}:{}", text.len(), text);
            let _ = write!(sql, "'{}'", text.replace('\'', "''"));
        }
        LiteralValue::Boolean(v) => {
            let _ = write!(canonical, "b{}", u8::from(*v));
            let _ = write!(sql, "{}", if *v { "TRUE" } else { "FALSE" });
        }
        LiteralValue::Null => {
            canonical.push('n');
            sql.push_str("NULL");
        }
        LiteralValue::Interval(interval) => {
            let _ = write!(
                canonical,
                "v{}:{}:{}",
                interval.months, interval.days, interval.nanoseconds
            );
            let _ = write!(
                sql,
                "INTERVAL '{} months {} days {} nanos'",
                interval.months, interval.days, interval.nanoseconds
            );
        }
    }
}

/// The canonical symbol for an operator a clustered expression may use.
///
/// Comparison and logical operators are absent on purpose: they produce a
/// boolean, and clustering on a boolean sorts rows into two groups, which
/// a zone map already separates without a column to hold it
fn binary_symbol(op: BinaryOperator) -> Option<&'static str> {
    Some(match op {
        BinaryOperator::Plus => "+",
        BinaryOperator::Minus => "-",
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Modulo => "%",
        BinaryOperator::Concat => "||",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::ids::{ColumnId, TableId};
    use zyron_common::TypeId;

    fn column(id: u16, name: &str, type_id: TypeId) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal: id,
            nullable: true,
            default_expr: None,
            max_length: None,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
        }
    }

    fn columns() -> Vec<ColumnEntry> {
        vec![
            column(0, "ts", TypeId::Timestamp),
            column(1, "name", TypeId::Varchar),
            column(2, "a", TypeId::Int64),
            column(3, "b", TypeId::Int64),
        ]
    }

    fn col(id: u16, type_id: TypeId) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(id),
            type_id,
            nullable: true,
            fractional_digits: None,
        })
    }

    fn text(value: &str) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::String(value.to_string()),
            type_id: TypeId::Varchar,
        }
    }

    fn call(name: &str, args: Vec<BoundExpr>) -> BoundExpr {
        BoundExpr::Function {
            name: name.to_string(),
            args,
            return_type: TypeId::Timestamp,
            distinct: false,
        }
    }

    fn add(left: BoundExpr, right: BoundExpr) -> BoundExpr {
        BoundExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOperator::Plus,
            right: Box::new(right),
            type_id: TypeId::Int64,
        }
    }

    fn hash_of(expr: &BoundExpr) -> u64 {
        canonicalize(expr, &columns())
            .expect("expression is clusterable")
            .canonical_hash
    }

    /// The unit of date_trunc is a keyword, so its case is a spelling
    #[test]
    fn test_keyword_argument_case_folds() {
        let lower = call("date_trunc", vec![text("day"), col(0, TypeId::Timestamp)]);
        let upper = call("date_trunc", vec![text("DAY"), col(0, TypeId::Timestamp)]);
        let mixed = call("date_trunc", vec![text("Day"), col(0, TypeId::Timestamp)]);
        assert_eq!(hash_of(&lower), hash_of(&upper));
        assert_eq!(hash_of(&lower), hash_of(&mixed));
    }

    /// A function name is case insensitive in SQL
    #[test]
    fn test_function_name_case_folds() {
        let lower = call("date_trunc", vec![text("day"), col(0, TypeId::Timestamp)]);
        let upper = call("DATE_TRUNC", vec![text("day"), col(0, TypeId::Timestamp)]);
        assert_eq!(hash_of(&lower), hash_of(&upper));
    }

    /// Identity is the column id, so renaming a column leaves every
    /// expression over it clustered where it was
    #[test]
    fn test_rename_does_not_change_identity() {
        let expr = call("date_trunc", vec![text("day"), col(0, TypeId::Timestamp)]);
        let before = canonicalize(&expr, &columns()).expect("clusterable");
        let mut renamed = columns();
        renamed[0].name = "event_time".to_string();
        let after = canonicalize(&expr, &renamed).expect("clusterable");
        assert_eq!(before.canonical_hash, after.canonical_hash);
        // The rendering follows the name, only the identity is fixed
        assert_ne!(before.sql, after.sql);
        assert!(after.sql.contains("event_time"));
    }

    /// Redundant grouping is something the parser applied, not something
    /// the expression means
    #[test]
    fn test_redundant_parentheses_do_not_change_identity() {
        let bare = add(col(2, TypeId::Int64), col(3, TypeId::Int64));
        let wrapped = BoundExpr::Nested(Box::new(BoundExpr::Nested(Box::new(add(
            col(2, TypeId::Int64),
            col(3, TypeId::Int64),
        )))));
        let inner_wrapped = add(
            BoundExpr::Nested(Box::new(col(2, TypeId::Int64))),
            col(3, TypeId::Int64),
        );
        assert_eq!(hash_of(&bare), hash_of(&wrapped));
        assert_eq!(hash_of(&bare), hash_of(&inner_wrapped));
    }

    /// Canonical form is built from the tree, so no spelling of whitespace
    /// can reach it
    #[test]
    fn test_canonical_form_carries_no_whitespace() {
        let expr = call("date_trunc", vec![text("day"), col(0, TypeId::Timestamp)]);
        let mut canonical = String::new();
        let mut sql = String::new();
        let mut sources = Vec::new();
        render(&expr, &columns(), &mut canonical, &mut sql, &mut sources).expect("renders");
        assert!(
            !canonical.contains(' '),
            "canonical form must not carry spacing: {canonical}"
        );
    }

    /// Addition over floats is not associative, so folding the operand
    /// order would let a stored column disagree with the expression a
    /// query wrote
    #[test]
    fn test_commutative_operands_keep_their_order() {
        let ab = add(col(2, TypeId::Int64), col(3, TypeId::Int64));
        let ba = add(col(3, TypeId::Int64), col(2, TypeId::Int64));
        assert_ne!(hash_of(&ab), hash_of(&ba));
    }

    /// SQL types an integer and a float differently, so the literals are
    /// different expressions
    #[test]
    fn test_numeric_literal_types_stay_distinct() {
        let int = add(
            col(2, TypeId::Int64),
            BoundExpr::Literal {
                value: LiteralValue::Integer(1),
                type_id: TypeId::Int64,
            },
        );
        let float = add(
            col(2, TypeId::Int64),
            BoundExpr::Literal {
                value: LiteralValue::Float(1.0),
                type_id: TypeId::Float64,
            },
        );
        assert_ne!(hash_of(&int), hash_of(&float));
    }

    /// A string argument that is not a keyword is data, so its case and
    /// its spacing are part of what the expression computes
    #[test]
    fn test_string_literals_are_preserved_byte_for_byte() {
        let upper = call("concat", vec![col(1, TypeId::Varchar), text("X")]);
        let lower = call("concat", vec![col(1, TypeId::Varchar), text("x")]);
        assert_ne!(
            hash_of(&upper),
            hash_of(&lower),
            "folding a data argument would change what the expression computes"
        );
        let spaced = call("concat", vec![col(1, TypeId::Varchar), text("a b")]);
        let tight = call("concat", vec![col(1, TypeId::Varchar), text("ab")]);
        assert_ne!(hash_of(&spaced), hash_of(&tight));
    }

    /// Two different expressions must not collide, and the sources have to
    /// name every column read, once each
    #[test]
    fn test_distinct_expressions_and_reported_sources() {
        let day = call("date_trunc", vec![text("day"), col(0, TypeId::Timestamp)]);
        let month = call("date_trunc", vec![text("month"), col(0, TypeId::Timestamp)]);
        assert_ne!(hash_of(&day), hash_of(&month));

        let both = add(col(2, TypeId::Int64), col(3, TypeId::Int64));
        let canonical = canonicalize(&both, &columns()).expect("clusterable");
        assert_eq!(canonical.source_columns, vec![2, 3]);

        let doubled = add(col(2, TypeId::Int64), col(2, TypeId::Int64));
        assert_eq!(
            canonicalize(&doubled, &columns())
                .expect("clusterable")
                .source_columns,
            vec![2]
        );
    }

    /// What clustering cannot target has to be refused rather than given
    /// an identity nothing can recompute
    #[test]
    fn test_unclusterable_expressions_are_refused() {
        // Volatile, two evaluations disagree
        assert!(canonicalize(&call("now", vec![]), &columns()).is_none());
        assert!(
            canonicalize(
                &add(col(2, TypeId::Int64), call("random", vec![])),
                &columns()
            )
            .is_none()
        );
        // A bare column is already clusterable as itself
        assert!(canonicalize(&col(0, TypeId::Timestamp), &columns()).is_none());
        assert!(
            canonicalize(
                &BoundExpr::Nested(Box::new(col(0, TypeId::Timestamp))),
                &columns()
            )
            .is_none()
        );
        // A constant is the same value in every row
        assert!(
            canonicalize(
                &BoundExpr::Literal {
                    value: LiteralValue::Integer(1),
                    type_id: TypeId::Int64,
                },
                &columns()
            )
            .is_none()
        );
        // An aggregate folds rows together and has no per-row value
        assert!(
            canonicalize(
                &BoundExpr::AggregateFunction {
                    name: "sum".into(),
                    args: vec![col(2, TypeId::Int64)],
                    distinct: false,
                    return_type: TypeId::Int64,
                    uda: None,
                },
                &columns()
            )
            .is_none()
        );
        // A column the table does not have
        assert!(
            canonicalize(
                &call("date_trunc", vec![text("day"), col(99, TypeId::Timestamp)]),
                &columns()
            )
            .is_none()
        );
    }

    /// The rendered form is what the write path re-parses to recompute the
    /// column, so a float that prints without its point comes back as an
    /// integer and binds at a different type than the one that was written
    #[test]
    fn test_a_float_literal_renders_with_its_point() {
        let expr = add(
            col(2, TypeId::Int64),
            BoundExpr::Literal {
                value: LiteralValue::Float(0.0),
                type_id: TypeId::Float64,
            },
        );
        let rendered = canonicalize(&expr, &columns()).expect("a float is a valid operand");
        assert!(
            rendered.sql.contains("0.0"),
            "an integral float has to keep its point, got {}",
            rendered.sql
        );

        // And a value with a fraction is unchanged by that
        let expr = add(
            col(2, TypeId::Int64),
            BoundExpr::Literal {
                value: LiteralValue::Float(1.5),
                type_id: TypeId::Float64,
            },
        );
        let rendered = canonicalize(&expr, &columns()).expect("canonical");
        assert!(rendered.sql.contains("1.5"), "{}", rendered.sql);
    }

    /// A decimal written as a mantissa and an exponent re-parses as a float
    #[test]
    fn test_a_decimal_literal_renders_with_its_point() {
        let expr = add(
            col(2, TypeId::Int64),
            BoundExpr::Literal {
                value: LiteralValue::Decimal {
                    digits: 12345,
                    scale: 2,
                },
                type_id: TypeId::Decimal,
            },
        );
        let rendered = canonicalize(&expr, &columns()).expect("canonical");
        assert!(
            rendered.sql.contains("123.45"),
            "a decimal has to render as a decimal, got {}",
            rendered.sql
        );
        assert!(
            !rendered.sql.contains('e'),
            "the exponent form re-parses as a float, got {}",
            rendered.sql
        );
    }

    /// The binder desugars bracket and operator syntax into function nodes.
    /// Rendering those back as calls stores text nothing parses
    #[test]
    fn test_desugared_syntax_renders_back_as_syntax() {
        let array = call(
            "array",
            vec![
                BoundExpr::Literal {
                    value: LiteralValue::Float(1.0),
                    type_id: TypeId::Float64,
                },
                col(2, TypeId::Int64),
            ],
        );
        let rendered = canonicalize(&array, &columns()).expect("canonical");
        assert_eq!(rendered.sql, "ARRAY[1.0, a]");

        let distance = call("vector_distance_l2", vec![col(2, TypeId::Int64), array]);
        let rendered = canonicalize(&distance, &columns()).expect("canonical");
        assert_eq!(rendered.sql, "(a <-> ARRAY[1.0, a])");

        let subscript = call(
            "array_subscript",
            vec![
                col(2, TypeId::Int64),
                BoundExpr::Literal {
                    value: LiteralValue::Integer(1),
                    type_id: TypeId::Int64,
                },
            ],
        );
        let rendered = canonicalize(&subscript, &columns()).expect("canonical");
        assert_eq!(rendered.sql, "a[1]");
    }

    /// Two spellings of one distance are one expression, and two different
    /// distances are two. The identity comes from the function name the
    /// binder produced, not from the symbol it renders back to
    #[test]
    fn test_distance_operators_keep_their_identities_apart() {
        let cols = columns();
        let l2 = canonicalize(
            &call(
                "vector_distance_l2",
                vec![col(2, TypeId::Int64), col(3, TypeId::Int64)],
            ),
            &cols,
        )
        .expect("canonical");
        let cosine = canonicalize(
            &call(
                "vector_distance_cosine",
                vec![col(2, TypeId::Int64), col(3, TypeId::Int64)],
            ),
            &cols,
        )
        .expect("canonical");
        assert_ne!(l2.canonical_hash, cosine.canonical_hash);
        assert_eq!(l2.source_columns, vec![2, 3]);
    }

    /// An operator node with the wrong arity is not the operator, and
    /// rendering it as one would produce text that means something else
    #[test]
    fn test_a_distance_node_of_the_wrong_arity_is_refused() {
        assert!(
            canonicalize(
                &call("vector_distance_l2", vec![col(2, TypeId::Int64)]),
                &columns()
            )
            .is_none()
        );
    }
}
