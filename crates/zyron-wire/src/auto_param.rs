//! Auto-parameterization scanner for DML literals.
//!
//! Real workloads issue the same query *shape* repeatedly with different
//! literal values: a bulk load sends `INSERT INTO t VALUES (1,'a'),(2,'b')`
//! then `INSERT INTO t VALUES (3,'c'),(4,'d')`. The literal text differs
//! every time, so a plan cache keyed on raw SQL text never hits.
//!
//! This scanner detects a small set of safe DML shapes and rewrites every
//! literal token into a `$N` placeholder, producing a normalized template
//! string plus the extracted literal values in order. The template is what
//! the plan cache keys on, so two executions of the same shape share one
//! cached plan; the extracted literals are bound as parameters at execute
//! time.
//!
//! The scanner is a cheap detector, not a parser. It reuses the real lexer
//! so string escapes, dollar-quoted strings, and comments tokenize exactly
//! as the parser sees them. On anything it does not positively recognize as
//! safe, it returns `None` and the caller falls through to the normal
//! parse/bind/plan path. "If unsure, do not parameterize" is the invariant
//! that keeps this safe.

use zyron_common::TypeId;
use zyron_executor::column::ScalarValue;
use zyron_parser::{Keyword, Lexer, SpannedToken, Token};

/// Cap on the number of literals a single statement may auto-parameterize.
/// A pathological client sending `VALUES (lit) x 1M` falls through to the
/// non-cached path rather than allocating an enormous parameter vector.
const MAX_LITERALS: usize = 32_768;

/// A query rewritten into a cacheable template plus its extracted literals.
pub struct TemplatedQuery {
    /// SQL with each literal replaced by a `$N` placeholder. Parses to a
    /// plan whose literal slots are `BoundExpr::Parameter`.
    pub template: String,
    /// Extracted literal values in `$1..$N` order, ready to bind as params.
    pub literals: Vec<ScalarValue>,
    /// Hash of the ordered literal `TypeId` sequence. Part of the cache key
    /// so a shape bound with int literals is not reused for float literals.
    pub type_kinds_hash: u64,
}

/// One extracted literal with the source byte range it occupied.
struct Literal {
    start: usize,
    end: usize,
    value: ScalarValue,
    type_id: TypeId,
}

/// Attempts to rewrite `sql` into a cacheable template. Returns `None` when
/// the statement is not a recognized safe DML shape, in which case the
/// caller must use the normal parse path.
pub fn templatize(sql: &str) -> Option<TemplatedQuery> {
    // Cheap first-token check: only INSERT/UPDATE/DELETE/SELECT are cacheable
    // shapes. Everything else (DDL, SET, transaction control, EXPLAIN, LISTEN,
    // PREPARE, cursors) bails after lexing a single token instead of paying a
    // full tokenization pass just to fall through.
    let mut probe = Lexer::new(sql);
    let kind = match probe.next_token().ok()?.token {
        Token::Keyword(Keyword::Insert) => Keyword::Insert,
        Token::Keyword(Keyword::Update) => Keyword::Update,
        Token::Keyword(Keyword::Delete) => Keyword::Delete,
        Token::Keyword(Keyword::Select) => Keyword::Select,
        _ => return None,
    };

    let tokens = lex_all(sql)?;
    if tokens.is_empty() {
        return None;
    }

    // INSERT parameterizes every value-tuple literal; UPDATE/DELETE/SELECT
    // parameterize only literals in explicitly-safe clauses (SET, WHERE) and
    // never in SELECT-list, GROUP BY, ORDER BY, LIMIT, or JOIN/ON positions,
    // where a literal can be a column ordinal or drive plan structure.
    match kind {
        Keyword::Insert => scan_insert_values(sql, &tokens),
        Keyword::Update => scan_clause_dml(sql, &tokens, StmtKind::Update),
        Keyword::Delete => scan_clause_dml(sql, &tokens, StmtKind::Delete),
        Keyword::Select => scan_clause_dml(sql, &tokens, StmtKind::Select),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StmtKind {
    Update,
    Delete,
    Select,
}

/// Region of an UPDATE/DELETE/SELECT statement. Literals are parameterized
/// only in `Set` (UPDATE assignment list) and `Where`. Every other region is
/// left untouched so column ordinals (`ORDER BY 1`), LIMIT/OFFSET counts, and
/// select-list constants keep their plan-shaping role.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Region {
    Other,
    Set,
    Where,
}

/// Lexes the entire input into spanned tokens. Returns `None` if lexing
/// fails (the normal parser will produce the user-facing error).
fn lex_all(sql: &str) -> Option<Vec<SpannedToken>> {
    let mut lexer = Lexer::new(sql);
    let mut out = Vec::new();
    loop {
        match lexer.next_token() {
            Ok(t) => {
                if matches!(t.token, Token::Eof) {
                    break;
                }
                out.push(t);
            }
            Err(_) => return None,
        }
    }
    Some(out)
}

/// Recognizes `INSERT INTO <ident> [(col, ...)] VALUES (lit, ...), (...) ...`
/// with literal-only value tuples. Rejects ON CONFLICT, RETURNING, and any
/// INSERT ... SELECT form by requiring the token stream to end right after
/// the final value tuple.
fn scan_insert_values(sql: &str, tokens: &[SpannedToken]) -> Option<TemplatedQuery> {
    let mut i = 0usize;
    expect_kw(tokens, &mut i, Keyword::Insert)?;
    expect_kw(tokens, &mut i, Keyword::Into)?;
    expect_ident(tokens, &mut i)?;

    // Optional schema-qualified name: <ident>.<ident>
    if matches!(tokens.get(i).map(|t| &t.token), Some(Token::Dot)) {
        i += 1;
        expect_ident(tokens, &mut i)?;
    }

    // Optional column list: ( ident, ident, ... )
    if matches!(tokens.get(i).map(|t| &t.token), Some(Token::LParen)) {
        i += 1;
        loop {
            expect_ident(tokens, &mut i)?;
            match tokens.get(i).map(|t| &t.token) {
                Some(Token::Comma) => {
                    i += 1;
                }
                Some(Token::RParen) => {
                    i += 1;
                    break;
                }
                _ => return None,
            }
        }
    }

    expect_kw(tokens, &mut i, Keyword::Values)?;

    let mut literals: Vec<Literal> = Vec::new();

    // One or more parenthesized literal tuples, comma-separated.
    loop {
        if !matches!(tokens.get(i).map(|t| &t.token), Some(Token::LParen)) {
            return None;
        }
        i += 1;

        // At least one value per tuple.
        loop {
            let lit = scan_literal(tokens, &mut i)?;
            if literals.len() >= MAX_LITERALS {
                return None;
            }
            literals.push(lit);
            match tokens.get(i).map(|t| &t.token) {
                Some(Token::Comma) => {
                    i += 1;
                }
                Some(Token::RParen) => {
                    i += 1;
                    break;
                }
                _ => return None,
            }
        }

        match tokens.get(i).map(|t| &t.token) {
            Some(Token::Comma) => {
                i += 1;
            }
            Some(Token::Semicolon) | None => break,
            _ => return None,
        }
    }

    // Reject anything trailing (ON CONFLICT, RETURNING, a second statement).
    match tokens.get(i).map(|t| &t.token) {
        None => {}
        Some(Token::Semicolon) => {
            // A trailing semicolon is fine only if it is the very last token.
            if i + 1 != tokens.len() {
                return None;
            }
        }
        _ => return None,
    }

    if literals.is_empty() {
        return None;
    }

    Some(build_template(sql, literals))
}

/// Returns true if a token produces a value, used to tell a binary minus
/// (`a - 5`) from a unary minus (`x = -5`). A minus is unary iff the previous
/// significant token is NOT value-producing.
fn is_value_token(t: &Token) -> bool {
    matches!(
        t,
        Token::Integer(_)
            | Token::Float(_)
            | Token::String(_)
            | Token::Ident(_)
            | Token::Parameter(_)
            | Token::RParen
            | Token::Keyword(Keyword::Null)
            | Token::Keyword(Keyword::True)
            | Token::Keyword(Keyword::False)
    )
}

/// Recognizes UPDATE/DELETE/SELECT and parameterizes literals only inside the
/// SET and WHERE clauses. Rejects subqueries, set operations (UNION/etc.),
/// and trailing multi-statements so the rewrite stays provably safe.
fn scan_clause_dml(sql: &str, tokens: &[SpannedToken], kind: StmtKind) -> Option<TemplatedQuery> {
    let mut region = Region::Other;
    let mut literals: Vec<Literal> = Vec::new();
    let mut prev_value = false;

    let mut idx = 0usize;
    while idx < tokens.len() {
        let tok = &tokens[idx].token;

        match tok {
            // A second SELECT means a subquery or a set-operation arm: not
            // handled, fall back to the full parser.
            Token::Keyword(Keyword::Select) if idx != 0 => return None,
            Token::Keyword(Keyword::Union)
            | Token::Keyword(Keyword::Intersect)
            | Token::Keyword(Keyword::Except) => return None,

            // A client-supplied $N placeholder already occupies a parameter
            // slot. Renumbering extracted literals from $1 would collide with
            // it (binding one literal to two positions), so this shape is not
            // cacheable: fall back to the full parser.
            Token::Parameter(_) => return None,

            // Region transitions.
            Token::Keyword(Keyword::Set) if kind == StmtKind::Update => {
                region = Region::Set;
                prev_value = false;
                idx += 1;
                continue;
            }
            Token::Keyword(Keyword::Where) => {
                region = Region::Where;
                prev_value = false;
                idx += 1;
                continue;
            }
            // Any clause that ends the SET/WHERE region. Literals beyond here
            // (ORDER BY ordinals, LIMIT counts, select-list constants, JOIN
            // conditions) are left untouched.
            Token::Keyword(Keyword::Group)
            | Token::Keyword(Keyword::Order)
            | Token::Keyword(Keyword::Having)
            | Token::Keyword(Keyword::Limit)
            | Token::Keyword(Keyword::Offset)
            | Token::Keyword(Keyword::Fetch)
            | Token::Keyword(Keyword::For)
            | Token::Keyword(Keyword::Returning)
            | Token::Keyword(Keyword::Join)
            | Token::Keyword(Keyword::On) => {
                region = Region::Other;
                prev_value = false;
                idx += 1;
                continue;
            }

            Token::Semicolon => {
                // Only a trailing semicolon is allowed.
                if idx + 1 != tokens.len() {
                    return None;
                }
                idx += 1;
                continue;
            }

            _ => {}
        }

        let parameterize = matches!(region, Region::Set | Region::Where);

        // Unary minus folding: only when the minus is unary (not preceded by a
        // value) and we are in a parameterize region.
        if parameterize && matches!(tok, Token::Minus) && !prev_value && idx + 1 < tokens.len() {
            let start = tokens[idx].span.offset;
            let num = &tokens[idx + 1];
            let end = num.span.offset + num.span.length;
            match &num.token {
                Token::Integer(v) => {
                    literals.push(Literal {
                        start,
                        end,
                        value: ScalarValue::Int64(v.wrapping_neg()),
                        type_id: TypeId::Int64,
                    });
                    prev_value = true;
                    idx += 2;
                    continue;
                }
                Token::Float(v) => {
                    literals.push(Literal {
                        start,
                        end,
                        value: ScalarValue::Float64(-*v),
                        type_id: TypeId::Float64,
                    });
                    prev_value = true;
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        }

        // A standalone literal in a parameterize region.
        if parameterize {
            if let Some(lit) = literal_at(&tokens[idx]) {
                if literals.len() >= MAX_LITERALS {
                    return None;
                }
                literals.push(lit);
                prev_value = true;
                idx += 1;
                continue;
            }
        }

        prev_value = is_value_token(tok);
        idx += 1;
    }

    // Even with no literals the template equals the SQL, which still gives an
    // exact-text plan cache entry for a repeated query.
    Some(build_template(sql, literals))
}

/// Extracts a literal from a single token (no unary-minus folding).
fn literal_at(st: &SpannedToken) -> Option<Literal> {
    let start = st.span.offset;
    let end = st.span.offset + st.span.length;
    let (value, type_id) = match &st.token {
        Token::Integer(v) => (ScalarValue::Int64(*v), TypeId::Int64),
        Token::Float(v) => (ScalarValue::Float64(*v), TypeId::Float64),
        Token::String(s) => (ScalarValue::Utf8(s.clone()), TypeId::Text),
        Token::Keyword(Keyword::Null) => (ScalarValue::Null, TypeId::Null),
        Token::Keyword(Keyword::True) => (ScalarValue::Boolean(true), TypeId::Boolean),
        Token::Keyword(Keyword::False) => (ScalarValue::Boolean(false), TypeId::Boolean),
        _ => return None,
    };
    Some(Literal {
        start,
        end,
        value,
        type_id,
    })
}

/// Scans a single literal at `*i`, advancing past it. Coalesces a leading
/// unary minus with the following number so `-1` becomes one signed scalar
/// rather than a unary-op expression (which would be non-cacheable).
fn scan_literal(tokens: &[SpannedToken], i: &mut usize) -> Option<Literal> {
    let tok = tokens.get(*i)?;
    let start = tok.span.offset;

    // Unary minus: fold with the next numeric literal.
    if matches!(tok.token, Token::Minus) {
        let num = tokens.get(*i + 1)?;
        let end = num.span.offset + num.span.length;
        let lit = match &num.token {
            Token::Integer(v) => Literal {
                start,
                end,
                value: ScalarValue::Int64(v.wrapping_neg()),
                type_id: TypeId::Int64,
            },
            Token::Float(v) => Literal {
                start,
                end,
                value: ScalarValue::Float64(-*v),
                type_id: TypeId::Float64,
            },
            _ => return None,
        };
        *i += 2;
        return Some(lit);
    }

    let end = tok.span.offset + tok.span.length;
    let lit = match &tok.token {
        Token::Integer(v) => Literal {
            start,
            end,
            value: ScalarValue::Int64(*v),
            type_id: TypeId::Int64,
        },
        Token::Float(v) => Literal {
            start,
            end,
            value: ScalarValue::Float64(*v),
            type_id: TypeId::Float64,
        },
        Token::String(s) => Literal {
            start,
            end,
            value: ScalarValue::Utf8(s.clone()),
            type_id: TypeId::Text,
        },
        Token::Keyword(Keyword::Null) => Literal {
            start,
            end,
            value: ScalarValue::Null,
            type_id: TypeId::Null,
        },
        Token::Keyword(Keyword::True) => Literal {
            start,
            end,
            value: ScalarValue::Boolean(true),
            type_id: TypeId::Boolean,
        },
        Token::Keyword(Keyword::False) => Literal {
            start,
            end,
            value: ScalarValue::Boolean(false),
            type_id: TypeId::Boolean,
        },
        // Identifiers, parameters, function calls, nested parens, casts,
        // etc. are not literals: the tuple is non-cacheable.
        _ => return None,
    };
    *i += 1;
    Some(lit)
}

/// Builds the template string by replacing each literal's byte range with
/// `$N`, copying every other byte verbatim. Literals arrive in source
/// order, which is also `$1..$N` order.
fn build_template(sql: &str, literals: Vec<Literal>) -> TemplatedQuery {
    let bytes = sql.as_bytes();
    let mut template = String::with_capacity(sql.len());
    let mut values = Vec::with_capacity(literals.len());
    let mut type_state: u64 = 0xcbf2_9ce4_8422_2325;
    let mut cursor = 0usize;

    for (idx, lit) in literals.into_iter().enumerate() {
        // Copy verbatim bytes preceding this literal.
        template.push_str(std::str::from_utf8(&bytes[cursor..lit.start]).unwrap_or(""));
        // Emit the placeholder ($1-based).
        template.push('$');
        template.push_str(&(idx + 1).to_string());
        cursor = lit.end;

        type_state = type_state.rotate_left(5) ^ (lit.type_id as u64).wrapping_add(1);
        values.push(lit.value);
    }
    template.push_str(std::str::from_utf8(&bytes[cursor..]).unwrap_or(""));

    TemplatedQuery {
        template,
        literals: values,
        type_kinds_hash: type_state,
    }
}

fn expect_kw(tokens: &[SpannedToken], i: &mut usize, kw: Keyword) -> Option<()> {
    match tokens.get(*i).map(|t| &t.token) {
        Some(Token::Keyword(k)) if *k == kw => {
            *i += 1;
            Some(())
        }
        _ => None,
    }
}

fn expect_ident(tokens: &[SpannedToken], i: &mut usize) -> Option<()> {
    match tokens.get(*i).map(|t| &t.token) {
        Some(Token::Ident(_)) => {
            *i += 1;
            Some(())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tq(sql: &str) -> Option<TemplatedQuery> {
        templatize(sql)
    }

    #[test]
    fn basic_insert_values_single_row() {
        let q = tq("INSERT INTO orders VALUES (1, 'paid', 3.5)").expect("cacheable");
        assert_eq!(q.template, "INSERT INTO orders VALUES ($1, $2, $3)");
        assert_eq!(q.literals.len(), 3);
        assert!(matches!(q.literals[0], ScalarValue::Int64(1)));
        assert!(matches!(q.literals[1], ScalarValue::Utf8(ref s) if s == "paid"));
        assert!(matches!(q.literals[2], ScalarValue::Float64(_)));
    }

    #[test]
    fn multi_row_insert() {
        let q = tq("INSERT INTO t VALUES (1,'a'),(2,'b')").expect("cacheable");
        assert_eq!(q.template, "INSERT INTO t VALUES ($1,$2),($3,$4)");
        assert_eq!(q.literals.len(), 4);
    }

    #[test]
    fn insert_with_column_list() {
        let q = tq("INSERT INTO t (id, name) VALUES (1, 'x')").expect("cacheable");
        assert_eq!(q.template, "INSERT INTO t (id, name) VALUES ($1, $2)");
        assert_eq!(q.literals.len(), 2);
    }

    #[test]
    fn schema_qualified_table() {
        let q = tq("INSERT INTO app.orders VALUES (1)").expect("cacheable");
        assert_eq!(q.template, "INSERT INTO app.orders VALUES ($1)");
    }

    #[test]
    fn null_literal() {
        let q = tq("INSERT INTO t VALUES (1, NULL)").expect("cacheable");
        assert_eq!(q.literals.len(), 2);
        assert!(matches!(q.literals[1], ScalarValue::Null));
    }

    #[test]
    fn boolean_literals() {
        let q = tq("INSERT INTO t VALUES (TRUE, FALSE)").expect("cacheable");
        assert!(matches!(q.literals[0], ScalarValue::Boolean(true)));
        assert!(matches!(q.literals[1], ScalarValue::Boolean(false)));
    }

    #[test]
    fn negative_numbers_folded() {
        let q = tq("INSERT INTO t VALUES (-1, -2.5)").expect("cacheable");
        assert!(matches!(q.literals[0], ScalarValue::Int64(-1)));
        assert!(matches!(q.literals[1], ScalarValue::Float64(v) if (v + 2.5).abs() < 1e-9));
        assert_eq!(q.template, "INSERT INTO t VALUES ($1, $2)");
    }

    #[test]
    fn escaped_quote_string() {
        // 'a''b' is the SQL escape for a'b
        let q = tq("INSERT INTO t VALUES ('a''b')").expect("cacheable");
        assert_eq!(q.literals.len(), 1);
        assert!(matches!(q.literals[0], ScalarValue::Utf8(ref s) if s == "a'b"));
    }

    #[test]
    fn type_kinds_distinguish_int_vs_float() {
        let a = tq("INSERT INTO t VALUES (1, 2)").unwrap();
        let b = tq("INSERT INTO t VALUES (1, 2.0)").unwrap();
        assert_eq!(a.template, b.template, "templates match");
        assert_ne!(
            a.type_kinds_hash, b.type_kinds_hash,
            "type kinds must differ so plans do not collide"
        );
    }

    #[test]
    fn reject_function_call_in_tuple() {
        assert!(tq("INSERT INTO t VALUES (1, now())").is_none());
    }

    #[test]
    fn reject_existing_parameter() {
        assert!(tq("INSERT INTO t VALUES (1, $1)").is_none());
    }

    #[test]
    fn reject_existing_parameter_in_clause_dml() {
        // A client-supplied $N in UPDATE/DELETE/SELECT must fall through, else
        // renumbering extracted literals from $1 collides with the existing
        // placeholder and binds one literal to two slots.
        assert!(tq("UPDATE t SET x = $1 WHERE y = 5").is_none());
        assert!(tq("DELETE FROM t WHERE y = $1").is_none());
        assert!(tq("SELECT a FROM t WHERE y = $1 AND z = 3").is_none());
    }

    #[test]
    fn reject_on_conflict() {
        assert!(tq("INSERT INTO t VALUES (1) ON CONFLICT DO NOTHING").is_none());
    }

    #[test]
    fn reject_returning() {
        assert!(tq("INSERT INTO t VALUES (1) RETURNING id").is_none());
    }

    #[test]
    fn reject_insert_select() {
        assert!(tq("INSERT INTO t SELECT * FROM s").is_none());
    }

    #[test]
    fn reject_multi_statement() {
        assert!(tq("INSERT INTO t VALUES (1); INSERT INTO t VALUES (2)").is_none());
    }

    #[test]
    fn reject_cast_in_tuple() {
        assert!(tq("INSERT INTO t VALUES (1::bigint)").is_none());
    }

    #[test]
    fn update_set_and_where() {
        let q = tq("UPDATE t SET status = 'paid', amount = 9 WHERE id = 42").expect("cacheable");
        assert_eq!(
            q.template,
            "UPDATE t SET status = $1, amount = $2 WHERE id = $3"
        );
        assert_eq!(q.literals.len(), 3);
        assert!(matches!(q.literals[0], ScalarValue::Utf8(ref s) if s == "paid"));
        assert!(matches!(q.literals[1], ScalarValue::Int64(9)));
        assert!(matches!(q.literals[2], ScalarValue::Int64(42)));
    }

    #[test]
    fn delete_where() {
        let q =
            tq("DELETE FROM orders WHERE customer_id = 7 AND status = 'old'").expect("cacheable");
        assert_eq!(
            q.template,
            "DELETE FROM orders WHERE customer_id = $1 AND status = $2"
        );
        assert_eq!(q.literals.len(), 2);
    }

    #[test]
    fn select_where_only() {
        let q = tq("SELECT id, amount FROM orders WHERE customer_id = 5").expect("cacheable");
        assert_eq!(
            q.template,
            "SELECT id, amount FROM orders WHERE customer_id = $1"
        );
        assert_eq!(q.literals.len(), 1);
        assert!(matches!(q.literals[0], ScalarValue::Int64(5)));
    }

    #[test]
    fn select_list_literal_not_parameterized() {
        // The constant 1 in the select list shapes output, not a filter value.
        let q = tq("SELECT 1 FROM orders WHERE id = 9").expect("cacheable");
        assert_eq!(q.template, "SELECT 1 FROM orders WHERE id = $1");
        assert_eq!(q.literals.len(), 1);
    }

    #[test]
    fn order_by_ordinal_not_parameterized() {
        // ORDER BY 1 is a column ordinal: must never become ORDER BY $1.
        let q = tq("SELECT id FROM t WHERE x = 5 ORDER BY 1").expect("cacheable");
        assert_eq!(q.template, "SELECT id FROM t WHERE x = $1 ORDER BY 1");
        assert_eq!(q.literals.len(), 1);
        assert!(matches!(q.literals[0], ScalarValue::Int64(5)));
    }

    #[test]
    fn limit_not_parameterized() {
        let q = tq("SELECT id FROM t WHERE x = 5 LIMIT 10").expect("cacheable");
        assert_eq!(q.template, "SELECT id FROM t WHERE x = $1 LIMIT 10");
        assert_eq!(q.literals.len(), 1);
    }

    #[test]
    fn where_in_list() {
        let q = tq("SELECT id FROM t WHERE x IN (1, 2, 3)").expect("cacheable");
        assert_eq!(q.template, "SELECT id FROM t WHERE x IN ($1, $2, $3)");
        assert_eq!(q.literals.len(), 3);
    }

    #[test]
    fn where_negative_literal() {
        let q = tq("SELECT id FROM t WHERE balance = -5").expect("cacheable");
        assert_eq!(q.template, "SELECT id FROM t WHERE balance = $1");
        assert!(matches!(q.literals[0], ScalarValue::Int64(-5)));
    }

    #[test]
    fn where_binary_minus_not_folded() {
        // `amount - 5` is binary subtraction: the 5 parameterizes positively,
        // the minus stays an operator.
        let q = tq("SELECT id FROM t WHERE amount - 5 > 0").expect("cacheable");
        assert_eq!(q.template, "SELECT id FROM t WHERE amount - $1 > $2");
        assert!(matches!(q.literals[0], ScalarValue::Int64(5)));
        assert!(matches!(q.literals[1], ScalarValue::Int64(0)));
    }

    #[test]
    fn reject_subquery() {
        assert!(tq("SELECT id FROM t WHERE x IN (SELECT y FROM s)").is_none());
    }

    #[test]
    fn reject_union() {
        assert!(tq("SELECT id FROM t WHERE x = 1 UNION SELECT id FROM s").is_none());
    }

    #[test]
    fn select_no_where_caches_as_template() {
        // No filter literals: template equals SQL, still a valid cache entry.
        let q = tq("SELECT count(*) FROM orders").expect("cacheable");
        assert_eq!(q.template, "SELECT count(*) FROM orders");
        assert_eq!(q.literals.len(), 0);
    }

    #[test]
    fn reject_non_dml() {
        assert!(tq("CREATE TABLE t (id INT)").is_none());
        assert!(tq("BEGIN").is_none());
        assert!(tq("VACUUM").is_none());
    }

    #[test]
    fn trailing_semicolon_ok() {
        let q = tq("INSERT INTO t VALUES (1);").expect("cacheable");
        assert_eq!(q.literals.len(), 1);
    }

    #[test]
    fn literal_cap_falls_through() {
        let mut sql = String::from("INSERT INTO t VALUES ");
        for i in 0..(MAX_LITERALS + 10) {
            if i > 0 {
                sql.push(',');
            }
            sql.push_str("(1)");
        }
        assert!(tq(&sql).is_none(), "over-cap insert must not cache");
    }
}
