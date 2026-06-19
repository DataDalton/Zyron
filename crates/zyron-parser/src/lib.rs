//! SQL parser for ZyronDB.
//!
//! Recursive descent parser with Pratt expression parsing.
//! Converts SQL text into a typed AST for query planning and execution.

pub mod ast;
pub mod expr_sql;
pub mod lexer;
pub mod parser;
pub mod simd_scan;
pub mod token;

pub use ast::*;
pub use expr_sql::expr_to_sql;
pub use lexer::Lexer;
pub use parser::Parser;
pub use token::{Keyword, Span, SpannedToken, Token};

/// Parses a SQL string into a list of semicolon-separated statements.
pub fn parse(sql: &str) -> zyron_common::Result<Vec<Statement>> {
    let mut parser = Parser::new(sql)?;
    parser.parse_statements()
}

/// Parses a single SQL scalar expression (e.g. a stored column default) into an
/// AST node. Used to round-trip column defaults and CHECK expressions that are
/// persisted as SQL text in the catalog.
pub fn parse_expr(input: &str) -> zyron_common::Result<Expr> {
    let mut parser = Parser::new(input)?;
    parser.parse_expr()
}
