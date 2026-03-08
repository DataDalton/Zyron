//! SQL parser for ZyronDB.
//!
//! Recursive descent parser with Pratt expression parsing.
//! Converts SQL text into a typed AST for query planning and execution.

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod token;

pub use ast::*;
pub use lexer::Lexer;
pub use parser::Parser;
pub use token::{Keyword, Span, SpannedToken, Token};

/// Parses a SQL string into a list of semicolon-separated statements.
pub fn parse(sql: &str) -> zyron_common::Result<Vec<Statement>> {
    let mut parser = Parser::new(sql)?;
    parser.parse_statements()
}
