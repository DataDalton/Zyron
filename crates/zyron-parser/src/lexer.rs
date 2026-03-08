//! SQL lexer that converts input text into a stream of tokens.

use zyron_common::{Result, ZyronError};

use crate::token::{Span, SpannedToken, Token, lookup_keyword};

/// SQL lexer. Produces tokens from an input SQL string.
pub struct Lexer<'a> {
    input: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: usize,
    column: usize,
    peeked: Option<SpannedToken>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
            line: 1,
            column: 1,
            peeked: None,
        }
    }

    /// Returns the current 1-based line number.
    pub fn line(&self) -> usize {
        self.line
    }

    /// Returns the current 1-based column number.
    pub fn column(&self) -> usize {
        self.column
    }

    /// Returns the next token, consuming it from the input.
    pub fn next_token(&mut self) -> Result<SpannedToken> {
        if let Some(tok) = self.peeked.take() {
            return Ok(tok);
        }
        self.scan_token()
    }

    /// Peeks at the next token without consuming it.
    pub fn peek_token(&mut self) -> Result<&SpannedToken> {
        if self.peeked.is_none() {
            let tok = self.scan_token()?;
            self.peeked = Some(tok);
        }
        Ok(self.peeked.as_ref().unwrap())
    }

    // -----------------------------------------------------------------------
    // Internal scanning
    // -----------------------------------------------------------------------

    fn scan_token(&mut self) -> Result<SpannedToken> {
        self.skip_whitespace_and_comments()?;

        if self.pos >= self.bytes.len() {
            return Ok(SpannedToken::new(Token::Eof, Span::new(self.pos, 0)));
        }

        let b = self.bytes[self.pos];

        match b {
            b'\'' => self.scan_string(),
            b'"' => self.scan_quoted_ident(),
            b'0'..=b'9' => self.scan_number(),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.scan_word(),
            _ => self.scan_operator_or_punctuation(),
        }
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<()> {
        loop {
            // Skip whitespace
            while self.pos < self.bytes.len() {
                match self.bytes[self.pos] {
                    b' ' | b'\t' | b'\r' => {
                        self.pos += 1;
                        self.column += 1;
                    }
                    b'\n' => {
                        self.pos += 1;
                        self.line += 1;
                        self.column = 1;
                    }
                    _ => break,
                }
            }

            if self.pos >= self.bytes.len() {
                return Ok(());
            }

            // Single-line comment: --
            if self.pos + 1 < self.bytes.len()
                && self.bytes[self.pos] == b'-'
                && self.bytes[self.pos + 1] == b'-'
            {
                self.pos += 2;
                self.column += 2;
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                    self.column += 1;
                }
                continue;
            }

            // Block comment: /* ... */
            if self.pos + 1 < self.bytes.len()
                && self.bytes[self.pos] == b'/'
                && self.bytes[self.pos + 1] == b'*'
            {
                let start_line = self.line;
                let start_col = self.column;
                self.pos += 2;
                self.column += 2;
                let mut depth = 1u32;

                while self.pos < self.bytes.len() && depth > 0 {
                    if self.pos + 1 < self.bytes.len()
                        && self.bytes[self.pos] == b'/'
                        && self.bytes[self.pos + 1] == b'*'
                    {
                        depth += 1;
                        self.pos += 2;
                        self.column += 2;
                    } else if self.pos + 1 < self.bytes.len()
                        && self.bytes[self.pos] == b'*'
                        && self.bytes[self.pos + 1] == b'/'
                    {
                        depth -= 1;
                        self.pos += 2;
                        self.column += 2;
                    } else if self.bytes[self.pos] == b'\n' {
                        self.pos += 1;
                        self.line += 1;
                        self.column = 1;
                    } else {
                        self.pos += 1;
                        self.column += 1;
                    }
                }

                if depth > 0 {
                    return Err(ZyronError::ParseError(format!(
                        "Unterminated block comment starting at line {}, column {}",
                        start_line, start_col
                    )));
                }
                continue;
            }

            // Not whitespace or comment
            return Ok(());
        }
    }

    fn scan_number(&mut self) -> Result<SpannedToken> {
        let start = self.pos;
        let start_col = self.column;

        // Consume digits
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
            self.pos += 1;
            self.column += 1;
        }

        // Check for decimal point followed by digit
        let is_float = self.pos < self.bytes.len()
            && self.bytes[self.pos] == b'.'
            && self.pos + 1 < self.bytes.len()
            && self.bytes[self.pos + 1].is_ascii_digit();

        if is_float {
            self.pos += 1; // consume '.'
            self.column += 1;
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                self.pos += 1;
                self.column += 1;
            }

            // Scientific notation: e/E followed by optional +/- and digits
            if self.pos < self.bytes.len()
                && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E')
            {
                self.pos += 1;
                self.column += 1;
                if self.pos < self.bytes.len()
                    && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-')
                {
                    self.pos += 1;
                    self.column += 1;
                }
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_digit() {
                    self.pos += 1;
                    self.column += 1;
                }
            }

            let text = &self.input[start..self.pos];
            let value: f64 = text.parse().map_err(|_| {
                ZyronError::ParseError(format!(
                    "Invalid float literal '{}' at line {}, column {}",
                    text, self.line, start_col
                ))
            })?;
            let span = Span::new(start, self.pos - start);
            return Ok(SpannedToken::new(Token::Float(value), span));
        }

        let text = &self.input[start..self.pos];
        let value: i64 = text.parse().map_err(|_| {
            ZyronError::ParseError(format!(
                "Integer literal '{}' overflows i64 at line {}, column {}",
                text, self.line, start_col
            ))
        })?;
        let span = Span::new(start, self.pos - start);
        Ok(SpannedToken::new(Token::Integer(value), span))
    }

    fn scan_string(&mut self) -> Result<SpannedToken> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        self.pos += 1; // consume opening quote
        self.column += 1;

        // Scan forward to find closing quote for zero-copy fast path.
        let content_start = self.pos;
        let mut has_escape = false;
        let mut scan = self.pos;
        while scan < self.bytes.len() {
            if self.bytes[scan] == b'\'' {
                if scan + 1 < self.bytes.len() && self.bytes[scan + 1] == b'\'' {
                    has_escape = true;
                    scan += 2;
                    continue;
                }
                break;
            }
            scan += 1;
        }
        if scan >= self.bytes.len() {
            return Err(ZyronError::ParseError(format!(
                "Unterminated string literal starting at line {}, column {}",
                start_line, start_col
            )));
        }

        let value = if !has_escape {
            // Fast path: no escaped quotes, direct slice copy
            let s = self.input[content_start..scan].to_string();
            // Update line/column tracking
            for &b in &self.bytes[content_start..scan] {
                if b == b'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
            }
            self.pos = scan + 1; // skip closing quote
            self.column += 1;
            s
        } else {
            // Slow path: has '' escapes, build string manually
            let mut s = String::with_capacity(scan - content_start);
            while self.pos < self.bytes.len() {
                let b = self.bytes[self.pos];
                if b == b'\'' {
                    self.pos += 1;
                    self.column += 1;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\'' {
                        s.push('\'');
                        self.pos += 1;
                        self.column += 1;
                        continue;
                    }
                    break;
                }
                if b == b'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                s.push(b as char);
                self.pos += 1;
            }
            s
        };

        let span = Span::new(start, self.pos - start);
        Ok(SpannedToken::new(Token::String(value), span))
    }

    fn scan_quoted_ident(&mut self) -> Result<SpannedToken> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;

        self.pos += 1; // consume opening quote
        self.column += 1;

        // Scan forward to find closing quote for zero-copy fast path.
        let content_start = self.pos;
        let mut has_escape = false;
        let mut scan = self.pos;
        while scan < self.bytes.len() {
            if self.bytes[scan] == b'"' {
                if scan + 1 < self.bytes.len() && self.bytes[scan + 1] == b'"' {
                    has_escape = true;
                    scan += 2;
                    continue;
                }
                break;
            }
            scan += 1;
        }
        if scan >= self.bytes.len() {
            return Err(ZyronError::ParseError(format!(
                "Unterminated quoted identifier starting at line {}, column {}",
                start_line, start_col
            )));
        }

        let value = if !has_escape {
            // Fast path: no escaped quotes, direct slice copy
            let s = self.input[content_start..scan].to_string();
            for &b in &self.bytes[content_start..scan] {
                if b == b'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
            }
            self.pos = scan + 1; // skip closing quote
            self.column += 1;
            s
        } else {
            // Slow path: has "" escapes, build string manually
            let mut s = String::with_capacity(scan - content_start);
            while self.pos < self.bytes.len() {
                let b = self.bytes[self.pos];
                if b == b'"' {
                    self.pos += 1;
                    self.column += 1;
                    if self.pos < self.bytes.len() && self.bytes[self.pos] == b'"' {
                        s.push('"');
                        self.pos += 1;
                        self.column += 1;
                        continue;
                    }
                    break;
                }
                if b == b'\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                s.push(b as char);
                self.pos += 1;
            }
            s
        };

        let span = Span::new(start, self.pos - start);
        Ok(SpannedToken::new(Token::Ident(value), span))
    }

    fn scan_word(&mut self) -> Result<SpannedToken> {
        let start = self.pos;

        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
            self.column += 1;
        }

        let word = &self.input[start..self.pos];
        let span = Span::new(start, self.pos - start);

        match lookup_keyword(word) {
            Some(kw) => Ok(SpannedToken::new(Token::Keyword(kw), span)),
            None => Ok(SpannedToken::new(Token::Ident(word.to_string()), span)),
        }
    }

    fn scan_operator_or_punctuation(&mut self) -> Result<SpannedToken> {
        let start = self.pos;
        let b = self.bytes[self.pos];
        let next = if self.pos + 1 < self.bytes.len() {
            Some(self.bytes[self.pos + 1])
        } else {
            None
        };
        let next2 = if self.pos + 2 < self.bytes.len() {
            Some(self.bytes[self.pos + 2])
        } else {
            None
        };

        let (token, len) = match (b, next, next2) {
            // Three-character operators (must be checked before 2-char prefixes)
            (b'#', Some(b'>'), Some(b'>')) => (Token::HashDoubleArrow, 3),
            (b'-', Some(b'>'), Some(b'>')) => (Token::DoubleArrow, 3),
            (b'<', Some(b'='), Some(b'>')) => (Token::CosineDistance, 3),
            (b'<', Some(b'-'), Some(b'>')) => (Token::L2Distance, 3),
            (b'<', Some(b'#'), Some(b'>')) => (Token::DotDistance, 3),

            // Two-character operators
            (b'?', Some(b'|'), _) => (Token::QuestionPipe, 2),
            (b'?', Some(b'&'), _) => (Token::QuestionAmp, 2),
            (b'|', Some(b'|'), _) => (Token::Concat, 2),
            (b'<', Some(b'='), _) => (Token::LtEq, 2),
            (b'>', Some(b'='), _) => (Token::GtEq, 2),
            (b'<', Some(b'>'), _) => (Token::Neq, 2),
            (b'!', Some(b'='), _) => (Token::Neq, 2),
            (b':', Some(b':'), _) => (Token::DoubleColon, 2),
            (b'=', Some(b'>'), _) => (Token::FatArrow, 2),
            (b'-', Some(b'>'), _) => (Token::Arrow, 2),
            (b'#', Some(b'>'), _) => (Token::HashArrow, 2),
            (b'@', Some(b'>'), _) => (Token::AtArrow, 2),
            (b'<', Some(b'@'), _) => (Token::ArrowAt, 2),

            // Single-character operators
            (b'+', _, _) => (Token::Plus, 1),
            (b'-', _, _) => (Token::Minus, 1),
            (b'*', _, _) => (Token::Star, 1),
            (b'/', _, _) => (Token::Slash, 1),
            (b'%', _, _) => (Token::Percent, 1),
            (b'=', _, _) => (Token::Eq, 1),
            (b'<', _, _) => (Token::Lt, 1),
            (b'>', _, _) => (Token::Gt, 1),
            (b'?', _, _) => (Token::Question, 1),

            // Punctuation
            (b',', _, _) => (Token::Comma, 1),
            (b';', _, _) => (Token::Semicolon, 1),
            (b'(', _, _) => (Token::LParen, 1),
            (b')', _, _) => (Token::RParen, 1),
            (b'[', _, _) => (Token::LBracket, 1),
            (b']', _, _) => (Token::RBracket, 1),
            (b'.', _, _) => (Token::Dot, 1),

            _ => {
                return Err(ZyronError::ParseError(format!(
                    "Unexpected character '{}' at line {}, column {}",
                    b as char, self.line, self.column
                )));
            }
        };

        self.pos += len;
        self.column += len;
        let span = Span::new(start, len);
        Ok(SpannedToken::new(token, span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Keyword;

    fn lex_all(input: &str) -> Result<Vec<Token>> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let st = lexer.next_token()?;
            if st.token == Token::Eof {
                break;
            }
            tokens.push(st.token);
        }
        Ok(tokens)
    }

    #[test]
    fn test_select_statement() {
        let tokens = lex_all("SELECT a, b FROM t WHERE x = 1").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Ident("a".to_string()),
                Token::Comma,
                Token::Ident("b".to_string()),
                Token::Keyword(Keyword::From),
                Token::Ident("t".to_string()),
                Token::Keyword(Keyword::Where),
                Token::Ident("x".to_string()),
                Token::Eq,
                Token::Integer(1),
            ]
        );
    }

    #[test]
    fn test_integer_literal() {
        let tokens = lex_all("42 0 999999").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Integer(42),
                Token::Integer(0),
                Token::Integer(999999)
            ]
        );
    }

    #[test]
    fn test_float_literal() {
        let tokens = lex_all("3.14 0.5 1.0e10 2.5E-3").unwrap();
        assert_eq!(tokens.len(), 4);
        assert!(matches!(tokens[0], Token::Float(_)));
        assert!(matches!(tokens[1], Token::Float(_)));
        assert!(matches!(tokens[2], Token::Float(_)));
        assert!(matches!(tokens[3], Token::Float(_)));
    }

    #[test]
    fn test_string_literal() {
        let tokens = lex_all("'hello' 'world'").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::String("hello".to_string()),
                Token::String("world".to_string()),
            ]
        );
    }

    #[test]
    fn test_string_escape() {
        let tokens = lex_all("'it''s a test'").unwrap();
        assert_eq!(tokens, vec![Token::String("it's a test".to_string())]);
    }

    #[test]
    fn test_quoted_identifier() {
        let tokens = lex_all("\"MyTable\" \"column name\"").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("MyTable".to_string()),
                Token::Ident("column name".to_string()),
            ]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = lex_all("+ - * / % = <> != < > <= >= || ::").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Percent,
                Token::Eq,
                Token::Neq,
                Token::Neq,
                Token::Lt,
                Token::Gt,
                Token::LtEq,
                Token::GtEq,
                Token::Concat,
                Token::DoubleColon,
            ]
        );
    }

    #[test]
    fn test_punctuation() {
        let tokens = lex_all(", ; ( ) .").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Comma,
                Token::Semicolon,
                Token::LParen,
                Token::RParen,
                Token::Dot,
            ]
        );
    }

    #[test]
    fn test_single_line_comment() {
        let tokens = lex_all("SELECT -- this is a comment\na").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Ident("a".to_string()),
            ]
        );
    }

    #[test]
    fn test_block_comment() {
        let tokens = lex_all("SELECT /* block comment */ a").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Ident("a".to_string()),
            ]
        );
    }

    #[test]
    fn test_nested_block_comment() {
        let tokens = lex_all("SELECT /* outer /* inner */ still comment */ a").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Ident("a".to_string()),
            ]
        );
    }

    #[test]
    fn test_unterminated_string() {
        let result = lex_all("'unterminated");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unterminated string"));
    }

    #[test]
    fn test_unterminated_block_comment() {
        let result = lex_all("/* unterminated");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unterminated block comment"));
    }

    #[test]
    fn test_unexpected_character() {
        let result = lex_all("SELECT @invalid");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unexpected character"));
    }

    #[test]
    fn test_peek_token() {
        let mut lexer = Lexer::new("SELECT a");
        let peeked = lexer.peek_token().unwrap().clone();
        assert_eq!(peeked.token, Token::Keyword(Keyword::Select));

        // Peeking again returns the same token
        let peeked2 = lexer.peek_token().unwrap().clone();
        assert_eq!(peeked, peeked2);

        // Consuming returns the peeked token
        let consumed = lexer.next_token().unwrap();
        assert_eq!(consumed.token, Token::Keyword(Keyword::Select));

        // Next token is 'a'
        let next = lexer.next_token().unwrap();
        assert_eq!(next.token, Token::Ident("a".to_string()));
    }

    #[test]
    fn test_eof() {
        let mut lexer = Lexer::new("");
        let tok = lexer.next_token().unwrap();
        assert_eq!(tok.token, Token::Eof);

        // Multiple EOF reads are safe
        let tok2 = lexer.next_token().unwrap();
        assert_eq!(tok2.token, Token::Eof);
    }

    #[test]
    fn test_keyword_case_insensitive() {
        let tokens = lex_all("select FROM where").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Select),
                Token::Keyword(Keyword::From),
                Token::Keyword(Keyword::Where),
            ]
        );
    }

    #[test]
    fn test_identifiers_case_preserved() {
        let tokens = lex_all("myTable MyColumn").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("myTable".to_string()),
                Token::Ident("MyColumn".to_string()),
            ]
        );
    }

    #[test]
    fn test_span_tracking() {
        let mut lexer = Lexer::new("SELECT a");
        let select = lexer.next_token().unwrap();
        assert_eq!(select.span, Span::new(0, 6));
        let a = lexer.next_token().unwrap();
        assert_eq!(a.span, Span::new(7, 1));
    }

    #[test]
    fn test_transaction_keywords() {
        let tokens = lex_all("BEGIN TRANSACTION COMMIT ROLLBACK SAVEPOINT RELEASE").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Begin),
                Token::Keyword(Keyword::Transaction),
                Token::Keyword(Keyword::Commit),
                Token::Keyword(Keyword::Rollback),
                Token::Keyword(Keyword::Savepoint),
                Token::Keyword(Keyword::Release),
            ]
        );
    }

    #[test]
    fn test_explain_analyze() {
        let tokens = lex_all("EXPLAIN ANALYZE SELECT 1").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Explain),
                Token::Keyword(Keyword::Analyze),
                Token::Keyword(Keyword::Select),
                Token::Integer(1),
            ]
        );
    }

    #[test]
    fn test_alter_table_keywords() {
        let tokens = lex_all("ALTER TABLE users ADD COLUMN email VARCHAR").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Keyword(Keyword::Alter),
                Token::Keyword(Keyword::Table),
                Token::Ident("users".to_string()),
                Token::Keyword(Keyword::Add),
                Token::Keyword(Keyword::Column),
                Token::Ident("email".to_string()),
                Token::Keyword(Keyword::Varchar),
            ]
        );
    }

    #[test]
    fn test_dot_not_float() {
        // A dot not followed by a digit is punctuation, not a float
        let tokens = lex_all("t.col").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Ident("t".to_string()),
                Token::Dot,
                Token::Ident("col".to_string()),
            ]
        );
    }

    #[test]
    fn test_integer_followed_by_dot_ident() {
        // "1.col" should be integer 1, dot, ident "col" (dot not followed by digit)
        let tokens = lex_all("1 .col").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Integer(1),
                Token::Dot,
                Token::Ident("col".to_string()),
            ]
        );
    }
}
