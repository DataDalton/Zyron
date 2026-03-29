//! Recursive descent SQL parser with Pratt expression parsing.

use zyron_common::{Result, ZyronError};

use crate::ast::*;
use crate::lexer::Lexer;
use crate::token::{Keyword, SpannedToken, Token};

/// SQL parser. Converts a token stream into an AST.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: SpannedToken,
    peek: SpannedToken,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given SQL input.
    pub fn new(input: &'a str) -> Result<Self> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        let peek = lexer.next_token()?;
        Ok(Self {
            lexer,
            current,
            peek,
        })
    }

    /// Parses a single SQL statement.
    pub fn parse_statement(&mut self) -> Result<Statement> {
        let stmt = match &self.current.token {
            Token::Keyword(Keyword::With) => self.parse_with_select(),
            Token::Keyword(Keyword::Select) => self.parse_select_statement(None),
            Token::Keyword(Keyword::Insert) => self.parse_insert(),
            Token::Keyword(Keyword::Update) => self.parse_update(),
            Token::Keyword(Keyword::Delete) => self.parse_delete(),
            Token::Keyword(Keyword::Create) => self.parse_create(),
            Token::Keyword(Keyword::Drop) => self.parse_drop(),
            Token::Keyword(Keyword::Alter) => self.parse_alter(),
            Token::Keyword(Keyword::Truncate) => self.parse_truncate(),
            Token::Keyword(Keyword::Begin) => self.parse_begin(),
            Token::Keyword(Keyword::Commit) => self.parse_commit(),
            Token::Keyword(Keyword::Rollback) => self.parse_rollback(),
            Token::Keyword(Keyword::Savepoint) => self.parse_savepoint(),
            Token::Keyword(Keyword::Release) => self.parse_release_savepoint(),
            Token::Keyword(Keyword::Explain) => self.parse_explain(),
            Token::Keyword(Keyword::Grant) => self.parse_grant(),
            Token::Keyword(Keyword::Revoke) => self.parse_revoke(),
            Token::Keyword(Keyword::Vacuum) => self.parse_vacuum(),
            Token::Keyword(Keyword::Reindex) => self.parse_reindex(),
            Token::Keyword(Keyword::Set) => self.parse_set_variable(),
            Token::Keyword(Keyword::Show) => self.parse_show(),
            Token::Keyword(Keyword::Copy) => self.parse_copy(),
            Token::Keyword(Keyword::Merge) => self.parse_merge(),
            Token::Keyword(Keyword::Prepare) => self.parse_prepare(),
            Token::Keyword(Keyword::Execute) => self.parse_execute(),
            Token::Keyword(Keyword::Deallocate) => self.parse_deallocate(),
            Token::Keyword(Keyword::Listen) => self.parse_listen(),
            Token::Keyword(Keyword::Notify) => self.parse_notify(),
            Token::Keyword(Keyword::Declare) => self.parse_declare_cursor(),
            Token::Keyword(Keyword::Fetch) => self.parse_fetch_cursor(),
            Token::Keyword(Keyword::Close) => self.parse_close_cursor(),
            Token::Keyword(Keyword::Comment) => self.parse_comment_on(),
            Token::Keyword(Keyword::Refresh) => self.parse_refresh_materialized_view(),
            Token::Keyword(Keyword::Do) => self.parse_do_block(),
            Token::Keyword(Keyword::Checkpoint) => self.parse_checkpoint(),
            Token::Keyword(Keyword::Values) => self.parse_values_query(),
            Token::Keyword(Keyword::Optimize) => self.parse_optimize_table(),
            Token::Keyword(Keyword::Pause) => self.parse_pause_schedule(),
            Token::Keyword(Keyword::Resume) => self.parse_resume_schedule(),
            Token::Keyword(Keyword::Run) => self.parse_run_pipeline(),
            Token::Keyword(Keyword::Archive) => self.parse_archive_table(),
            Token::Keyword(Keyword::Restore) => self.parse_restore_table(),
            Token::Keyword(Keyword::Analyze) => self.parse_analyze(),
            Token::Keyword(Keyword::Use) => self.parse_use_branch(),
            Token::Keyword(Keyword::Call) => self.parse_call(),
            _ => Err(self.error(&format!(
                "Expected a statement, found {}",
                self.current.token
            ))),
        }?;
        Ok(stmt)
    }

    /// Parses multiple semicolon-separated statements.
    pub fn parse_statements(&mut self) -> Result<Vec<Statement>> {
        let mut stmts = Vec::new();

        while self.current.token != Token::Eof {
            match self.parse_statement() {
                Ok(stmt) => stmts.push(stmt),
                Err(e) => {
                    // Attempt recovery: skip to next semicolon or statement keyword
                    if !self.recover_to_next_statement() {
                        return Err(e);
                    }
                    // If we recovered, continue parsing but report this error
                    // For now, return the error on first failure
                    return Err(e);
                }
            }

            // Consume optional trailing semicolons
            while self.current.token == Token::Semicolon {
                self.advance()?;
            }
        }

        Ok(stmts)
    }

    // -----------------------------------------------------------------------
    // Token management
    // -----------------------------------------------------------------------

    fn advance(&mut self) -> Result<SpannedToken> {
        let prev = std::mem::replace(
            &mut self.current,
            std::mem::replace(&mut self.peek, self.lexer.next_token()?),
        );
        Ok(prev)
    }

    fn expect_token(&mut self, expected: &Token) -> Result<SpannedToken> {
        if &self.current.token == expected {
            self.advance()
        } else {
            Err(self.error(&format!(
                "Expected {}, found {}",
                expected, self.current.token
            )))
        }
    }

    fn expect_keyword(&mut self, kw: Keyword) -> Result<SpannedToken> {
        if self.current.token == Token::Keyword(kw) {
            self.advance()
        } else {
            Err(self.error(&format!("Expected {}, found {}", kw, self.current.token)))
        }
    }

    fn consume_keyword(&mut self, kw: Keyword) -> Result<bool> {
        if self.current.token == Token::Keyword(kw) {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn at_keyword(&self, kw: Keyword) -> bool {
        self.current.token == Token::Keyword(kw)
    }

    fn at_token(&self, token: &Token) -> bool {
        &self.current.token == token
    }

    fn parse_ident(&mut self) -> Result<String> {
        match &self.current.token {
            Token::Ident(_) => {
                let tok = self.advance()?;
                if let Token::Ident(name) = tok.token {
                    Ok(name)
                } else {
                    unreachable!()
                }
            }
            // Allow keywords to be used as identifiers in certain contexts
            Token::Keyword(kw) => {
                let name = keyword_to_ident_str(*kw);
                if let Some(s) = name {
                    self.advance()?;
                    Ok(s.to_string())
                } else {
                    Err(self.error(&format!("Expected identifier, found keyword {}", kw)))
                }
            }
            _ => Err(self.error(&format!(
                "Expected identifier, found {}",
                self.current.token
            ))),
        }
    }

    fn parse_comma_separated<T>(
        &mut self,
        mut parse_fn: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let mut items = vec![parse_fn(self)?];
        while self.current.token == Token::Comma {
            self.advance()?;
            items.push(parse_fn(self)?);
        }
        Ok(items)
    }

    fn error(&self, msg: &str) -> ZyronError {
        ZyronError::ParseError(format!(
            "{} at line {}, column {}",
            msg,
            self.lexer.line(),
            self.lexer.column()
        ))
    }

    fn recover_to_next_statement(&mut self) -> bool {
        loop {
            match &self.current.token {
                Token::Eof => return false,
                Token::Semicolon => {
                    let _ = self.advance();
                    return self.current.token != Token::Eof;
                }
                Token::Keyword(
                    Keyword::Select
                    | Keyword::Insert
                    | Keyword::Update
                    | Keyword::Delete
                    | Keyword::Create
                    | Keyword::Drop
                    | Keyword::Alter
                    | Keyword::Truncate
                    | Keyword::Begin
                    | Keyword::Commit
                    | Keyword::Rollback
                    | Keyword::Explain
                    | Keyword::With
                    | Keyword::Grant
                    | Keyword::Revoke
                    | Keyword::Vacuum
                    | Keyword::Reindex
                    | Keyword::Set
                    | Keyword::Show
                    | Keyword::Copy
                    | Keyword::Merge
                    | Keyword::Prepare
                    | Keyword::Execute
                    | Keyword::Deallocate
                    | Keyword::Listen
                    | Keyword::Notify
                    | Keyword::Declare
                    | Keyword::Fetch
                    | Keyword::Close
                    | Keyword::Comment
                    | Keyword::Refresh
                    | Keyword::Do
                    | Keyword::Checkpoint
                    | Keyword::Values
                    | Keyword::Optimize
                    | Keyword::Pause
                    | Keyword::Resume
                    | Keyword::Run
                    | Keyword::Archive
                    | Keyword::Restore
                    | Keyword::Analyze,
                ) => return true,
                _ => {
                    if self.advance().is_err() {
                        return false;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // SELECT
    // -----------------------------------------------------------------------

    /// Parses WITH ... SELECT (CTE followed by SELECT).
    fn parse_with_select(&mut self) -> Result<Statement> {
        let with_clause = self.parse_with_clause()?;
        self.parse_select_statement(Some(with_clause))
    }

    /// Parses a WITH clause: WITH [RECURSIVE] name [(cols)] AS (select), ...
    fn parse_with_clause(&mut self) -> Result<WithClause> {
        self.expect_keyword(Keyword::With)?;
        let recursive = self.consume_keyword(Keyword::Recursive)?;

        let ctes = self.parse_comma_separated(|p| {
            let name = p.parse_ident()?;
            let columns = if p.at_token(&Token::LParen) {
                p.advance()?;
                let cols = p.parse_comma_separated(|p2| p2.parse_ident())?;
                p.expect_token(&Token::RParen)?;
                cols
            } else {
                vec![]
            };
            p.expect_keyword(Keyword::As)?;
            p.expect_token(&Token::LParen)?;
            let query = Box::new(p.parse_select_body(None)?);
            p.expect_token(&Token::RParen)?;
            Ok(Cte {
                name,
                columns,
                query,
            })
        })?;

        Ok(WithClause { recursive, ctes })
    }

    /// Parses SELECT ... as a statement (wraps in Statement::Select).
    fn parse_select_statement(&mut self, with: Option<WithClause>) -> Result<Statement> {
        let select = self.parse_select_body(with)?;
        Ok(Statement::Select(Box::new(select)))
    }

    /// Parses a SELECT body (used by both top-level SELECT and subqueries).
    /// Handles SELECT core, set operations (UNION/INTERSECT/EXCEPT), and ORDER BY/LIMIT/OFFSET.
    fn parse_select_body(&mut self, with: Option<WithClause>) -> Result<SelectStatement> {
        let mut stmt = self.parse_select_core(with)?;

        // Handle set operations: UNION/INTERSECT/EXCEPT [ALL] SELECT ...
        let mut set_ops = Vec::new();
        loop {
            let op = if self.at_keyword(Keyword::Union) {
                SetOpType::Union
            } else if self.at_keyword(Keyword::Intersect) {
                SetOpType::Intersect
            } else if self.at_keyword(Keyword::Except) {
                SetOpType::Except
            } else {
                break;
            };
            self.advance()?;
            let all = self.consume_keyword(Keyword::All)?;
            let right = self.parse_select_core(None)?;
            set_ops.push(SetOpItem {
                op,
                all,
                right: Box::new(right),
            });
        }
        stmt.set_ops = set_ops;

        // ORDER BY (applies to the entire set operation result)
        stmt.order_by = if self.at_keyword(Keyword::Order) {
            self.advance()?;
            self.expect_keyword(Keyword::By)?;
            self.parse_comma_separated(|p| p.parse_order_by_expr())?
        } else {
            vec![]
        };

        stmt.limit = if self.consume_keyword(Keyword::Limit)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        stmt.offset = if self.consume_keyword(Keyword::Offset)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        // FETCH FIRST/NEXT n [PERCENT] ROWS ONLY | WITH TIES
        stmt.fetch = if self.at_keyword(Keyword::Fetch) {
            self.advance()?;
            if !self.consume_keyword(Keyword::First)? {
                self.consume_keyword(Keyword::Next)?;
            }
            let count = Box::new(self.parse_expr()?);
            let percent = self.consume_keyword(Keyword::Percent)?;
            if !self.consume_keyword(Keyword::Rows)? {
                self.consume_keyword(Keyword::Row)?;
            }
            let with_ties = if self.at_keyword(Keyword::With) {
                self.advance()?;
                let ident = self.parse_ident()?;
                if ident.to_ascii_uppercase() != "TIES" {
                    return Err(self.error("Expected TIES after WITH"));
                }
                true
            } else {
                self.expect_keyword(Keyword::Only)?;
                false
            };
            Some(FetchFirst {
                count,
                percent,
                with_ties,
            })
        } else {
            None
        };

        // FOR UPDATE/SHARE/NO KEY UPDATE/KEY SHARE
        stmt.for_clause = if self.at_keyword(Keyword::For) {
            self.advance()?;
            let lock_type = if self.consume_keyword(Keyword::Update)? {
                ForLockType::Update
            } else if self.consume_keyword(Keyword::Share)? {
                ForLockType::Share
            } else if self.at_keyword(Keyword::No) {
                self.advance()?;
                self.expect_keyword(Keyword::Key)?;
                self.expect_keyword(Keyword::Update)?;
                ForLockType::NoKeyUpdate
            } else {
                self.expect_keyword(Keyword::Key)?;
                self.expect_keyword(Keyword::Share)?;
                ForLockType::KeyShare
            };
            let tables = Vec::new();
            let wait = if self.consume_keyword(Keyword::Nowait)? {
                ForWait::Nowait
            } else if self.at_keyword(Keyword::Skip) {
                self.advance()?;
                self.expect_keyword(Keyword::Locked)?;
                ForWait::SkipLocked
            } else {
                ForWait::Wait
            };
            Some(ForClause {
                lock_type,
                tables,
                wait,
            })
        } else {
            None
        };

        Ok(stmt)
    }

    /// Parses SELECT through HAVING (no ORDER BY/LIMIT/OFFSET or set operations).
    fn parse_select_core(&mut self, with: Option<WithClause>) -> Result<SelectStatement> {
        self.expect_keyword(Keyword::Select)?;

        let distinct = self.consume_keyword(Keyword::Distinct)?;
        let mut distinct_on = Vec::new();
        if distinct && self.at_keyword(Keyword::On) {
            self.advance()?;
            self.expect_token(&Token::LParen)?;
            distinct_on = self.parse_comma_separated(|p| p.parse_expr())?;
            self.expect_token(&Token::RParen)?;
        }
        if !distinct {
            self.consume_keyword(Keyword::All)?;
        }

        let projections = self.parse_comma_separated(|p| p.parse_select_item())?;

        // FROM clause (optional for expressions like SELECT 1)
        let from = if self.consume_keyword(Keyword::From)? {
            self.parse_comma_separated(|p| p.parse_table_ref())?
        } else {
            vec![]
        };

        let where_clause = if self.consume_keyword(Keyword::Where)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let mut group_by = vec![];
        let mut group_by_sets = None;
        if self.at_keyword(Keyword::Group) {
            self.advance()?;
            self.expect_keyword(Keyword::By)?;

            if self.at_keyword(Keyword::Rollup) {
                self.advance()?;
                self.expect_token(&Token::LParen)?;
                let exprs = self.parse_comma_separated(|p| p.parse_expr())?;
                self.expect_token(&Token::RParen)?;
                group_by_sets = Some(GroupBySets::Rollup(exprs));
            } else if self.at_keyword(Keyword::Cube) {
                self.advance()?;
                self.expect_token(&Token::LParen)?;
                let exprs = self.parse_comma_separated(|p| p.parse_expr())?;
                self.expect_token(&Token::RParen)?;
                group_by_sets = Some(GroupBySets::Cube(exprs));
            } else if self.at_keyword(Keyword::Grouping) {
                self.advance()?;
                self.expect_keyword(Keyword::Sets)?;
                self.expect_token(&Token::LParen)?;
                let sets = self.parse_comma_separated(|p| {
                    p.expect_token(&Token::LParen)?;
                    let exprs = if p.at_token(&Token::RParen) {
                        vec![]
                    } else {
                        p.parse_comma_separated(|p2| p2.parse_expr())?
                    };
                    p.expect_token(&Token::RParen)?;
                    Ok(exprs)
                })?;
                self.expect_token(&Token::RParen)?;
                group_by_sets = Some(GroupBySets::GroupingSets(sets));
            } else {
                group_by = self.parse_comma_separated(|p| p.parse_expr())?;
            }
        }

        let having = if self.consume_keyword(Keyword::Having)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let qualify = if self.consume_keyword(Keyword::Qualify)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(SelectStatement {
            with,
            distinct,
            distinct_on,
            projections,
            from,
            where_clause,
            group_by,
            group_by_sets,
            having,
            qualify,
            set_ops: vec![],
            order_by: vec![],
            limit: None,
            offset: None,
            fetch: None,
            for_clause: None,
        })
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        // Check for unqualified wildcard
        if self.at_token(&Token::Star) {
            self.advance()?;
            return Ok(SelectItem::Wildcard);
        }

        // Check for qualified wildcard: ident.*
        if let Token::Ident(ref name) = self.current.token {
            if self.peek.token == Token::Dot {
                let name = name.clone();
                // Need to look further: is it ident.* or ident.column?
                // Save state and check
                let saved_name = name.clone();
                self.advance()?; // consume ident
                self.advance()?; // consume dot
                if self.at_token(&Token::Star) {
                    self.advance()?; // consume *
                    return Ok(SelectItem::QualifiedWildcard(saved_name));
                }
                // Not a wildcard, it is ident.column. Re-construct as QualifiedIdentifier expr.
                let column = self.parse_ident()?;
                let expr = Expr::QualifiedIdentifier {
                    table: saved_name,
                    column,
                };
                let alias = self.parse_optional_alias()?;
                return Ok(SelectItem::Expr(expr, alias));
            }
        }

        let expr = self.parse_expr()?;
        let alias = self.parse_optional_alias()?;
        Ok(SelectItem::Expr(expr, alias))
    }

    fn parse_optional_alias(&mut self) -> Result<Option<String>> {
        if self.consume_keyword(Keyword::As)? {
            Ok(Some(self.parse_ident()?))
        } else if let Token::Ident(_) = &self.current.token {
            // Implicit alias (identifier without AS)
            Ok(Some(self.parse_ident()?))
        } else {
            Ok(None)
        }
    }

    // -----------------------------------------------------------------------
    // Table references and JOINs
    // -----------------------------------------------------------------------

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let mut left = self.parse_base_table_ref()?;

        // Loop to handle chained JOINs
        loop {
            // Check for NATURAL prefix
            let natural = if self.at_keyword(Keyword::Natural) {
                self.advance()?;
                true
            } else {
                false
            };

            let join_type = match &self.current.token {
                Token::Keyword(Keyword::Inner) => {
                    self.advance()?;
                    self.expect_keyword(Keyword::Join)?;
                    Some(JoinType::Inner)
                }
                Token::Keyword(Keyword::Left) => {
                    self.advance()?;
                    self.consume_keyword(Keyword::Outer)?;
                    self.expect_keyword(Keyword::Join)?;
                    Some(JoinType::Left)
                }
                Token::Keyword(Keyword::Right) => {
                    self.advance()?;
                    self.consume_keyword(Keyword::Outer)?;
                    self.expect_keyword(Keyword::Join)?;
                    Some(JoinType::Right)
                }
                Token::Keyword(Keyword::Full) => {
                    self.advance()?;
                    self.consume_keyword(Keyword::Outer)?;
                    self.expect_keyword(Keyword::Join)?;
                    Some(JoinType::Full)
                }
                Token::Keyword(Keyword::Cross) => {
                    self.advance()?;
                    self.expect_keyword(Keyword::Join)?;
                    Some(JoinType::Cross)
                }
                Token::Keyword(Keyword::Join) => {
                    self.advance()?;
                    Some(JoinType::Inner)
                }
                _ => None,
            };

            // If NATURAL was specified but no join keyword followed, treat as NATURAL JOIN (inner)
            let jt = match join_type {
                Some(jt) => jt,
                None => {
                    if natural {
                        JoinType::Inner
                    } else {
                        break;
                    }
                }
            };

            let right = self.parse_base_table_ref()?;
            let condition = if natural {
                JoinCondition::Natural
            } else if jt == JoinType::Cross {
                JoinCondition::None
            } else if self.consume_keyword(Keyword::Using)? {
                self.expect_token(&Token::LParen)?;
                let cols = self.parse_comma_separated(|p| p.parse_ident())?;
                self.expect_token(&Token::RParen)?;
                JoinCondition::Using(cols)
            } else {
                self.expect_keyword(Keyword::On)?;
                JoinCondition::On(Box::new(self.parse_expr()?))
            };
            left = TableRef::Join(Box::new(JoinTableRef {
                left,
                join_type: jt,
                right,
                condition,
            }));
        }

        Ok(left)
    }

    fn parse_base_table_ref(&mut self) -> Result<TableRef> {
        // LATERAL subquery or table function
        if self.at_keyword(Keyword::Lateral) {
            self.advance()?;
            let inner = self.parse_base_table_ref()?;
            return Ok(TableRef::Lateral {
                subquery: Box::new(inner),
            });
        }

        // Subquery in FROM: (SELECT ...) [AS] alias
        if self.at_token(&Token::LParen) {
            self.advance()?;
            if self.at_keyword(Keyword::Select) || self.at_keyword(Keyword::With) {
                let query = if self.at_keyword(Keyword::With) {
                    let with_clause = self.parse_with_clause()?;
                    self.parse_select_body(Some(with_clause))?
                } else {
                    self.parse_select_body(None)?
                };
                self.expect_token(&Token::RParen)?;
                self.consume_keyword(Keyword::As)?;
                let alias = self.parse_ident()?;
                return Ok(TableRef::Subquery {
                    query: Box::new(query),
                    alias,
                });
            }
            // Not a subquery, error (parenthesized table refs not supported)
            return Err(self.error("Expected SELECT after '(' in FROM clause"));
        }

        let name = self.parse_ident()?;

        // Check for time travel: AS OF TIMESTAMP expr, VERSION AS OF expr,
        // FOR SYSTEM_TIME BETWEEN, FOR APPLICATION_TIME, FOR PORTION OF
        let as_of = if self.at_keyword(Keyword::Version)
            && self.peek.token == Token::Keyword(Keyword::As)
        {
            self.advance()?; // VERSION
            self.advance()?; // AS
            self.expect_keyword(Keyword::Of)?;
            let expr = self.parse_expr()?;
            Some(Box::new(AsOf::Version(expr)))
        } else if self.at_keyword(Keyword::As) && self.peek.token == Token::Keyword(Keyword::Of) {
            self.advance()?; // AS
            self.advance()?; // OF
            // Optionally consume TIMESTAMP or TIMESTAMPTZ keyword
            let _ = self.consume_keyword(Keyword::Timestamp)?;
            let _ = self.consume_keyword(Keyword::Timestamptz)?;
            let expr = self.parse_expr()?;
            Some(Box::new(AsOf::Timestamp(expr)))
        } else if self.at_keyword(Keyword::For)
            && (self.peek.token == Token::Keyword(Keyword::System)
                || self.peek.token == Token::Keyword(Keyword::Versioning))
        {
            self.advance()?; // FOR
            self.advance()?; // SYSTEM or VERSIONING
            self.expect_keyword(Keyword::Between)?;
            let start = self.parse_expr()?;
            self.expect_keyword(Keyword::And)?;
            let end = self.parse_expr()?;
            Some(Box::new(AsOf::SystemTime { start, end }))
        } else if self.at_keyword(Keyword::For)
            && self.peek.token == Token::Keyword(Keyword::Portion)
        {
            // FOR PORTION OF period FROM start TO end
            self.advance()?; // FOR
            self.advance()?; // PORTION
            self.expect_keyword(Keyword::Of)?;
            let period = self.parse_ident()?;
            self.expect_keyword(Keyword::From)?;
            let start = self.parse_expr()?;
            self.expect_keyword(Keyword::To)?;
            let end = self.parse_expr()?;
            Some(Box::new(AsOf::ForPortionOf { period, start, end }))
        } else if self.at_keyword(Keyword::For)
            && matches!(&self.peek.token, Token::Ident(s) if s.eq_ignore_ascii_case("application_time"))
        {
            self.advance()?; // FOR
            self.advance()?; // APPLICATION_TIME (ident)
            if self.consume_keyword(Keyword::Between)? {
                let start = self.parse_expr()?;
                self.expect_keyword(Keyword::And)?;
                let end = self.parse_expr()?;
                Some(Box::new(AsOf::ApplicationTime { start, end }))
            } else {
                // AS OF expr
                self.expect_keyword(Keyword::As)?;
                self.expect_keyword(Keyword::Of)?;
                let expr = self.parse_expr()?;
                let cloned = expr.clone();
                Some(Box::new(AsOf::ApplicationTime {
                    start: expr,
                    end: cloned,
                }))
            }
        } else {
            None
        };

        let alias = if self.consume_keyword(Keyword::As)? {
            Some(self.parse_ident()?)
        } else if let Token::Ident(_) = &self.current.token {
            // Check that this is not a keyword that starts a clause
            if !self.is_clause_keyword() {
                Some(self.parse_ident()?)
            } else {
                None
            }
        } else {
            None
        };
        Ok(TableRef::Table { name, alias, as_of })
    }

    fn is_clause_keyword(&self) -> bool {
        matches!(
            &self.current.token,
            Token::Keyword(
                Keyword::Where
                    | Keyword::Group
                    | Keyword::Having
                    | Keyword::Order
                    | Keyword::Limit
                    | Keyword::Offset
                    | Keyword::Inner
                    | Keyword::Left
                    | Keyword::Right
                    | Keyword::Full
                    | Keyword::Cross
                    | Keyword::Join
                    | Keyword::Natural
                    | Keyword::On
                    | Keyword::Using
                    | Keyword::Set
                    | Keyword::Union
                    | Keyword::Intersect
                    | Keyword::Except
                    | Keyword::Returning
                    | Keyword::For
                    | Keyword::Qualify
                    | Keyword::Version
            )
        )
    }

    // -----------------------------------------------------------------------
    // INSERT
    // -----------------------------------------------------------------------

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Insert)?;
        self.expect_keyword(Keyword::Into)?;
        let table = self.parse_ident()?;

        // Optional column list
        let columns = if self.at_token(&Token::LParen) {
            self.advance()?;
            let cols = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            cols
        } else {
            vec![]
        };

        // INSERT INTO ... SELECT or INSERT INTO ... VALUES
        let source = if self.at_keyword(Keyword::Select) || self.at_keyword(Keyword::With) {
            let query = if self.at_keyword(Keyword::With) {
                let with_clause = self.parse_with_clause()?;
                self.parse_select_body(Some(with_clause))?
            } else {
                self.parse_select_body(None)?
            };
            InsertSource::Query(Box::new(query))
        } else {
            self.expect_keyword(Keyword::Values)?;
            let values = self.parse_comma_separated(|p| {
                p.expect_token(&Token::LParen)?;
                let exprs = p.parse_comma_separated(|p2| p2.parse_expr())?;
                p.expect_token(&Token::RParen)?;
                Ok(exprs)
            })?;
            InsertSource::Values(values)
        };

        // ON CONFLICT
        let on_conflict = if self.at_keyword(Keyword::On)
            && self.peek.token == Token::Keyword(Keyword::Conflict)
        {
            self.advance()?; // ON
            self.advance()?; // CONFLICT
            let columns = if self.at_token(&Token::LParen) {
                self.advance()?;
                let cols = self.parse_comma_separated(|p| p.parse_ident())?;
                self.expect_token(&Token::RParen)?;
                cols
            } else {
                vec![]
            };
            self.expect_keyword(Keyword::Do)?;
            let action = if self.consume_keyword(Keyword::Nothing)? {
                ConflictAction::DoNothing
            } else {
                self.expect_keyword(Keyword::Update)?;
                self.expect_keyword(Keyword::Set)?;
                let assignments = self.parse_comma_separated(|p| p.parse_assignment())?;
                ConflictAction::DoUpdate(assignments)
            };
            Some(OnConflict { columns, action })
        } else {
            None
        };
        let returning = self.parse_returning()?;

        Ok(Statement::Insert(Box::new(InsertStatement {
            table,
            columns,
            source,
            on_conflict,
            returning,
        })))
    }

    // -----------------------------------------------------------------------
    // UPDATE
    // -----------------------------------------------------------------------

    fn parse_update(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Update)?;
        let table = self.parse_ident()?;
        self.expect_keyword(Keyword::Set)?;

        let assignments = self.parse_comma_separated(|p| p.parse_assignment())?;

        let where_clause = if self.consume_keyword(Keyword::Where)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let returning = self.parse_returning()?;

        Ok(Statement::Update(Box::new(UpdateStatement {
            table,
            assignments,
            where_clause,
            returning,
        })))
    }

    fn parse_assignment(&mut self) -> Result<Assignment> {
        let column = self.parse_ident()?;
        self.expect_token(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Assignment { column, value })
    }

    // -----------------------------------------------------------------------
    // DELETE
    // -----------------------------------------------------------------------

    fn parse_delete(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Delete)?;
        self.expect_keyword(Keyword::From)?;
        let table = self.parse_ident()?;

        let where_clause = if self.consume_keyword(Keyword::Where)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let returning = self.parse_returning()?;

        Ok(Statement::Delete(Box::new(DeleteStatement {
            table,
            where_clause,
            returning,
        })))
    }

    // -----------------------------------------------------------------------
    // CREATE
    // -----------------------------------------------------------------------

    fn parse_create(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Create)?;

        // CREATE UNIQUE INDEX
        if self.at_keyword(Keyword::Unique) {
            self.advance()?;
            return self.parse_create_index(true);
        }

        // CREATE OR REPLACE VIEW/FUNCTION/PROCEDURE
        if self.at_keyword(Keyword::Or) {
            self.advance()?;
            self.expect_keyword(Keyword::Replace)?;
            return match &self.current.token {
                Token::Keyword(Keyword::View) => self.parse_create_view(true),
                Token::Keyword(Keyword::Function) => self.parse_create_function(true),
                Token::Keyword(Keyword::Procedure) => self.parse_create_procedure(true),
                _ => Err(self.error(&format!(
                    "Expected VIEW, FUNCTION, or PROCEDURE after CREATE OR REPLACE, found {}",
                    self.current.token
                ))),
            };
        }

        match &self.current.token {
            Token::Keyword(Keyword::Table) => self.parse_create_table(),
            Token::Keyword(Keyword::Index) => self.parse_create_index(false),
            Token::Keyword(Keyword::View) => self.parse_create_view(false),
            Token::Keyword(Keyword::Schema) => self.parse_create_schema(),
            Token::Keyword(Keyword::Sequence) => self.parse_create_sequence(),
            Token::Keyword(Keyword::Materialized) => self.parse_create_materialized_view(),
            Token::Keyword(Keyword::Schedule) => self.parse_create_schedule(),
            Token::Keyword(Keyword::User) => self.parse_create_user(),
            Token::Keyword(Keyword::Role) => self.parse_create_role(),
            Token::Keyword(Keyword::Pipeline) => self.parse_create_pipeline(),
            Token::Keyword(Keyword::Fulltext) => self.parse_create_fulltext_index(),
            Token::Keyword(Keyword::Vector) => self.parse_create_vector_index(),
            Token::Keyword(Keyword::Branch) => self.parse_create_branch(),
            Token::Keyword(Keyword::Version) => self.parse_create_version(),
            Token::Keyword(Keyword::Replication) => self.parse_create_replication_slot(),
            Token::Keyword(Keyword::Cdc) => self.parse_create_cdc(),
            Token::Keyword(Keyword::Publication) => self.parse_create_publication(),
            Token::Keyword(Keyword::Trigger) => self.parse_create_trigger(),
            Token::Keyword(Keyword::Function) => self.parse_create_function(false),
            Token::Keyword(Keyword::Aggregate) => self.parse_create_aggregate(),
            Token::Keyword(Keyword::Procedure) => self.parse_create_procedure(false),
            Token::Keyword(Keyword::Event) => self.parse_create_event_handler(),
            _ => Err(self.error(&format!(
                "Expected TABLE, INDEX, VIEW, SCHEMA, SEQUENCE, MATERIALIZED, SCHEDULE, USER, ROLE, PIPELINE, FULLTEXT, VECTOR, BRANCH, VERSION, REPLICATION, CDC, PUBLICATION, TRIGGER, FUNCTION, AGGREGATE, PROCEDURE, or EVENT after CREATE, found {}",
                self.current.token
            ))),
        }
    }

    fn parse_create_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Table)?;

        let if_not_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let name = self.parse_ident()?;
        self.expect_token(&Token::LParen)?;

        let mut columns = Vec::new();
        let mut constraints = Vec::new();

        loop {
            // Check for table-level constraints
            if self.at_keyword(Keyword::Primary)
                || self.at_keyword(Keyword::Unique)
                || self.at_keyword(Keyword::Check)
                || self.at_keyword(Keyword::Foreign)
                || self.at_keyword(Keyword::Constraint)
            {
                constraints.push(self.parse_table_constraint()?);
            } else if self.at_token(&Token::RParen) {
                break;
            } else {
                columns.push(self.parse_column_def()?);
            }

            if !self.at_token(&Token::RParen) {
                self.expect_token(&Token::Comma)?;
            }
        }

        self.expect_token(&Token::RParen)?;

        // Parse WITH (options) if present
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            self.expect_token(&Token::LParen)?;
            options = self.parse_comma_separated(|p| p.parse_table_option())?;
            self.expect_token(&Token::RParen)?;
        }

        Ok(Statement::CreateTable(Box::new(CreateTableStatement {
            name,
            if_not_exists,
            columns,
            constraints,
            options,
        })))
    }

    fn parse_create_index(&mut self, unique: bool) -> Result<Statement> {
        self.expect_keyword(Keyword::Index)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table = self.parse_ident()?;
        self.expect_token(&Token::LParen)?;
        let columns = self.parse_comma_separated(|p| p.parse_order_by_expr())?;
        self.expect_token(&Token::RParen)?;

        Ok(Statement::CreateIndex(Box::new(CreateIndexStatement {
            name,
            table,
            columns,
            unique,
        })))
    }

    // -----------------------------------------------------------------------
    // DROP
    // -----------------------------------------------------------------------

    fn parse_drop(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Drop)?;

        match &self.current.token {
            Token::Keyword(Keyword::Table) => self.parse_drop_table(),
            Token::Keyword(Keyword::Index) => self.parse_drop_index(),
            Token::Keyword(Keyword::View) => self.parse_drop_view(),
            Token::Keyword(Keyword::Schema) => self.parse_drop_schema(),
            Token::Keyword(Keyword::Sequence) => self.parse_drop_sequence(),
            Token::Keyword(Keyword::Materialized) => self.parse_drop_materialized_view(),
            Token::Keyword(Keyword::Schedule) => self.parse_drop_schedule(),
            Token::Keyword(Keyword::User) => self.parse_drop_user(),
            Token::Keyword(Keyword::Role) => self.parse_drop_role(),
            Token::Keyword(Keyword::Pipeline) => self.parse_drop_pipeline(),
            Token::Keyword(Keyword::Branch) => self.parse_drop_branch(),
            Token::Keyword(Keyword::Replication) => self.parse_drop_replication_slot(),
            Token::Keyword(Keyword::Cdc) => self.parse_drop_cdc(),
            Token::Keyword(Keyword::Publication) => self.parse_drop_publication(),
            Token::Keyword(Keyword::Trigger) => self.parse_drop_trigger(),
            Token::Keyword(Keyword::Function) => self.parse_drop_function(),
            Token::Keyword(Keyword::Aggregate) => self.parse_drop_aggregate(),
            Token::Keyword(Keyword::Procedure) => self.parse_drop_procedure(),
            Token::Keyword(Keyword::Event) => self.parse_drop_event_handler(),
            _ => Err(self.error(&format!(
                "Expected TABLE, INDEX, VIEW, SCHEMA, SEQUENCE, MATERIALIZED, SCHEDULE, USER, ROLE, PIPELINE, BRANCH, REPLICATION, CDC, PUBLICATION, TRIGGER, FUNCTION, AGGREGATE, PROCEDURE, or EVENT after DROP, found {}",
                self.current.token
            ))),
        }
    }

    fn parse_drop_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Table)?;

        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let name = self.parse_ident()?;

        Ok(Statement::DropTable(Box::new(DropTableStatement {
            name,
            if_exists,
        })))
    }

    fn parse_drop_index(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Index)?;

        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };

        let name = self.parse_ident()?;

        Ok(Statement::DropIndex(Box::new(DropIndexStatement {
            name,
            if_exists,
        })))
    }

    // -----------------------------------------------------------------------
    // ALTER TABLE
    // -----------------------------------------------------------------------

    fn parse_alter(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Alter)?;
        match &self.current.token {
            Token::Keyword(Keyword::Table) => self.parse_alter_table(),
            Token::Keyword(Keyword::Index) => self.parse_alter_index(),
            Token::Keyword(Keyword::Sequence) => self.parse_alter_sequence(),
            Token::Keyword(Keyword::View) => self.parse_alter_view(),
            Token::Keyword(Keyword::User) => self.parse_alter_user(),
            Token::Keyword(Keyword::Role) => self.parse_alter_role(),
            Token::Keyword(Keyword::System) => self.parse_alter_system(),
            Token::Keyword(Keyword::Publication) => self.parse_alter_publication(),
            _ => Err(self.error(&format!(
                "Expected TABLE, INDEX, SEQUENCE, VIEW, USER, ROLE, SYSTEM, or PUBLICATION after ALTER, found {}",
                self.current.token
            ))),
        }
    }

    fn parse_alter_system(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::System)?;
        self.expect_keyword(Keyword::Set)?;
        let name = self.parse_ident()?;
        self.expect_token(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Statement::AlterSystemSet(Box::new(
            AlterSystemSetStatement { name, value },
        )))
    }

    fn parse_alter_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Table)?;
        let name = self.parse_ident()?;

        // Handle SET TTL and DROP TTL before the standard ALTER TABLE operations
        if self.at_keyword(Keyword::Set) && self.peek.token == Token::Keyword(Keyword::Ttl) {
            self.advance()?; // SET
            self.advance()?; // TTL
            let action = if self.consume_keyword(Keyword::Archive)? {
                TtlAction::Archive
            } else {
                TtlAction::Delete
            };
            let duration = self.parse_ttl_duration()?;
            self.expect_keyword(Keyword::On)?;
            let column = self.parse_ident()?;
            return Ok(Statement::AlterTableTtl(Box::new(AlterTableTtlStatement {
                table: name,
                operation: TtlOperation::Set {
                    duration,
                    column,
                    action,
                },
            })));
        }

        if self.at_keyword(Keyword::Drop) && self.peek.token == Token::Keyword(Keyword::Ttl) {
            self.advance()?; // DROP
            self.advance()?; // TTL
            return Ok(Statement::AlterTableTtl(Box::new(AlterTableTtlStatement {
                table: name,
                operation: TtlOperation::Drop,
            })));
        }

        // SET (key = value, ...) table options
        if self.at_keyword(Keyword::Set) && self.peek.token == Token::LParen {
            self.advance()?; // SET
            return self.parse_alter_table_options(name);
        }

        // ADD EXPECTATION
        if self.at_keyword(Keyword::Add) && self.peek.token == Token::Keyword(Keyword::Expectation)
        {
            self.advance()?; // ADD
            self.advance()?; // EXPECTATION
            return self.parse_add_expectation(name);
        }

        // DROP EXPECTATION
        if self.at_keyword(Keyword::Drop) && self.peek.token == Token::Keyword(Keyword::Expectation)
        {
            self.advance()?; // DROP
            self.advance()?; // EXPECTATION
            return self.parse_drop_expectation(name);
        }

        // ENABLE
        if self.at_keyword(Keyword::Enable) {
            self.advance()?; // ENABLE
            return self.parse_enable_feature(name);
        }

        // DISABLE
        if self.at_keyword(Keyword::Disable) {
            self.advance()?; // DISABLE
            return self.parse_disable_feature(name);
        }

        let operation = match &self.current.token {
            Token::Keyword(Keyword::Add) => {
                self.advance()?;
                if self.at_keyword(Keyword::Constraint)
                    || self.at_keyword(Keyword::Primary)
                    || self.at_keyword(Keyword::Unique)
                    || self.at_keyword(Keyword::Check)
                    || self.at_keyword(Keyword::Foreign)
                {
                    AlterTableOperation::AddConstraint(self.parse_table_constraint()?)
                } else {
                    self.consume_keyword(Keyword::Column)?;
                    AlterTableOperation::AddColumn(self.parse_column_def()?)
                }
            }
            Token::Keyword(Keyword::Drop) => {
                self.advance()?;
                if self.at_keyword(Keyword::Constraint) {
                    self.advance()?;
                    let if_exists = if self.consume_keyword(Keyword::If)? {
                        self.expect_keyword(Keyword::Exists)?;
                        true
                    } else {
                        false
                    };
                    let constraint_name = self.parse_ident()?;
                    AlterTableOperation::DropConstraint {
                        name: constraint_name,
                        if_exists,
                    }
                } else {
                    self.consume_keyword(Keyword::Column)?;
                    let if_exists = if self.consume_keyword(Keyword::If)? {
                        self.expect_keyword(Keyword::Exists)?;
                        true
                    } else {
                        false
                    };
                    let col_name = self.parse_ident()?;
                    AlterTableOperation::DropColumn {
                        name: col_name,
                        if_exists,
                    }
                }
            }
            Token::Keyword(Keyword::Rename) => {
                self.advance()?;
                if self.consume_keyword(Keyword::Column)? {
                    let old_name = self.parse_ident()?;
                    self.expect_keyword(Keyword::To)?;
                    let new_name = self.parse_ident()?;
                    AlterTableOperation::RenameColumn { old_name, new_name }
                } else if self.consume_keyword(Keyword::To)? {
                    let new_name = self.parse_ident()?;
                    AlterTableOperation::RenameTable { new_name }
                } else {
                    // RENAME old_name TO new_name (without COLUMN keyword)
                    let old_name = self.parse_ident()?;
                    self.expect_keyword(Keyword::To)?;
                    let new_name = self.parse_ident()?;
                    AlterTableOperation::RenameColumn { old_name, new_name }
                }
            }
            Token::Keyword(Keyword::Alter) => {
                self.advance()?;
                self.consume_keyword(Keyword::Column)?;
                let column = self.parse_ident()?;

                if self.consume_keyword(Keyword::Set)? {
                    if self.consume_keyword(Keyword::Not)? {
                        self.expect_keyword(Keyword::Null)?;
                        AlterTableOperation::AlterColumnSetNotNull { column }
                    } else if self.consume_keyword(Keyword::Default)? {
                        let default = self.parse_expr()?;
                        AlterTableOperation::AlterColumnSetDefault { column, default }
                    } else {
                        return Err(
                            self.error("Expected NOT NULL or DEFAULT after ALTER COLUMN ... SET")
                        );
                    }
                } else if self.consume_keyword(Keyword::Drop)? {
                    if self.consume_keyword(Keyword::Not)? {
                        self.expect_keyword(Keyword::Null)?;
                        AlterTableOperation::AlterColumnDropNotNull { column }
                    } else if self.consume_keyword(Keyword::Default)? {
                        AlterTableOperation::AlterColumnDropDefault { column }
                    } else {
                        return Err(
                            self.error("Expected NOT NULL or DEFAULT after ALTER COLUMN ... DROP")
                        );
                    }
                } else if self.consume_keyword(Keyword::Type)? {
                    let data_type = self.parse_data_type()?;
                    AlterTableOperation::AlterColumnSetType { column, data_type }
                } else {
                    return Err(self.error("Expected SET, DROP, or TYPE after ALTER COLUMN name"));
                }
            }
            _ => {
                return Err(self.error(&format!(
                    "Expected ADD, DROP, RENAME, or ALTER after ALTER TABLE name, found {}",
                    self.current.token
                )));
            }
        };

        Ok(Statement::AlterTable(Box::new(AlterTableStatement {
            name,
            operation,
        })))
    }

    // -----------------------------------------------------------------------
    // TRUNCATE
    // -----------------------------------------------------------------------

    fn parse_truncate(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Truncate)?;
        self.consume_keyword(Keyword::Table)?;
        let table = self.parse_ident()?;
        Ok(Statement::Truncate(Box::new(TruncateStatement { table })))
    }

    // -----------------------------------------------------------------------
    // Transaction control
    // -----------------------------------------------------------------------

    fn parse_begin(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Begin)?;
        self.consume_keyword(Keyword::Transaction)?;
        Ok(Statement::Begin(Box::new(BeginStatement {})))
    }

    fn parse_commit(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Commit)?;
        Ok(Statement::Commit(Box::new(CommitStatement {})))
    }

    fn parse_rollback(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Rollback)?;

        let savepoint = if self.consume_keyword(Keyword::To)? {
            self.consume_keyword(Keyword::Savepoint)?;
            Some(self.parse_ident()?)
        } else {
            None
        };

        Ok(Statement::Rollback(Box::new(RollbackStatement {
            savepoint,
        })))
    }

    fn parse_savepoint(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Savepoint)?;
        let name = self.parse_ident()?;
        Ok(Statement::Savepoint(Box::new(SavepointStatement { name })))
    }

    fn parse_release_savepoint(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Release)?;
        self.consume_keyword(Keyword::Savepoint)?;
        let name = self.parse_ident()?;
        Ok(Statement::ReleaseSavepoint(Box::new(
            ReleaseSavepointStatement { name },
        )))
    }

    // -----------------------------------------------------------------------
    // EXPLAIN
    // -----------------------------------------------------------------------

    fn parse_explain(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Explain)?;

        let mut analyze = false;
        let mut costs = true;
        let mut buffers = false;
        let mut timing = true;
        let mut format = None;

        // Check for parenthesized options: EXPLAIN (ANALYZE, COSTS, BUFFERS, TIMING, FORMAT TEXT)
        if self.at_token(&Token::LParen) {
            self.advance()?;
            loop {
                let option_name = self.parse_explain_option_name()?;
                match option_name.to_uppercase().as_str() {
                    "ANALYZE" => analyze = self.parse_explain_bool_value(true)?,
                    "COSTS" => costs = self.parse_explain_bool_value(true)?,
                    "BUFFERS" => buffers = self.parse_explain_bool_value(true)?,
                    "TIMING" => timing = self.parse_explain_bool_value(true)?,
                    "FORMAT" => {
                        let fmt_name = self.parse_explain_option_name()?;
                        format = Some(fmt_name);
                    }
                    _ => {
                        return Err(self.error(&format!("Unknown EXPLAIN option: {}", option_name)));
                    }
                }

                if !self.at_token(&Token::Comma) {
                    break;
                }
                self.advance()?;
            }
            self.expect_token(&Token::RParen)?;
        } else {
            // Legacy syntax: EXPLAIN [ANALYZE] <statement>
            analyze = self.consume_keyword(Keyword::Analyze)?;
        }

        let statement = self.parse_statement()?;
        Ok(Statement::Explain(Box::new(ExplainStatement {
            analyze,
            costs,
            buffers,
            timing,
            format,
            statement: Box::new(statement),
        })))
    }

    /// Parses an optional ON/OFF value for an EXPLAIN boolean option.
    /// If the next token is ON or OFF (as keyword or identifier), consumes it.
    /// Otherwise returns the default value.
    fn parse_explain_bool_value(&mut self, default: bool) -> Result<bool> {
        // Check for ON keyword
        if self.at_keyword(Keyword::On) {
            self.advance()?;
            return Ok(true);
        }
        // Check for OFF or TRUE/FALSE as identifiers
        match &self.current.token {
            Token::Ident(s) => {
                let upper = s.to_uppercase();
                if upper == "OFF" || upper == "FALSE" {
                    self.advance()?;
                    return Ok(false);
                }
                if upper == "TRUE" {
                    self.advance()?;
                    return Ok(true);
                }
            }
            _ => {}
        }
        Ok(default)
    }

    /// Parses an EXPLAIN option name. Handles both identifiers and keywords
    /// that can appear as option names (ANALYZE is a keyword).
    fn parse_explain_option_name(&mut self) -> Result<String> {
        match &self.current.token {
            Token::Ident(_) => {
                let tok = self.advance()?;
                if let Token::Ident(name) = tok.token {
                    Ok(name)
                } else {
                    Err(self.error("Expected option name"))
                }
            }
            Token::Keyword(kw) => {
                let name = format!("{}", kw);
                self.advance()?;
                Ok(name)
            }
            _ => Err(self.error("Expected EXPLAIN option name")),
        }
    }

    // -----------------------------------------------------------------------
    // Column and type definitions
    // -----------------------------------------------------------------------

    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.parse_ident()?;
        let data_type = self.parse_data_type()?;

        let mut nullable = None;
        let mut default = None;
        let mut constraints = Vec::new();

        // Parse inline column constraints
        loop {
            if self.consume_keyword(Keyword::Not)? {
                self.expect_keyword(Keyword::Null)?;
                nullable = Some(false);
                constraints.push(ColumnConstraint::NotNull);
            } else if self.at_keyword(Keyword::Null) {
                self.advance()?;
                nullable = Some(true);
            } else if self.at_keyword(Keyword::Primary) {
                self.advance()?;
                self.expect_keyword(Keyword::Key)?;
                constraints.push(ColumnConstraint::PrimaryKey);
            } else if self.at_keyword(Keyword::Unique) {
                self.advance()?;
                constraints.push(ColumnConstraint::Unique);
            } else if self.consume_keyword(Keyword::Default)? {
                let expr = self.parse_expr()?;
                default = Some(expr.clone());
                constraints.push(ColumnConstraint::Default(expr));
            } else if self.at_keyword(Keyword::Check) {
                self.advance()?;
                self.expect_token(&Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect_token(&Token::RParen)?;
                constraints.push(ColumnConstraint::Check(expr));
            } else if self.at_keyword(Keyword::References) {
                self.advance()?;
                let table = self.parse_ident()?;
                self.expect_token(&Token::LParen)?;
                let column = self.parse_ident()?;
                self.expect_token(&Token::RParen)?;
                constraints.push(ColumnConstraint::References { table, column });
            } else {
                break;
            }
        }

        Ok(ColumnDef {
            name,
            data_type,
            nullable,
            default,
            constraints,
        })
    }

    fn parse_table_constraint(&mut self) -> Result<TableConstraint> {
        // Optional CONSTRAINT name
        if self.consume_keyword(Keyword::Constraint)? {
            // Consume constraint name (we store it in the AST through the variants)
            self.parse_ident()?;
        }

        if self.at_keyword(Keyword::Primary) {
            self.advance()?;
            self.expect_keyword(Keyword::Key)?;
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::PrimaryKey(columns))
        } else if self.at_keyword(Keyword::Unique) {
            self.advance()?;
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::Unique(columns))
        } else if self.at_keyword(Keyword::Check) {
            self.advance()?;
            self.expect_token(&Token::LParen)?;
            let expr = self.parse_expr()?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::Check(expr))
        } else if self.at_keyword(Keyword::Foreign) {
            self.advance()?;
            self.expect_keyword(Keyword::Key)?;
            self.expect_token(&Token::LParen)?;
            let columns = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            self.expect_keyword(Keyword::References)?;
            let ref_table = self.parse_ident()?;
            self.expect_token(&Token::LParen)?;
            let ref_columns = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            Ok(TableConstraint::ForeignKey {
                columns,
                ref_table,
                ref_columns,
            })
        } else {
            Err(self.error(&format!(
                "Expected PRIMARY KEY, UNIQUE, CHECK, or FOREIGN KEY, found {}",
                self.current.token
            )))
        }
    }

    fn parse_data_type(&mut self) -> Result<DataType> {
        let base = self.parse_base_data_type()?;
        // Check for array suffix: type[]
        if self.at_token(&Token::LBracket) && self.peek.token == Token::RBracket {
            self.advance()?;
            self.advance()?;
            Ok(DataType::Array(Box::new(base)))
        } else {
            Ok(base)
        }
    }

    fn parse_base_data_type(&mut self) -> Result<DataType> {
        match &self.current.token {
            Token::Keyword(Keyword::Boolean) => {
                self.advance()?;
                Ok(DataType::Boolean)
            }
            Token::Keyword(Keyword::Smallint) => {
                self.advance()?;
                Ok(DataType::SmallInt)
            }
            Token::Keyword(Keyword::Int) | Token::Keyword(Keyword::Integer) => {
                self.advance()?;
                Ok(DataType::Int)
            }
            Token::Keyword(Keyword::Bigint) => {
                self.advance()?;
                Ok(DataType::BigInt)
            }
            Token::Keyword(Keyword::Real) => {
                self.advance()?;
                Ok(DataType::Real)
            }
            Token::Keyword(Keyword::Double) => {
                self.advance()?;
                self.expect_keyword(Keyword::Precision)?;
                Ok(DataType::DoublePrecision)
            }
            Token::Keyword(Keyword::Float) => {
                self.advance()?;
                let prec = self.parse_optional_type_param_u32()?;
                Ok(DataType::Float(prec))
            }
            Token::Keyword(Keyword::Decimal) => {
                self.advance()?;
                let (p, s) = self.parse_optional_precision_scale()?;
                Ok(DataType::Decimal(p, s))
            }
            Token::Keyword(Keyword::Numeric) => {
                self.advance()?;
                let (p, s) = self.parse_optional_precision_scale()?;
                Ok(DataType::Numeric(p, s))
            }
            Token::Keyword(Keyword::Char) => {
                self.advance()?;
                let len = self.parse_optional_type_param_usize()?;
                Ok(DataType::Char(len))
            }
            Token::Keyword(Keyword::Varchar) => {
                self.advance()?;
                let len = self.parse_optional_type_param_usize()?;
                Ok(DataType::Varchar(len))
            }
            Token::Keyword(Keyword::Text) => {
                self.advance()?;
                Ok(DataType::Text)
            }
            Token::Keyword(Keyword::Binary) => {
                self.advance()?;
                let len = self.parse_optional_type_param_usize()?;
                Ok(DataType::Binary(len))
            }
            Token::Keyword(Keyword::Varbinary) => {
                self.advance()?;
                let len = self.parse_optional_type_param_usize()?;
                Ok(DataType::Varbinary(len))
            }
            Token::Keyword(Keyword::Bytea) => {
                self.advance()?;
                Ok(DataType::Bytea)
            }
            Token::Keyword(Keyword::Date) => {
                self.advance()?;
                Ok(DataType::Date)
            }
            Token::Keyword(Keyword::Time) => {
                self.advance()?;
                Ok(DataType::Time)
            }
            Token::Keyword(Keyword::Timestamp) => {
                self.advance()?;
                // TIMESTAMP WITH TIME ZONE -> TimestampTz
                if self.at_keyword(Keyword::With) {
                    self.advance()?;
                    self.expect_keyword(Keyword::Time)?;
                    self.expect_keyword(Keyword::Zone)?;
                    Ok(DataType::TimestampTz)
                } else {
                    Ok(DataType::Timestamp)
                }
            }
            Token::Keyword(Keyword::Timestamptz) => {
                self.advance()?;
                Ok(DataType::TimestampTz)
            }
            Token::Keyword(Keyword::Interval) => {
                self.advance()?;
                Ok(DataType::Interval)
            }
            Token::Keyword(Keyword::Uuid) => {
                self.advance()?;
                Ok(DataType::Uuid)
            }
            Token::Keyword(Keyword::Json) => {
                self.advance()?;
                Ok(DataType::Json)
            }
            Token::Keyword(Keyword::Jsonb) => {
                self.advance()?;
                Ok(DataType::Jsonb)
            }
            Token::Keyword(Keyword::Tinyint) => {
                self.advance()?;
                Ok(DataType::TinyInt)
            }
            Token::Keyword(Keyword::Int128) => {
                self.advance()?;
                Ok(DataType::Int128)
            }
            Token::Keyword(Keyword::Uint8) => {
                self.advance()?;
                Ok(DataType::UInt8)
            }
            Token::Keyword(Keyword::Uint16) => {
                self.advance()?;
                Ok(DataType::UInt16)
            }
            Token::Keyword(Keyword::Uint32) => {
                self.advance()?;
                Ok(DataType::UInt32)
            }
            Token::Keyword(Keyword::Uint64) => {
                self.advance()?;
                Ok(DataType::UInt64)
            }
            Token::Keyword(Keyword::Uint128) => {
                self.advance()?;
                Ok(DataType::UInt128)
            }
            Token::Keyword(Keyword::Vector) => {
                self.advance()?;
                let dim = if self.at_token(&Token::LParen) {
                    self.advance()?;
                    let n = self.parse_integer_value()? as usize;
                    self.expect_token(&Token::RParen)?;
                    Some(n)
                } else {
                    None
                };
                Ok(DataType::Vector(dim))
            }
            _ => Err(self.error(&format!("Expected data type, found {}", self.current.token))),
        }
    }

    fn parse_optional_type_param_u32(&mut self) -> Result<Option<u32>> {
        if self.at_token(&Token::LParen) {
            self.advance()?;
            let val = self.parse_integer_value()? as u32;
            self.expect_token(&Token::RParen)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    fn parse_optional_type_param_usize(&mut self) -> Result<Option<usize>> {
        if self.at_token(&Token::LParen) {
            self.advance()?;
            let val = self.parse_integer_value()? as usize;
            self.expect_token(&Token::RParen)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    fn parse_optional_precision_scale(&mut self) -> Result<(Option<u8>, Option<u8>)> {
        if self.at_token(&Token::LParen) {
            self.advance()?;
            let precision = self.parse_integer_value()? as u8;
            let scale = if self.at_token(&Token::Comma) {
                self.advance()?;
                Some(self.parse_integer_value()? as u8)
            } else {
                None
            };
            self.expect_token(&Token::RParen)?;
            Ok((Some(precision), scale))
        } else {
            Ok((None, None))
        }
    }

    fn parse_integer_value(&mut self) -> Result<i64> {
        match &self.current.token {
            Token::Integer(n) => {
                let val = *n;
                self.advance()?;
                Ok(val)
            }
            _ => Err(self.error(&format!("Expected integer, found {}", self.current.token))),
        }
    }

    fn parse_string_literal(&mut self) -> Result<String> {
        match &self.current.token {
            Token::String(s) => {
                let val = s.clone();
                self.advance()?;
                Ok(val)
            }
            _ => Err(self.error(&format!(
                "Expected string literal, found {}",
                self.current.token
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // ORDER BY
    // -----------------------------------------------------------------------

    fn parse_order_by_expr(&mut self) -> Result<OrderByExpr> {
        let expr = self.parse_expr()?;

        let asc = if self.consume_keyword(Keyword::Asc)? {
            Some(true)
        } else if self.consume_keyword(Keyword::Desc)? {
            Some(false)
        } else {
            None
        };

        let nulls_first = if self.at_keyword(Keyword::Nulls) {
            self.advance()?;
            if self.consume_keyword(Keyword::First)? {
                Some(true)
            } else if self.consume_keyword(Keyword::Last)? {
                Some(false)
            } else {
                return Err(self.error("Expected FIRST or LAST after NULLS"));
            }
        } else {
            None
        };

        Ok(OrderByExpr {
            expr,
            asc,
            nulls_first,
        })
    }

    // -----------------------------------------------------------------------
    // Pratt expression parser
    // -----------------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr> {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            // Check for postfix-like operators: IS [NOT] NULL, [NOT] IN, [NOT] BETWEEN, [NOT] LIKE
            let (new_lhs, matched) = self.try_parse_postfix(lhs, min_bp)?;
            lhs = new_lhs;
            if matched {
                continue;
            }

            // Check for infix operators
            let (op, l_bp, r_bp) = match self.infix_binding_power() {
                Some(bp) => bp,
                None => break,
            };

            if l_bp < min_bp {
                break;
            }

            self.advance()?;
            let rhs = self.parse_expr_bp(r_bp)?;
            lhs = Expr::BinaryOp {
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
            };
        }

        // Check for :: cast operator
        while self.at_token(&Token::DoubleColon) {
            self.advance()?;
            let data_type = self.parse_data_type()?;
            lhs = Expr::Cast {
                expr: Box::new(lhs),
                data_type,
            };
        }

        // Array subscript: expr[index]
        while self.at_token(&Token::LBracket) {
            self.advance()?;
            let index = self.parse_expr()?;
            self.expect_token(&Token::RBracket)?;
            lhs = Expr::ArraySubscript {
                array: Box::new(lhs),
                index: Box::new(index),
            };
        }

        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr> {
        match &self.current.token {
            Token::Integer(n) => {
                let val = *n;
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Integer(val)))
            }
            Token::Float(f) => {
                let val = *f;
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Float(val)))
            }
            Token::String(s) => {
                let val = s.clone();
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::String(val)))
            }
            Token::Keyword(Keyword::True) => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Boolean(true)))
            }
            Token::Keyword(Keyword::False) => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Boolean(false)))
            }
            Token::Keyword(Keyword::Null) => {
                self.advance()?;
                Ok(Expr::Literal(LiteralValue::Null))
            }
            Token::Ident(_) => {
                let name = self.parse_ident()?;
                self.parse_ident_continuation(name)
            }
            // Specific keyword handlers must come before the generic keyword-as-ident case
            Token::Keyword(Keyword::Not) => {
                // NOT EXISTS (SELECT ...)
                if self.peek.token == Token::Keyword(Keyword::Exists) {
                    self.advance()?; // consume NOT
                    self.advance()?; // consume EXISTS
                    self.expect_token(&Token::LParen)?;
                    let query = if self.at_keyword(Keyword::With) {
                        let with_clause = self.parse_with_clause()?;
                        self.parse_select_body(Some(with_clause))?
                    } else {
                        self.parse_select_body(None)?
                    };
                    self.expect_token(&Token::RParen)?;
                    return Ok(Expr::Exists {
                        query: Box::new(query),
                        negated: true,
                    });
                }
                self.advance()?;
                let expr = self.parse_expr_bp(6)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(expr),
                })
            }
            Token::Keyword(Keyword::Exists) => {
                self.advance()?;
                self.expect_token(&Token::LParen)?;
                let query = if self.at_keyword(Keyword::With) {
                    let with_clause = self.parse_with_clause()?;
                    self.parse_select_body(Some(with_clause))?
                } else {
                    self.parse_select_body(None)?
                };
                self.expect_token(&Token::RParen)?;
                Ok(Expr::Exists {
                    query: Box::new(query),
                    negated: false,
                })
            }
            Token::Keyword(Keyword::Case) => self.parse_case_expr(),
            Token::Keyword(Keyword::Cast) => self.parse_cast_expr(),
            Token::Keyword(Keyword::Array) => {
                self.advance()?;
                self.expect_token(&Token::LBracket)?;
                if self.at_token(&Token::RBracket) {
                    self.advance()?;
                    return Ok(Expr::ArrayConstructor(vec![]));
                }
                let elements = self.parse_comma_separated(|p| p.parse_expr())?;
                self.expect_token(&Token::RBracket)?;
                Ok(Expr::ArrayConstructor(elements))
            }
            Token::Keyword(Keyword::Match) => {
                self.advance()?; // MATCH
                self.expect_token(&Token::LParen)?;
                let columns = self.parse_comma_separated(|p| p.parse_ident())?;
                self.expect_token(&Token::RParen)?;
                self.expect_keyword(Keyword::Against)?;
                self.expect_token(&Token::LParen)?;
                // Parse query as a prefix-only expression to avoid IN being consumed as a postfix operator
                let query = Box::new(self.parse_prefix()?);
                let mode = if self.consume_keyword(Keyword::In)? {
                    // Parse mode words until RParen
                    let mut mode_str = self.parse_ident()?;
                    while !self.at_token(&Token::RParen) {
                        mode_str.push(' ');
                        mode_str.push_str(&self.parse_ident()?);
                    }
                    Some(mode_str)
                } else {
                    None
                };
                self.expect_token(&Token::RParen)?;
                Ok(Expr::MatchAgainst {
                    columns,
                    query,
                    mode,
                })
            }
            // Keywords used as identifiers in expression context
            Token::Keyword(kw) if keyword_to_ident_str(*kw).is_some() => {
                let name = self.parse_ident()?;
                self.parse_ident_continuation(name)
            }
            Token::LParen => {
                self.advance()?;
                // Check for subquery: (SELECT ...)
                if self.at_keyword(Keyword::Select) || self.at_keyword(Keyword::With) {
                    let query = if self.at_keyword(Keyword::With) {
                        let with_clause = self.parse_with_clause()?;
                        self.parse_select_body(Some(with_clause))?
                    } else {
                        self.parse_select_body(None)?
                    };
                    self.expect_token(&Token::RParen)?;
                    return Ok(Expr::Subquery(Box::new(query)));
                }
                let expr = self.parse_expr()?;
                self.expect_token(&Token::RParen)?;
                Ok(Expr::Nested(Box::new(expr)))
            }
            Token::Minus => {
                self.advance()?;
                let expr = self.parse_expr_bp(15)?;
                Ok(Expr::UnaryOp {
                    op: UnaryOperator::Minus,
                    expr: Box::new(expr),
                })
            }
            Token::Star => {
                // Star in expression context (for COUNT(*))
                self.advance()?;
                Ok(Expr::Identifier("*".to_string()))
            }
            _ => Err(self.error(&format!(
                "Expected expression, found {}",
                self.current.token
            ))),
        }
    }

    fn parse_ident_continuation(&mut self, name: String) -> Result<Expr> {
        // Check for function call: name(
        if self.at_token(&Token::LParen) {
            return self.parse_function_call(name);
        }

        // Check for qualified identifier: name.column
        if self.at_token(&Token::Dot) {
            self.advance()?;
            let column = self.parse_ident()?;
            return Ok(Expr::QualifiedIdentifier {
                table: name,
                column,
            });
        }

        Ok(Expr::Identifier(name))
    }

    fn parse_function_call(&mut self, name: String) -> Result<Expr> {
        self.expect_token(&Token::LParen)?;

        let distinct = self.consume_keyword(Keyword::Distinct)?;

        if self.at_token(&Token::RParen) {
            self.advance()?;
            let func_expr = Expr::Function {
                name,
                args: vec![],
                distinct,
            };
            if self.at_keyword(Keyword::Over) {
                return self.parse_window_function(func_expr);
            }
            return Ok(func_expr);
        }

        // Handle COUNT(*)
        if self.at_token(&Token::Star) {
            self.advance()?;
            self.expect_token(&Token::RParen)?;
            let func_expr = Expr::Function {
                name,
                args: vec![FunctionArg::Unnamed(Expr::Identifier("*".to_string()))],
                distinct,
            };
            if self.at_keyword(Keyword::Over) {
                return self.parse_window_function(func_expr);
            }
            return Ok(func_expr);
        }

        let args = self.parse_comma_separated(|p| p.parse_function_arg())?;
        self.expect_token(&Token::RParen)?;
        let func_expr = Expr::Function {
            name,
            args,
            distinct,
        };

        // Check for OVER (window function)
        if self.at_keyword(Keyword::Over) {
            return self.parse_window_function(func_expr);
        }

        Ok(func_expr)
    }

    fn parse_function_arg(&mut self) -> Result<FunctionArg> {
        // Check for named arg: ident => expr
        if let Token::Ident(name) = &self.current.token {
            if self.peek.token == Token::FatArrow {
                let name = name.clone();
                self.advance()?; // consume ident
                self.advance()?; // consume =>
                let value = self.parse_expr()?;
                return Ok(FunctionArg::Named { name, value });
            }
        }
        // Check for keyword as name => expr (e.g., rate => 0.10)
        if let Token::Keyword(kw) = &self.current.token {
            if keyword_to_ident_str(*kw).is_some() && self.peek.token == Token::FatArrow {
                let name = keyword_to_ident_str(*kw).unwrap().to_string();
                self.advance()?; // consume keyword
                self.advance()?; // consume =>
                let value = self.parse_expr()?;
                return Ok(FunctionArg::Named { name, value });
            }
        }
        let expr = self.parse_expr()?;
        Ok(FunctionArg::Unnamed(expr))
    }

    fn parse_window_function(&mut self, function: Expr) -> Result<Expr> {
        self.expect_keyword(Keyword::Over)?;
        self.expect_token(&Token::LParen)?;

        // PARTITION BY
        let partition_by = if self.at_keyword(Keyword::Partition) {
            self.advance()?;
            self.expect_keyword(Keyword::By)?;
            self.parse_comma_separated(|p| p.parse_expr())?
        } else {
            vec![]
        };

        // ORDER BY
        let order_by = if self.at_keyword(Keyword::Order) {
            self.advance()?;
            self.expect_keyword(Keyword::By)?;
            self.parse_comma_separated(|p| p.parse_order_by_expr())?
        } else {
            vec![]
        };

        // Window frame: ROWS/RANGE ...
        let frame = if self.at_keyword(Keyword::Rows) || self.at_keyword(Keyword::Range) {
            Some(self.parse_window_frame()?)
        } else {
            None
        };

        self.expect_token(&Token::RParen)?;

        Ok(Expr::WindowFunction {
            function: Box::new(function),
            partition_by,
            order_by,
            frame,
        })
    }

    fn parse_window_frame(&mut self) -> Result<WindowFrame> {
        let mode = if self.consume_keyword(Keyword::Rows)? {
            WindowFrameMode::Rows
        } else {
            self.expect_keyword(Keyword::Range)?;
            WindowFrameMode::Range
        };

        // BETWEEN start AND end, or just a single bound
        if self.consume_keyword(Keyword::Between)? {
            let start = self.parse_window_frame_bound()?;
            self.expect_keyword(Keyword::And)?;
            let end = self.parse_window_frame_bound()?;
            Ok(WindowFrame {
                mode,
                start,
                end: Some(end),
            })
        } else {
            let start = self.parse_window_frame_bound()?;
            Ok(WindowFrame {
                mode,
                start,
                end: None,
            })
        }
    }

    fn parse_window_frame_bound(&mut self) -> Result<WindowFrameBound> {
        // CURRENT ROW
        if self.at_keyword(Keyword::Current) {
            self.advance()?;
            self.expect_keyword(Keyword::Row)?;
            return Ok(WindowFrameBound::CurrentRow);
        }

        // UNBOUNDED PRECEDING/FOLLOWING
        if self.at_keyword(Keyword::Unbounded) {
            self.advance()?;
            let direction = self.parse_frame_direction()?;
            return Ok(WindowFrameBound::Unbounded(direction));
        }

        // N PRECEDING/FOLLOWING
        let n = self.parse_integer_value()? as u64;
        let direction = self.parse_frame_direction()?;
        Ok(WindowFrameBound::Offset(n, direction))
    }

    fn parse_frame_direction(&mut self) -> Result<WindowFrameDirection> {
        if self.consume_keyword(Keyword::Preceding)? {
            Ok(WindowFrameDirection::Preceding)
        } else if self.consume_keyword(Keyword::Following)? {
            Ok(WindowFrameDirection::Following)
        } else {
            Err(self.error("Expected PRECEDING or FOLLOWING"))
        }
    }

    fn parse_case_expr(&mut self) -> Result<Expr> {
        self.expect_keyword(Keyword::Case)?;

        // Optional operand for simple CASE
        let operand = if !self.at_keyword(Keyword::When) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let mut conditions = Vec::new();
        while self.consume_keyword(Keyword::When)? {
            let condition = self.parse_expr()?;
            self.expect_keyword(Keyword::Then)?;
            let result = self.parse_expr()?;
            conditions.push(WhenClause { condition, result });
        }

        let else_result = if self.consume_keyword(Keyword::Else)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        self.expect_keyword(Keyword::End)?;

        Ok(Expr::Case {
            operand,
            conditions,
            else_result,
        })
    }

    fn parse_cast_expr(&mut self) -> Result<Expr> {
        self.expect_keyword(Keyword::Cast)?;
        self.expect_token(&Token::LParen)?;
        let expr = self.parse_expr()?;
        self.expect_keyword(Keyword::As)?;
        let data_type = self.parse_data_type()?;
        self.expect_token(&Token::RParen)?;
        Ok(Expr::Cast {
            expr: Box::new(expr),
            data_type,
        })
    }

    // Binding power table for infix binary operators.
    // Returns (BinaryOperator, left_bp, right_bp).
    fn infix_binding_power(&self) -> Option<(BinaryOperator, u8, u8)> {
        match &self.current.token {
            Token::Keyword(Keyword::Or) => Some((BinaryOperator::Or, 2, 3)),
            Token::Keyword(Keyword::And) => Some((BinaryOperator::And, 4, 5)),
            Token::Eq => Some((BinaryOperator::Eq, 7, 8)),
            Token::Neq => Some((BinaryOperator::Neq, 7, 8)),
            Token::Lt => Some((BinaryOperator::Lt, 7, 8)),
            Token::Gt => Some((BinaryOperator::Gt, 7, 8)),
            Token::LtEq => Some((BinaryOperator::LtEq, 7, 8)),
            Token::GtEq => Some((BinaryOperator::GtEq, 7, 8)),
            Token::Plus => Some((BinaryOperator::Plus, 11, 12)),
            Token::Minus => Some((BinaryOperator::Minus, 11, 12)),
            Token::Concat => Some((BinaryOperator::Concat, 11, 12)),
            Token::Star => Some((BinaryOperator::Multiply, 13, 14)),
            Token::Slash => Some((BinaryOperator::Divide, 13, 14)),
            Token::Percent => Some((BinaryOperator::Modulo, 13, 14)),
            _ => None,
        }
    }

    // Try to parse postfix-like operators: IS [NOT] NULL, [NOT] IN, [NOT] BETWEEN, [NOT] LIKE
    fn try_parse_postfix(&mut self, lhs: Expr, min_bp: u8) -> Result<(Expr, bool)> {
        let bp: u8 = 9;
        if bp < min_bp {
            return Ok((lhs, false));
        }

        // IS [NOT] NULL
        if self.at_keyword(Keyword::Is) {
            self.advance()?;
            let negated = self.consume_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Null)?;
            return Ok((
                Expr::IsNull {
                    expr: Box::new(lhs),
                    negated,
                },
                true,
            ));
        }

        // NOT IN / NOT BETWEEN / NOT LIKE
        if self.at_keyword(Keyword::Not) {
            match &self.peek.token {
                Token::Keyword(Keyword::In) => {
                    self.advance()?; // consume NOT
                    self.advance()?; // consume IN
                    self.expect_token(&Token::LParen)?;
                    // Check for subquery: NOT IN (SELECT ...)
                    if self.at_keyword(Keyword::Select) || self.at_keyword(Keyword::With) {
                        let query = if self.at_keyword(Keyword::With) {
                            let with_clause = self.parse_with_clause()?;
                            self.parse_select_body(Some(with_clause))?
                        } else {
                            self.parse_select_body(None)?
                        };
                        self.expect_token(&Token::RParen)?;
                        return Ok((
                            Expr::InSubquery {
                                expr: Box::new(lhs),
                                query: Box::new(query),
                                negated: true,
                            },
                            true,
                        ));
                    }
                    let list = self.parse_comma_separated(|p| p.parse_expr())?;
                    self.expect_token(&Token::RParen)?;
                    return Ok((
                        Expr::InList {
                            expr: Box::new(lhs),
                            list,
                            negated: true,
                        },
                        true,
                    ));
                }
                Token::Keyword(Keyword::Between) => {
                    self.advance()?; // consume NOT
                    self.advance()?; // consume BETWEEN
                    let low = self.parse_expr_bp(10)?;
                    self.expect_keyword(Keyword::And)?;
                    let high = self.parse_expr_bp(10)?;
                    return Ok((
                        Expr::Between {
                            expr: Box::new(lhs),
                            low: Box::new(low),
                            high: Box::new(high),
                            negated: true,
                        },
                        true,
                    ));
                }
                Token::Keyword(Keyword::Like) => {
                    self.advance()?; // consume NOT
                    self.advance()?; // consume LIKE
                    let pattern = self.parse_expr_bp(10)?;
                    return Ok((
                        Expr::Like {
                            expr: Box::new(lhs),
                            pattern: Box::new(pattern),
                            negated: true,
                        },
                        true,
                    ));
                }
                Token::Keyword(Keyword::Ilike) => {
                    self.advance()?; // consume NOT
                    self.advance()?; // consume ILIKE
                    let pattern = self.parse_expr_bp(10)?;
                    return Ok((
                        Expr::ILike {
                            expr: Box::new(lhs),
                            pattern: Box::new(pattern),
                            negated: true,
                        },
                        true,
                    ));
                }
                _ => {}
            }
        }

        // IN (list) or IN (SELECT ...)
        if self.at_keyword(Keyword::In) {
            self.advance()?;
            self.expect_token(&Token::LParen)?;
            // Check for subquery: IN (SELECT ...)
            if self.at_keyword(Keyword::Select) || self.at_keyword(Keyword::With) {
                let query = if self.at_keyword(Keyword::With) {
                    let with_clause = self.parse_with_clause()?;
                    self.parse_select_body(Some(with_clause))?
                } else {
                    self.parse_select_body(None)?
                };
                self.expect_token(&Token::RParen)?;
                return Ok((
                    Expr::InSubquery {
                        expr: Box::new(lhs),
                        query: Box::new(query),
                        negated: false,
                    },
                    true,
                ));
            }
            let list = self.parse_comma_separated(|p| p.parse_expr())?;
            self.expect_token(&Token::RParen)?;
            return Ok((
                Expr::InList {
                    expr: Box::new(lhs),
                    list,
                    negated: false,
                },
                true,
            ));
        }

        // BETWEEN low AND high
        if self.at_keyword(Keyword::Between) {
            self.advance()?;
            let low = self.parse_expr_bp(10)?;
            self.expect_keyword(Keyword::And)?;
            let high = self.parse_expr_bp(10)?;
            return Ok((
                Expr::Between {
                    expr: Box::new(lhs),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated: false,
                },
                true,
            ));
        }

        // LIKE pattern
        if self.at_keyword(Keyword::Like) {
            self.advance()?;
            let pattern = self.parse_expr_bp(10)?;
            return Ok((
                Expr::Like {
                    expr: Box::new(lhs),
                    pattern: Box::new(pattern),
                    negated: false,
                },
                true,
            ));
        }

        // ILIKE pattern (case-insensitive LIKE)
        if self.at_keyword(Keyword::Ilike) {
            self.advance()?;
            let pattern = self.parse_expr_bp(10)?;
            return Ok((
                Expr::ILike {
                    expr: Box::new(lhs),
                    pattern: Box::new(pattern),
                    negated: false,
                },
                true,
            ));
        }

        // Vector distance operators
        match &self.current.token {
            Token::CosineDistance | Token::L2Distance | Token::DotDistance => {
                let (l_bp, r_bp) = (7, 8);
                if l_bp >= min_bp {
                    let op = match &self.current.token {
                        Token::CosineDistance => VectorDistanceOp::Cosine,
                        Token::L2Distance => VectorDistanceOp::L2,
                        Token::DotDistance => VectorDistanceOp::DotProduct,
                        _ => unreachable!(),
                    };
                    self.advance()?;
                    let rhs = self.parse_expr_bp(r_bp)?;
                    return Ok((
                        Expr::VectorDistance {
                            left: Box::new(lhs),
                            op,
                            right: Box::new(rhs),
                        },
                        true,
                    ));
                }
            }
            _ => {}
        }

        // JSON operators
        self.try_parse_json_operator(lhs)
    }

    // -----------------------------------------------------------------------
    // RETURNING clause helper
    // -----------------------------------------------------------------------

    fn parse_returning(&mut self) -> Result<Option<Vec<SelectItem>>> {
        if self.consume_keyword(Keyword::Returning)? {
            Ok(Some(self.parse_comma_separated(|p| p.parse_select_item())?))
        } else {
            Ok(None)
        }
    }

    // -----------------------------------------------------------------------
    // CREATE/DROP VIEW
    // -----------------------------------------------------------------------

    fn parse_create_view(&mut self, or_replace: bool) -> Result<Statement> {
        self.expect_keyword(Keyword::View)?;
        let name = self.parse_ident()?;
        let columns = if self.at_token(&Token::LParen) {
            self.advance()?;
            let cols = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            cols
        } else {
            vec![]
        };
        self.expect_keyword(Keyword::As)?;
        let query = Box::new(self.parse_select_body(None)?);
        Ok(Statement::CreateView(Box::new(CreateViewStatement {
            name,
            columns,
            query,
            or_replace,
        })))
    }

    fn parse_drop_view(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::View)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropView(Box::new(DropViewStatement {
            name,
            if_exists,
        })))
    }

    // -----------------------------------------------------------------------
    // GRANT / REVOKE
    // -----------------------------------------------------------------------

    fn parse_grant(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Grant)?;
        let privileges = self.parse_privilege_list()?;
        self.expect_keyword(Keyword::On)?;
        self.consume_keyword(Keyword::Table)?;
        let on_table = self.parse_ident()?;
        self.expect_keyword(Keyword::To)?;
        let to = self.parse_ident()?;
        Ok(Statement::Grant(Box::new(GrantStatement {
            privileges,
            on_table,
            to,
        })))
    }

    fn parse_revoke(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Revoke)?;
        let privileges = self.parse_privilege_list()?;
        self.expect_keyword(Keyword::On)?;
        self.consume_keyword(Keyword::Table)?;
        let on_table = self.parse_ident()?;
        self.expect_keyword(Keyword::From)?;
        let from = self.parse_ident()?;
        Ok(Statement::Revoke(Box::new(RevokeStatement {
            privileges,
            on_table,
            from,
        })))
    }

    fn parse_privilege_list(&mut self) -> Result<Vec<Privilege>> {
        if self.at_keyword(Keyword::All) {
            self.advance()?;
            self.consume_keyword(Keyword::Privileges)?;
            return Ok(vec![Privilege::All]);
        }
        self.parse_comma_separated(|p| match &p.current.token {
            Token::Keyword(Keyword::Select) => {
                p.advance()?;
                Ok(Privilege::Select)
            }
            Token::Keyword(Keyword::Insert) => {
                p.advance()?;
                Ok(Privilege::Insert)
            }
            Token::Keyword(Keyword::Update) => {
                p.advance()?;
                Ok(Privilege::Update)
            }
            Token::Keyword(Keyword::Delete) => {
                p.advance()?;
                Ok(Privilege::Delete)
            }
            _ => Err(p.error(&format!("Expected privilege, found {}", p.current.token))),
        })
    }

    // -----------------------------------------------------------------------
    // CREATE/DROP SCHEMA
    // -----------------------------------------------------------------------

    fn parse_create_schema(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Schema)?;
        let if_not_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::CreateSchema(Box::new(CreateSchemaStatement {
            name,
            if_not_exists,
        })))
    }

    fn parse_drop_schema(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Schema)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        let cascade = self.consume_keyword(Keyword::Cascade)?;
        if !cascade {
            self.consume_keyword(Keyword::Restrict)?;
        }
        Ok(Statement::DropSchema(Box::new(DropSchemaStatement {
            name,
            if_exists,
            cascade,
        })))
    }

    // -----------------------------------------------------------------------
    // CREATE/DROP SEQUENCE
    // -----------------------------------------------------------------------

    fn parse_create_sequence(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Sequence)?;
        let if_not_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        let mut increment = None;
        let mut min_value = None;
        let mut max_value = None;
        let mut start = None;
        let mut cache = None;
        let mut cycle = false;
        loop {
            if self.consume_keyword(Keyword::Increment)? {
                self.consume_keyword(Keyword::By)?;
                increment = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Minvalue)? {
                min_value = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Maxvalue)? {
                max_value = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Start)? {
                self.consume_keyword(Keyword::With)?;
                start = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Cache)? {
                cache = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Cycle)? {
                cycle = true;
            } else if self.at_keyword(Keyword::No) {
                self.advance()?;
                if self.consume_keyword(Keyword::Minvalue)? {
                    // NO MINVALUE: leave min_value as None
                } else if self.consume_keyword(Keyword::Maxvalue)? {
                    // NO MAXVALUE: leave max_value as None
                } else if self.consume_keyword(Keyword::Cycle)? {
                    cycle = false;
                } else {
                    return Err(self.error("Expected MINVALUE, MAXVALUE, or CYCLE after NO"));
                }
            } else {
                break;
            }
        }
        Ok(Statement::CreateSequence(Box::new(
            CreateSequenceStatement {
                name,
                if_not_exists,
                increment,
                min_value,
                max_value,
                start,
                cache,
                cycle,
            },
        )))
    }

    fn parse_drop_sequence(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Sequence)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropSequence(Box::new(DropSequenceStatement {
            name,
            if_exists,
        })))
    }

    // -----------------------------------------------------------------------
    // VACUUM / REINDEX
    // -----------------------------------------------------------------------

    fn parse_vacuum(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Vacuum)?;
        let table = if self.current.token != Token::Eof && self.current.token != Token::Semicolon {
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(Statement::Vacuum(Box::new(VacuumStatement { table })))
    }

    fn parse_analyze(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Analyze)?;
        let table = if self.current.token != Token::Eof && self.current.token != Token::Semicolon {
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(Statement::Analyze(Box::new(AnalyzeStatement { table })))
    }

    fn parse_reindex(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Reindex)?;
        let target = if self.consume_keyword(Keyword::Index)? {
            ReindexTarget::Index(self.parse_ident()?)
        } else {
            self.consume_keyword(Keyword::Table)?;
            ReindexTarget::Table(self.parse_ident()?)
        };
        Ok(Statement::Reindex(Box::new(ReindexStatement { target })))
    }

    // -----------------------------------------------------------------------
    // SET / SHOW
    // -----------------------------------------------------------------------

    fn parse_set_variable(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Set)?;
        let name = self.parse_ident()?;
        // SET name = value or SET name TO value
        if self.consume_keyword(Keyword::To)? {
            // TO form
        } else {
            self.expect_token(&Token::Eq)?;
        }
        let value = self.parse_expr()?;
        Ok(Statement::SetVariable(Box::new(SetVariableStatement {
            name,
            value,
        })))
    }

    fn parse_show(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Show)?;
        let name = if self.consume_keyword(Keyword::All)? {
            "all".to_string()
        } else {
            self.parse_ident()?
        };
        Ok(Statement::Show(Box::new(ShowStatement { name })))
    }

    // -----------------------------------------------------------------------
    // COPY
    // -----------------------------------------------------------------------

    fn parse_copy(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Copy)?;
        let table = self.parse_ident()?;
        let columns = if self.at_token(&Token::LParen) {
            self.advance()?;
            let cols = self.parse_comma_separated(|p| p.parse_ident())?;
            self.expect_token(&Token::RParen)?;
            cols
        } else {
            vec![]
        };
        let direction = if self.consume_keyword(Keyword::From)? {
            let target = self.parse_copy_target()?;
            CopyDirection::From(target)
        } else {
            self.expect_keyword(Keyword::To)?;
            let target = self.parse_copy_target()?;
            CopyDirection::To(target)
        };
        Ok(Statement::Copy(Box::new(CopyStatement {
            table,
            columns,
            direction,
        })))
    }

    // -----------------------------------------------------------------------
    // JSON operators
    // -----------------------------------------------------------------------

    fn try_parse_json_operator(&mut self, lhs: Expr) -> Result<(Expr, bool)> {
        match &self.current.token {
            Token::Arrow => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonAccess {
                        left: Box::new(lhs),
                        op: JsonOperator::Arrow,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::DoubleArrow => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonAccess {
                        left: Box::new(lhs),
                        op: JsonOperator::DoubleArrow,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::HashArrow => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonAccess {
                        left: Box::new(lhs),
                        op: JsonOperator::HashArrow,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::HashDoubleArrow => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonAccess {
                        left: Box::new(lhs),
                        op: JsonOperator::HashDoubleArrow,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::AtArrow => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonContains {
                        left: Box::new(lhs),
                        op: JsonContainsOp::Contains,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::ArrowAt => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonContains {
                        left: Box::new(lhs),
                        op: JsonContainsOp::ContainedBy,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::Question => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonExists {
                        left: Box::new(lhs),
                        op: JsonExistsOp::Exists,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::QuestionPipe => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonExists {
                        left: Box::new(lhs),
                        op: JsonExistsOp::ExistsAny,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            Token::QuestionAmp => {
                self.advance()?;
                let right = self.parse_expr_bp(10)?;
                Ok((
                    Expr::JsonExists {
                        left: Box::new(lhs),
                        op: JsonExistsOp::ExistsAll,
                        right: Box::new(right),
                    },
                    true,
                ))
            }
            _ => Ok((lhs, false)),
        }
    }

    // -----------------------------------------------------------------------
    // MERGE
    // -----------------------------------------------------------------------

    fn parse_merge(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Merge)?;

        // MERGE BRANCH source INTO target
        if self.consume_keyword(Keyword::Branch)? {
            return self.parse_merge_branch();
        }

        self.expect_keyword(Keyword::Into)?;
        let target = self.parse_ident()?;
        self.expect_keyword(Keyword::Using)?;
        let source = self.parse_base_table_ref()?;
        self.expect_keyword(Keyword::On)?;
        let on = self.parse_expr()?;
        let mut clauses = Vec::new();
        while self.at_keyword(Keyword::When) {
            self.advance()?;
            if self.consume_keyword(Keyword::Matched)? {
                let condition = if self.consume_keyword(Keyword::And)? {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                self.expect_keyword(Keyword::Then)?;
                let action = if self.consume_keyword(Keyword::Update)? {
                    self.expect_keyword(Keyword::Set)?;
                    let assignments = self.parse_comma_separated(|p| p.parse_assignment())?;
                    MergeAction::Update(assignments)
                } else if self.consume_keyword(Keyword::Delete)? {
                    MergeAction::Delete
                } else {
                    return Err(self.error("Expected UPDATE or DELETE after THEN in WHEN MATCHED"));
                };
                clauses.push(MergeClause::WhenMatched { condition, action });
            } else {
                self.expect_keyword(Keyword::Not)?;
                self.expect_keyword(Keyword::Matched)?;
                let condition = if self.consume_keyword(Keyword::And)? {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                self.expect_keyword(Keyword::Then)?;
                if self.consume_keyword(Keyword::Insert)? {
                    let columns = if self.at_token(&Token::LParen) {
                        self.advance()?;
                        let cols = self.parse_comma_separated(|p| p.parse_ident())?;
                        self.expect_token(&Token::RParen)?;
                        cols
                    } else {
                        vec![]
                    };
                    self.expect_keyword(Keyword::Values)?;
                    self.expect_token(&Token::LParen)?;
                    let values = self.parse_comma_separated(|p| p.parse_expr())?;
                    self.expect_token(&Token::RParen)?;
                    clauses.push(MergeClause::WhenNotMatched {
                        condition,
                        action: MergeAction::Insert { columns, values },
                    });
                } else {
                    return Err(self.error("Expected INSERT after THEN in WHEN NOT MATCHED"));
                }
            }
        }
        Ok(Statement::Merge(Box::new(MergeStatement {
            target,
            source,
            on,
            clauses,
        })))
    }

    // -----------------------------------------------------------------------
    // Prepared statements
    // -----------------------------------------------------------------------

    fn parse_prepare(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Prepare)?;
        let name = self.parse_ident()?;
        let param_types = if self.at_token(&Token::LParen) {
            self.advance()?;
            let types = self.parse_comma_separated(|p| p.parse_data_type())?;
            self.expect_token(&Token::RParen)?;
            types
        } else {
            vec![]
        };
        self.expect_keyword(Keyword::As)?;
        let statement = Box::new(self.parse_statement()?);
        Ok(Statement::Prepare(Box::new(PrepareStatement {
            name,
            param_types,
            statement,
        })))
    }

    fn parse_execute(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Execute)?;
        let name = self.parse_ident()?;
        let params = if self.at_token(&Token::LParen) {
            self.advance()?;
            let p = self.parse_comma_separated(|p| p.parse_expr())?;
            self.expect_token(&Token::RParen)?;
            p
        } else {
            vec![]
        };
        Ok(Statement::Execute(Box::new(ExecuteStatement {
            name,
            params,
        })))
    }

    fn parse_deallocate(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Deallocate)?;
        self.consume_keyword(Keyword::Prepare)?;
        if self.consume_keyword(Keyword::All)? {
            Ok(Statement::Deallocate(Box::new(DeallocateStatement {
                name: None,
                all: true,
            })))
        } else {
            let name = self.parse_ident()?;
            Ok(Statement::Deallocate(Box::new(DeallocateStatement {
                name: Some(name),
                all: false,
            })))
        }
    }

    // -----------------------------------------------------------------------
    // LISTEN / NOTIFY
    // -----------------------------------------------------------------------

    fn parse_listen(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Listen)?;
        let channel = self.parse_ident()?;
        Ok(Statement::Listen(Box::new(ListenStatement { channel })))
    }

    fn parse_notify(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Notify)?;
        let channel = self.parse_ident()?;
        let payload = if self.at_token(&Token::Comma) {
            self.advance()?;
            if let Token::String(s) = &self.current.token {
                let p = s.clone();
                self.advance()?;
                Some(p)
            } else {
                return Err(self.error("Expected string payload after comma in NOTIFY"));
            }
        } else {
            None
        };
        Ok(Statement::Notify(Box::new(NotifyStatement {
            channel,
            payload,
        })))
    }

    // -----------------------------------------------------------------------
    // Cursors
    // -----------------------------------------------------------------------

    fn parse_declare_cursor(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Declare)?;
        let name = self.parse_ident()?;
        let scroll = if self.at_keyword(Keyword::Scroll) {
            self.advance()?;
            Some(true)
        } else if self.at_keyword(Keyword::No) && self.peek.token == Token::Keyword(Keyword::Scroll)
        {
            self.advance()?;
            self.advance()?;
            Some(false)
        } else {
            None
        };
        self.expect_keyword(Keyword::Cursor)?;
        let hold = if self.at_keyword(Keyword::With) {
            self.advance()?;
            self.expect_keyword(Keyword::Hold)?;
            Some(true)
        } else if self.at_keyword(Keyword::Without) {
            self.advance()?;
            self.expect_keyword(Keyword::Hold)?;
            Some(false)
        } else {
            None
        };
        self.expect_keyword(Keyword::For)?;
        let query = Box::new(self.parse_select_body(None)?);
        Ok(Statement::DeclareCursor(Box::new(DeclareCursorStatement {
            name,
            scroll,
            hold,
            query,
        })))
    }

    fn parse_fetch_cursor(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Fetch)?;
        let direction = if self.consume_keyword(Keyword::Next)? {
            FetchDirection::Next
        } else if self.at_keyword(Keyword::First) {
            self.advance()?;
            FetchDirection::First
        } else if self.at_keyword(Keyword::Last) {
            self.advance()?;
            FetchDirection::Last
        } else if self.consume_keyword(Keyword::All)? {
            FetchDirection::All
        } else if self.consume_keyword(Keyword::Absolute)? {
            let n = self.parse_integer_value()?;
            FetchDirection::Absolute(n)
        } else if self.consume_keyword(Keyword::Relative)? {
            let n = self.parse_integer_value()?;
            FetchDirection::Relative(n)
        } else if self.consume_keyword(Keyword::Forward)? {
            if self.consume_keyword(Keyword::All)? {
                FetchDirection::Forward(None)
            } else if let Token::Integer(_) = &self.current.token {
                let n = self.parse_integer_value()?;
                FetchDirection::Forward(Some(n))
            } else {
                FetchDirection::Forward(Some(1))
            }
        } else if self.consume_keyword(Keyword::Backward)? {
            if self.consume_keyword(Keyword::All)? {
                FetchDirection::Backward(None)
            } else if let Token::Integer(_) = &self.current.token {
                let n = self.parse_integer_value()?;
                FetchDirection::Backward(Some(n))
            } else {
                FetchDirection::Backward(Some(1))
            }
        } else {
            FetchDirection::Next
        };
        if !self.consume_keyword(Keyword::From)? {
            self.consume_keyword(Keyword::In)?;
        }
        let cursor = self.parse_ident()?;
        Ok(Statement::FetchCursor(Box::new(FetchCursorStatement {
            direction,
            cursor,
        })))
    }

    fn parse_close_cursor(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Close)?;
        if self.consume_keyword(Keyword::All)? {
            Ok(Statement::CloseCursor(Box::new(CloseCursorStatement {
                name: None,
                all: true,
            })))
        } else {
            let name = self.parse_ident()?;
            Ok(Statement::CloseCursor(Box::new(CloseCursorStatement {
                name: Some(name),
                all: false,
            })))
        }
    }

    // -----------------------------------------------------------------------
    // COMMENT ON
    // -----------------------------------------------------------------------

    fn parse_comment_on(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Comment)?;
        self.expect_keyword(Keyword::On)?;
        let (object_type, name, column) = if self.consume_keyword(Keyword::Table)? {
            (CommentObjectType::Table, self.parse_ident()?, None)
        } else if self.at_keyword(Keyword::Column) {
            self.advance()?;
            let table = self.parse_ident()?;
            self.expect_token(&Token::Dot)?;
            let col = self.parse_ident()?;
            (CommentObjectType::Column, table, Some(col))
        } else if self.consume_keyword(Keyword::Index)? {
            (CommentObjectType::Index, self.parse_ident()?, None)
        } else if self.consume_keyword(Keyword::Schema)? {
            (CommentObjectType::Schema, self.parse_ident()?, None)
        } else if self.consume_keyword(Keyword::Sequence)? {
            (CommentObjectType::Sequence, self.parse_ident()?, None)
        } else if self.consume_keyword(Keyword::View)? {
            (CommentObjectType::View, self.parse_ident()?, None)
        } else {
            return Err(self.error(
                "Expected TABLE, COLUMN, INDEX, SCHEMA, SEQUENCE, or VIEW after COMMENT ON",
            ));
        };
        self.expect_keyword(Keyword::Is)?;
        let comment = if self.at_keyword(Keyword::Null) {
            self.advance()?;
            None
        } else if let Token::String(s) = &self.current.token {
            let c = s.clone();
            self.advance()?;
            Some(c)
        } else {
            return Err(self.error("Expected string or NULL after IS"));
        };
        Ok(Statement::CommentOn(Box::new(CommentOnStatement {
            object_type,
            name,
            column,
            comment,
        })))
    }

    // -----------------------------------------------------------------------
    // ALTER INDEX / SEQUENCE / VIEW
    // -----------------------------------------------------------------------

    fn parse_alter_index(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Index)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::Rename)?;
        self.expect_keyword(Keyword::To)?;
        let new_name = self.parse_ident()?;
        Ok(Statement::AlterIndex(Box::new(AlterIndexStatement {
            name,
            operation: AlterIndexOperation::Rename { new_name },
        })))
    }

    fn parse_alter_sequence(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Sequence)?;
        let name = self.parse_ident()?;
        let mut increment = None;
        let mut min_value = None;
        let mut max_value = None;
        let mut start = None;
        let restart = None;
        let mut cache = None;
        let mut cycle = None;
        loop {
            if self.consume_keyword(Keyword::Increment)? {
                self.consume_keyword(Keyword::By)?;
                increment = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Minvalue)? {
                min_value = Some(Some(self.parse_integer_value()?));
            } else if self.consume_keyword(Keyword::Maxvalue)? {
                max_value = Some(Some(self.parse_integer_value()?));
            } else if self.consume_keyword(Keyword::Start)? {
                self.consume_keyword(Keyword::With)?;
                start = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Cache)? {
                cache = Some(self.parse_integer_value()?);
            } else if self.consume_keyword(Keyword::Cycle)? {
                cycle = Some(true);
            } else if self.at_keyword(Keyword::No) {
                self.advance()?;
                if self.consume_keyword(Keyword::Minvalue)? {
                    min_value = Some(None);
                } else if self.consume_keyword(Keyword::Maxvalue)? {
                    max_value = Some(None);
                } else if self.consume_keyword(Keyword::Cycle)? {
                    cycle = Some(false);
                } else {
                    return Err(self.error("Expected MINVALUE, MAXVALUE, or CYCLE after NO"));
                }
            } else {
                break;
            }
        }
        Ok(Statement::AlterSequence(Box::new(AlterSequenceStatement {
            name,
            increment,
            min_value,
            max_value,
            start,
            restart,
            cache,
            cycle,
        })))
    }

    fn parse_alter_view(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::View)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::Rename)?;
        self.expect_keyword(Keyword::To)?;
        let new_name = self.parse_ident()?;
        Ok(Statement::AlterView(Box::new(AlterViewStatement {
            name,
            operation: AlterViewOperation::Rename { new_name },
        })))
    }

    // -----------------------------------------------------------------------
    // Materialized views
    // -----------------------------------------------------------------------

    fn parse_create_materialized_view(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Materialized)?;
        self.expect_keyword(Keyword::View)?;
        let if_not_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Not)?;
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::As)?;
        let query = Box::new(self.parse_select_body(None)?);
        Ok(Statement::CreateMaterializedView(Box::new(
            CreateMaterializedViewStatement {
                name,
                if_not_exists,
                query,
            },
        )))
    }

    fn parse_drop_materialized_view(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Materialized)?;
        self.expect_keyword(Keyword::View)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropMaterializedView(Box::new(
            DropMaterializedViewStatement { name, if_exists },
        )))
    }

    fn parse_refresh_materialized_view(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Refresh)?;
        self.expect_keyword(Keyword::Materialized)?;
        self.expect_keyword(Keyword::View)?;
        let concurrently = if let Token::Ident(ref s) = self.current.token {
            if s.to_ascii_uppercase() == "CONCURRENTLY" {
                self.advance()?;
                true
            } else {
                false
            }
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::RefreshMaterializedView(Box::new(
            RefreshMaterializedViewStatement { name, concurrently },
        )))
    }

    // -----------------------------------------------------------------------
    // DO block
    // -----------------------------------------------------------------------

    fn parse_do_block(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Do)?;
        let body = if let Token::String(s) = &self.current.token {
            let b = s.clone();
            self.advance()?;
            b
        } else {
            return Err(self.error("Expected string body for DO block"));
        };
        let language = if self.consume_keyword(Keyword::Language)? {
            Some(self.parse_ident()?)
        } else {
            None
        };
        Ok(Statement::DoBlock(Box::new(DoBlockStatement {
            body,
            language,
        })))
    }

    // -----------------------------------------------------------------------
    // CHECKPOINT
    // -----------------------------------------------------------------------

    fn parse_checkpoint(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Checkpoint)?;
        Ok(Statement::Checkpoint(Box::new(CheckpointStatement {})))
    }

    // -----------------------------------------------------------------------
    // VALUES query
    // -----------------------------------------------------------------------

    fn parse_values_query(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Values)?;
        let rows = self.parse_comma_separated(|p| {
            p.expect_token(&Token::LParen)?;
            let exprs = p.parse_comma_separated(|p2| p2.parse_expr())?;
            p.expect_token(&Token::RParen)?;
            Ok(exprs)
        })?;
        Ok(Statement::ValuesQuery(Box::new(ValuesQueryStatement {
            rows,
        })))
    }

    // -----------------------------------------------------------------------
    // COPY
    // -----------------------------------------------------------------------

    fn parse_copy_target(&mut self) -> Result<CopyTarget> {
        if self.consume_keyword(Keyword::Stdin)? {
            Ok(CopyTarget::Stdin)
        } else if self.consume_keyword(Keyword::Stdout)? {
            Ok(CopyTarget::Stdout)
        } else if let Token::String(s) = &self.current.token {
            let path = s.clone();
            self.advance()?;
            Ok(CopyTarget::File(path))
        } else {
            Err(self.error(&format!(
                "Expected file path, STDIN, or STDOUT, found {}",
                self.current.token
            )))
        }
    }

    // -----------------------------------------------------------------------
    // TTL / Schedule / Optimize
    // -----------------------------------------------------------------------

    fn parse_ttl_duration(&mut self) -> Result<TtlDuration> {
        let value = self.parse_integer_value()?;
        let unit = if self.consume_keyword(Keyword::Days)? {
            TtlUnit::Days
        } else if self.consume_keyword(Keyword::Hours)? {
            TtlUnit::Hours
        } else if self.consume_keyword(Keyword::Minutes)? {
            TtlUnit::Minutes
        } else if self.consume_keyword(Keyword::Seconds)? {
            TtlUnit::Seconds
        } else {
            return Err(self.error("Expected DAYS, HOURS, MINUTES, or SECONDS"));
        };
        Ok(TtlDuration { value, unit })
    }

    fn parse_create_schedule(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Schedule)?;
        let name = self.parse_ident()?;
        let interval = if self.consume_keyword(Keyword::Every)? {
            ScheduleInterval::Every(self.parse_ttl_duration()?)
        } else if self.consume_keyword(Keyword::Cron)? {
            let expr = if let Token::String(s) = &self.current.token {
                let e = s.clone();
                self.advance()?;
                e
            } else {
                return Err(self.error("Expected cron expression string after CRON"));
            };
            ScheduleInterval::Cron(expr)
        } else {
            return Err(self.error("Expected EVERY or CRON after schedule name"));
        };
        self.expect_keyword(Keyword::Do)?;
        let body = Box::new(self.parse_statement()?);
        Ok(Statement::CreateSchedule(Box::new(
            CreateScheduleStatement {
                name,
                interval,
                body,
            },
        )))
    }

    fn parse_drop_schedule(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Schedule)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropSchedule(Box::new(DropScheduleStatement {
            name,
            if_exists,
        })))
    }

    fn parse_pause_schedule(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Pause)?;
        self.expect_keyword(Keyword::Schedule)?;
        let name = self.parse_ident()?;
        Ok(Statement::PauseSchedule(Box::new(PauseScheduleStatement {
            name,
        })))
    }

    fn parse_resume_schedule(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Resume)?;
        self.expect_keyword(Keyword::Schedule)?;
        let name = self.parse_ident()?;
        Ok(Statement::ResumeSchedule(Box::new(
            ResumeScheduleStatement { name },
        )))
    }

    fn parse_optimize_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Optimize)?;
        self.consume_keyword(Keyword::Table)?;
        let table = self.parse_ident()?;
        Ok(Statement::OptimizeTable(Box::new(OptimizeTableStatement {
            table,
        })))
    }

    // -----------------------------------------------------------------------
    // consume_token helper
    // -----------------------------------------------------------------------

    fn consume_token(&mut self, expected: &Token) -> Result<bool> {
        if self.current.token == *expected {
            self.advance()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // -----------------------------------------------------------------------
    // Table options parsing helper
    // -----------------------------------------------------------------------

    fn parse_table_option(&mut self) -> Result<TableOption> {
        let key = self.parse_ident()?;
        self.expect_token(&Token::Eq)?;
        let value = match &self.current.token {
            Token::String(s) => {
                let v = TableOptionValue::String(s.clone());
                self.advance()?;
                v
            }
            Token::Integer(n) => {
                let v = TableOptionValue::Integer(*n);
                self.advance()?;
                v
            }
            Token::Keyword(Keyword::True) => {
                self.advance()?;
                TableOptionValue::Boolean(true)
            }
            Token::Keyword(Keyword::False) => {
                self.advance()?;
                TableOptionValue::Boolean(false)
            }
            Token::LBracket => {
                // Parse string list: ['a', 'b', 'c']
                self.advance()?;
                let mut items = vec![];
                loop {
                    if self.at_token(&Token::RBracket) {
                        break;
                    }
                    if let Token::String(s) = &self.current.token {
                        items.push(s.clone());
                        self.advance()?;
                    } else {
                        return Err(self.error("Expected string in list"));
                    }
                    if !self.consume_token(&Token::Comma)? {
                        break;
                    }
                }
                self.expect_token(&Token::RBracket)?;
                TableOptionValue::StringList(items)
            }
            _ => {
                // Try as identifier
                let ident = self.parse_ident()?;
                TableOptionValue::Identifier(ident)
            }
        };
        Ok(TableOption { key, value })
    }

    // -----------------------------------------------------------------------
    // Security: CREATE/ALTER/DROP USER and ROLE
    // -----------------------------------------------------------------------

    fn parse_create_user(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::User)?;
        let name = self.parse_ident()?;
        let mut password = None;
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            loop {
                if self.consume_keyword(Keyword::Password)? {
                    if let Token::String(pw) = &self.current.token {
                        password = Some(pw.clone());
                        self.advance()?;
                    } else {
                        return Err(self.error("Expected password string"));
                    }
                } else if self.consume_keyword(Keyword::Superuser)? {
                    options.push(UserOption::Superuser(true));
                } else if self.consume_keyword(Keyword::Login)? {
                    options.push(UserOption::Login(true));
                } else if self.consume_keyword(Keyword::Nologin)? {
                    options.push(UserOption::Login(false));
                } else if self.consume_keyword(Keyword::Valid)? {
                    self.expect_keyword(Keyword::Until)?;
                    if let Token::String(until) = &self.current.token {
                        options.push(UserOption::ValidUntil(until.clone()));
                        self.advance()?;
                    } else {
                        return Err(self.error("Expected date string after VALID UNTIL"));
                    }
                } else {
                    break;
                }
            }
        }
        Ok(Statement::CreateUser(Box::new(CreateUserStatement {
            name,
            password,
            options,
        })))
    }

    fn parse_create_role(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Role)?;
        let name = self.parse_ident()?;
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            loop {
                if self.consume_keyword(Keyword::Superuser)? {
                    options.push(UserOption::Superuser(true));
                } else if self.consume_keyword(Keyword::Login)? {
                    options.push(UserOption::Login(true));
                } else if self.consume_keyword(Keyword::Nologin)? {
                    options.push(UserOption::Login(false));
                } else {
                    break;
                }
            }
        }
        Ok(Statement::CreateRole(Box::new(CreateRoleStatement {
            name,
            options,
        })))
    }

    fn parse_drop_user(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::User)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropUser(Box::new(DropUserStatement {
            name,
            if_exists,
        })))
    }

    fn parse_drop_role(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Role)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropRole(Box::new(DropRoleStatement {
            name,
            if_exists,
        })))
    }

    fn parse_alter_user(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::User)?;
        let name = self.parse_ident()?;
        let operation = if self.consume_keyword(Keyword::Set)? {
            self.expect_keyword(Keyword::Password)?;
            if let Token::String(pw) = &self.current.token {
                let pw = pw.clone();
                self.advance()?;
                AlterUserOperation::SetPassword(pw)
            } else {
                return Err(self.error("Expected password string"));
            }
        } else if self.consume_keyword(Keyword::Rename)? {
            self.expect_keyword(Keyword::To)?;
            let new_name = self.parse_ident()?;
            AlterUserOperation::Rename { new_name }
        } else if self.consume_keyword(Keyword::With)? {
            if self.consume_keyword(Keyword::Superuser)? {
                AlterUserOperation::SetOption(UserOption::Superuser(true))
            } else if self.consume_keyword(Keyword::Login)? {
                AlterUserOperation::SetOption(UserOption::Login(true))
            } else if self.consume_keyword(Keyword::Nologin)? {
                AlterUserOperation::SetOption(UserOption::Login(false))
            } else {
                return Err(self.error("Expected user option after WITH"));
            }
        } else {
            return Err(self.error("Expected SET, RENAME, or WITH after ALTER USER name"));
        };
        Ok(Statement::AlterUser(Box::new(AlterUserStatement {
            name,
            operation,
        })))
    }

    fn parse_alter_role(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Role)?;
        let name = self.parse_ident()?;
        let operation = if self.consume_keyword(Keyword::Rename)? {
            self.expect_keyword(Keyword::To)?;
            let new_name = self.parse_ident()?;
            AlterUserOperation::Rename { new_name }
        } else if self.consume_keyword(Keyword::With)? {
            if self.consume_keyword(Keyword::Superuser)? {
                AlterUserOperation::SetOption(UserOption::Superuser(true))
            } else if self.consume_keyword(Keyword::Login)? {
                AlterUserOperation::SetOption(UserOption::Login(true))
            } else if self.consume_keyword(Keyword::Nologin)? {
                AlterUserOperation::SetOption(UserOption::Login(false))
            } else {
                return Err(self.error("Expected role option after WITH"));
            }
        } else {
            return Err(self.error("Expected RENAME or WITH after ALTER ROLE name"));
        };
        Ok(Statement::AlterRole(Box::new(AlterRoleStatement {
            name,
            operation,
        })))
    }

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------

    fn parse_create_pipeline(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Pipeline)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::As)?;
        self.expect_token(&Token::LParen)?;
        let stages = self.parse_comma_separated(|p| p.parse_pipeline_stage())?;
        self.expect_token(&Token::RParen)?;
        Ok(Statement::CreatePipeline(Box::new(
            CreatePipelineStatement { name, stages },
        )))
    }

    fn parse_pipeline_stage(&mut self) -> Result<PipelineStage> {
        self.expect_keyword(Keyword::Stage)?;
        let name = self.parse_ident()?;
        self.expect_token(&Token::LParen)?;

        let mut source = String::new();
        let mut target = String::new();
        let mut mode = None;
        let mut transform = None;
        let mut expectations = vec![];

        loop {
            if self.at_token(&Token::RParen) {
                self.advance()?;
                break;
            }
            if self.consume_keyword(Keyword::Source)? {
                source = self.parse_ident()?;
                self.consume_token(&Token::Comma)?;
            } else if self.consume_keyword(Keyword::Target)? {
                target = self.parse_ident()?;
                self.consume_token(&Token::Comma)?;
            } else if self.consume_keyword(Keyword::Mode)? {
                mode = Some(self.parse_ident()?);
                self.consume_token(&Token::Comma)?;
            } else if self.consume_keyword(Keyword::Transform)? {
                self.expect_keyword(Keyword::As)?;
                self.expect_token(&Token::LParen)?;
                let query = self.parse_select_body(None)?;
                transform = Some(Box::new(query));
                self.expect_token(&Token::RParen)?;
                self.consume_token(&Token::Comma)?;
            } else if self.consume_keyword(Keyword::Expect)? {
                // Parse inline expectations
                let expr = self.parse_expr()?;
                expectations.push(PipelineExpectation { expr });
                self.consume_token(&Token::Comma)?;
            } else {
                return Err(
                    self.error("Expected SOURCE, TARGET, MODE, TRANSFORM, or EXPECT in STAGE")
                );
            }
        }

        Ok(PipelineStage {
            name,
            source,
            target,
            mode,
            transform,
            expectations,
        })
    }

    fn parse_run_pipeline(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Run)?;
        self.expect_keyword(Keyword::Pipeline)?;
        let name = self.parse_ident()?;
        let stage = if self.consume_keyword(Keyword::Stage)? {
            Some(self.parse_ident()?)
        } else {
            None
        };
        let preview_limit = if self.consume_keyword(Keyword::Preview)? {
            self.expect_keyword(Keyword::Limit)?;
            Some(self.parse_integer_value()? as u64)
        } else {
            None
        };
        Ok(Statement::RunPipeline(Box::new(RunPipelineStatement {
            name,
            stage,
            preview_limit,
        })))
    }

    fn parse_drop_pipeline(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Pipeline)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropPipeline(Box::new(DropPipelineStatement {
            name,
            if_exists,
        })))
    }

    // -----------------------------------------------------------------------
    // Archive / Restore
    // -----------------------------------------------------------------------

    fn parse_archive_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Archive)?;
        self.expect_keyword(Keyword::Table)?;
        let table = self.parse_ident()?;
        let where_clause = if self.consume_keyword(Keyword::Where)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        self.expect_keyword(Keyword::To)?;
        let destination = if let Token::String(s) = &self.current.token {
            let s = s.clone();
            self.advance()?;
            s
        } else {
            return Err(self.error("Expected destination path string after TO"));
        };
        Ok(Statement::ArchiveTable(Box::new(ArchiveTableStatement {
            table,
            where_clause,
            destination,
        })))
    }

    fn parse_restore_table(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Restore)?;
        self.expect_keyword(Keyword::Table)?;
        let table = self.parse_ident()?;
        self.expect_keyword(Keyword::From)?;
        let source = if let Token::String(s) = &self.current.token {
            let s = s.clone();
            self.advance()?;
            s
        } else {
            return Err(self.error("Expected source path string after FROM"));
        };
        let into_table = if self.consume_keyword(Keyword::Into)? {
            Some(self.parse_ident()?)
        } else {
            None
        };

        // Optional: TO VERSION AS OF expr or TO TIMESTAMP AS OF expr
        let mut at_version = None;
        let mut at_timestamp = None;
        if self.consume_keyword(Keyword::To)? {
            if self.consume_keyword(Keyword::Version)? {
                self.expect_keyword(Keyword::As)?;
                self.expect_keyword(Keyword::Of)?;
                at_version = Some(self.parse_expr()?);
            } else if self.consume_keyword(Keyword::Timestamp)? {
                self.expect_keyword(Keyword::As)?;
                self.expect_keyword(Keyword::Of)?;
                at_timestamp = Some(self.parse_expr()?);
            } else {
                return Err(self.error("Expected VERSION or TIMESTAMP after TO in RESTORE TABLE"));
            }
        }

        Ok(Statement::RestoreTable(Box::new(RestoreTableStatement {
            table,
            source,
            into_table,
            at_version,
            at_timestamp,
        })))
    }

    // -----------------------------------------------------------------------
    // Branching / Versioning
    // -----------------------------------------------------------------------

    fn parse_create_branch(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Branch)?;
        let name = self.parse_ident()?;
        let from_branch = if self.consume_keyword(Keyword::From)? {
            Some(self.parse_ident()?)
        } else {
            None
        };
        let at_version = if self.at_keyword(Keyword::At) {
            self.advance()?; // AT
            self.expect_keyword(Keyword::Version)?;
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::CreateBranch(Box::new(CreateBranchStatement {
            name,
            from_branch,
            at_version,
        })))
    }

    fn parse_merge_branch(&mut self) -> Result<Statement> {
        let source = self.parse_ident()?;
        self.expect_keyword(Keyword::Into)?;
        let into_target = self.parse_ident()?;
        Ok(Statement::MergeBranch(Box::new(MergeBranchStatement {
            source,
            into_target,
        })))
    }

    fn parse_drop_branch(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Branch)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropBranch(Box::new(DropBranchStatement {
            name,
            if_exists,
        })))
    }

    fn parse_use_branch(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Use)?;
        self.expect_keyword(Keyword::Branch)?;
        let name = self.parse_ident()?;
        Ok(Statement::UseBranch(Box::new(UseBranchStatement { name })))
    }

    fn parse_create_version(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Version)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table = self.parse_ident()?;
        let at_version = if self.at_keyword(Keyword::As) {
            self.advance()?; // AS
            self.expect_keyword(Keyword::Of)?;
            self.expect_keyword(Keyword::Version)?;
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(Statement::CreateVersion(Box::new(CreateVersionStatement {
            name,
            table,
            at_version,
        })))
    }

    // -----------------------------------------------------------------------
    // CDC: Replication Slots, CDC Streams, CDC Ingest, Publications
    // -----------------------------------------------------------------------

    fn parse_create_replication_slot(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Replication)?;
        self.expect_keyword(Keyword::Slot)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::Plugin)?;
        let plugin = match &self.current.token {
            Token::String(s) => {
                let v = s.clone();
                self.advance()?;
                v
            }
            _ => return Err(self.error("Expected plugin name string after PLUGIN")),
        };
        let mut table_filter = vec![];
        if self.consume_keyword(Keyword::For)? {
            self.expect_keyword(Keyword::Table)?;
            table_filter = self.parse_comma_separated(|p| p.parse_ident())?;
        }
        Ok(Statement::CreateReplicationSlot(Box::new(
            CreateReplicationSlotStatement {
                name,
                plugin,
                table_filter,
            },
        )))
    }

    fn parse_drop_replication_slot(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Replication)?;
        self.expect_keyword(Keyword::Slot)?;
        let name = self.parse_ident()?;
        Ok(Statement::DropReplicationSlot(Box::new(
            DropReplicationSlotStatement { name },
        )))
    }

    fn parse_create_cdc(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Cdc)?;
        match &self.current.token {
            Token::Keyword(Keyword::Stream) => self.parse_create_cdc_stream(),
            Token::Keyword(Keyword::Ingest) => self.parse_create_cdc_ingest(),
            _ => Err(self.error("Expected STREAM or INGEST after CREATE CDC")),
        }
    }

    fn parse_create_cdc_stream(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Stream)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table_name = self.parse_ident()?;
        self.expect_keyword(Keyword::To)?;
        let sink_type = self.parse_ident()?;
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            self.expect_token(&Token::LParen)?;
            options = self.parse_comma_separated(|p| p.parse_table_option())?;
            self.expect_token(&Token::RParen)?;
        }
        Ok(Statement::CreateCdcStream(Box::new(
            CreateCdcStreamStatement {
                name,
                table_name,
                sink_type,
                options,
            },
        )))
    }

    fn parse_create_cdc_ingest(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Ingest)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::From)?;
        let source_type = self.parse_ident()?;
        self.expect_keyword(Keyword::Into)?;
        let target_table = self.parse_ident()?;
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            self.expect_token(&Token::LParen)?;
            options = self.parse_comma_separated(|p| p.parse_table_option())?;
            self.expect_token(&Token::RParen)?;
        }
        Ok(Statement::CreateCdcIngest(Box::new(
            CreateCdcIngestStatement {
                name,
                source_type,
                target_table,
                options,
            },
        )))
    }

    fn parse_drop_cdc(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Cdc)?;
        match &self.current.token {
            Token::Keyword(Keyword::Stream) => {
                self.advance()?;
                let name = self.parse_ident()?;
                Ok(Statement::DropCdcStream(Box::new(DropCdcStreamStatement {
                    name,
                })))
            }
            Token::Keyword(Keyword::Ingest) => {
                self.advance()?;
                let name = self.parse_ident()?;
                Ok(Statement::DropCdcIngest(Box::new(DropCdcIngestStatement {
                    name,
                })))
            }
            _ => Err(self.error("Expected STREAM or INGEST after DROP CDC")),
        }
    }

    fn parse_create_publication(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Publication)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::For)?;
        let mut tables = vec![];
        let mut all_tables = false;
        if self.consume_keyword(Keyword::All)? {
            self.expect_keyword(Keyword::Table)?;
            all_tables = true;
        } else {
            self.expect_keyword(Keyword::Table)?;
            tables = self.parse_comma_separated(|p| p.parse_ident())?;
        }
        let include_ddl = if self.consume_keyword(Keyword::Include)? {
            self.expect_keyword(Keyword::Ddl)?;
            true
        } else {
            false
        };
        Ok(Statement::CreatePublication(Box::new(
            CreatePublicationStatement {
                name,
                tables,
                all_tables,
                include_ddl,
            },
        )))
    }

    fn parse_alter_publication(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Publication)?;
        let name = self.parse_ident()?;
        let action = if self.consume_keyword(Keyword::Add)? {
            self.expect_keyword(Keyword::Table)?;
            let table = self.parse_ident()?;
            AlterPublicationAction::AddTable(table)
        } else if self.at_keyword(Keyword::Drop) {
            self.advance()?;
            self.expect_keyword(Keyword::Table)?;
            let table = self.parse_ident()?;
            AlterPublicationAction::DropTable(table)
        } else {
            return Err(self.error("Expected ADD or DROP after ALTER PUBLICATION name"));
        };
        Ok(Statement::AlterPublication(Box::new(
            AlterPublicationStatement { name, action },
        )))
    }

    fn parse_drop_publication(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Publication)?;
        let name = self.parse_ident()?;
        Ok(Statement::DropPublication(Box::new(
            DropPublicationStatement { name },
        )))
    }

    // -----------------------------------------------------------------------
    // Triggers
    // -----------------------------------------------------------------------

    /// CREATE TRIGGER name {BEFORE|AFTER|INSTEAD OF} {INSERT|UPDATE|DELETE|TRUNCATE} [OR ...]
    /// ON table [REFERENCING OLD TABLE AS name NEW TABLE AS name]
    /// FOR EACH {ROW|STATEMENT} [WHEN (condition)]
    /// [PRIORITY n] EXECUTE FUNCTION func_name(args...)
    fn parse_create_trigger(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Trigger)?;
        let name = self.parse_ident()?;

        // Parse timing: BEFORE | AFTER | INSTEAD OF
        let timing = if self.consume_keyword(Keyword::Before)? {
            TriggerTiming::Before
        } else if self.consume_keyword(Keyword::After)? {
            TriggerTiming::After
        } else if self.consume_keyword(Keyword::Instead)? {
            self.expect_keyword(Keyword::Of)?;
            TriggerTiming::InsteadOf
        } else {
            return Err(self.error("Expected BEFORE, AFTER, or INSTEAD OF"));
        };

        // Parse events: INSERT [OR UPDATE [OR DELETE [OR TRUNCATE]]]
        let mut events = vec![self.parse_trigger_event()?];
        while self.consume_keyword(Keyword::Or)? {
            events.push(self.parse_trigger_event()?);
        }

        self.expect_keyword(Keyword::On)?;
        let table = self.parse_ident()?;

        // Optional REFERENCING clause
        let referencing = if self.consume_keyword(Keyword::Referencing)? {
            let mut old_table = None;
            let mut new_table = None;
            // Can have OLD TABLE AS name and/or NEW TABLE AS name in any order
            for _ in 0..2 {
                if self.consume_keyword(Keyword::Old)? {
                    self.expect_keyword(Keyword::Table)?;
                    self.expect_keyword(Keyword::As)?;
                    old_table = Some(self.parse_ident()?);
                } else if self.consume_keyword(Keyword::New)? {
                    self.expect_keyword(Keyword::Table)?;
                    self.expect_keyword(Keyword::As)?;
                    new_table = Some(self.parse_ident()?);
                } else {
                    break;
                }
            }
            Some(TransitionTables {
                old_table,
                new_table,
            })
        } else {
            None
        };

        // FOR EACH ROW | FOR EACH STATEMENT
        self.expect_keyword(Keyword::For)?;
        self.expect_keyword(Keyword::Each)?;
        let for_each = if self.consume_keyword(Keyword::Row)? {
            TriggerGranularity::Row
        } else {
            // "STATEMENT" is not a keyword, parse as ident
            let ident = self.parse_ident()?;
            if ident.eq_ignore_ascii_case("statement") {
                TriggerGranularity::Statement
            } else {
                return Err(self.error("Expected ROW or STATEMENT after FOR EACH"));
            }
        };

        // Optional WHEN (condition)
        let when_condition = if self.consume_keyword(Keyword::When)? {
            self.expect_token(&Token::LParen)?;
            let expr = self.parse_expr()?;
            self.expect_token(&Token::RParen)?;
            Some(Box::new(expr))
        } else {
            None
        };

        // Optional PRIORITY n
        let priority = if self.consume_keyword(Keyword::Priority)? {
            Some(self.parse_integer_value()? as u32)
        } else {
            None
        };

        // EXECUTE FUNCTION func_name(args...)
        self.expect_keyword(Keyword::Execute)?;
        self.expect_keyword(Keyword::Function)?;
        let execute_function = self.parse_ident()?;

        let mut args = Vec::new();
        if self.current.token == Token::LParen {
            self.advance()?;
            if self.current.token != Token::RParen {
                args = self.parse_comma_separated(|p| p.parse_expr())?;
            }
            self.expect_token(&Token::RParen)?;
        }

        Ok(Statement::CreateTrigger(Box::new(CreateTriggerStatement {
            name,
            timing,
            events,
            table,
            for_each,
            when_condition,
            referencing,
            execute_function,
            args,
            priority,
            enabled: true,
        })))
    }

    fn parse_trigger_event(&mut self) -> Result<TriggerEvent> {
        if self.consume_keyword(Keyword::Insert)? {
            Ok(TriggerEvent::Insert)
        } else if self.consume_keyword(Keyword::Update)? {
            Ok(TriggerEvent::Update)
        } else if self.consume_keyword(Keyword::Delete)? {
            Ok(TriggerEvent::Delete)
        } else if self.consume_keyword(Keyword::Truncate)? {
            Ok(TriggerEvent::Truncate)
        } else {
            Err(self.error("Expected INSERT, UPDATE, DELETE, or TRUNCATE"))
        }
    }

    /// DROP TRIGGER [IF EXISTS] name ON table [CASCADE|RESTRICT]
    fn parse_drop_trigger(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Trigger)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table = self.parse_ident()?;
        let drop_behavior = self.parse_optional_drop_behavior()?;
        Ok(Statement::DropTrigger(Box::new(DropTriggerStatement {
            name,
            table,
            if_exists,
            drop_behavior,
        })))
    }

    fn parse_optional_drop_behavior(&mut self) -> Result<Option<DropBehavior>> {
        if self.consume_keyword(Keyword::Cascade)? {
            Ok(Some(DropBehavior::Cascade))
        } else if self.consume_keyword(Keyword::Restrict)? {
            Ok(Some(DropBehavior::Restrict))
        } else {
            Ok(None)
        }
    }

    // -----------------------------------------------------------------------
    // User-defined functions
    // -----------------------------------------------------------------------

    /// CREATE [OR REPLACE] FUNCTION name(params) RETURNS type
    /// AS $$ body $$ LANGUAGE {SQL|RUST|RUST_VECTORIZED} [IMMUTABLE|STABLE|VOLATILE]
    /// [LIBRARY 'path' SYMBOL 'name']
    fn parse_create_function(&mut self, or_replace: bool) -> Result<Statement> {
        self.expect_keyword(Keyword::Function)?;
        let name = self.parse_ident()?;

        // Parse parameters
        self.expect_token(&Token::LParen)?;
        let params = if self.current.token == Token::RParen {
            Vec::new()
        } else {
            self.parse_comma_separated(|p| p.parse_function_param())?
        };
        self.expect_token(&Token::RParen)?;

        // RETURNS type | RETURNS TABLE(col type, ...) | RETURNS SETOF type
        self.expect_keyword(Keyword::Returns)?;
        let return_type = if self.consume_keyword(Keyword::Table)? {
            self.expect_token(&Token::LParen)?;
            let cols = self.parse_comma_separated(|p| p.parse_function_param())?;
            self.expect_token(&Token::RParen)?;
            FunctionReturnType::Table(cols)
        } else if self.consume_keyword(Keyword::Setof)? {
            FunctionReturnType::SetOf(self.parse_data_type()?)
        } else {
            FunctionReturnType::Scalar(self.parse_data_type()?)
        };

        // AS $$ body $$
        self.expect_keyword(Keyword::As)?;
        let body = self.parse_dollar_quoted_string()?;

        // LANGUAGE {SQL|RUST|RUST_VECTORIZED}
        self.expect_keyword(Keyword::Language)?;
        let language = if self.consume_keyword(Keyword::Sql)? {
            FunctionLanguage::Sql
        } else if self.consume_keyword(Keyword::RustVectorized)? {
            FunctionLanguage::RustVectorized
        } else if self.consume_keyword(Keyword::Rust)? {
            FunctionLanguage::Rust
        } else {
            return Err(self.error("Expected SQL, RUST, or RUST_VECTORIZED"));
        };

        // Optional volatility: IMMUTABLE | STABLE | VOLATILE
        let volatility = if self.consume_keyword(Keyword::Immutable)? {
            Volatility::Immutable
        } else if self.consume_keyword(Keyword::Stable)? {
            Volatility::Stable
        } else if self.consume_keyword(Keyword::Volatile)? {
            Volatility::Volatile
        } else {
            Volatility::Volatile // default
        };

        // Optional LIBRARY 'path' SYMBOL 'symbol'
        let mut rust_library = None;
        let mut rust_symbol = None;
        if self.at_ident_eq("library") {
            self.advance()?;
            rust_library = Some(self.parse_string_literal()?);
            self.expect_keyword(Keyword::Symbol)?;
            rust_symbol = Some(self.parse_string_literal()?);
        }

        Ok(Statement::CreateFunction(Box::new(
            CreateFunctionStatement {
                name,
                or_replace,
                params,
                return_type,
                language,
                body,
                volatility,
                rust_library,
                rust_symbol,
            },
        )))
    }

    fn parse_function_param(&mut self) -> Result<FunctionParam> {
        let name = self.parse_ident()?;
        let data_type = self.parse_data_type()?;
        let default_value = if self.consume_keyword(Keyword::Default)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        Ok(FunctionParam {
            name,
            data_type,
            default_value,
        })
    }

    /// Parse a dollar-quoted string: $$ ... $$ or $tag$ ... $tag$
    fn parse_dollar_quoted_string(&mut self) -> Result<String> {
        // Expect current token to be a dollar-quoted string or a string literal
        match &self.current.token {
            Token::String(s) => {
                let body = s.clone();
                self.advance()?;
                Ok(body)
            }
            _ => {
                // Try to consume $$ delimited content from raw token stream
                // For simplicity, also accept a regular string literal
                Err(self.error(
                    "Expected a dollar-quoted string ($$...$$) or string literal for function body",
                ))
            }
        }
    }

    fn at_ident_eq(&self, expected: &str) -> bool {
        match &self.current.token {
            Token::Ident(s) => s.eq_ignore_ascii_case(expected),
            Token::Keyword(kw) => {
                keyword_to_ident_str(*kw).map_or(false, |s| s.eq_ignore_ascii_case(expected))
            }
            _ => false,
        }
    }

    /// DROP FUNCTION [IF EXISTS] name [CASCADE|RESTRICT]
    fn parse_drop_function(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Function)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        let drop_behavior = self.parse_optional_drop_behavior()?;
        Ok(Statement::DropFunction(Box::new(DropFunctionStatement {
            name,
            if_exists,
            drop_behavior,
        })))
    }

    // -----------------------------------------------------------------------
    // User-defined aggregates
    // -----------------------------------------------------------------------

    /// CREATE AGGREGATE name(params) (
    ///   SFUNC = sfunc_name,
    ///   STYPE = state_type
    ///   [, FINALFUNC = finalfunc_name]
    ///   [, COMBINEFUNC = combinefunc_name]
    ///   [, INITCOND = 'initial_value']
    /// )
    fn parse_create_aggregate(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Aggregate)?;
        let name = self.parse_ident()?;

        // Parse input parameters
        self.expect_token(&Token::LParen)?;
        let params = if self.current.token == Token::RParen {
            Vec::new()
        } else {
            self.parse_comma_separated(|p| p.parse_function_param())?
        };
        self.expect_token(&Token::RParen)?;

        // Parse aggregate options in parentheses
        self.expect_token(&Token::LParen)?;

        let mut sfunc = None;
        let mut stype = None;
        let mut finalfunc = None;
        let mut combinefunc = None;
        let mut initcond = None;

        loop {
            if self.consume_keyword(Keyword::Sfunc)? {
                self.expect_token(&Token::Eq)?;
                sfunc = Some(self.parse_ident()?);
            } else if self.consume_keyword(Keyword::Stype)? {
                self.expect_token(&Token::Eq)?;
                stype = Some(self.parse_data_type()?);
            } else if self.consume_keyword(Keyword::Finalfunc)? {
                self.expect_token(&Token::Eq)?;
                finalfunc = Some(self.parse_ident()?);
            } else if self.consume_keyword(Keyword::Combinefunc)? {
                self.expect_token(&Token::Eq)?;
                combinefunc = Some(self.parse_ident()?);
            } else if self.consume_keyword(Keyword::Initcond)? {
                self.expect_token(&Token::Eq)?;
                initcond = Some(self.parse_string_literal()?);
            } else {
                break;
            }
            // Consume optional comma between options
            if self.current.token == Token::Comma {
                self.advance()?;
            }
        }
        self.expect_token(&Token::RParen)?;

        let sfunc = sfunc.ok_or_else(|| self.error("SFUNC is required in CREATE AGGREGATE"))?;
        let stype = stype.ok_or_else(|| self.error("STYPE is required in CREATE AGGREGATE"))?;

        Ok(Statement::CreateAggregate(Box::new(
            CreateAggregateStatement {
                name,
                params,
                sfunc,
                stype,
                finalfunc,
                combinefunc,
                initcond,
            },
        )))
    }

    /// DROP AGGREGATE [IF EXISTS] name [CASCADE|RESTRICT]
    fn parse_drop_aggregate(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Aggregate)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        let drop_behavior = self.parse_optional_drop_behavior()?;
        Ok(Statement::DropAggregate(Box::new(DropAggregateStatement {
            name,
            if_exists,
            drop_behavior,
        })))
    }

    // -----------------------------------------------------------------------
    // Stored procedures
    // -----------------------------------------------------------------------

    /// CREATE [OR REPLACE] PROCEDURE name(params)
    /// AS $$ body $$ LANGUAGE {SQL|PLSQL|RUST}
    /// [SECURITY {DEFINER|INVOKER}]
    fn parse_create_procedure(&mut self, or_replace: bool) -> Result<Statement> {
        self.expect_keyword(Keyword::Procedure)?;
        let name = self.parse_ident()?;

        // Parse parameters
        self.expect_token(&Token::LParen)?;
        let params = if self.current.token == Token::RParen {
            Vec::new()
        } else {
            self.parse_comma_separated(|p| p.parse_function_param())?
        };
        self.expect_token(&Token::RParen)?;

        // AS $$ body $$
        self.expect_keyword(Keyword::As)?;
        let body = self.parse_dollar_quoted_string()?;

        // LANGUAGE {SQL|PLSQL|RUST}
        self.expect_keyword(Keyword::Language)?;
        let language = if self.consume_keyword(Keyword::Sql)? {
            ProcedureLanguage::Sql
        } else if self.consume_keyword(Keyword::Plsql)? {
            ProcedureLanguage::PlSql
        } else if self.consume_keyword(Keyword::Rust)? {
            ProcedureLanguage::Rust
        } else {
            return Err(self.error("Expected SQL, PLSQL, or RUST"));
        };

        // Optional SECURITY {DEFINER|INVOKER}
        let security = if self.consume_keyword(Keyword::Security)? {
            if self.consume_keyword(Keyword::Definer)? {
                SecurityMode::Definer
            } else if self.consume_keyword(Keyword::Invoker)? {
                SecurityMode::Invoker
            } else {
                return Err(self.error("Expected DEFINER or INVOKER after SECURITY"));
            }
        } else {
            SecurityMode::Invoker // default
        };

        Ok(Statement::CreateProcedure(Box::new(
            CreateProcedureStatement {
                name,
                or_replace,
                params,
                language,
                body,
                security,
            },
        )))
    }

    /// DROP PROCEDURE [IF EXISTS] name [CASCADE|RESTRICT]
    fn parse_drop_procedure(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Procedure)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        let drop_behavior = self.parse_optional_drop_behavior()?;
        Ok(Statement::DropProcedure(Box::new(DropProcedureStatement {
            name,
            if_exists,
            drop_behavior,
        })))
    }

    /// CALL procedure_name(args...)
    fn parse_call(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Call)?;
        let name = self.parse_ident()?;
        self.expect_token(&Token::LParen)?;
        let args = if self.current.token == Token::RParen {
            Vec::new()
        } else {
            self.parse_comma_separated(|p| p.parse_expr())?
        };
        self.expect_token(&Token::RParen)?;
        Ok(Statement::Call(Box::new(CallStatement { name, args })))
    }

    // -----------------------------------------------------------------------
    // Event handlers
    // -----------------------------------------------------------------------

    /// CREATE EVENT HANDLER name WHEN event_type EXECUTE FUNCTION func_name
    fn parse_create_event_handler(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Event)?;
        self.expect_keyword(Keyword::Handler)?;
        let name = self.parse_ident()?;

        self.expect_keyword(Keyword::When)?;
        let event_type = self.parse_ident()?;

        let condition = if self.consume_keyword(Keyword::And)? {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        self.expect_keyword(Keyword::Execute)?;
        self.expect_keyword(Keyword::Function)?;
        let execute_function = self.parse_ident()?;

        Ok(Statement::CreateEventHandler(Box::new(
            CreateEventHandlerStatement {
                name,
                event_type,
                condition,
                execute_function,
            },
        )))
    }

    /// DROP EVENT HANDLER [IF EXISTS] name
    fn parse_drop_event_handler(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Event)?;
        self.expect_keyword(Keyword::Handler)?;
        let if_exists = if self.consume_keyword(Keyword::If)? {
            self.expect_keyword(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_ident()?;
        Ok(Statement::DropEventHandler(Box::new(
            DropEventHandlerStatement { name, if_exists },
        )))
    }

    // -----------------------------------------------------------------------
    // ALTER TABLE extensions
    // -----------------------------------------------------------------------

    fn parse_alter_table_options(&mut self, table: String) -> Result<Statement> {
        self.expect_token(&Token::LParen)?;
        let options = self.parse_comma_separated(|p| p.parse_table_option())?;
        self.expect_token(&Token::RParen)?;
        Ok(Statement::AlterTableOptions(Box::new(
            AlterTableOptionsStatement { table, options },
        )))
    }

    fn parse_add_expectation(&mut self, table: String) -> Result<Statement> {
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::Expect)?;
        let expr = self.parse_expr()?;
        self.expect_keyword(Keyword::On)?;
        self.expect_keyword(Keyword::Violation)?;
        let on_violation = if self.consume_keyword(Keyword::Fail)? {
            ViolationAction::Fail
        } else if self.consume_keyword(Keyword::Quarantine)? {
            ViolationAction::Quarantine
        } else if self.at_keyword(Keyword::Drop) {
            self.advance()?;
            ViolationAction::Drop
        } else {
            // Default: treat unknown or WARN as Warn
            let _ = self.parse_ident();
            ViolationAction::Warn
        };
        Ok(Statement::AddExpectation(Box::new(
            AddExpectationStatement {
                table,
                name,
                expr,
                on_violation,
            },
        )))
    }

    fn parse_drop_expectation(&mut self, table: String) -> Result<Statement> {
        let name = self.parse_ident()?;
        Ok(Statement::DropExpectation(Box::new(
            DropExpectationStatement { table, name },
        )))
    }

    fn parse_enable_feature(&mut self, table: String) -> Result<Statement> {
        let feature = self.parse_ident()?;
        Ok(Statement::EnableFeature(Box::new(EnableFeatureStatement {
            table,
            feature,
        })))
    }

    fn parse_disable_feature(&mut self, table: String) -> Result<Statement> {
        let feature = self.parse_ident()?;
        Ok(Statement::DisableFeature(Box::new(
            DisableFeatureStatement { table, feature },
        )))
    }

    // -----------------------------------------------------------------------
    // Search indexes
    // -----------------------------------------------------------------------

    fn parse_create_fulltext_index(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Fulltext)?;
        self.expect_keyword(Keyword::Index)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table = self.parse_ident()?;
        self.expect_token(&Token::LParen)?;
        let columns = self.parse_comma_separated(|p| p.parse_ident())?;
        self.expect_token(&Token::RParen)?;
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            self.expect_token(&Token::LParen)?;
            options = self.parse_comma_separated(|p| p.parse_table_option())?;
            self.expect_token(&Token::RParen)?;
        }
        Ok(Statement::CreateFulltextIndex(Box::new(
            CreateFulltextIndexStatement {
                name,
                table,
                columns,
                options,
            },
        )))
    }

    fn parse_create_vector_index(&mut self) -> Result<Statement> {
        self.expect_keyword(Keyword::Vector)?;
        self.expect_keyword(Keyword::Index)?;
        let name = self.parse_ident()?;
        self.expect_keyword(Keyword::On)?;
        let table = self.parse_ident()?;
        self.expect_token(&Token::LParen)?;
        let column = self.parse_ident()?;
        self.expect_token(&Token::RParen)?;
        let mut options = vec![];
        if self.consume_keyword(Keyword::With)? {
            self.expect_token(&Token::LParen)?;
            options = self.parse_comma_separated(|p| p.parse_table_option())?;
            self.expect_token(&Token::RParen)?;
        }
        Ok(Statement::CreateVectorIndex(Box::new(
            CreateVectorIndexStatement {
                name,
                table,
                column,
                options,
            },
        )))
    }
}

/// Maps keywords that can be used as identifiers in non-keyword position.
/// Returns the string representation, or None if the keyword cannot be used as an identifier.
fn keyword_to_ident_str(kw: Keyword) -> Option<&'static str> {
    match kw {
        // Data type keywords commonly used as identifiers
        Keyword::Type => Some("type"),
        Keyword::Column => Some("column"),
        Keyword::First => Some("first"),
        Keyword::Last => Some("last"),
        Keyword::Key => Some("key"),
        Keyword::Values => Some("values"),
        Keyword::Set => Some("set"),
        Keyword::Add => Some("add"),
        Keyword::Index => Some("index"),
        Keyword::Cascade => Some("cascade"),
        Keyword::Restrict => Some("restrict"),
        Keyword::Release => Some("release"),
        Keyword::Transaction => Some("transaction"),
        Keyword::Analyze => Some("analyze"),
        Keyword::Savepoint => Some("savepoint"),
        // Data type keywords that may appear as identifiers
        Keyword::Int => Some("int"),
        Keyword::Integer => Some("integer"),
        Keyword::Smallint => Some("smallint"),
        Keyword::Bigint => Some("bigint"),
        Keyword::Real => Some("real"),
        Keyword::Float => Some("float"),
        Keyword::Boolean => Some("boolean"),
        Keyword::Char => Some("char"),
        Keyword::Varchar => Some("varchar"),
        Keyword::Text => Some("text"),
        Keyword::Decimal => Some("decimal"),
        Keyword::Numeric => Some("numeric"),
        Keyword::Date => Some("date"),
        Keyword::Time => Some("time"),
        Keyword::Timestamp => Some("timestamp"),
        Keyword::Timestamptz => Some("timestamptz"),
        Keyword::Interval => Some("interval"),
        Keyword::Uuid => Some("uuid"),
        Keyword::Json => Some("json"),
        Keyword::Jsonb => Some("jsonb"),
        Keyword::Binary => Some("binary"),
        Keyword::Varbinary => Some("varbinary"),
        Keyword::Bytea => Some("bytea"),
        Keyword::Double => Some("double"),
        Keyword::Precision => Some("precision"),
        Keyword::Zone => Some("zone"),
        Keyword::To => Some("to"),
        Keyword::Rename => Some("rename"),
        // Window function keywords
        Keyword::Over => Some("over"),
        Keyword::Partition => Some("partition"),
        Keyword::Rows => Some("rows"),
        Keyword::Range => Some("range"),
        Keyword::Unbounded => Some("unbounded"),
        Keyword::Preceding => Some("preceding"),
        Keyword::Following => Some("following"),
        Keyword::Current => Some("current"),
        Keyword::Row => Some("row"),
        // CTE keyword
        Keyword::Recursive => Some("recursive"),
        // Set operation keywords
        Keyword::Union => Some("union"),
        Keyword::Intersect => Some("intersect"),
        Keyword::Except => Some("except"),
        // View keywords
        Keyword::View => Some("view"),
        Keyword::Replace => Some("replace"),
        // DCL keywords
        Keyword::Grant => Some("grant"),
        Keyword::Revoke => Some("revoke"),
        Keyword::Privileges => Some("privileges"),
        Keyword::Public => Some("public"),
        // RETURNING
        Keyword::Returning => Some("returning"),
        // ON CONFLICT keywords
        Keyword::Conflict => Some("conflict"),
        Keyword::Do => Some("do"),
        Keyword::Nothing => Some("nothing"),
        // Join keywords
        Keyword::Natural => Some("natural"),
        Keyword::Using => Some("using"),
        // Schema keywords
        Keyword::Schema => Some("schema"),
        // Sequence keywords
        Keyword::Sequence => Some("sequence"),
        Keyword::Increment => Some("increment"),
        Keyword::Minvalue => Some("minvalue"),
        Keyword::Maxvalue => Some("maxvalue"),
        Keyword::Start => Some("start"),
        Keyword::Cache => Some("cache"),
        Keyword::Cycle => Some("cycle"),
        Keyword::No => Some("no"),
        // Maintenance keywords
        Keyword::Vacuum => Some("vacuum"),
        Keyword::Reindex => Some("reindex"),
        // Session keywords
        Keyword::Show => Some("show"),
        // Copy keywords
        Keyword::Copy => Some("copy"),
        Keyword::Stdin => Some("stdin"),
        Keyword::Stdout => Some("stdout"),
        // Row locking
        Keyword::For => Some("for"),
        Keyword::Share => Some("share"),
        Keyword::Lock => Some("lock"),
        Keyword::Nowait => Some("nowait"),
        Keyword::Skip => Some("skip"),
        Keyword::Locked => Some("locked"),
        // MERGE
        Keyword::Merge => Some("merge"),
        Keyword::Matched => Some("matched"),
        // Prepared statements
        Keyword::Prepare => Some("prepare"),
        Keyword::Execute => Some("execute"),
        Keyword::Deallocate => Some("deallocate"),
        // FETCH
        Keyword::Fetch => Some("fetch"),
        Keyword::Next => Some("next"),
        Keyword::Only => Some("only"),
        Keyword::Percent => Some("percent"),
        // LATERAL
        Keyword::Lateral => Some("lateral"),
        // Array
        Keyword::Array => Some("array"),
        Keyword::Any => Some("any"),
        Keyword::Some => Some("some"),
        // LISTEN/NOTIFY
        Keyword::Listen => Some("listen"),
        Keyword::Notify => Some("notify"),
        Keyword::Payload => Some("payload"),
        // TABLESAMPLE
        Keyword::Tablesample => Some("tablesample"),
        Keyword::Bernoulli => Some("bernoulli"),
        Keyword::System => Some("system"),
        // Cursors
        Keyword::Declare => Some("declare"),
        Keyword::Cursor => Some("cursor"),
        Keyword::Close => Some("close"),
        Keyword::Scroll => Some("scroll"),
        Keyword::Hold => Some("hold"),
        Keyword::Without => Some("without"),
        Keyword::Absolute => Some("absolute"),
        Keyword::Relative => Some("relative"),
        Keyword::Forward => Some("forward"),
        Keyword::Backward => Some("backward"),
        // COMMENT ON
        Keyword::Comment => Some("comment"),
        // Materialized views
        Keyword::Refresh => Some("refresh"),
        Keyword::Materialized => Some("materialized"),
        // CHECKPOINT
        Keyword::Checkpoint => Some("checkpoint"),
        // DO block
        Keyword::Language => Some("language"),
        // ZyronDB custom
        Keyword::Segments => Some("segments"),
        Keyword::Status => Some("status"),
        Keyword::Buffer => Some("buffer"),
        Keyword::Pool => Some("pool"),
        Keyword::Transactions => Some("transactions"),
        Keyword::Format => Some("format"),
        Keyword::Storage => Some("storage"),
        Keyword::Columnar => Some("columnar"),
        Keyword::Heap => Some("heap"),
        // TTL / data retention
        Keyword::Ttl => Some("ttl"),
        Keyword::Days => Some("days"),
        Keyword::Hours => Some("hours"),
        Keyword::Minutes => Some("minutes"),
        Keyword::Seconds => Some("seconds"),
        Keyword::Archive => Some("archive"),
        Keyword::Retain => Some("retain"),
        Keyword::Expire => Some("expire"),
        // Scheduling
        Keyword::Schedule => Some("schedule"),
        Keyword::Every => Some("every"),
        Keyword::Pause => Some("pause"),
        Keyword::Resume => Some("resume"),
        Keyword::Cron => Some("cron"),
        // Optimize
        Keyword::Optimize => Some("optimize"),
        // GROUP BY extensions
        Keyword::Rollup => Some("rollup"),
        Keyword::Cube => Some("cube"),
        Keyword::Grouping => Some("grouping"),
        Keyword::Sets => Some("sets"),
        Keyword::Qualify => Some("qualify"),
        // Time travel
        Keyword::Versioning => Some("versioning"),
        Keyword::Period => Some("period"),
        Keyword::Of => Some("of"),
        // Pipeline / Medallion
        Keyword::Pipeline => Some("pipeline"),
        Keyword::Stage => Some("stage"),
        Keyword::Source => Some("source"),
        Keyword::Target => Some("target"),
        Keyword::Mode => Some("mode"),
        Keyword::Transform => Some("transform"),
        Keyword::Expect => Some("expect"),
        Keyword::Expectation => Some("expectation"),
        Keyword::Violation => Some("violation"),
        Keyword::Fail => Some("fail"),
        Keyword::Quarantine => Some("quarantine"),
        Keyword::Run => Some("run"),
        Keyword::Restore => Some("restore"),
        // Security
        Keyword::User => Some("user"),
        Keyword::Role => Some("role"),
        Keyword::Password => Some("password"),
        Keyword::Login => Some("login"),
        Keyword::Superuser => Some("superuser"),
        Keyword::Valid => Some("valid"),
        Keyword::Until => Some("until"),
        Keyword::Nologin => Some("nologin"),
        // Search indexes
        Keyword::Fulltext => Some("fulltext"),
        Keyword::Match => Some("match"),
        Keyword::Against => Some("against"),
        // Feature toggles
        Keyword::Enable => Some("enable"),
        Keyword::Disable => Some("disable"),
        Keyword::Feed => Some("feed"),
        Keyword::Change => Some("change"),
        // CDC
        Keyword::Replication => Some("replication"),
        Keyword::Slot => Some("slot"),
        Keyword::Plugin => Some("plugin"),
        Keyword::Cdc => Some("cdc"),
        Keyword::Stream => Some("stream"),
        Keyword::Ingest => Some("ingest"),
        Keyword::Publication => Some("publication"),
        Keyword::Include => Some("include"),
        Keyword::Ddl => Some("ddl"),
        // New data types
        Keyword::Tinyint => Some("tinyint"),
        Keyword::Int128 => Some("int128"),
        Keyword::Uint8 => Some("uint8"),
        Keyword::Uint16 => Some("uint16"),
        Keyword::Uint32 => Some("uint32"),
        Keyword::Uint64 => Some("uint64"),
        Keyword::Uint128 => Some("uint128"),
        Keyword::Vector => Some("vector"),
        // Branching / Versioning
        Keyword::Branch => Some("branch"),
        Keyword::Version => Some("version"),
        Keyword::Portion => Some("portion"),
        Keyword::Use => Some("use"),
        Keyword::At => Some("at"),
        // Triggers, UDFs, procedures, aggregates
        Keyword::Trigger => Some("trigger"),
        Keyword::Before => Some("before"),
        Keyword::After => Some("after"),
        Keyword::Instead => Some("instead"),
        Keyword::Each => Some("each"),
        Keyword::Referencing => Some("referencing"),
        Keyword::Old => Some("old"),
        Keyword::New => Some("new"),
        Keyword::Priority => Some("priority"),
        Keyword::Function => Some("function"),
        Keyword::Returns => Some("returns"),
        Keyword::Immutable => Some("immutable"),
        Keyword::Volatile => Some("volatile"),
        Keyword::Stable => Some("stable"),
        Keyword::Setof => Some("setof"),
        Keyword::Procedure => Some("procedure"),
        Keyword::Call => Some("call"),
        Keyword::Aggregate => Some("aggregate"),
        Keyword::Sfunc => Some("sfunc"),
        Keyword::Stype => Some("stype"),
        Keyword::Finalfunc => Some("finalfunc"),
        Keyword::Combinefunc => Some("combinefunc"),
        Keyword::Initcond => Some("initcond"),
        Keyword::Definer => Some("definer"),
        Keyword::Security => Some("security"),
        Keyword::Invoker => Some("invoker"),
        Keyword::Handler => Some("handler"),
        Keyword::Event => Some("event"),
        Keyword::Preview => Some("preview"),
        Keyword::Plsql => Some("plsql"),
        Keyword::Sql => Some("sql"),
        Keyword::Symbol => Some("symbol"),
        Keyword::Rust => Some("rust"),
        Keyword::RustVectorized => Some("rust_vectorized"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_one(sql: &str) -> Statement {
        let mut parser = Parser::new(sql).unwrap();
        parser.parse_statement().unwrap()
    }

    fn parse_all(sql: &str) -> Vec<Statement> {
        let mut parser = Parser::new(sql).unwrap();
        parser.parse_statements().unwrap()
    }

    fn parse_err(sql: &str) -> String {
        let mut parser = Parser::new(sql).unwrap();
        parser.parse_statement().unwrap_err().to_string()
    }

    // -----------------------------------------------------------------------
    // SELECT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_star() {
        let stmt = parse_one("SELECT * FROM users");
        match stmt {
            Statement::Select(s) => {
                assert!(!s.distinct);
                assert_eq!(s.projections, vec![SelectItem::Wildcard]);
                assert_eq!(s.from.len(), 1);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_columns() {
        let stmt = parse_one("SELECT a, b, c FROM t");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.projections.len(), 3);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_distinct() {
        let stmt = parse_one("SELECT DISTINCT name FROM users");
        match stmt {
            Statement::Select(s) => {
                assert!(s.distinct);
                assert_eq!(s.projections.len(), 1);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_with_alias() {
        let stmt = parse_one("SELECT a AS x, b y FROM t");
        match stmt {
            Statement::Select(s) => {
                assert!(
                    matches!(&s.projections[0], SelectItem::Expr(_, Some(alias)) if alias == "x")
                );
                assert!(
                    matches!(&s.projections[1], SelectItem::Expr(_, Some(alias)) if alias == "y")
                );
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_qualified_wildcard() {
        let stmt = parse_one("SELECT t.* FROM t");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(&s.projections[0], SelectItem::QualifiedWildcard(t) if t == "t"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_where() {
        let stmt = parse_one("SELECT * FROM t WHERE x > 5");
        match stmt {
            Statement::Select(s) => {
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_order_by() {
        let stmt = parse_one("SELECT * FROM t ORDER BY a ASC, b DESC NULLS FIRST");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.order_by.len(), 2);
                assert_eq!(s.order_by[0].asc, Some(true));
                assert_eq!(s.order_by[1].asc, Some(false));
                assert_eq!(s.order_by[1].nulls_first, Some(true));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_limit_offset() {
        let stmt = parse_one("SELECT * FROM t LIMIT 10 OFFSET 20");
        match stmt {
            Statement::Select(s) => {
                assert!(s.limit.is_some());
                assert!(s.offset.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_group_by_having() {
        let stmt = parse_one("SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.group_by.len(), 1);
                assert!(s.having.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_expression_only() {
        let stmt = parse_one("SELECT 1 + 2");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.projections.len(), 1);
                assert!(s.from.is_empty());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // JOIN tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inner_join() {
        let stmt = parse_one("SELECT * FROM a INNER JOIN b ON a.id = b.a_id");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(&s.from[0], TableRef::Join(j) if j.join_type == JoinType::Inner));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_left_join() {
        let stmt = parse_one("SELECT * FROM a LEFT JOIN b ON a.id = b.a_id");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(&s.from[0], TableRef::Join(j) if j.join_type == JoinType::Left));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_right_join() {
        let stmt = parse_one("SELECT * FROM a RIGHT JOIN b ON a.id = b.a_id");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(&s.from[0], TableRef::Join(j) if j.join_type == JoinType::Right));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_full_join() {
        let stmt = parse_one("SELECT * FROM a FULL JOIN b ON a.id = b.a_id");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(&s.from[0], TableRef::Join(j) if j.join_type == JoinType::Full));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_cross_join() {
        let stmt = parse_one("SELECT * FROM a CROSS JOIN b");
        match stmt {
            Statement::Select(s) => {
                assert!(
                    matches!(&s.from[0], TableRef::Join(j) if j.join_type == JoinType::Cross && j.condition == JoinCondition::None)
                );
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_implicit_inner_join() {
        let stmt = parse_one("SELECT * FROM a JOIN b ON a.id = b.a_id");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(&s.from[0], TableRef::Join(j) if j.join_type == JoinType::Inner));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_table_alias() {
        let stmt = parse_one("SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id");
        match stmt {
            Statement::Select(s) => {
                if let TableRef::Join(j) = &s.from[0] {
                    assert!(matches!(&j.left, TableRef::Table { alias: Some(a), .. } if a == "u"));
                    assert!(matches!(&j.right, TableRef::Table { alias: Some(a), .. } if a == "o"));
                } else {
                    panic!("Expected JOIN");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // INSERT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_values() {
        let stmt = parse_one("INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25)");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "users");
                assert_eq!(i.columns, vec!["name", "age"]);
                match &i.source {
                    InsertSource::Values(values) => {
                        assert_eq!(values.len(), 2);
                        assert_eq!(values[0].len(), 2);
                    }
                    _ => panic!("Expected VALUES"),
                }
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_without_columns() {
        let stmt = parse_one("INSERT INTO t VALUES (1, 2, 3)");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "t");
                assert!(i.columns.is_empty());
                match &i.source {
                    InsertSource::Values(values) => assert_eq!(values.len(), 1),
                    _ => panic!("Expected VALUES"),
                }
            }
            _ => panic!("Expected INSERT"),
        }
    }

    // -----------------------------------------------------------------------
    // UPDATE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_update() {
        let stmt = parse_one("UPDATE users SET name = 'Alice', age = 30 WHERE id = 1");
        match stmt {
            Statement::Update(u) => {
                assert_eq!(u.table, "users");
                assert_eq!(u.assignments.len(), 2);
                assert_eq!(u.assignments[0].column, "name");
                assert!(u.where_clause.is_some());
            }
            _ => panic!("Expected UPDATE"),
        }
    }

    #[test]
    fn test_update_without_where() {
        let stmt = parse_one("UPDATE t SET x = 1");
        match stmt {
            Statement::Update(u) => {
                assert!(u.where_clause.is_none());
            }
            _ => panic!("Expected UPDATE"),
        }
    }

    // -----------------------------------------------------------------------
    // DELETE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_delete() {
        let stmt = parse_one("DELETE FROM users WHERE id = 1");
        match stmt {
            Statement::Delete(d) => {
                assert_eq!(d.table, "users");
                assert!(d.where_clause.is_some());
            }
            _ => panic!("Expected DELETE"),
        }
    }

    #[test]
    fn test_delete_without_where() {
        let stmt = parse_one("DELETE FROM t");
        match stmt {
            Statement::Delete(d) => {
                assert!(d.where_clause.is_none());
            }
            _ => panic!("Expected DELETE"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE TABLE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_table() {
        let stmt = parse_one(
            "CREATE TABLE users (
                id INT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email TEXT UNIQUE,
                age INT DEFAULT 0,
                active BOOLEAN
            )",
        );
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.name, "users");
                assert!(!ct.if_not_exists);
                assert_eq!(ct.columns.len(), 5);
                assert_eq!(ct.columns[0].name, "id");
                assert_eq!(ct.columns[0].data_type, DataType::Int);
                assert!(
                    ct.columns[0]
                        .constraints
                        .contains(&ColumnConstraint::PrimaryKey)
                );
                assert_eq!(ct.columns[1].data_type, DataType::Varchar(Some(255)));
                assert!(
                    ct.columns[1]
                        .constraints
                        .contains(&ColumnConstraint::NotNull)
                );
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let stmt = parse_one("CREATE TABLE IF NOT EXISTS t (id INT)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(ct.if_not_exists);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_create_table_with_constraints() {
        let stmt = parse_one(
            "CREATE TABLE orders (
                id INT,
                user_id INT,
                amount DECIMAL(10, 2),
                PRIMARY KEY (id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )",
        );
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 3);
                assert_eq!(ct.constraints.len(), 2);
                assert!(matches!(
                    &ct.constraints[0],
                    TableConstraint::PrimaryKey(cols) if cols == &["id"]
                ));
                assert!(matches!(
                    &ct.constraints[1],
                    TableConstraint::ForeignKey { ref_table, .. } if ref_table == "users"
                ));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_create_table_timestamp_with_time_zone() {
        let stmt = parse_one("CREATE TABLE t (created_at TIMESTAMP WITH TIME ZONE)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::TimestampTz);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_create_table_double_precision() {
        let stmt = parse_one("CREATE TABLE t (val DOUBLE PRECISION)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::DoublePrecision);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // DROP TABLE / INDEX tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drop_table() {
        let stmt = parse_one("DROP TABLE users");
        match stmt {
            Statement::DropTable(dt) => {
                assert_eq!(dt.name, "users");
                assert!(!dt.if_exists);
            }
            _ => panic!("Expected DROP TABLE"),
        }
    }

    #[test]
    fn test_drop_table_if_exists() {
        let stmt = parse_one("DROP TABLE IF EXISTS users");
        match stmt {
            Statement::DropTable(dt) => {
                assert!(dt.if_exists);
            }
            _ => panic!("Expected DROP TABLE"),
        }
    }

    #[test]
    fn test_create_index() {
        let stmt = parse_one("CREATE INDEX idx_name ON users (name)");
        match stmt {
            Statement::CreateIndex(ci) => {
                assert_eq!(ci.name, "idx_name");
                assert_eq!(ci.table, "users");
                assert!(!ci.unique);
                assert_eq!(ci.columns.len(), 1);
            }
            _ => panic!("Expected CREATE INDEX"),
        }
    }

    #[test]
    fn test_create_unique_index() {
        let stmt = parse_one("CREATE UNIQUE INDEX idx_email ON users (email)");
        match stmt {
            Statement::CreateIndex(ci) => {
                assert!(ci.unique);
            }
            _ => panic!("Expected CREATE INDEX"),
        }
    }

    #[test]
    fn test_drop_index() {
        let stmt = parse_one("DROP INDEX idx_name");
        match stmt {
            Statement::DropIndex(di) => {
                assert_eq!(di.name, "idx_name");
                assert!(!di.if_exists);
            }
            _ => panic!("Expected DROP INDEX"),
        }
    }

    #[test]
    fn test_drop_index_if_exists() {
        let stmt = parse_one("DROP INDEX IF EXISTS idx_name");
        match stmt {
            Statement::DropIndex(di) => {
                assert!(di.if_exists);
            }
            _ => panic!("Expected DROP INDEX"),
        }
    }

    // -----------------------------------------------------------------------
    // ALTER TABLE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_table_add_column() {
        let stmt = parse_one("ALTER TABLE users ADD COLUMN email VARCHAR(255) NOT NULL");
        match stmt {
            Statement::AlterTable(at) => {
                assert_eq!(at.name, "users");
                assert!(matches!(at.operation, AlterTableOperation::AddColumn(_)));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_add_column_without_keyword() {
        let stmt = parse_one("ALTER TABLE users ADD email TEXT");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(at.operation, AlterTableOperation::AddColumn(_)));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_drop_column() {
        let stmt = parse_one("ALTER TABLE users DROP COLUMN email");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::DropColumn { ref name, if_exists: false } if name == "email"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_drop_column_if_exists() {
        let stmt = parse_one("ALTER TABLE users DROP COLUMN IF EXISTS email");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::DropColumn {
                        if_exists: true,
                        ..
                    }
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_rename_column() {
        let stmt = parse_one("ALTER TABLE users RENAME COLUMN fname TO first_name");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::RenameColumn { ref old_name, ref new_name }
                    if old_name == "fname" && new_name == "first_name"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_rename_to() {
        let stmt = parse_one("ALTER TABLE users RENAME TO customers");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::RenameTable { ref new_name } if new_name == "customers"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_alter_column_set_default() {
        let stmt = parse_one("ALTER TABLE users ALTER COLUMN active SET DEFAULT TRUE");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::AlterColumnSetDefault { ref column, .. }
                    if column == "active"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_alter_column_drop_default() {
        let stmt = parse_one("ALTER TABLE users ALTER COLUMN active DROP DEFAULT");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::AlterColumnDropDefault { ref column }
                    if column == "active"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_alter_column_set_not_null() {
        let stmt = parse_one("ALTER TABLE users ALTER COLUMN name SET NOT NULL");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::AlterColumnSetNotNull { ref column }
                    if column == "name"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_alter_column_drop_not_null() {
        let stmt = parse_one("ALTER TABLE users ALTER COLUMN name DROP NOT NULL");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::AlterColumnDropNotNull { ref column }
                    if column == "name"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_alter_column_type() {
        let stmt = parse_one("ALTER TABLE users ALTER COLUMN age TYPE BIGINT");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::AlterColumnSetType { ref column, ref data_type }
                    if column == "age" && *data_type == DataType::BigInt
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_add_constraint() {
        let stmt = parse_one(
            "ALTER TABLE orders ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id)",
        );
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::AddConstraint(TableConstraint::ForeignKey { .. })
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    #[test]
    fn test_alter_table_drop_constraint() {
        let stmt = parse_one("ALTER TABLE orders DROP CONSTRAINT fk_user");
        match stmt {
            Statement::AlterTable(at) => {
                assert!(matches!(
                    at.operation,
                    AlterTableOperation::DropConstraint { ref name, if_exists: false }
                    if name == "fk_user"
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // TRUNCATE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncate() {
        let stmt = parse_one("TRUNCATE TABLE users");
        match stmt {
            Statement::Truncate(t) => {
                assert_eq!(t.table, "users");
            }
            _ => panic!("Expected TRUNCATE"),
        }
    }

    #[test]
    fn test_truncate_without_table_keyword() {
        let stmt = parse_one("TRUNCATE users");
        match stmt {
            Statement::Truncate(t) => {
                assert_eq!(t.table, "users");
            }
            _ => panic!("Expected TRUNCATE"),
        }
    }

    // -----------------------------------------------------------------------
    // Transaction control tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_begin() {
        let stmt = parse_one("BEGIN");
        assert!(matches!(stmt, Statement::Begin(_)));
    }

    #[test]
    fn test_begin_transaction() {
        let stmt = parse_one("BEGIN TRANSACTION");
        assert!(matches!(stmt, Statement::Begin(_)));
    }

    #[test]
    fn test_commit() {
        let stmt = parse_one("COMMIT");
        assert!(matches!(stmt, Statement::Commit(_)));
    }

    #[test]
    fn test_rollback() {
        let stmt = parse_one("ROLLBACK");
        match stmt {
            Statement::Rollback(r) => {
                assert!(r.savepoint.is_none());
            }
            _ => panic!("Expected ROLLBACK"),
        }
    }

    #[test]
    fn test_rollback_to_savepoint() {
        let stmt = parse_one("ROLLBACK TO SAVEPOINT sp1");
        match stmt {
            Statement::Rollback(r) => {
                assert_eq!(r.savepoint, Some("sp1".to_string()));
            }
            _ => panic!("Expected ROLLBACK"),
        }
    }

    #[test]
    fn test_rollback_to_without_savepoint_keyword() {
        let stmt = parse_one("ROLLBACK TO sp1");
        match stmt {
            Statement::Rollback(r) => {
                assert_eq!(r.savepoint, Some("sp1".to_string()));
            }
            _ => panic!("Expected ROLLBACK"),
        }
    }

    #[test]
    fn test_savepoint() {
        let stmt = parse_one("SAVEPOINT sp1");
        match stmt {
            Statement::Savepoint(s) => {
                assert_eq!(s.name, "sp1");
            }
            _ => panic!("Expected SAVEPOINT"),
        }
    }

    #[test]
    fn test_release_savepoint() {
        let stmt = parse_one("RELEASE SAVEPOINT sp1");
        match stmt {
            Statement::ReleaseSavepoint(r) => {
                assert_eq!(r.name, "sp1");
            }
            _ => panic!("Expected RELEASE SAVEPOINT"),
        }
    }

    #[test]
    fn test_release_without_savepoint_keyword() {
        let stmt = parse_one("RELEASE sp1");
        match stmt {
            Statement::ReleaseSavepoint(r) => {
                assert_eq!(r.name, "sp1");
            }
            _ => panic!("Expected RELEASE SAVEPOINT"),
        }
    }

    // -----------------------------------------------------------------------
    // EXPLAIN tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_explain() {
        let stmt = parse_one("EXPLAIN SELECT * FROM t");
        match stmt {
            Statement::Explain(e) => {
                assert!(!e.analyze);
                assert!(matches!(*e.statement, Statement::Select(_)));
            }
            _ => panic!("Expected EXPLAIN"),
        }
    }

    #[test]
    fn test_explain_analyze() {
        let stmt = parse_one("EXPLAIN ANALYZE SELECT * FROM t");
        match stmt {
            Statement::Explain(e) => {
                assert!(e.analyze);
                assert!(matches!(*e.statement, Statement::Select(_)));
            }
            _ => panic!("Expected EXPLAIN"),
        }
    }

    // -----------------------------------------------------------------------
    // Expression tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_arithmetic_precedence() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let stmt = parse_one("SELECT 1 + 2 * 3");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::BinaryOp { op, right, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, BinaryOperator::Plus);
                    assert!(matches!(
                        right.as_ref(),
                        Expr::BinaryOp {
                            op: BinaryOperator::Multiply,
                            ..
                        }
                    ));
                } else {
                    panic!("Expected BinaryOp");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parenthesized_precedence() {
        // (1 + 2) * 3 should parse as (1 + 2) * 3
        let stmt = parse_one("SELECT (1 + 2) * 3");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::BinaryOp { op, left, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, BinaryOperator::Multiply);
                    assert!(matches!(left.as_ref(), Expr::Nested(_)));
                } else {
                    panic!("Expected BinaryOp");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_boolean_precedence() {
        // a AND b OR c should parse as (a AND b) OR c
        let stmt = parse_one("SELECT a AND b OR c");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::BinaryOp { op, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, BinaryOperator::Or);
                } else {
                    panic!("Expected BinaryOp");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_precedence() {
        // NOT a AND b should parse as (NOT a) AND b
        let stmt = parse_one("SELECT NOT a AND b");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::BinaryOp { op, left, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, BinaryOperator::And);
                    assert!(matches!(
                        left.as_ref(),
                        Expr::UnaryOp {
                            op: UnaryOperator::Not,
                            ..
                        }
                    ));
                } else {
                    panic!("Expected BinaryOp");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_comparison_operators() {
        parse_one("SELECT * FROM t WHERE a = 1");
        parse_one("SELECT * FROM t WHERE a <> 1");
        parse_one("SELECT * FROM t WHERE a != 1");
        parse_one("SELECT * FROM t WHERE a < 1");
        parse_one("SELECT * FROM t WHERE a > 1");
        parse_one("SELECT * FROM t WHERE a <= 1");
        parse_one("SELECT * FROM t WHERE a >= 1");
    }

    #[test]
    fn test_is_null() {
        let stmt = parse_one("SELECT * FROM t WHERE x IS NULL");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::IsNull { negated: false, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_is_not_null() {
        let stmt = parse_one("SELECT * FROM t WHERE x IS NOT NULL");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::IsNull { negated: true, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_in_list() {
        let stmt = parse_one("SELECT * FROM t WHERE x IN (1, 2, 3)");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::InList { negated: false, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_in_list() {
        let stmt = parse_one("SELECT * FROM t WHERE x NOT IN (1, 2)");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::InList { negated: true, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_between() {
        let stmt = parse_one("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::Between { negated: false, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_between() {
        let stmt = parse_one("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::Between { negated: true, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_like() {
        let stmt = parse_one("SELECT * FROM t WHERE name LIKE '%alice%'");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::Like { negated: false, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_like() {
        let stmt = parse_one("SELECT * FROM t WHERE name NOT LIKE '%test%'");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::Like { negated: true, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_ilike() {
        let stmt = parse_one("SELECT * FROM t WHERE name ILIKE '%alice%'");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::ILike { negated: false, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_ilike() {
        let stmt = parse_one("SELECT * FROM t WHERE name NOT ILIKE '%test%'");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    s.where_clause.as_deref(),
                    Some(Expr::ILike { negated: true, .. })
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_cast() {
        let stmt = parse_one("SELECT CAST(x AS INT)");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    &s.projections[0],
                    SelectItem::Expr(
                        Expr::Cast {
                            data_type: DataType::Int,
                            ..
                        },
                        _
                    )
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_double_colon_cast() {
        let stmt = parse_one("SELECT x::INT");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    &s.projections[0],
                    SelectItem::Expr(
                        Expr::Cast {
                            data_type: DataType::Int,
                            ..
                        },
                        _
                    )
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_case_searched() {
        let stmt =
            parse_one("SELECT CASE WHEN x > 0 THEN 'pos' WHEN x < 0 THEN 'neg' ELSE 'zero' END");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(
                    Expr::Case {
                        operand,
                        conditions,
                        else_result,
                    },
                    _,
                ) = &s.projections[0]
                {
                    assert!(operand.is_none());
                    assert_eq!(conditions.len(), 2);
                    assert!(else_result.is_some());
                } else {
                    panic!("Expected CASE expr");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_case_simple() {
        let stmt = parse_one("SELECT CASE x WHEN 1 THEN 'one' WHEN 2 THEN 'two' END");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(
                    Expr::Case {
                        operand,
                        conditions,
                        else_result,
                    },
                    _,
                ) = &s.projections[0]
                {
                    assert!(operand.is_some());
                    assert_eq!(conditions.len(), 2);
                    assert!(else_result.is_none());
                } else {
                    panic!("Expected CASE expr");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_function_call() {
        let stmt = parse_one("SELECT COUNT(*), SUM(amount), UPPER(name)");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.projections.len(), 3);
                assert!(
                    matches!(&s.projections[0], SelectItem::Expr(Expr::Function { name, .. }, _) if name == "COUNT")
                );
                assert!(
                    matches!(&s.projections[1], SelectItem::Expr(Expr::Function { name, .. }, _) if name == "SUM")
                );
                assert!(
                    matches!(&s.projections[2], SelectItem::Expr(Expr::Function { name, .. }, _) if name == "UPPER")
                );
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_function_distinct() {
        let stmt = parse_one("SELECT COUNT(DISTINCT category)");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::Function { distinct, .. }, _) = &s.projections[0] {
                    assert!(*distinct);
                } else {
                    panic!("Expected Function");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_qualified_identifier() {
        let stmt = parse_one("SELECT t.col FROM t");
        match stmt {
            Statement::Select(s) => {
                assert!(matches!(
                    &s.projections[0],
                    SelectItem::Expr(Expr::QualifiedIdentifier { table, column }, _)
                    if table == "t" && column == "col"
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_string_concat() {
        let stmt = parse_one("SELECT 'hello' || ' ' || 'world'");
        match stmt {
            Statement::Select(s) => {
                // Should be left-associative: ('hello' || ' ') || 'world'
                if let SelectItem::Expr(Expr::BinaryOp { op, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, BinaryOperator::Concat);
                } else {
                    panic!("Expected BinaryOp");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_unary_minus() {
        let stmt = parse_one("SELECT -5, -(a + b)");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.projections.len(), 2);
                assert!(matches!(
                    &s.projections[0],
                    SelectItem::Expr(
                        Expr::UnaryOp {
                            op: UnaryOperator::Minus,
                            ..
                        },
                        _
                    )
                ));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // Multi-statement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_statement() {
        let stmts = parse_all("SELECT 1; SELECT 2; SELECT 3");
        assert_eq!(stmts.len(), 3);
    }

    #[test]
    fn test_trailing_semicolons() {
        let stmts = parse_all("SELECT 1;;; SELECT 2;");
        assert_eq!(stmts.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Error tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_without_from_with_where() {
        // FROM is optional, so SELECT * WHERE x = 1 is valid (empty FROM, WHERE parsed)
        let stmt = parse_one("SELECT * WHERE x = 1");
        match stmt {
            Statement::Select(s) => {
                assert!(s.from.is_empty());
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_error_unclosed_paren() {
        let err = parse_err("SELECT (1 + 2");
        assert!(err.contains("Parse error"));
    }

    #[test]
    fn test_error_invalid_statement() {
        let err = parse_err("INVALID SQL");
        assert!(err.contains("Parse error"));
    }

    #[test]
    fn test_error_includes_position() {
        let err = parse_err("SELECT 1 + ");
        assert!(err.contains("line"));
        assert!(err.contains("column"));
    }

    // -----------------------------------------------------------------------
    // CTE (WITH ... AS) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cte_simple() {
        let stmt = parse_one(
            "WITH active AS (SELECT * FROM users WHERE active = true) SELECT * FROM active",
        );
        match stmt {
            Statement::Select(s) => {
                let with = s.with.unwrap();
                assert!(!with.recursive);
                assert_eq!(with.ctes.len(), 1);
                assert_eq!(with.ctes[0].name, "active");
                assert!(with.ctes[0].columns.is_empty());
                assert_eq!(s.from.len(), 1);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_cte_with_columns() {
        let stmt = parse_one("WITH t (a, b) AS (SELECT 1, 2) SELECT * FROM t");
        match stmt {
            Statement::Select(s) => {
                let with = s.with.unwrap();
                assert_eq!(with.ctes[0].columns, vec!["a", "b"]);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_cte_multiple() {
        let stmt = parse_one("WITH a AS (SELECT 1 AS x), b AS (SELECT 2 AS y) SELECT * FROM a, b");
        match stmt {
            Statement::Select(s) => {
                let with = s.with.unwrap();
                assert_eq!(with.ctes.len(), 2);
                assert_eq!(with.ctes[0].name, "a");
                assert_eq!(with.ctes[1].name, "b");
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_cte_recursive() {
        let stmt = parse_one("WITH RECURSIVE nums AS (SELECT 1 AS n) SELECT * FROM nums");
        match stmt {
            Statement::Select(s) => {
                let with = s.with.unwrap();
                assert!(with.recursive);
                assert_eq!(with.ctes[0].name, "nums");
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // Subquery tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_subquery() {
        let stmt = parse_one("SELECT (SELECT 1)");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::Subquery(q), _) => {
                    assert_eq!(q.projections.len(), 1);
                }
                _ => panic!("Expected scalar subquery"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_subquery_in_from() {
        let stmt = parse_one("SELECT t.x FROM (SELECT 1 AS x) AS t");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Subquery { query, alias } => {
                    assert_eq!(alias, "t");
                    assert_eq!(query.projections.len(), 1);
                }
                _ => panic!("Expected subquery in FROM"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_subquery_in_from_without_as() {
        let stmt = parse_one("SELECT t.x FROM (SELECT 1 AS x) t");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Subquery { alias, .. } => assert_eq!(alias, "t"),
                _ => panic!("Expected subquery in FROM"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_exists() {
        let stmt = parse_one(
            "SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders WHERE orders.user_id = users.id)",
        );
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::Exists { negated, .. }) => assert!(!negated),
                _ => panic!("Expected EXISTS"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_exists() {
        let stmt = parse_one(
            "SELECT * FROM users WHERE NOT EXISTS (SELECT 1 FROM banned WHERE banned.id = users.id)",
        );
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::Exists { negated, .. }) => assert!(negated),
                _ => panic!("Expected NOT EXISTS"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_in_subquery() {
        let stmt = parse_one("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)");
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::InSubquery { negated, .. }) => assert!(!negated),
                _ => panic!("Expected IN subquery"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_not_in_subquery() {
        let stmt = parse_one("SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM banned)");
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::InSubquery { negated, .. }) => assert!(negated),
                _ => panic!("Expected NOT IN subquery"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_in_list_still_works() {
        // Verify that IN with a plain list still works
        let stmt = parse_one("SELECT * FROM t WHERE x IN (1, 2, 3)");
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::InList { list, negated, .. }) => {
                    assert!(!negated);
                    assert_eq!(list.len(), 3);
                }
                _ => panic!("Expected IN list"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // Window function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_window_function_row_number() {
        let stmt = parse_one("SELECT row_number() OVER (ORDER BY id) FROM users");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(
                    Expr::WindowFunction {
                        partition_by,
                        order_by,
                        frame,
                        ..
                    },
                    _,
                ) => {
                    assert!(partition_by.is_empty());
                    assert_eq!(order_by.len(), 1);
                    assert!(frame.is_none());
                }
                _ => panic!("Expected window function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_window_function_partition_by() {
        let stmt =
            parse_one("SELECT rank() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(
                    Expr::WindowFunction {
                        partition_by,
                        order_by,
                        ..
                    },
                    _,
                ) => {
                    assert_eq!(partition_by.len(), 1);
                    assert_eq!(order_by.len(), 1);
                    assert_eq!(order_by[0].asc, Some(false));
                }
                _ => panic!("Expected window function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_window_function_with_frame() {
        let stmt = parse_one(
            "SELECT sum(amount) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        );
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::WindowFunction { frame, .. }, _) => {
                    let f = frame.as_ref().unwrap();
                    assert_eq!(f.mode, WindowFrameMode::Rows);
                    assert_eq!(
                        f.start,
                        WindowFrameBound::Unbounded(WindowFrameDirection::Preceding)
                    );
                    assert_eq!(f.end, Some(WindowFrameBound::CurrentRow));
                }
                _ => panic!("Expected window function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_window_function_offset_frame() {
        let stmt = parse_one(
            "SELECT avg(val) OVER (ORDER BY id ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING) FROM t",
        );
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::WindowFunction { frame, .. }, _) => {
                    let f = frame.as_ref().unwrap();
                    assert_eq!(
                        f.start,
                        WindowFrameBound::Offset(3, WindowFrameDirection::Preceding)
                    );
                    assert_eq!(
                        f.end,
                        Some(WindowFrameBound::Offset(1, WindowFrameDirection::Following))
                    );
                }
                _ => panic!("Expected window function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_window_function_range_frame() {
        let stmt = parse_one("SELECT sum(x) OVER (ORDER BY id RANGE UNBOUNDED PRECEDING) FROM t");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::WindowFunction { frame, .. }, _) => {
                    let f = frame.as_ref().unwrap();
                    assert_eq!(f.mode, WindowFrameMode::Range);
                    assert_eq!(
                        f.start,
                        WindowFrameBound::Unbounded(WindowFrameDirection::Preceding)
                    );
                    assert!(f.end.is_none());
                }
                _ => panic!("Expected window function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_window_function_empty_over() {
        let stmt = parse_one("SELECT count(*) OVER () FROM t");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(
                    Expr::WindowFunction {
                        partition_by,
                        order_by,
                        frame,
                        ..
                    },
                    _,
                ) => {
                    assert!(partition_by.is_empty());
                    assert!(order_by.is_empty());
                    assert!(frame.is_none());
                }
                _ => panic!("Expected window function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // INSERT INTO ... SELECT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_select() {
        let stmt = parse_one("INSERT INTO archive SELECT * FROM users WHERE active = false");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "archive");
                assert!(i.columns.is_empty());
                match &i.source {
                    InsertSource::Query(q) => {
                        assert_eq!(q.projections, vec![SelectItem::Wildcard]);
                        assert!(q.where_clause.is_some());
                    }
                    _ => panic!("Expected SELECT source"),
                }
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_select_with_columns() {
        let stmt = parse_one("INSERT INTO archive (name, email) SELECT name, email FROM users");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.table, "archive");
                assert_eq!(i.columns, vec!["name", "email"]);
                assert!(matches!(&i.source, InsertSource::Query(_)));
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_select_with_cte() {
        let stmt = parse_one(
            "INSERT INTO summary WITH active AS (SELECT * FROM users WHERE active = true) SELECT count(*) FROM active",
        );
        match stmt {
            Statement::Insert(i) => match &i.source {
                InsertSource::Query(q) => {
                    assert!(q.with.is_some());
                }
                _ => panic!("Expected SELECT source"),
            },
            _ => panic!("Expected INSERT"),
        }
    }

    // -----------------------------------------------------------------------
    // UNION / INTERSECT / EXCEPT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_union() {
        let stmt = parse_one("SELECT a FROM t1 UNION SELECT b FROM t2");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.set_ops.len(), 1);
                assert_eq!(s.set_ops[0].op, SetOpType::Union);
                assert!(!s.set_ops[0].all);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_union_all() {
        let stmt = parse_one("SELECT a FROM t1 UNION ALL SELECT b FROM t2");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.set_ops.len(), 1);
                assert!(s.set_ops[0].all);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_intersect() {
        let stmt = parse_one("SELECT a FROM t1 INTERSECT SELECT b FROM t2");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.set_ops[0].op, SetOpType::Intersect);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_except() {
        let stmt = parse_one("SELECT a FROM t1 EXCEPT SELECT b FROM t2");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.set_ops[0].op, SetOpType::Except);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_union_chain() {
        let stmt = parse_one(
            "SELECT a FROM t1 UNION SELECT b FROM t2 UNION ALL SELECT c FROM t3 ORDER BY 1",
        );
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.set_ops.len(), 2);
                assert!(!s.set_ops[0].all);
                assert!(s.set_ops[1].all);
                assert_eq!(s.order_by.len(), 1);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // RETURNING tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_returning() {
        let stmt = parse_one("INSERT INTO users (name) VALUES ('Alice') RETURNING id, name");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.returning.unwrap().len(), 2);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_insert_returning_star() {
        let stmt = parse_one("INSERT INTO users (name) VALUES ('Alice') RETURNING *");
        match stmt {
            Statement::Insert(i) => {
                assert_eq!(i.returning.unwrap(), vec![SelectItem::Wildcard]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_update_returning() {
        let stmt = parse_one("UPDATE users SET name = 'Bob' WHERE id = 1 RETURNING *");
        match stmt {
            Statement::Update(u) => assert!(u.returning.is_some()),
            _ => panic!("Expected UPDATE"),
        }
    }

    #[test]
    fn test_delete_returning() {
        let stmt = parse_one("DELETE FROM users WHERE id = 1 RETURNING id");
        match stmt {
            Statement::Delete(d) => {
                assert_eq!(d.returning.unwrap().len(), 1);
            }
            _ => panic!("Expected DELETE"),
        }
    }

    // -----------------------------------------------------------------------
    // ON CONFLICT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_on_conflict_do_nothing() {
        let stmt = parse_one("INSERT INTO users (name) VALUES ('Alice') ON CONFLICT DO NOTHING");
        match stmt {
            Statement::Insert(i) => {
                let oc = i.on_conflict.unwrap();
                assert!(oc.columns.is_empty());
                assert!(matches!(oc.action, ConflictAction::DoNothing));
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_on_conflict_do_update() {
        let stmt = parse_one(
            "INSERT INTO users (id, name) VALUES (1, 'Alice') ON CONFLICT (id) DO UPDATE SET name = 'Bob'",
        );
        match stmt {
            Statement::Insert(i) => {
                let oc = i.on_conflict.unwrap();
                assert_eq!(oc.columns, vec!["id"]);
                match &oc.action {
                    ConflictAction::DoUpdate(a) => assert_eq!(a.len(), 1),
                    _ => panic!("Expected DO UPDATE"),
                }
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_on_conflict_with_returning() {
        let stmt = parse_one(
            "INSERT INTO users (name) VALUES ('Alice') ON CONFLICT DO NOTHING RETURNING *",
        );
        match stmt {
            Statement::Insert(i) => {
                assert!(i.on_conflict.is_some());
                assert!(i.returning.is_some());
            }
            _ => panic!("Expected INSERT"),
        }
    }

    // -----------------------------------------------------------------------
    // NATURAL JOIN / USING tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_natural_join() {
        let stmt = parse_one("SELECT * FROM a NATURAL JOIN b");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Join(j) => {
                    assert_eq!(j.join_type, JoinType::Inner);
                    assert_eq!(j.condition, JoinCondition::Natural);
                }
                _ => panic!("Expected JOIN"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_natural_left_join() {
        let stmt = parse_one("SELECT * FROM a NATURAL LEFT JOIN b");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Join(j) => {
                    assert_eq!(j.join_type, JoinType::Left);
                    assert_eq!(j.condition, JoinCondition::Natural);
                }
                _ => panic!("Expected JOIN"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_join_using() {
        let stmt = parse_one("SELECT * FROM a JOIN b USING (id, name)");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Join(j) => match &j.condition {
                    JoinCondition::Using(cols) => {
                        assert_eq!(*cols, vec!["id".to_string(), "name".to_string()])
                    }
                    _ => panic!("Expected USING"),
                },
                _ => panic!("Expected JOIN"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE/DROP VIEW tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_view() {
        let stmt = parse_one("CREATE VIEW active_users AS SELECT * FROM users WHERE active = true");
        match stmt {
            Statement::CreateView(v) => {
                assert_eq!(v.name, "active_users");
                assert!(!v.or_replace);
                assert!(v.columns.is_empty());
            }
            _ => panic!("Expected CREATE VIEW"),
        }
    }

    #[test]
    fn test_create_or_replace_view() {
        let stmt = parse_one("CREATE OR REPLACE VIEW v AS SELECT 1");
        match stmt {
            Statement::CreateView(v) => assert!(v.or_replace),
            _ => panic!("Expected CREATE VIEW"),
        }
    }

    #[test]
    fn test_create_view_with_columns() {
        let stmt = parse_one("CREATE VIEW v (a, b) AS SELECT 1, 2");
        match stmt {
            Statement::CreateView(v) => assert_eq!(v.columns, vec!["a", "b"]),
            _ => panic!("Expected CREATE VIEW"),
        }
    }

    #[test]
    fn test_drop_view() {
        let stmt = parse_one("DROP VIEW my_view");
        match stmt {
            Statement::DropView(v) => {
                assert_eq!(v.name, "my_view");
                assert!(!v.if_exists);
            }
            _ => panic!("Expected DROP VIEW"),
        }
    }

    #[test]
    fn test_drop_view_if_exists() {
        let stmt = parse_one("DROP VIEW IF EXISTS my_view");
        match stmt {
            Statement::DropView(v) => assert!(v.if_exists),
            _ => panic!("Expected DROP VIEW"),
        }
    }

    // -----------------------------------------------------------------------
    // GRANT / REVOKE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grant_select() {
        let stmt = parse_one("GRANT SELECT ON users TO reader");
        match stmt {
            Statement::Grant(g) => {
                assert_eq!(g.privileges, vec![Privilege::Select]);
                assert_eq!(g.on_table, "users");
                assert_eq!(g.to, "reader");
            }
            _ => panic!("Expected GRANT"),
        }
    }

    #[test]
    fn test_grant_multiple() {
        let stmt = parse_one("GRANT SELECT, INSERT, UPDATE ON users TO writer");
        match stmt {
            Statement::Grant(g) => {
                assert_eq!(
                    g.privileges,
                    vec![Privilege::Select, Privilege::Insert, Privilege::Update]
                );
            }
            _ => panic!("Expected GRANT"),
        }
    }

    #[test]
    fn test_grant_all() {
        let stmt = parse_one("GRANT ALL PRIVILEGES ON users TO admin");
        match stmt {
            Statement::Grant(g) => assert_eq!(g.privileges, vec![Privilege::All]),
            _ => panic!("Expected GRANT"),
        }
    }

    #[test]
    fn test_revoke() {
        let stmt = parse_one("REVOKE DELETE ON users FROM reader");
        match stmt {
            Statement::Revoke(r) => {
                assert_eq!(r.privileges, vec![Privilege::Delete]);
                assert_eq!(r.on_table, "users");
                assert_eq!(r.from, "reader");
            }
            _ => panic!("Expected REVOKE"),
        }
    }

    #[test]
    fn test_grant_on_table() {
        let stmt = parse_one("GRANT SELECT ON TABLE users TO reader");
        match stmt {
            Statement::Grant(g) => assert_eq!(g.on_table, "users"),
            _ => panic!("Expected GRANT"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE/DROP SCHEMA tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_schema() {
        let stmt = parse_one("CREATE SCHEMA myschema");
        match stmt {
            Statement::CreateSchema(s) => {
                assert_eq!(s.name, "myschema");
                assert!(!s.if_not_exists);
            }
            _ => panic!("Expected CREATE SCHEMA"),
        }
    }

    #[test]
    fn test_create_schema_if_not_exists() {
        let stmt = parse_one("CREATE SCHEMA IF NOT EXISTS myschema");
        match stmt {
            Statement::CreateSchema(s) => assert!(s.if_not_exists),
            _ => panic!("Expected CREATE SCHEMA"),
        }
    }

    #[test]
    fn test_drop_schema_cascade() {
        let stmt = parse_one("DROP SCHEMA IF EXISTS myschema CASCADE");
        match stmt {
            Statement::DropSchema(s) => {
                assert!(s.if_exists);
                assert!(s.cascade);
            }
            _ => panic!("Expected DROP SCHEMA"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE/DROP SEQUENCE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_sequence_simple() {
        let stmt = parse_one("CREATE SEQUENCE my_seq");
        match stmt {
            Statement::CreateSequence(s) => {
                assert_eq!(s.name, "my_seq");
                assert!(s.increment.is_none());
            }
            _ => panic!("Expected CREATE SEQUENCE"),
        }
    }

    #[test]
    fn test_create_sequence_with_options() {
        let stmt = parse_one(
            "CREATE SEQUENCE IF NOT EXISTS my_seq INCREMENT BY 2 START WITH 100 MINVALUE 1 MAXVALUE 1000 CACHE 10 CYCLE",
        );
        match stmt {
            Statement::CreateSequence(s) => {
                assert!(s.if_not_exists);
                assert_eq!(s.increment, Some(2));
                assert_eq!(s.start, Some(100));
                assert_eq!(s.min_value, Some(1));
                assert_eq!(s.max_value, Some(1000));
                assert_eq!(s.cache, Some(10));
                assert!(s.cycle);
            }
            _ => panic!("Expected CREATE SEQUENCE"),
        }
    }

    #[test]
    fn test_drop_sequence_if_exists() {
        let stmt = parse_one("DROP SEQUENCE IF EXISTS my_seq");
        match stmt {
            Statement::DropSequence(s) => assert!(s.if_exists),
            _ => panic!("Expected DROP SEQUENCE"),
        }
    }

    // -----------------------------------------------------------------------
    // VACUUM / REINDEX tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vacuum() {
        let stmt = parse_one("VACUUM");
        match stmt {
            Statement::Vacuum(v) => assert!(v.table.is_none()),
            _ => panic!("Expected VACUUM"),
        }
    }

    #[test]
    fn test_vacuum_table() {
        let stmt = parse_one("VACUUM users");
        match stmt {
            Statement::Vacuum(v) => assert_eq!(v.table.unwrap(), "users"),
            _ => panic!("Expected VACUUM"),
        }
    }

    #[test]
    fn test_reindex_table() {
        let stmt = parse_one("REINDEX TABLE users");
        match stmt {
            Statement::Reindex(r) => {
                assert_eq!(r.target, ReindexTarget::Table("users".to_string()));
            }
            _ => panic!("Expected REINDEX"),
        }
    }

    #[test]
    fn test_reindex_index() {
        let stmt = parse_one("REINDEX INDEX idx_name");
        match stmt {
            Statement::Reindex(r) => {
                assert_eq!(r.target, ReindexTarget::Index("idx_name".to_string()));
            }
            _ => panic!("Expected REINDEX"),
        }
    }

    // -----------------------------------------------------------------------
    // SET / SHOW tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_variable_eq() {
        let stmt = parse_one("SET search_path = 'public'");
        match stmt {
            Statement::SetVariable(s) => {
                assert_eq!(s.name, "search_path");
            }
            _ => panic!("Expected SET"),
        }
    }

    #[test]
    fn test_set_variable_to() {
        let stmt = parse_one("SET timezone TO 'UTC'");
        match stmt {
            Statement::SetVariable(s) => assert_eq!(s.name, "timezone"),
            _ => panic!("Expected SET"),
        }
    }

    #[test]
    fn test_show() {
        let stmt = parse_one("SHOW search_path");
        match stmt {
            Statement::Show(s) => assert_eq!(s.name, "search_path"),
            _ => panic!("Expected SHOW"),
        }
    }

    // -----------------------------------------------------------------------
    // COPY tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_copy_from_stdin() {
        let stmt = parse_one("COPY users FROM STDIN");
        match stmt {
            Statement::Copy(c) => {
                assert_eq!(c.table, "users");
                assert!(matches!(
                    c.direction,
                    CopyDirection::From(CopyTarget::Stdin)
                ));
            }
            _ => panic!("Expected COPY"),
        }
    }

    #[test]
    fn test_copy_to_stdout() {
        let stmt = parse_one("COPY users TO STDOUT");
        match stmt {
            Statement::Copy(c) => {
                assert!(matches!(c.direction, CopyDirection::To(CopyTarget::Stdout)));
            }
            _ => panic!("Expected COPY"),
        }
    }

    #[test]
    fn test_copy_from_file() {
        let stmt = parse_one("COPY users FROM '/tmp/data.csv'");
        match stmt {
            Statement::Copy(c) => match &c.direction {
                CopyDirection::From(CopyTarget::File(p)) => assert_eq!(p, "/tmp/data.csv"),
                _ => panic!("Expected FROM file"),
            },
            _ => panic!("Expected COPY"),
        }
    }

    #[test]
    fn test_copy_with_columns() {
        let stmt = parse_one("COPY users (name, email) FROM STDIN");
        match stmt {
            Statement::Copy(c) => assert_eq!(c.columns, vec!["name", "email"]),
            _ => panic!("Expected COPY"),
        }
    }

    // -----------------------------------------------------------------------
    // FOR UPDATE/SHARE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_for_update() {
        let stmt = parse_one("SELECT * FROM users FOR UPDATE");
        match stmt {
            Statement::Select(s) => {
                let fc = s.for_clause.unwrap();
                assert_eq!(fc.lock_type, ForLockType::Update);
                assert_eq!(fc.wait, ForWait::Wait);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_for_share_nowait() {
        let stmt = parse_one("SELECT * FROM users FOR SHARE NOWAIT");
        match stmt {
            Statement::Select(s) => {
                let fc = s.for_clause.unwrap();
                assert_eq!(fc.lock_type, ForLockType::Share);
                assert_eq!(fc.wait, ForWait::Nowait);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_for_update_skip_locked() {
        let stmt = parse_one("SELECT * FROM users FOR UPDATE SKIP LOCKED");
        match stmt {
            Statement::Select(s) => {
                let fc = s.for_clause.unwrap();
                assert_eq!(fc.lock_type, ForLockType::Update);
                assert_eq!(fc.wait, ForWait::SkipLocked);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_for_no_key_update() {
        let stmt = parse_one("SELECT * FROM users FOR NO KEY UPDATE");
        match stmt {
            Statement::Select(s) => {
                let fc = s.for_clause.unwrap();
                assert_eq!(fc.lock_type, ForLockType::NoKeyUpdate);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_for_key_share() {
        let stmt = parse_one("SELECT * FROM users FOR KEY SHARE");
        match stmt {
            Statement::Select(s) => {
                let fc = s.for_clause.unwrap();
                assert_eq!(fc.lock_type, ForLockType::KeyShare);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // FETCH FIRST/NEXT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fetch_first_rows_only() {
        let stmt = parse_one("SELECT * FROM users FETCH FIRST 10 ROWS ONLY");
        match stmt {
            Statement::Select(s) => {
                let f = s.fetch.unwrap();
                assert!(!f.percent);
                assert!(!f.with_ties);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_fetch_next_row_only() {
        let stmt = parse_one("SELECT * FROM users FETCH NEXT 1 ROW ONLY");
        match stmt {
            Statement::Select(s) => assert!(s.fetch.is_some()),
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_fetch_first_percent() {
        let stmt = parse_one("SELECT * FROM users FETCH FIRST 10 PERCENT ROWS ONLY");
        match stmt {
            Statement::Select(s) => {
                let f = s.fetch.unwrap();
                assert!(f.percent);
                assert!(!f.with_ties);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // DISTINCT ON tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_distinct_on() {
        let stmt = parse_one("SELECT DISTINCT ON (dept) name, dept FROM employees");
        match stmt {
            Statement::Select(s) => {
                assert!(s.distinct);
                assert_eq!(s.distinct_on.len(), 1);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // MERGE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_basic() {
        let stmt = parse_one(
            "MERGE INTO target USING source ON target.id = source.id \
             WHEN MATCHED THEN UPDATE SET name = source.name \
             WHEN NOT MATCHED THEN INSERT (id, name) VALUES (source.id, source.name)",
        );
        match stmt {
            Statement::Merge(m) => {
                assert_eq!(m.target, "target");
                assert_eq!(m.clauses.len(), 2);
                assert!(matches!(&m.clauses[0], MergeClause::WhenMatched { .. }));
                assert!(matches!(&m.clauses[1], MergeClause::WhenNotMatched { .. }));
            }
            _ => panic!("Expected MERGE"),
        }
    }

    #[test]
    fn test_merge_with_delete() {
        let stmt = parse_one(
            "MERGE INTO target USING source ON target.id = source.id \
             WHEN MATCHED AND source.deleted = TRUE THEN DELETE",
        );
        match stmt {
            Statement::Merge(m) => {
                assert_eq!(m.clauses.len(), 1);
                match &m.clauses[0] {
                    MergeClause::WhenMatched { condition, action } => {
                        assert!(condition.is_some());
                        assert!(matches!(action, MergeAction::Delete));
                    }
                    _ => panic!("Expected WHEN MATCHED"),
                }
            }
            _ => panic!("Expected MERGE"),
        }
    }

    // -----------------------------------------------------------------------
    // PREPARE / EXECUTE / DEALLOCATE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_prepare() {
        let stmt = parse_one("PREPARE my_plan (INT, VARCHAR) AS SELECT * FROM users WHERE id = 1");
        match stmt {
            Statement::Prepare(p) => {
                assert_eq!(p.name, "my_plan");
                assert_eq!(p.param_types.len(), 2);
            }
            _ => panic!("Expected PREPARE"),
        }
    }

    #[test]
    fn test_prepare_no_params() {
        let stmt = parse_one("PREPARE my_plan AS SELECT 1");
        match stmt {
            Statement::Prepare(p) => {
                assert_eq!(p.name, "my_plan");
                assert!(p.param_types.is_empty());
            }
            _ => panic!("Expected PREPARE"),
        }
    }

    #[test]
    fn test_execute() {
        let stmt = parse_one("EXECUTE my_plan (1, 'hello')");
        match stmt {
            Statement::Execute(e) => {
                assert_eq!(e.name, "my_plan");
                assert_eq!(e.params.len(), 2);
            }
            _ => panic!("Expected EXECUTE"),
        }
    }

    #[test]
    fn test_execute_no_params() {
        let stmt = parse_one("EXECUTE my_plan");
        match stmt {
            Statement::Execute(e) => {
                assert_eq!(e.name, "my_plan");
                assert!(e.params.is_empty());
            }
            _ => panic!("Expected EXECUTE"),
        }
    }

    #[test]
    fn test_deallocate() {
        let stmt = parse_one("DEALLOCATE my_plan");
        match stmt {
            Statement::Deallocate(d) => {
                assert_eq!(d.name, Some("my_plan".to_string()));
                assert!(!d.all);
            }
            _ => panic!("Expected DEALLOCATE"),
        }
    }

    #[test]
    fn test_deallocate_all() {
        let stmt = parse_one("DEALLOCATE ALL");
        match stmt {
            Statement::Deallocate(d) => {
                assert!(d.all);
                assert!(d.name.is_none());
            }
            _ => panic!("Expected DEALLOCATE"),
        }
    }

    #[test]
    fn test_deallocate_prepare() {
        let stmt = parse_one("DEALLOCATE PREPARE my_plan");
        match stmt {
            Statement::Deallocate(d) => {
                assert_eq!(d.name, Some("my_plan".to_string()));
            }
            _ => panic!("Expected DEALLOCATE"),
        }
    }

    // -----------------------------------------------------------------------
    // LATERAL join tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lateral_subquery() {
        let stmt = parse_one(
            "SELECT * FROM users, LATERAL (SELECT * FROM orders WHERE orders.user_id = users.id) AS o",
        );
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.from.len(), 2);
                assert!(matches!(&s.from[1], TableRef::Lateral { .. }));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // Array tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_array_constructor() {
        let stmt = parse_one("SELECT ARRAY[1, 2, 3]");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::ArrayConstructor(elems), _) => {
                    assert_eq!(elems.len(), 3);
                }
                _ => panic!("Expected array constructor"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_array_subscript() {
        let stmt = parse_one("SELECT arr[1] FROM t");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::ArraySubscript { .. }, _) => {}
                _ => panic!("Expected array subscript"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_array_type() {
        let stmt = parse_one("CREATE TABLE t (tags TEXT[])");
        match stmt {
            Statement::CreateTable(ct) => {
                assert!(matches!(ct.columns[0].data_type, DataType::Array(_)));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // JSON operator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_arrow() {
        let stmt = parse_one("SELECT data -> 'key' FROM t");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::JsonAccess { op, .. }, _) => {
                    assert_eq!(*op, JsonOperator::Arrow);
                }
                _ => panic!("Expected JSON access"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_json_double_arrow() {
        let stmt = parse_one("SELECT data ->> 'key' FROM t");
        match stmt {
            Statement::Select(s) => match &s.projections[0] {
                SelectItem::Expr(Expr::JsonAccess { op, .. }, _) => {
                    assert_eq!(*op, JsonOperator::DoubleArrow);
                }
                _ => panic!("Expected JSON access"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_json_contains() {
        let stmt = parse_one("SELECT * FROM t WHERE data @> '{\"key\": 1}'");
        match stmt {
            Statement::Select(s) => {
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_json_exists() {
        let stmt = parse_one("SELECT * FROM t WHERE data ? 'key'");
        match stmt {
            Statement::Select(s) => {
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // LISTEN / NOTIFY tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_listen() {
        let stmt = parse_one("LISTEN my_channel");
        match stmt {
            Statement::Listen(l) => assert_eq!(l.channel, "my_channel"),
            _ => panic!("Expected LISTEN"),
        }
    }

    #[test]
    fn test_notify() {
        let stmt = parse_one("NOTIFY my_channel");
        match stmt {
            Statement::Notify(n) => {
                assert_eq!(n.channel, "my_channel");
                assert!(n.payload.is_none());
            }
            _ => panic!("Expected NOTIFY"),
        }
    }

    #[test]
    fn test_notify_with_payload() {
        let stmt = parse_one("NOTIFY my_channel, 'hello world'");
        match stmt {
            Statement::Notify(n) => {
                assert_eq!(n.channel, "my_channel");
                assert_eq!(n.payload.unwrap(), "hello world");
            }
            _ => panic!("Expected NOTIFY"),
        }
    }

    // -----------------------------------------------------------------------
    // Cursor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_declare_cursor() {
        let stmt = parse_one("DECLARE my_cursor CURSOR FOR SELECT * FROM users");
        match stmt {
            Statement::DeclareCursor(d) => {
                assert_eq!(d.name, "my_cursor");
                assert!(d.scroll.is_none());
                assert!(d.hold.is_none());
            }
            _ => panic!("Expected DECLARE"),
        }
    }

    #[test]
    fn test_declare_scroll_cursor_with_hold() {
        let stmt = parse_one("DECLARE my_cursor SCROLL CURSOR WITH HOLD FOR SELECT * FROM users");
        match stmt {
            Statement::DeclareCursor(d) => {
                assert_eq!(d.scroll, Some(true));
                assert_eq!(d.hold, Some(true));
            }
            _ => panic!("Expected DECLARE"),
        }
    }

    #[test]
    fn test_declare_no_scroll_cursor() {
        let stmt = parse_one("DECLARE my_cursor NO SCROLL CURSOR FOR SELECT * FROM users");
        match stmt {
            Statement::DeclareCursor(d) => {
                assert_eq!(d.scroll, Some(false));
            }
            _ => panic!("Expected DECLARE"),
        }
    }

    #[test]
    fn test_fetch_cursor_next() {
        let stmt = parse_one("FETCH NEXT FROM my_cursor");
        match stmt {
            Statement::FetchCursor(f) => {
                assert_eq!(f.direction, FetchDirection::Next);
                assert_eq!(f.cursor, "my_cursor");
            }
            _ => panic!("Expected FETCH"),
        }
    }

    #[test]
    fn test_fetch_cursor_absolute() {
        let stmt = parse_one("FETCH ABSOLUTE 5 FROM my_cursor");
        match stmt {
            Statement::FetchCursor(f) => {
                assert_eq!(f.direction, FetchDirection::Absolute(5));
            }
            _ => panic!("Expected FETCH"),
        }
    }

    #[test]
    fn test_fetch_cursor_forward_all() {
        let stmt = parse_one("FETCH FORWARD ALL FROM my_cursor");
        match stmt {
            Statement::FetchCursor(f) => {
                assert_eq!(f.direction, FetchDirection::Forward(None));
            }
            _ => panic!("Expected FETCH"),
        }
    }

    #[test]
    fn test_close_cursor() {
        let stmt = parse_one("CLOSE my_cursor");
        match stmt {
            Statement::CloseCursor(c) => {
                assert_eq!(c.name, Some("my_cursor".to_string()));
                assert!(!c.all);
            }
            _ => panic!("Expected CLOSE"),
        }
    }

    #[test]
    fn test_close_all_cursors() {
        let stmt = parse_one("CLOSE ALL");
        match stmt {
            Statement::CloseCursor(c) => assert!(c.all),
            _ => panic!("Expected CLOSE"),
        }
    }

    // -----------------------------------------------------------------------
    // COMMENT ON tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_comment_on_table() {
        let stmt = parse_one("COMMENT ON TABLE users IS 'User accounts'");
        match stmt {
            Statement::CommentOn(c) => {
                assert_eq!(c.object_type, CommentObjectType::Table);
                assert_eq!(c.name, "users");
                assert_eq!(c.comment, Some("User accounts".to_string()));
            }
            _ => panic!("Expected COMMENT ON"),
        }
    }

    #[test]
    fn test_comment_on_column() {
        let stmt = parse_one("COMMENT ON COLUMN users.email IS 'Email address'");
        match stmt {
            Statement::CommentOn(c) => {
                assert_eq!(c.object_type, CommentObjectType::Column);
                assert_eq!(c.name, "users");
                assert_eq!(c.column, Some("email".to_string()));
            }
            _ => panic!("Expected COMMENT ON"),
        }
    }

    #[test]
    fn test_comment_on_null() {
        let stmt = parse_one("COMMENT ON TABLE users IS NULL");
        match stmt {
            Statement::CommentOn(c) => assert!(c.comment.is_none()),
            _ => panic!("Expected COMMENT ON"),
        }
    }

    // -----------------------------------------------------------------------
    // ALTER INDEX / SEQUENCE / VIEW tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_index_rename() {
        let stmt = parse_one("ALTER INDEX idx_old RENAME TO idx_new");
        match stmt {
            Statement::AlterIndex(a) => {
                assert_eq!(a.name, "idx_old");
                assert!(matches!(
                    a.operation,
                    AlterIndexOperation::Rename { new_name } if new_name == "idx_new"
                ));
            }
            _ => panic!("Expected ALTER INDEX"),
        }
    }

    #[test]
    fn test_alter_sequence() {
        let stmt = parse_one("ALTER SEQUENCE my_seq INCREMENT BY 5 MINVALUE 1 MAXVALUE 1000");
        match stmt {
            Statement::AlterSequence(a) => {
                assert_eq!(a.name, "my_seq");
                assert_eq!(a.increment, Some(5));
                assert_eq!(a.min_value, Some(Some(1)));
                assert_eq!(a.max_value, Some(Some(1000)));
            }
            _ => panic!("Expected ALTER SEQUENCE"),
        }
    }

    #[test]
    fn test_alter_sequence_no_minvalue() {
        let stmt = parse_one("ALTER SEQUENCE my_seq NO MINVALUE NO MAXVALUE");
        match stmt {
            Statement::AlterSequence(a) => {
                assert_eq!(a.min_value, Some(None));
                assert_eq!(a.max_value, Some(None));
            }
            _ => panic!("Expected ALTER SEQUENCE"),
        }
    }

    #[test]
    fn test_alter_view_rename() {
        let stmt = parse_one("ALTER VIEW old_view RENAME TO new_view");
        match stmt {
            Statement::AlterView(a) => {
                assert_eq!(a.name, "old_view");
                assert!(matches!(
                    a.operation,
                    AlterViewOperation::Rename { new_name } if new_name == "new_view"
                ));
            }
            _ => panic!("Expected ALTER VIEW"),
        }
    }

    // -----------------------------------------------------------------------
    // Materialized view tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_materialized_view() {
        let stmt = parse_one("CREATE MATERIALIZED VIEW my_view AS SELECT * FROM users");
        match stmt {
            Statement::CreateMaterializedView(v) => {
                assert_eq!(v.name, "my_view");
                assert!(!v.if_not_exists);
            }
            _ => panic!("Expected CREATE MATERIALIZED VIEW"),
        }
    }

    #[test]
    fn test_create_materialized_view_if_not_exists() {
        let stmt = parse_one("CREATE MATERIALIZED VIEW IF NOT EXISTS mv AS SELECT * FROM users");
        match stmt {
            Statement::CreateMaterializedView(v) => assert!(v.if_not_exists),
            _ => panic!("Expected CREATE MATERIALIZED VIEW"),
        }
    }

    #[test]
    fn test_drop_materialized_view() {
        let stmt = parse_one("DROP MATERIALIZED VIEW IF EXISTS my_view");
        match stmt {
            Statement::DropMaterializedView(v) => {
                assert_eq!(v.name, "my_view");
                assert!(v.if_exists);
            }
            _ => panic!("Expected DROP MATERIALIZED VIEW"),
        }
    }

    #[test]
    fn test_refresh_materialized_view() {
        let stmt = parse_one("REFRESH MATERIALIZED VIEW my_view");
        match stmt {
            Statement::RefreshMaterializedView(r) => {
                assert_eq!(r.name, "my_view");
                assert!(!r.concurrently);
            }
            _ => panic!("Expected REFRESH MATERIALIZED VIEW"),
        }
    }

    // -----------------------------------------------------------------------
    // DO block tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_do_block() {
        let stmt = parse_one("DO 'BEGIN RAISE NOTICE ''hello''; END'");
        match stmt {
            Statement::DoBlock(d) => {
                assert!(!d.body.is_empty());
                assert!(d.language.is_none());
            }
            _ => panic!("Expected DO"),
        }
    }

    #[test]
    fn test_do_block_with_language() {
        let stmt = parse_one("DO 'body' LANGUAGE plpgsql");
        match stmt {
            Statement::DoBlock(d) => {
                assert_eq!(d.body, "body");
                assert_eq!(d.language, Some("plpgsql".to_string()));
            }
            _ => panic!("Expected DO"),
        }
    }

    // -----------------------------------------------------------------------
    // CHECKPOINT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint() {
        let stmt = parse_one("CHECKPOINT");
        assert!(matches!(stmt, Statement::Checkpoint(_)));
    }

    // -----------------------------------------------------------------------
    // VALUES query tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_values_query() {
        let stmt = parse_one("VALUES (1, 'a'), (2, 'b'), (3, 'c')");
        match stmt {
            Statement::ValuesQuery(v) => {
                assert_eq!(v.rows.len(), 3);
                assert_eq!(v.rows[0].len(), 2);
            }
            _ => panic!("Expected VALUES"),
        }
    }

    // -----------------------------------------------------------------------
    // Lexer tests for new tokens
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_operator_tokens() {
        use crate::lexer::Lexer;
        let mut lex = Lexer::new("-> ->> #> #>> @> <@ ? ?| ?&");
        let tokens: Vec<Token> = std::iter::from_fn(|| {
            let t = lex.next_token().ok()?;
            if t.token == Token::Eof {
                None
            } else {
                Some(t.token)
            }
        })
        .collect();
        assert_eq!(
            tokens,
            vec![
                Token::Arrow,
                Token::DoubleArrow,
                Token::HashArrow,
                Token::HashDoubleArrow,
                Token::AtArrow,
                Token::ArrowAt,
                Token::Question,
                Token::QuestionPipe,
                Token::QuestionAmp,
            ]
        );
    }

    #[test]
    fn test_bracket_tokens() {
        use crate::lexer::Lexer;
        let mut lex = Lexer::new("[1]");
        let t1 = lex.next_token().unwrap();
        assert_eq!(t1.token, Token::LBracket);
        let t2 = lex.next_token().unwrap();
        assert_eq!(t2.token, Token::Integer(1));
        let t3 = lex.next_token().unwrap();
        assert_eq!(t3.token, Token::RBracket);
    }

    // -----------------------------------------------------------------------
    // SHOW with multi-word (ZyronDB custom)
    // -----------------------------------------------------------------------

    #[test]
    fn test_show_wal_status() {
        let stmt = parse_one("SHOW wal_status");
        match stmt {
            Statement::Show(s) => assert_eq!(s.name, "wal_status"),
            _ => panic!("Expected SHOW"),
        }
    }

    #[test]
    fn test_show_buffer_pool() {
        let stmt = parse_one("SHOW buffer_pool");
        match stmt {
            Statement::Show(s) => assert_eq!(s.name, "buffer_pool"),
            _ => panic!("Expected SHOW"),
        }
    }

    // -----------------------------------------------------------------------
    // ALTER TABLE SET STORAGE (ZyronDB custom)
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_table_set_storage_columnar() {
        // Using ALTER TABLE ... ALTER COLUMN ... SET TYPE to test ALTER still works
        let stmt = parse_one("ALTER TABLE users ALTER COLUMN data TYPE JSONB");
        match stmt {
            Statement::AlterTable(a) => {
                assert_eq!(a.name, "users");
                assert!(matches!(
                    a.operation,
                    AlterTableOperation::AlterColumnSetType { .. }
                ));
            }
            _ => panic!("Expected ALTER TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // TTL
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_table_set_ttl_delete() {
        let stmt = parse_one("ALTER TABLE events SET TTL 30 DAYS ON created_at");
        match stmt {
            Statement::AlterTableTtl(a) => {
                assert_eq!(a.table, "events");
                match a.operation {
                    TtlOperation::Set {
                        duration,
                        column,
                        action,
                    } => {
                        assert_eq!(duration.value, 30);
                        assert_eq!(duration.unit, TtlUnit::Days);
                        assert_eq!(column, "created_at");
                        assert_eq!(action, TtlAction::Delete);
                    }
                    _ => panic!("Expected Set"),
                }
            }
            _ => panic!("Expected AlterTableTtl"),
        }
    }

    #[test]
    fn test_alter_table_set_ttl_archive() {
        let stmt = parse_one("ALTER TABLE logs SET TTL ARCHIVE 90 DAYS ON updated_at");
        match stmt {
            Statement::AlterTableTtl(a) => {
                assert_eq!(a.table, "logs");
                match a.operation {
                    TtlOperation::Set {
                        duration,
                        column,
                        action,
                    } => {
                        assert_eq!(duration.value, 90);
                        assert_eq!(duration.unit, TtlUnit::Days);
                        assert_eq!(column, "updated_at");
                        assert_eq!(action, TtlAction::Archive);
                    }
                    _ => panic!("Expected Set"),
                }
            }
            _ => panic!("Expected AlterTableTtl"),
        }
    }

    #[test]
    fn test_alter_table_set_ttl_hours() {
        let stmt = parse_one("ALTER TABLE sessions SET TTL 24 HOURS ON last_seen");
        match stmt {
            Statement::AlterTableTtl(a) => {
                assert_eq!(a.table, "sessions");
                match a.operation {
                    TtlOperation::Set {
                        duration,
                        column,
                        action,
                    } => {
                        assert_eq!(duration.value, 24);
                        assert_eq!(duration.unit, TtlUnit::Hours);
                        assert_eq!(column, "last_seen");
                        assert_eq!(action, TtlAction::Delete);
                    }
                    _ => panic!("Expected Set"),
                }
            }
            _ => panic!("Expected AlterTableTtl"),
        }
    }

    #[test]
    fn test_alter_table_set_ttl_minutes() {
        let stmt = parse_one("ALTER TABLE cache SET TTL 15 MINUTES ON expires_at");
        match stmt {
            Statement::AlterTableTtl(a) => match a.operation {
                TtlOperation::Set { duration, .. } => {
                    assert_eq!(duration.value, 15);
                    assert_eq!(duration.unit, TtlUnit::Minutes);
                }
                _ => panic!("Expected Set"),
            },
            _ => panic!("Expected AlterTableTtl"),
        }
    }

    #[test]
    fn test_alter_table_set_ttl_seconds() {
        let stmt = parse_one("ALTER TABLE temp SET TTL 300 SECONDS ON ts");
        match stmt {
            Statement::AlterTableTtl(a) => match a.operation {
                TtlOperation::Set { duration, .. } => {
                    assert_eq!(duration.value, 300);
                    assert_eq!(duration.unit, TtlUnit::Seconds);
                }
                _ => panic!("Expected Set"),
            },
            _ => panic!("Expected AlterTableTtl"),
        }
    }

    #[test]
    fn test_alter_table_drop_ttl() {
        let stmt = parse_one("ALTER TABLE events DROP TTL");
        match stmt {
            Statement::AlterTableTtl(a) => {
                assert_eq!(a.table, "events");
                assert_eq!(a.operation, TtlOperation::Drop);
            }
            _ => panic!("Expected AlterTableTtl"),
        }
    }

    // -----------------------------------------------------------------------
    // SCHEDULE
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_schedule_every() {
        let stmt = parse_one("CREATE SCHEDULE auto_vacuum EVERY 5 MINUTES DO VACUUM");
        match stmt {
            Statement::CreateSchedule(s) => {
                assert_eq!(s.name, "auto_vacuum");
                match s.interval {
                    ScheduleInterval::Every(d) => {
                        assert_eq!(d.value, 5);
                        assert_eq!(d.unit, TtlUnit::Minutes);
                    }
                    _ => panic!("Expected Every"),
                }
                assert!(matches!(*s.body, Statement::Vacuum(_)));
            }
            _ => panic!("Expected CreateSchedule"),
        }
    }

    #[test]
    fn test_create_schedule_cron() {
        let stmt = parse_one(
            "CREATE SCHEDULE cleanup CRON '0 */5 * * *' DO DELETE FROM logs WHERE age > 7",
        );
        match stmt {
            Statement::CreateSchedule(s) => {
                assert_eq!(s.name, "cleanup");
                match &s.interval {
                    ScheduleInterval::Cron(expr) => assert_eq!(expr, "0 */5 * * *"),
                    _ => panic!("Expected Cron"),
                }
                assert!(matches!(*s.body, Statement::Delete(_)));
            }
            _ => panic!("Expected CreateSchedule"),
        }
    }

    #[test]
    fn test_create_schedule_every_hours() {
        let stmt = parse_one("CREATE SCHEDULE hourly_reindex EVERY 1 HOURS DO REINDEX TABLE users");
        match stmt {
            Statement::CreateSchedule(s) => {
                assert_eq!(s.name, "hourly_reindex");
                match s.interval {
                    ScheduleInterval::Every(d) => {
                        assert_eq!(d.value, 1);
                        assert_eq!(d.unit, TtlUnit::Hours);
                    }
                    _ => panic!("Expected Every"),
                }
                assert!(matches!(*s.body, Statement::Reindex(_)));
            }
            _ => panic!("Expected CreateSchedule"),
        }
    }

    #[test]
    fn test_drop_schedule() {
        let stmt = parse_one("DROP SCHEDULE auto_vacuum");
        match stmt {
            Statement::DropSchedule(s) => {
                assert_eq!(s.name, "auto_vacuum");
                assert!(!s.if_exists);
            }
            _ => panic!("Expected DropSchedule"),
        }
    }

    #[test]
    fn test_drop_schedule_if_exists() {
        let stmt = parse_one("DROP SCHEDULE IF EXISTS old_job");
        match stmt {
            Statement::DropSchedule(s) => {
                assert_eq!(s.name, "old_job");
                assert!(s.if_exists);
            }
            _ => panic!("Expected DropSchedule"),
        }
    }

    #[test]
    fn test_pause_schedule() {
        let stmt = parse_one("PAUSE SCHEDULE auto_vacuum");
        match stmt {
            Statement::PauseSchedule(s) => {
                assert_eq!(s.name, "auto_vacuum");
            }
            _ => panic!("Expected PauseSchedule"),
        }
    }

    #[test]
    fn test_resume_schedule() {
        let stmt = parse_one("RESUME SCHEDULE auto_vacuum");
        match stmt {
            Statement::ResumeSchedule(s) => {
                assert_eq!(s.name, "auto_vacuum");
            }
            _ => panic!("Expected ResumeSchedule"),
        }
    }

    // -----------------------------------------------------------------------
    // OPTIMIZE
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_table() {
        let stmt = parse_one("OPTIMIZE TABLE users");
        match stmt {
            Statement::OptimizeTable(o) => {
                assert_eq!(o.table, "users");
            }
            _ => panic!("Expected OptimizeTable"),
        }
    }

    #[test]
    fn test_optimize_without_table_keyword() {
        let stmt = parse_one("OPTIMIZE users");
        match stmt {
            Statement::OptimizeTable(o) => {
                assert_eq!(o.table, "users");
            }
            _ => panic!("Expected OptimizeTable"),
        }
    }

    // -----------------------------------------------------------------------
    // New data types
    // -----------------------------------------------------------------------

    #[test]
    fn test_data_type_tinyint() {
        let stmt = parse_one("CREATE TABLE t (x TINYINT)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::TinyInt);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_data_type_int128() {
        let stmt = parse_one("CREATE TABLE t (x INT128)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::Int128);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_data_type_uint8_to_uint128() {
        let stmt = parse_one("CREATE TABLE t (a UINT8, b UINT16, c UINT32, d UINT64, e UINT128)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::UInt8);
                assert_eq!(ct.columns[1].data_type, DataType::UInt16);
                assert_eq!(ct.columns[2].data_type, DataType::UInt32);
                assert_eq!(ct.columns[3].data_type, DataType::UInt64);
                assert_eq!(ct.columns[4].data_type, DataType::UInt128);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_data_type_vector_with_dimension() {
        let stmt = parse_one("CREATE TABLE t (embedding VECTOR(1536))");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::Vector(Some(1536)));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_data_type_vector_without_dimension() {
        let stmt = parse_one("CREATE TABLE t (embedding VECTOR)");
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.columns[0].data_type, DataType::Vector(None));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // Named function arguments
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_function_args() {
        let stmt = parse_one("SELECT NPV(rate => 0.10, cash_flows => ARRAY[1, 2, 3])");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::Function { name, args, .. }, _) = &s.projections[0] {
                    assert_eq!(name, "NPV");
                    assert_eq!(args.len(), 2);
                    assert!(matches!(&args[0], FunctionArg::Named { name, .. } if name == "rate"));
                    assert!(
                        matches!(&args[1], FunctionArg::Named { name, .. } if name == "cash_flows")
                    );
                } else {
                    panic!("Expected Function");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_mixed_function_args() {
        let stmt = parse_one("SELECT myfunc(1, key => 'value')");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::Function { args, .. }, _) = &s.projections[0] {
                    assert_eq!(args.len(), 2);
                    assert!(matches!(&args[0], FunctionArg::Unnamed(_)));
                    assert!(matches!(&args[1], FunctionArg::Named { name, .. } if name == "key"));
                } else {
                    panic!("Expected Function");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // GROUP BY ROLLUP / CUBE / GROUPING SETS
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_by_rollup() {
        let stmt = parse_one("SELECT a, SUM(b) FROM t GROUP BY ROLLUP(a)");
        match stmt {
            Statement::Select(s) => {
                assert!(s.group_by.is_empty());
                match s.group_by_sets {
                    Some(GroupBySets::Rollup(exprs)) => {
                        assert_eq!(exprs.len(), 1);
                    }
                    _ => panic!("Expected ROLLUP"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_group_by_cube() {
        let stmt = parse_one("SELECT a, b, SUM(c) FROM t GROUP BY CUBE(a, b)");
        match stmt {
            Statement::Select(s) => match s.group_by_sets {
                Some(GroupBySets::Cube(exprs)) => {
                    assert_eq!(exprs.len(), 2);
                }
                _ => panic!("Expected CUBE"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_group_by_grouping_sets() {
        let stmt = parse_one("SELECT a, b, SUM(c) FROM t GROUP BY GROUPING SETS ((a), (b), ())");
        match stmt {
            Statement::Select(s) => {
                match s.group_by_sets {
                    Some(GroupBySets::GroupingSets(sets)) => {
                        assert_eq!(sets.len(), 3);
                        assert_eq!(sets[0].len(), 1);
                        assert_eq!(sets[1].len(), 1);
                        assert_eq!(sets[2].len(), 0); // empty set ()
                    }
                    _ => panic!("Expected GROUPING SETS"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // QUALIFY
    // -----------------------------------------------------------------------

    #[test]
    fn test_qualify() {
        let stmt = parse_one(
            "SELECT *, ROW_NUMBER() OVER (PARTITION BY a ORDER BY b) AS rn FROM t QUALIFY rn = 1",
        );
        match stmt {
            Statement::Select(s) => {
                assert!(s.qualify.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE TABLE WITH options
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_table_with_options() {
        let stmt = parse_one(
            "CREATE TABLE t (id INT) WITH (change_data_feed = true, retention_period = '90 days')",
        );
        match stmt {
            Statement::CreateTable(ct) => {
                assert_eq!(ct.name, "t");
                assert_eq!(ct.options.len(), 2);
                assert_eq!(ct.options[0].key, "change_data_feed");
                assert_eq!(ct.options[0].value, TableOptionValue::Boolean(true));
                assert_eq!(ct.options[1].key, "retention_period");
                assert_eq!(
                    ct.options[1].value,
                    TableOptionValue::String("90 days".to_string())
                );
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // AS OF TIMESTAMP (time travel)
    // -----------------------------------------------------------------------

    #[test]
    fn test_as_of_timestamp() {
        let stmt = parse_one("SELECT * FROM orders AS OF TIMESTAMP '2024-01-01'");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Table { name, as_of, .. } => {
                    assert_eq!(name, "orders");
                    assert!(matches!(as_of.as_deref(), Some(AsOf::Timestamp(_))));
                }
                _ => panic!("Expected Table"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // Vector distance operators
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_cosine_distance() {
        let stmt = parse_one("SELECT * FROM items ORDER BY embedding <=> query_vec LIMIT 10");
        match stmt {
            Statement::Select(s) => {
                assert_eq!(s.order_by.len(), 1);
                match &s.order_by[0].expr {
                    Expr::VectorDistance { op, .. } => {
                        assert_eq!(*op, VectorDistanceOp::Cosine);
                    }
                    _ => panic!("Expected VectorDistance"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_vector_l2_distance() {
        let stmt = parse_one("SELECT embedding <-> query_vec AS dist FROM items");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::VectorDistance { op, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, VectorDistanceOp::L2);
                } else {
                    panic!("Expected VectorDistance");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_vector_dot_distance() {
        let stmt = parse_one("SELECT embedding <#> query_vec AS dist FROM items");
        match stmt {
            Statement::Select(s) => {
                if let SelectItem::Expr(Expr::VectorDistance { op, .. }, _) = &s.projections[0] {
                    assert_eq!(*op, VectorDistanceOp::DotProduct);
                } else {
                    panic!("Expected VectorDistance");
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE/ALTER/DROP USER
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_user() {
        let stmt = parse_one("CREATE USER alice WITH PASSWORD 'secret123' SUPERUSER LOGIN");
        match stmt {
            Statement::CreateUser(u) => {
                assert_eq!(u.name, "alice");
                assert_eq!(u.password, Some("secret123".to_string()));
                assert_eq!(u.options.len(), 2);
                assert!(matches!(&u.options[0], UserOption::Superuser(true)));
                assert!(matches!(&u.options[1], UserOption::Login(true)));
            }
            _ => panic!("Expected CREATE USER"),
        }
    }

    #[test]
    fn test_create_user_with_valid_until() {
        let stmt = parse_one("CREATE USER bob WITH PASSWORD 'pw' VALID UNTIL '2025-12-31'");
        match stmt {
            Statement::CreateUser(u) => {
                assert_eq!(u.name, "bob");
                assert!(
                    u.options
                        .iter()
                        .any(|o| matches!(o, UserOption::ValidUntil(d) if d == "2025-12-31"))
                );
            }
            _ => panic!("Expected CREATE USER"),
        }
    }

    #[test]
    fn test_drop_user() {
        let stmt = parse_one("DROP USER alice");
        match stmt {
            Statement::DropUser(u) => {
                assert_eq!(u.name, "alice");
                assert!(!u.if_exists);
            }
            _ => panic!("Expected DROP USER"),
        }
    }

    #[test]
    fn test_drop_user_if_exists() {
        let stmt = parse_one("DROP USER IF EXISTS alice");
        match stmt {
            Statement::DropUser(u) => {
                assert!(u.if_exists);
            }
            _ => panic!("Expected DROP USER"),
        }
    }

    #[test]
    fn test_alter_user_set_password() {
        let stmt = parse_one("ALTER USER alice SET PASSWORD 'newpass'");
        match stmt {
            Statement::AlterUser(u) => {
                assert_eq!(u.name, "alice");
                assert!(
                    matches!(u.operation, AlterUserOperation::SetPassword(ref pw) if pw == "newpass")
                );
            }
            _ => panic!("Expected ALTER USER"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE/ALTER/DROP ROLE
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_role() {
        let stmt = parse_one("CREATE ROLE admin WITH SUPERUSER LOGIN");
        match stmt {
            Statement::CreateRole(r) => {
                assert_eq!(r.name, "admin");
                assert_eq!(r.options.len(), 2);
            }
            _ => panic!("Expected CREATE ROLE"),
        }
    }

    #[test]
    fn test_drop_role() {
        let stmt = parse_one("DROP ROLE admin");
        match stmt {
            Statement::DropRole(r) => {
                assert_eq!(r.name, "admin");
                assert!(!r.if_exists);
            }
            _ => panic!("Expected DROP ROLE"),
        }
    }

    #[test]
    fn test_drop_role_if_exists() {
        let stmt = parse_one("DROP ROLE IF EXISTS admin");
        match stmt {
            Statement::DropRole(r) => {
                assert!(r.if_exists);
            }
            _ => panic!("Expected DROP ROLE"),
        }
    }

    #[test]
    fn test_alter_role_rename() {
        let stmt = parse_one("ALTER ROLE admin RENAME TO superadmin");
        match stmt {
            Statement::AlterRole(r) => {
                assert_eq!(r.name, "admin");
                assert!(
                    matches!(r.operation, AlterUserOperation::Rename { ref new_name } if new_name == "superadmin")
                );
            }
            _ => panic!("Expected ALTER ROLE"),
        }
    }

    // -----------------------------------------------------------------------
    // PIPELINE
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_pipeline() {
        let stmt = parse_one(
            "CREATE PIPELINE etl AS (\
                STAGE ingest (\
                    SOURCE raw_data, \
                    TARGET staging, \
                    MODE append\
                )\
            )",
        );
        match stmt {
            Statement::CreatePipeline(p) => {
                assert_eq!(p.name, "etl");
                assert_eq!(p.stages.len(), 1);
                assert_eq!(p.stages[0].name, "ingest");
                assert_eq!(p.stages[0].source, "raw_data");
                assert_eq!(p.stages[0].target, "staging");
                assert_eq!(p.stages[0].mode, Some("append".to_string()));
            }
            _ => panic!("Expected CREATE PIPELINE"),
        }
    }

    #[test]
    fn test_run_pipeline() {
        let stmt = parse_one("RUN PIPELINE etl");
        match stmt {
            Statement::RunPipeline(r) => {
                assert_eq!(r.name, "etl");
                assert!(r.stage.is_none());
            }
            _ => panic!("Expected RUN PIPELINE"),
        }
    }

    #[test]
    fn test_run_pipeline_with_stage() {
        let stmt = parse_one("RUN PIPELINE etl STAGE ingest");
        match stmt {
            Statement::RunPipeline(r) => {
                assert_eq!(r.name, "etl");
                assert_eq!(r.stage, Some("ingest".to_string()));
            }
            _ => panic!("Expected RUN PIPELINE"),
        }
    }

    #[test]
    fn test_drop_pipeline() {
        let stmt = parse_one("DROP PIPELINE etl");
        match stmt {
            Statement::DropPipeline(p) => {
                assert_eq!(p.name, "etl");
                assert!(!p.if_exists);
            }
            _ => panic!("Expected DROP PIPELINE"),
        }
    }

    #[test]
    fn test_drop_pipeline_if_exists() {
        let stmt = parse_one("DROP PIPELINE IF EXISTS etl");
        match stmt {
            Statement::DropPipeline(p) => {
                assert!(p.if_exists);
            }
            _ => panic!("Expected DROP PIPELINE"),
        }
    }

    // -----------------------------------------------------------------------
    // ARCHIVE / RESTORE
    // -----------------------------------------------------------------------

    #[test]
    fn test_archive_table() {
        let stmt =
            parse_one("ARCHIVE TABLE events WHERE created_at < '2023-01-01' TO '/archive/events'");
        match stmt {
            Statement::ArchiveTable(a) => {
                assert_eq!(a.table, "events");
                assert!(a.where_clause.is_some());
                assert_eq!(a.destination, "/archive/events");
            }
            _ => panic!("Expected ARCHIVE TABLE"),
        }
    }

    #[test]
    fn test_archive_table_no_where() {
        let stmt = parse_one("ARCHIVE TABLE old_data TO '/backup/old'");
        match stmt {
            Statement::ArchiveTable(a) => {
                assert_eq!(a.table, "old_data");
                assert!(a.where_clause.is_none());
                assert_eq!(a.destination, "/backup/old");
            }
            _ => panic!("Expected ARCHIVE TABLE"),
        }
    }

    #[test]
    fn test_restore_table() {
        let stmt = parse_one("RESTORE TABLE events FROM '/archive/events'");
        match stmt {
            Statement::RestoreTable(r) => {
                assert_eq!(r.table, "events");
                assert_eq!(r.source, "/archive/events");
                assert!(r.into_table.is_none());
            }
            _ => panic!("Expected RESTORE TABLE"),
        }
    }

    #[test]
    fn test_restore_table_into() {
        let stmt = parse_one("RESTORE TABLE events FROM '/archive/events' INTO events_restored");
        match stmt {
            Statement::RestoreTable(r) => {
                assert_eq!(r.table, "events");
                assert_eq!(r.into_table, Some("events_restored".to_string()));
            }
            _ => panic!("Expected RESTORE TABLE"),
        }
    }

    // -----------------------------------------------------------------------
    // ALTER TABLE SET (options)
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_table_set_options() {
        let stmt = parse_one("ALTER TABLE events SET (retention_days = 90, compress = true)");
        match stmt {
            Statement::AlterTableOptions(a) => {
                assert_eq!(a.table, "events");
                assert_eq!(a.options.len(), 2);
                assert_eq!(a.options[0].key, "retention_days");
                assert_eq!(a.options[0].value, TableOptionValue::Integer(90));
                assert_eq!(a.options[1].key, "compress");
                assert_eq!(a.options[1].value, TableOptionValue::Boolean(true));
            }
            _ => panic!("Expected AlterTableOptions"),
        }
    }

    // -----------------------------------------------------------------------
    // ADD/DROP EXPECTATION
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_table_add_expectation() {
        let stmt = parse_one(
            "ALTER TABLE orders ADD EXPECTATION positive_amount EXPECT amount > 0 ON VIOLATION FAIL",
        );
        match stmt {
            Statement::AddExpectation(a) => {
                assert_eq!(a.table, "orders");
                assert_eq!(a.name, "positive_amount");
                assert_eq!(a.on_violation, ViolationAction::Fail);
            }
            _ => panic!("Expected AddExpectation"),
        }
    }

    #[test]
    fn test_alter_table_add_expectation_quarantine() {
        let stmt = parse_one(
            "ALTER TABLE orders ADD EXPECTATION valid_email EXPECT email IS NOT NULL ON VIOLATION QUARANTINE",
        );
        match stmt {
            Statement::AddExpectation(a) => {
                assert_eq!(a.name, "valid_email");
                assert_eq!(a.on_violation, ViolationAction::Quarantine);
            }
            _ => panic!("Expected AddExpectation"),
        }
    }

    #[test]
    fn test_alter_table_drop_expectation() {
        let stmt = parse_one("ALTER TABLE orders DROP EXPECTATION positive_amount");
        match stmt {
            Statement::DropExpectation(d) => {
                assert_eq!(d.table, "orders");
                assert_eq!(d.name, "positive_amount");
            }
            _ => panic!("Expected DropExpectation"),
        }
    }

    // -----------------------------------------------------------------------
    // ENABLE / DISABLE
    // -----------------------------------------------------------------------

    #[test]
    fn test_alter_table_enable_feature() {
        let stmt = parse_one("ALTER TABLE events ENABLE change_data_feed");
        match stmt {
            Statement::EnableFeature(e) => {
                assert_eq!(e.table, "events");
                assert_eq!(e.feature, "change_data_feed");
            }
            _ => panic!("Expected EnableFeature"),
        }
    }

    #[test]
    fn test_alter_table_disable_feature() {
        let stmt = parse_one("ALTER TABLE events DISABLE change_data_feed");
        match stmt {
            Statement::DisableFeature(d) => {
                assert_eq!(d.table, "events");
                assert_eq!(d.feature, "change_data_feed");
            }
            _ => panic!("Expected DisableFeature"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE FULLTEXT INDEX
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_fulltext_index() {
        let stmt = parse_one(
            "CREATE FULLTEXT INDEX ft_docs ON documents (title, body) WITH (analyzer = 'standard')",
        );
        match stmt {
            Statement::CreateFulltextIndex(f) => {
                assert_eq!(f.name, "ft_docs");
                assert_eq!(f.table, "documents");
                assert_eq!(f.columns, vec!["title", "body"]);
                assert_eq!(f.options.len(), 1);
                assert_eq!(f.options[0].key, "analyzer");
            }
            _ => panic!("Expected CREATE FULLTEXT INDEX"),
        }
    }

    #[test]
    fn test_create_fulltext_index_no_options() {
        let stmt = parse_one("CREATE FULLTEXT INDEX ft_docs ON documents (title)");
        match stmt {
            Statement::CreateFulltextIndex(f) => {
                assert_eq!(f.name, "ft_docs");
                assert!(f.options.is_empty());
            }
            _ => panic!("Expected CREATE FULLTEXT INDEX"),
        }
    }

    // -----------------------------------------------------------------------
    // CREATE VECTOR INDEX
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_vector_index() {
        let stmt = parse_one(
            "CREATE VECTOR INDEX vec_idx ON items (embedding) WITH (metric = 'cosine', lists = 100)",
        );
        match stmt {
            Statement::CreateVectorIndex(v) => {
                assert_eq!(v.name, "vec_idx");
                assert_eq!(v.table, "items");
                assert_eq!(v.column, "embedding");
                assert_eq!(v.options.len(), 2);
                assert_eq!(v.options[0].key, "metric");
                assert_eq!(v.options[1].key, "lists");
            }
            _ => panic!("Expected CREATE VECTOR INDEX"),
        }
    }

    #[test]
    fn test_create_vector_index_no_options() {
        let stmt = parse_one("CREATE VECTOR INDEX vec_idx ON items (embedding)");
        match stmt {
            Statement::CreateVectorIndex(v) => {
                assert_eq!(v.name, "vec_idx");
                assert!(v.options.is_empty());
            }
            _ => panic!("Expected CREATE VECTOR INDEX"),
        }
    }

    // -----------------------------------------------------------------------
    // MATCH AGAINST
    // -----------------------------------------------------------------------

    #[test]
    fn test_match_against() {
        let stmt = parse_one("SELECT * FROM docs WHERE MATCH(title, body) AGAINST ('search term')");
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::MatchAgainst { columns, mode, .. }) => {
                    assert_eq!(*columns, vec!["title".to_string(), "body".to_string()]);
                    assert!(mode.is_none());
                }
                _ => panic!("Expected MatchAgainst"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_match_against_with_mode() {
        let stmt = parse_one(
            "SELECT * FROM docs WHERE MATCH(title) AGAINST ('search' IN NATURAL LANGUAGE MODE)",
        );
        match stmt {
            Statement::Select(s) => match s.where_clause.as_deref() {
                Some(Expr::MatchAgainst { columns, mode, .. }) => {
                    assert_eq!(columns.len(), 1);
                    assert_eq!(mode.as_deref(), Some("natural language mode"));
                }
                _ => panic!("Expected MatchAgainst"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    // -----------------------------------------------------------------------
    // Branching / Versioning
    // -----------------------------------------------------------------------

    #[test]
    fn test_version_as_of() {
        let stmt = parse_one("SELECT * FROM orders VERSION AS OF 42");
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Table { name, as_of, .. } => {
                    assert_eq!(name, "orders");
                    assert!(matches!(as_of.as_deref(), Some(AsOf::Version(_))));
                }
                _ => panic!("Expected Table"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_create_branch_from_at_version() {
        let stmt = parse_one("CREATE BRANCH dev FROM main AT VERSION 10");
        match stmt {
            Statement::CreateBranch(b) => {
                assert_eq!(b.name, "dev");
                assert_eq!(b.from_branch.as_deref(), Some("main"));
                assert!(b.at_version.is_some());
            }
            _ => panic!("Expected CreateBranch"),
        }
    }

    #[test]
    fn test_merge_branch_into() {
        let stmt = parse_one("MERGE BRANCH dev INTO main");
        match stmt {
            Statement::MergeBranch(m) => {
                assert_eq!(m.source, "dev");
                assert_eq!(m.into_target, "main");
            }
            _ => panic!("Expected MergeBranch"),
        }
    }

    #[test]
    fn test_drop_branch_if_exists() {
        let stmt = parse_one("DROP BRANCH IF EXISTS dev");
        match stmt {
            Statement::DropBranch(d) => {
                assert_eq!(d.name, "dev");
                assert!(d.if_exists);
            }
            _ => panic!("Expected DropBranch"),
        }
    }

    #[test]
    fn test_use_branch() {
        let stmt = parse_one("USE BRANCH dev");
        match stmt {
            Statement::UseBranch(u) => {
                assert_eq!(u.name, "dev");
            }
            _ => panic!("Expected UseBranch"),
        }
    }

    #[test]
    fn test_create_version() {
        let stmt = parse_one("CREATE VERSION v1 ON orders AS OF VERSION 5");
        match stmt {
            Statement::CreateVersion(v) => {
                assert_eq!(v.name, "v1");
                assert_eq!(v.table, "orders");
                assert!(v.at_version.is_some());
            }
            _ => panic!("Expected CreateVersion"),
        }
    }

    #[test]
    fn test_create_branch_simple() {
        let stmt = parse_one("CREATE BRANCH feature_x");
        match stmt {
            Statement::CreateBranch(b) => {
                assert_eq!(b.name, "feature_x");
                assert!(b.from_branch.is_none());
                assert!(b.at_version.is_none());
            }
            _ => panic!("Expected CreateBranch"),
        }
    }

    #[test]
    fn test_drop_branch_no_if_exists() {
        let stmt = parse_one("DROP BRANCH dev");
        match stmt {
            Statement::DropBranch(d) => {
                assert_eq!(d.name, "dev");
                assert!(!d.if_exists);
            }
            _ => panic!("Expected DropBranch"),
        }
    }

    #[test]
    fn test_for_portion_of() {
        let stmt = parse_one(
            "SELECT * FROM emp FOR PORTION OF employment_period FROM '2024-01-01' TO '2024-06-01'",
        );
        match stmt {
            Statement::Select(s) => match &s.from[0] {
                TableRef::Table { as_of, .. } => match as_of.as_deref() {
                    Some(AsOf::ForPortionOf { period, .. }) => {
                        assert_eq!(period, "employment_period");
                    }
                    _ => panic!("Expected ForPortionOf"),
                },
                _ => panic!("Expected Table"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_restore_table_with_version() {
        let stmt = parse_one("RESTORE TABLE orders FROM '/backup/orders' TO VERSION AS OF 7");
        match stmt {
            Statement::RestoreTable(r) => {
                assert_eq!(r.table, "orders");
                assert_eq!(r.source, "/backup/orders");
                assert!(r.at_version.is_some());
                assert!(r.at_timestamp.is_none());
            }
            _ => panic!("Expected RestoreTable"),
        }
    }

    // -----------------------------------------------------------------------
    // CDC statement tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_replication_slot() {
        let stmt = parse_one("CREATE REPLICATION SLOT my_slot PLUGIN 'wal2json'");
        match stmt {
            Statement::CreateReplicationSlot(s) => {
                assert_eq!(s.name, "my_slot");
                assert_eq!(s.plugin, "wal2json");
                assert!(s.table_filter.is_empty());
            }
            _ => panic!("Expected CreateReplicationSlot"),
        }
    }

    #[test]
    fn test_create_replication_slot_with_filter() {
        let stmt =
            parse_one("CREATE REPLICATION SLOT my_slot PLUGIN 'wal2json' FOR TABLE orders, users");
        match stmt {
            Statement::CreateReplicationSlot(s) => {
                assert_eq!(s.name, "my_slot");
                assert_eq!(s.plugin, "wal2json");
                assert_eq!(s.table_filter, vec!["orders", "users"]);
            }
            _ => panic!("Expected CreateReplicationSlot"),
        }
    }

    #[test]
    fn test_drop_replication_slot() {
        let stmt = parse_one("DROP REPLICATION SLOT my_slot");
        match stmt {
            Statement::DropReplicationSlot(s) => {
                assert_eq!(s.name, "my_slot");
            }
            _ => panic!("Expected DropReplicationSlot"),
        }
    }

    #[test]
    fn test_create_cdc_stream() {
        let stmt = parse_one(
            "CREATE CDC STREAM order_stream ON orders TO kafka WITH (topic = 'order_events', format = 'avro')",
        );
        match stmt {
            Statement::CreateCdcStream(s) => {
                assert_eq!(s.name, "order_stream");
                assert_eq!(s.table_name, "orders");
                assert_eq!(s.sink_type, "kafka");
                assert_eq!(s.options.len(), 2);
                assert_eq!(s.options[0].key, "topic");
                assert_eq!(s.options[1].key, "format");
            }
            _ => panic!("Expected CreateCdcStream"),
        }
    }

    #[test]
    fn test_create_cdc_stream_no_options() {
        let stmt = parse_one("CREATE CDC STREAM order_stream ON orders TO kafka");
        match stmt {
            Statement::CreateCdcStream(s) => {
                assert_eq!(s.name, "order_stream");
                assert_eq!(s.table_name, "orders");
                assert_eq!(s.sink_type, "kafka");
                assert!(s.options.is_empty());
            }
            _ => panic!("Expected CreateCdcStream"),
        }
    }

    #[test]
    fn test_drop_cdc_stream() {
        let stmt = parse_one("DROP CDC STREAM order_stream");
        match stmt {
            Statement::DropCdcStream(s) => {
                assert_eq!(s.name, "order_stream");
            }
            _ => panic!("Expected DropCdcStream"),
        }
    }

    #[test]
    fn test_create_cdc_ingest() {
        let stmt = parse_one(
            "CREATE CDC INGEST kafka_ingest FROM kafka INTO raw_events WITH (topic = 'events', group_id = 'cdc_group')",
        );
        match stmt {
            Statement::CreateCdcIngest(s) => {
                assert_eq!(s.name, "kafka_ingest");
                assert_eq!(s.source_type, "kafka");
                assert_eq!(s.target_table, "raw_events");
                assert_eq!(s.options.len(), 2);
                assert_eq!(s.options[0].key, "topic");
                assert_eq!(s.options[1].key, "group_id");
            }
            _ => panic!("Expected CreateCdcIngest"),
        }
    }

    #[test]
    fn test_create_cdc_ingest_no_options() {
        let stmt = parse_one("CREATE CDC INGEST kafka_ingest FROM kafka INTO raw_events");
        match stmt {
            Statement::CreateCdcIngest(s) => {
                assert_eq!(s.name, "kafka_ingest");
                assert_eq!(s.source_type, "kafka");
                assert_eq!(s.target_table, "raw_events");
                assert!(s.options.is_empty());
            }
            _ => panic!("Expected CreateCdcIngest"),
        }
    }

    #[test]
    fn test_drop_cdc_ingest() {
        let stmt = parse_one("DROP CDC INGEST kafka_ingest");
        match stmt {
            Statement::DropCdcIngest(s) => {
                assert_eq!(s.name, "kafka_ingest");
            }
            _ => panic!("Expected DropCdcIngest"),
        }
    }

    #[test]
    fn test_create_publication_for_tables() {
        let stmt = parse_one("CREATE PUBLICATION my_pub FOR TABLE orders, users");
        match stmt {
            Statement::CreatePublication(p) => {
                assert_eq!(p.name, "my_pub");
                assert_eq!(p.tables, vec!["orders", "users"]);
                assert!(!p.all_tables);
                assert!(!p.include_ddl);
            }
            _ => panic!("Expected CreatePublication"),
        }
    }

    #[test]
    fn test_create_publication_all_tables() {
        let stmt = parse_one("CREATE PUBLICATION my_pub FOR ALL TABLE");
        match stmt {
            Statement::CreatePublication(p) => {
                assert_eq!(p.name, "my_pub");
                assert!(p.all_tables);
                assert!(p.tables.is_empty());
                assert!(!p.include_ddl);
            }
            _ => panic!("Expected CreatePublication"),
        }
    }

    #[test]
    fn test_create_publication_include_ddl() {
        let stmt = parse_one("CREATE PUBLICATION my_pub FOR TABLE orders INCLUDE DDL");
        match stmt {
            Statement::CreatePublication(p) => {
                assert_eq!(p.name, "my_pub");
                assert_eq!(p.tables, vec!["orders"]);
                assert!(p.include_ddl);
            }
            _ => panic!("Expected CreatePublication"),
        }
    }

    #[test]
    fn test_alter_publication_add_table() {
        let stmt = parse_one("ALTER PUBLICATION my_pub ADD TABLE users");
        match stmt {
            Statement::AlterPublication(a) => {
                assert_eq!(a.name, "my_pub");
                assert_eq!(
                    a.action,
                    AlterPublicationAction::AddTable("users".to_string())
                );
            }
            _ => panic!("Expected AlterPublication"),
        }
    }

    #[test]
    fn test_alter_publication_drop_table() {
        let stmt = parse_one("ALTER PUBLICATION my_pub DROP TABLE users");
        match stmt {
            Statement::AlterPublication(a) => {
                assert_eq!(a.name, "my_pub");
                assert_eq!(
                    a.action,
                    AlterPublicationAction::DropTable("users".to_string())
                );
            }
            _ => panic!("Expected AlterPublication"),
        }
    }

    #[test]
    fn test_drop_publication() {
        let stmt = parse_one("DROP PUBLICATION my_pub");
        match stmt {
            Statement::DropPublication(p) => {
                assert_eq!(p.name, "my_pub");
            }
            _ => panic!("Expected DropPublication"),
        }
    }
}
