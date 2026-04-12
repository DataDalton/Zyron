//! RETURNING clause with OLD/NEW value resolution for DML operations.
//!
//! For UPDATE: old.* = values before the update, new.* = values after.
//! For INSERT: old.* = NULL, new.* = inserted values.
//! For DELETE: old.* = deleted values, new.* = NULL.

// ---------------------------------------------------------------------------
// ReturnSource
// ---------------------------------------------------------------------------

/// Source of a return column value in a RETURNING clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnSource {
    /// Value from before the mutation (pre-image).
    Old,
    /// Value from after the mutation (post-image).
    New,
    /// Arbitrary expression evaluated against available row data.
    Expr,
}

// ---------------------------------------------------------------------------
// ReturnColumn
// ---------------------------------------------------------------------------

/// A single column in a RETURNING clause.
#[derive(Debug, Clone)]
pub struct ReturnColumn {
    pub name: String,
    pub source: ReturnSource,
    /// Index into the table schema for Old/New sources.
    pub column_index: usize,
    pub alias: Option<String>,
}

impl ReturnColumn {
    /// Returns the output name (alias if present, otherwise name).
    pub fn output_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.name)
    }
}

// ---------------------------------------------------------------------------
// ReturnClause
// ---------------------------------------------------------------------------

/// Parsed RETURNING clause containing one or more return columns.
#[derive(Debug, Clone)]
pub struct ReturnClause {
    pub columns: Vec<ReturnColumn>,
}

impl ReturnClause {
    pub fn new(columns: Vec<ReturnColumn>) -> Self {
        Self { columns }
    }

    /// Returns true if any column references the old (pre-image) row.
    pub fn needs_old_values(&self) -> bool {
        self.columns.iter().any(|c| c.source == ReturnSource::Old)
    }

    /// Returns true if any column references the new (post-image) row.
    pub fn needs_new_values(&self) -> bool {
        self.columns.iter().any(|c| c.source == ReturnSource::New)
    }
}

// ---------------------------------------------------------------------------
// OldNewResolver
// ---------------------------------------------------------------------------

/// Captures pre-update and post-update values during DML operations.
///
/// Resolves RETURNING columns by picking values from old or new row data.
/// Row data is represented as byte slices (serialized tuple values).
pub struct OldNewResolver {
    return_clause: ReturnClause,
    num_columns: usize,
}

impl OldNewResolver {
    pub fn new(return_clause: ReturnClause, num_columns: usize) -> Self {
        Self {
            return_clause,
            num_columns,
        }
    }

    /// Resolves a single row given optional old and new byte arrays.
    /// Returns pairs of (output_name, value_bytes) where value_bytes is
    /// None for NULL (e.g., old.* on INSERT).
    pub fn resolve_row<'a>(
        &self,
        old_row: Option<&'a [Vec<u8>]>,
        new_row: Option<&'a [Vec<u8>]>,
    ) -> Vec<(&str, Option<&'a [u8]>)> {
        let mut result = Vec::with_capacity(self.return_clause.columns.len());

        for col in &self.return_clause.columns {
            let value = match col.source {
                ReturnSource::Old => old_row
                    .and_then(|row| row.get(col.column_index))
                    .map(|v| v.as_slice()),
                ReturnSource::New => new_row
                    .and_then(|row| row.get(col.column_index))
                    .map(|v| v.as_slice()),
                ReturnSource::Expr => {
                    // Expression evaluation is handled by the executor.
                    // The resolver returns None, and the caller evaluates the expression.
                    None
                }
            };
            result.push((col.output_name(), value));
        }

        result
    }

    /// Returns the return clause reference.
    pub fn return_clause(&self) -> &ReturnClause {
        &self.return_clause
    }

    /// Returns the number of output columns.
    pub fn output_column_count(&self) -> usize {
        self.return_clause.columns.len()
    }

    /// Returns the number of table columns.
    pub fn table_column_count(&self) -> usize {
        self.num_columns
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clause() -> ReturnClause {
        ReturnClause::new(vec![
            ReturnColumn {
                name: "x".into(),
                source: ReturnSource::Old,
                column_index: 0,
                alias: Some("prev_x".into()),
            },
            ReturnColumn {
                name: "x".into(),
                source: ReturnSource::New,
                column_index: 0,
                alias: Some("curr_x".into()),
            },
        ])
    }

    #[test]
    fn test_return_column_output_name() {
        let col = ReturnColumn {
            name: "x".into(),
            source: ReturnSource::Old,
            column_index: 0,
            alias: Some("prev".into()),
        };
        assert_eq!(col.output_name(), "prev");

        let col_no_alias = ReturnColumn {
            name: "x".into(),
            source: ReturnSource::New,
            column_index: 0,
            alias: None,
        };
        assert_eq!(col_no_alias.output_name(), "x");
    }

    #[test]
    fn test_return_clause_needs() {
        let clause = make_clause();
        assert!(clause.needs_old_values());
        assert!(clause.needs_new_values());

        let insert_clause = ReturnClause::new(vec![ReturnColumn {
            name: "id".into(),
            source: ReturnSource::New,
            column_index: 0,
            alias: None,
        }]);
        assert!(!insert_clause.needs_old_values());
        assert!(insert_clause.needs_new_values());
    }

    #[test]
    fn test_resolve_update_row() {
        let clause = make_clause();
        let resolver = OldNewResolver::new(clause, 2);

        let old_row = vec![vec![10u8], vec![20u8]];
        let new_row = vec![vec![30u8], vec![40u8]];

        let result = resolver.resolve_row(Some(&old_row), Some(&new_row));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "prev_x");
        assert_eq!(result[0].1, Some(&[10u8][..]));
        assert_eq!(result[1].0, "curr_x");
        assert_eq!(result[1].1, Some(&[30u8][..]));
    }

    #[test]
    fn test_resolve_insert_row() {
        let clause = make_clause();
        let resolver = OldNewResolver::new(clause, 2);

        let new_row = vec![vec![30u8], vec![40u8]];

        let result = resolver.resolve_row(None, Some(&new_row));
        assert_eq!(result[0].0, "prev_x");
        assert_eq!(result[0].1, None); // old is NULL for INSERT
        assert_eq!(result[1].0, "curr_x");
        assert_eq!(result[1].1, Some(&[30u8][..]));
    }

    #[test]
    fn test_resolve_delete_row() {
        let clause = make_clause();
        let resolver = OldNewResolver::new(clause, 2);

        let old_row = vec![vec![10u8], vec![20u8]];

        let result = resolver.resolve_row(Some(&old_row), None);
        assert_eq!(result[0].0, "prev_x");
        assert_eq!(result[0].1, Some(&[10u8][..]));
        assert_eq!(result[1].0, "curr_x");
        assert_eq!(result[1].1, None); // new is NULL for DELETE
    }

    #[test]
    fn test_resolver_counts() {
        let clause = make_clause();
        let resolver = OldNewResolver::new(clause, 5);
        assert_eq!(resolver.output_column_count(), 2);
        assert_eq!(resolver.table_column_count(), 5);
    }
}
