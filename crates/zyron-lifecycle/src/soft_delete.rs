//! Soft-delete helpers. Pure functions that produce parser `Expr` and
//! `Assignment` nodes for the binder to inject. No state.

use zyron_catalog::schema::{LifecycleConfig, TableEntry};
use zyron_parser::ast::{Assignment, BinaryOperator, Expr, LiteralValue};

/// Resolved soft-delete configuration for a table.
#[derive(Debug, Clone)]
pub struct SoftDeleteConfig {
    pub is_deleted_column: String,
    pub deleted_at_column: String,
}

/// Returns the soft-delete configuration when enabled. Column names are
/// resolved from the table entry's columns by the stored column ids.
pub fn soft_delete_config(entry: &TableEntry) -> Option<SoftDeleteConfig> {
    let lc: &LifecycleConfig = &entry.lifecycle;
    if !lc.soft_delete_enabled {
        return None;
    }
    let is_deleted = column_name_by_id(entry, lc.soft_delete_is_deleted_col_id)
        .unwrap_or_else(|| "is_deleted".to_string());
    let deleted_at = column_name_by_id(entry, lc.soft_delete_deleted_at_col_id)
        .unwrap_or_else(|| "deleted_at".to_string());
    Some(SoftDeleteConfig {
        is_deleted_column: is_deleted,
        deleted_at_column: deleted_at,
    })
}

fn column_name_by_id(entry: &TableEntry, col_id: u32) -> Option<String> {
    if col_id == 0 {
        return None;
    }
    entry
        .columns
        .iter()
        .find(|c| c.id.0 as u32 == col_id)
        .map(|c| c.name.clone())
}

/// `<is_deleted_column> = false`
pub fn build_is_deleted_false_predicate(cfg: &SoftDeleteConfig) -> Expr {
    eq(
        Expr::Identifier(cfg.is_deleted_column.clone()),
        Expr::Literal(LiteralValue::Boolean(false)),
    )
}

/// `<is_deleted_column> = true`
pub fn build_is_deleted_true_predicate(cfg: &SoftDeleteConfig) -> Expr {
    eq(
        Expr::Identifier(cfg.is_deleted_column.clone()),
        Expr::Literal(LiteralValue::Boolean(true)),
    )
}

/// Assignments that mark a row soft-deleted: `is_deleted = true,
/// deleted_at = now()`.
pub fn build_soft_delete_assignments(cfg: &SoftDeleteConfig) -> Vec<Assignment> {
    vec![
        Assignment {
            column: cfg.is_deleted_column.clone(),
            value: Expr::Literal(LiteralValue::Boolean(true)),
        },
        Assignment {
            column: cfg.deleted_at_column.clone(),
            value: Expr::Function {
                name: "now".to_string(),
                args: vec![],
                distinct: false,
            },
        },
    ]
}

/// Assignments that undo a soft delete (RESTORE FROM t WHERE ...).
pub fn build_restore_assignments(cfg: &SoftDeleteConfig) -> Vec<Assignment> {
    vec![
        Assignment {
            column: cfg.is_deleted_column.clone(),
            value: Expr::Literal(LiteralValue::Boolean(false)),
        },
        Assignment {
            column: cfg.deleted_at_column.clone(),
            value: Expr::Literal(LiteralValue::Null),
        },
    ]
}

/// Whether soft-deleted rows on this table are archived before hard purge.
pub fn should_archive_on_purge(entry: &TableEntry) -> bool {
    entry.lifecycle.archive_on_purge
}

fn eq(left: Expr, right: Expr) -> Expr {
    Expr::BinaryOp {
        left: Box::new(left),
        op: BinaryOperator::Eq,
        right: Box::new(right),
    }
}

/// Combines two optional predicates with AND.
pub fn and_combine(a: Option<Expr>, b: Option<Expr>) -> Option<Expr> {
    match (a, b) {
        (Some(x), Some(y)) => Some(Expr::BinaryOp {
            left: Box::new(x),
            op: BinaryOperator::And,
            right: Box::new(y),
        }),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}
