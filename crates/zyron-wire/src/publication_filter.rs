//! ABAC row-filter enforcement on publication change streams.
//!
//! A publication-target ABAC policy attaches a WHERE predicate to a
//! publication. Rows streamed to a subscriber must satisfy that predicate, so
//! the subscriber only sees the rows the policy permits. This is the streaming
//! counterpart to table-target ABAC, which the planner injects at query bind
//! time.
//!
//! The filter is compiled once per subscription: each member table's predicate
//! is parsed and bound against that table's schema, and projection pushdown
//! prunes the decoded columns to exactly those the predicate references. When
//! no policy applies to the subscriber, `build` returns `None` and the
//! streaming hot path pays nothing.

use std::collections::HashMap;

use zyron_auth::{AbacPolicy, SecurityManager};
use zyron_catalog::{Catalog, ColumnEntry, PublicationEntry, TableId};
use zyron_common::{Result, ZyronError};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::PhysicalPlan;

/// Compiled row predicate for one table in a publication. Combines the
/// publication's own WHERE filter, the per-table WHERE filter, and any ABAC
/// policy predicate into one bound expression.
struct CompiledTablePredicate {
    /// Columns the predicate references, in decode order. The predicate's
    /// ColumnRefs resolve by position in this slice.
    output_columns: Vec<LogicalColumn>,
    /// Full table schema, used to decode the NSM tuple bytes.
    table_columns: Vec<ColumnEntry>,
    predicate: BoundExpr,
}

/// Per-table column projection. Carries the full table schema and the indices
/// of the published columns so the streaming path re-encodes each tuple keeping
/// only the published column values and nulling the rest.
struct CompiledProjection {
    table_columns: Vec<ColumnEntry>,
    /// Column ordinals that remain published. Columns outside this set are
    /// nulled in the re-encoded tuple.
    keep: Vec<usize>,
}

/// Per-subscription row filter enforcing a publication's WHERE predicates, ABAC
/// policies, and column projection on the change stream. Built once at
/// subscribe time.
pub struct PublicationRowFilter {
    per_table: HashMap<u32, CompiledTablePredicate>,
    projections: HashMap<u32, CompiledProjection>,
}

impl PublicationRowFilter {
    /// Compiles the publication's ABAC policies that apply to `subscriber_role`
    /// into per-table predicates. Returns `Ok(None)` when no policy applies, so
    /// the caller can skip filtering entirely.
    ///
    /// Fails closed: if a policy predicate cannot be bound against a member
    /// table (for example it references a column the table does not have), the
    /// subscription is rejected rather than streaming unfiltered rows.
    pub async fn build(
        catalog: &Catalog,
        sm: &SecurityManager,
        publication: &PublicationEntry,
        table_ids: &[u32],
        subscriber_role: u32,
    ) -> Result<Option<Self>> {
        let policies: Vec<AbacPolicy> = sm
            .abac_store
            .policies_for_publication(publication.id.0)
            .into_iter()
            .filter(|p| p.enabled && role_applies(&p.roles, subscriber_role))
            .collect();

        // Per-table WHERE predicates and column projections stored on the
        // publication's table map.
        let pub_tables = catalog.get_publication_tables(publication.id);

        let mut per_table = HashMap::with_capacity(table_ids.len());
        let mut projections = HashMap::new();
        for &tid in table_ids {
            let table = catalog.get_table_by_id(TableId(tid))?;
            let schema = catalog.get_schema_by_id(table.schema_id)?;

            // Collect every predicate that applies to this table: the ABAC
            // combination, the publication-level WHERE, and the per-table
            // WHERE. A row must satisfy all of them (AND), so the subscriber
            // only sees rows that pass every filter.
            let mut conjuncts: Vec<String> = Vec::new();
            if !policies.is_empty() {
                conjuncts.push(format!("({})", combine_predicates(&policies)));
            }
            if let Some(p) = publication.where_predicate.as_deref() {
                if !p.is_empty() {
                    conjuncts.push(format!("({p})"));
                }
            }
            if let Some(pt) = pub_tables.iter().find(|pt| pt.table_id.0 == tid) {
                if let Some(p) = pt.where_predicate.as_deref() {
                    if !p.is_empty() {
                        conjuncts.push(format!("({p})"));
                    }
                }
            }

            if !conjuncts.is_empty() {
                let combined = conjuncts.join(" AND ");
                let sql = format!(
                    "SELECT 1 FROM \"{}\".\"{}\" WHERE {}",
                    schema.name, table.name, combined
                );
                let stmt = zyron_parser::parse(&sql)
                    .map_err(|e| {
                        ZyronError::PlanError(format!(
                            "publication predicate '{combined}' does not bind against table '{}': {e}",
                            table.name
                        ))
                    })?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        ZyronError::PlanError("publication predicate produced no statement".into())
                    })?;

                let plan = zyron_planner::plan(
                    catalog,
                    schema.database_id,
                    vec![schema.name.clone()],
                    stmt,
                    None,
                )
                .await?;

                let (output_columns, predicate) = extract_filter(&plan).ok_or_else(|| {
                    ZyronError::PlanError(format!(
                        "publication predicate '{combined}' produced no row filter for table '{}'",
                        table.name
                    ))
                })?;

                per_table.insert(
                    tid,
                    CompiledTablePredicate {
                        output_columns,
                        table_columns: table.columns.clone(),
                        predicate,
                    },
                );
            }

            // Column projection: the published-column set is the per-table
            // column list, or the publication-level projection when the table
            // has no specific list. An empty set means publish all columns.
            let projected: Vec<String> = pub_tables
                .iter()
                .find(|pt| pt.table_id.0 == tid)
                .map(|pt| pt.columns.clone())
                .filter(|c| !c.is_empty())
                .unwrap_or_else(|| publication.columns_projection.clone());
            if !projected.is_empty() {
                let keep: Vec<usize> = table
                    .columns
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| projected.iter().any(|p| p == &c.name))
                    .map(|(i, _)| i)
                    .collect();
                // Only install a projection when it actually drops columns.
                if keep.len() < table.columns.len() {
                    projections.insert(
                        tid,
                        CompiledProjection {
                            table_columns: table.columns.clone(),
                            keep,
                        },
                    );
                }
            }
        }

        if per_table.is_empty() && projections.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self {
            per_table,
            projections,
        }))
    }

    /// Whether this filter has a predicate for the given table.
    #[inline]
    pub fn covers(&self, table_id: u32) -> bool {
        self.per_table.contains_key(&table_id)
    }

    /// Returns a keep mask for `rows` (encoded tuple bytes) of one table, or
    /// `None` when no policy covers the table (caller keeps every row). The
    /// rows are decoded once and the predicate is evaluated vectorized.
    pub fn mask_for(&self, table_id: u32, rows: &[&[u8]]) -> Result<Option<Vec<bool>>> {
        match self.per_table.get(&table_id) {
            None => Ok(None),
            Some(c) => Ok(Some(zyron_executor::evaluate_row_filter(
                &c.output_columns,
                &c.table_columns,
                &c.predicate,
                rows,
            )?)),
        }
    }

    /// Whether this filter projects columns for the given table.
    #[inline]
    pub fn projects(&self, table_id: u32) -> bool {
        self.projections.contains_key(&table_id)
    }

    /// Re-encodes one NSM tuple keeping only the published column values and
    /// nulling the rest, preserving the table's column count so the subscriber
    /// decodes against the unchanged schema. Returns the original bytes
    /// unchanged when the tuple is malformed or no projection applies.
    pub fn project_row(&self, table_id: u32, row: &[u8]) -> Vec<u8> {
        let Some(proj) = self.projections.get(&table_id) else {
            return row.to_vec();
        };
        project_nsm_tuple(row, &proj.table_columns, &proj.keep).unwrap_or_else(|| row.to_vec())
    }
}

/// Re-encodes an NSM tuple nulling every column whose index is absent from
/// `keep`. Kept columns retain their bytes verbatim. Returns None if the tuple
/// does not span its declared schema (malformed record).
fn project_nsm_tuple(data: &[u8], columns: &[ColumnEntry], keep: &[usize]) -> Option<Vec<u8>> {
    let num_cols = columns.len();
    let null_bitmap_len = num_cols.div_ceil(8);
    if data.len() < null_bitmap_len {
        return None;
    }
    let src_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;

    // Build the new tuple: fresh null bitmap, then each column's value bytes.
    let mut body: Vec<u8> = Vec::with_capacity(data.len());
    let mut out_bitmap = vec![0u8; null_bitmap_len];

    for (i, col) in columns.iter().enumerate() {
        let src_null = (src_bitmap[i / 8] >> (i % 8)) & 1 == 1;
        let keep_col = keep.contains(&i);
        let phys = col.physical_type_id();

        if let Some(fixed) = phys.fixed_size() {
            if offset + fixed > data.len() {
                return None;
            }
            if !keep_col {
                // Null the column: set the bit and write zero-filled bytes so
                // the fixed width is preserved for the decoder.
                out_bitmap[i / 8] |= 1 << (i % 8);
                body.extend(std::iter::repeat(0u8).take(fixed));
            } else {
                if src_null {
                    out_bitmap[i / 8] |= 1 << (i % 8);
                }
                body.extend_from_slice(&data[offset..offset + fixed]);
            }
            offset += fixed;
        } else {
            if offset + 4 > data.len() {
                return None;
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + len > data.len() {
                return None;
            }
            if !keep_col {
                // Null the column: set the bit and emit a zero-length value.
                out_bitmap[i / 8] |= 1 << (i % 8);
                body.extend_from_slice(&0u32.to_le_bytes());
            } else {
                if src_null {
                    out_bitmap[i / 8] |= 1 << (i % 8);
                }
                body.extend_from_slice(&data[offset - 4..offset + len]);
            }
            offset += len;
        }
    }

    let mut out = Vec::with_capacity(null_bitmap_len + body.len());
    out.extend_from_slice(&out_bitmap);
    out.extend_from_slice(&body);
    Some(out)
}

/// A policy applies to a subscriber when it targets no specific role, or the
/// subscriber's role is among its target roles.
fn role_applies(policy_roles: &[zyron_auth::RoleId], subscriber_role: u32) -> bool {
    policy_roles.is_empty() || policy_roles.iter().any(|r| r.0 == subscriber_role)
}

/// Combines policies into one SQL predicate. A row is visible when it matches
/// any permissive policy (OR) and every restrictive policy (AND). With no
/// permissive policy the permissive part is unrestricted, leaving only the
/// restrictive conjunction.
fn combine_predicates(policies: &[AbacPolicy]) -> String {
    // Each policy predicate is individually parenthesized so its internal
    // operators cannot rebind across the combinator.
    let permissive: Vec<String> = policies
        .iter()
        .filter(|p| p.permissive)
        .map(|p| format!("({})", p.predicate))
        .collect();
    let restrictive: Vec<String> = policies
        .iter()
        .filter(|p| !p.permissive)
        .map(|p| format!("({})", p.predicate))
        .collect();

    let mut parts: Vec<String> = Vec::new();
    // The whole permissive disjunction is one AND term, wrapped so SQL's
    // AND-binds-tighter-than-OR precedence cannot let a permissive match
    // bypass a restrictive policy (`(a) OR (b) AND (r)` would parse as
    // `a OR (b AND r)`).
    match permissive.len() {
        0 => {}
        1 => parts.push(permissive.into_iter().next().unwrap()),
        _ => parts.push(format!("({})", permissive.join(" OR "))),
    }
    parts.extend(restrictive);
    parts.join(" AND ")
}

/// Extracts the scan output columns and the row predicate from a planned
/// `SELECT 1 FROM t WHERE <pred>`. Projection pushdown prunes the scan columns
/// to those the predicate references; predicate pushdown places the predicate
/// on the scan or a Filter node.
fn extract_filter(plan: &PhysicalPlan) -> Option<(Vec<LogicalColumn>, BoundExpr)> {
    let (cols, pred) = collect(plan);
    Some((cols?, pred?))
}

fn collect(plan: &PhysicalPlan) -> (Option<Vec<LogicalColumn>>, Option<BoundExpr>) {
    match plan {
        PhysicalPlan::SeqScan {
            columns, predicate, ..
        }
        | PhysicalPlan::ParallelSeqScan {
            columns, predicate, ..
        }
        | PhysicalPlan::HybridScan {
            columns, predicate, ..
        } => (Some(columns.clone()), predicate.clone()),
        PhysicalPlan::Filter {
            predicate, child, ..
        } => {
            let (cols, child_pred) = collect(child);
            (cols, Some(and_opt(predicate.clone(), child_pred)))
        }
        PhysicalPlan::Project { child, .. } => collect(child),
        _ => (None, None),
    }
}

/// Combines two predicates with AND, dropping the second when absent.
fn and_opt(left: BoundExpr, right: Option<BoundExpr>) -> BoundExpr {
    match right {
        None => left,
        Some(r) => BoundExpr::BinaryOp {
            left: Box::new(left),
            op: zyron_parser::ast::BinaryOperator::And,
            right: Box::new(r),
            type_id: zyron_common::TypeId::Boolean,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_auth::{AbacTarget, RoleId};

    fn policy(name: &str, predicate: &str, permissive: bool, roles: Vec<RoleId>) -> AbacPolicy {
        AbacPolicy {
            id: 0,
            name: name.to_string(),
            table_id: 1,
            target: AbacTarget::Publication,
            predicate: predicate.to_string(),
            enabled: true,
            permissive,
            roles,
        }
    }

    #[test]
    fn role_applies_empty_means_all() {
        assert!(role_applies(&[], 7));
        assert!(role_applies(&[RoleId(7)], 7));
        assert!(!role_applies(&[RoleId(8)], 7));
        assert!(role_applies(&[RoleId(8), RoleId(7)], 7));
    }

    #[test]
    fn combine_single_permissive() {
        let p = vec![policy("a", "region = 'us'", true, vec![])];
        assert_eq!(combine_predicates(&p), "(region = 'us')");
    }

    #[test]
    fn combine_permissive_or() {
        let p = vec![
            policy("a", "region = 'us'", true, vec![]),
            policy("b", "region = 'eu'", true, vec![]),
        ];
        // A row is visible if it matches any permissive policy. The disjunction
        // is wrapped as one unit.
        assert_eq!(
            combine_predicates(&p),
            "((region = 'us') OR (region = 'eu'))"
        );
    }

    #[test]
    fn combine_permissive_and_restrictive() {
        let p = vec![
            policy("a", "region = 'us'", true, vec![]),
            policy("r", "tier > 2", false, vec![]),
        ];
        // Visible rows match the permissive set AND every restrictive policy.
        assert_eq!(combine_predicates(&p), "(region = 'us') AND (tier > 2)");
    }

    #[test]
    fn combine_multi_permissive_with_restrictive_is_grouped() {
        let p = vec![
            policy("a", "region = 'us'", true, vec![]),
            policy("b", "region = 'eu'", true, vec![]),
            policy("r", "tier > 2", false, vec![]),
        ];
        // The permissive OR must be one parenthesized term so a row matching a
        // permissive policy still has to satisfy the restrictive policy. Without
        // the wrap this would parse as `a OR (b AND r)` and leak rows.
        assert_eq!(
            combine_predicates(&p),
            "((region = 'us') OR (region = 'eu')) AND (tier > 2)"
        );
    }

    #[test]
    fn combine_only_restrictive() {
        let p = vec![policy("r", "tier > 2", false, vec![])];
        assert_eq!(combine_predicates(&p), "(tier > 2)");
    }
}
