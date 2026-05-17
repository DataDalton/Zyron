//! Bridges the SecurityManager's RLS / ABAC / row-ownership policy stores to
//! the planner's `RowSecurityProvider` so user queries get row-security
//! predicates injected at bind time. Internal/admin queries do not construct
//! this provider and therefore bypass row security (by design).

use std::sync::Arc;

use zyron_auth::{PolicyType, RoleId, SecurityContext, SecurityManager};
use zyron_planner::{RowPredicate, RowSecurityProvider};

pub struct SmRowSecurityProvider {
    sm: Arc<SecurityManager>,
    roles: Vec<RoleId>,
    current_role: RoleId,
}

impl SmRowSecurityProvider {
    pub fn new(sm: Arc<SecurityManager>, sc: &SecurityContext) -> Self {
        Self {
            sm,
            roles: sc.effective_roles.clone(),
            current_role: sc.current_role,
        }
    }

    fn role_applies(&self, policy_roles: &[RoleId]) -> bool {
        policy_roles.is_empty() || policy_roles.iter().any(|r| self.roles.contains(r))
    }
}

impl RowSecurityProvider for SmRowSecurityProvider {
    fn row_predicates(&self, table_id: u32) -> Vec<RowPredicate> {
        let mut out = Vec::new();

        // RLS using-predicates (SELECT visibility, also applied to
        // UPDATE/DELETE to restrict which rows are touched).
        for p in self.sm.rls_store.policies_for_table(table_id) {
            if !p.enabled || !self.role_applies(&p.roles) {
                continue;
            }
            if let Some(expr) = &p.using_expr {
                out.push(RowPredicate {
                    sql: expr.clone(),
                    permissive: matches!(p.policy_type, PolicyType::Permissive),
                });
            }
        }

        // ABAC predicates.
        for p in self.sm.abac_store.policies_for_table(table_id) {
            if !p.enabled || !self.role_applies(&p.roles) {
                continue;
            }
            out.push(RowPredicate {
                sql: p.predicate.clone(),
                permissive: p.permissive,
            });
        }

        // Row ownership: a non-admin role only sees its own rows.
        if self.sm.row_ownership_store.is_enabled(table_id)
            && !self
                .sm
                .row_ownership_store
                .is_admin(table_id, self.current_role)
        {
            if let Some(cfg) = self.sm.row_ownership_store.get_config(table_id) {
                out.push(RowPredicate {
                    sql: format!("\"{}\" = {}", cfg.owner_column, self.current_role.0),
                    permissive: false,
                });
            }
        }

        out
    }

    fn has_row_security(&self, table_id: u32) -> bool {
        !self.sm.rls_store.policies_for_table(table_id).is_empty()
            || !self.sm.abac_store.policies_for_table(table_id).is_empty()
            || self.sm.row_ownership_store.is_enabled(table_id)
    }
}
