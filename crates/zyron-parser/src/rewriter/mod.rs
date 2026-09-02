//! The user-object rewriter.
//!
//! A release that renames something a user-authored object references
//! registers a rewriter beside the deprecation. On upgrade the substrate
//! parses each object, runs every registered rewriter against the parsed
//! statement, and classifies what came back.
//!
//! Nothing is applied without a decision. A safe rewrite is a mechanical
//! rename that preserves meaning and applies on its own under the default
//! policy. An ambiguous one could change behavior in a narrow case and waits
//! for an admin. An unsafe one has no defensible automatic mapping and stops
//! the upgrade until the object is rewritten by hand or the breakage is
//! accepted explicitly.
//!
//! A dry run is a pure function of the object's text and the registered
//! rewriters, so running it twice returns the same answer and changes
//! nothing

pub mod walk;

use zyron_common::format::rewrite::{
    ObjectKind, RewriteCategory, RewriteDisposition, UserObjectRewritePolicy,
};

use crate::ast::Statement;

pub use walk::{RenameTarget, query_of, rename};

/// Applies a rewrite to a parsed statement, returning how many places
/// changed. Zero means the rewriter had nothing to do with this object
pub type RewriteFn = fn(&mut Statement) -> usize;

/// Renders the human-readable diff an admin reads before acknowledging
pub type DiffFn = fn(before: &Statement, after: &Statement) -> String;

/// One registered rewrite
#[derive(Clone, Copy)]
pub struct UserObjectRewrite {
    /// The name the audit trail and the queue record
    pub name: &'static str,
    /// The Zyron version the change lands in
    pub from_version: &'static str,
    pub to_version: &'static str,
    /// Which kinds of object this rewriter looks at
    pub target: &'static [ObjectKind],
    pub rewriter: RewriteFn,
    pub category: RewriteCategory,
    /// One line naming what the rewrite does
    pub description: &'static str,
    pub dry_run_diff_generator: DiffFn,
}

impl std::fmt::Debug for UserObjectRewrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserObjectRewrite")
            .field("name", &self.name)
            .field("from_version", &self.from_version)
            .field("to_version", &self.to_version)
            .field("target", &self.target)
            .field("category", &self.category)
            .field("description", &self.description)
            .finish()
    }
}

inventory::collect!(UserObjectRewrite);

/// One rewriter's verdict on one object
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedRewrite {
    pub object_name: String,
    pub object_kind: ObjectKind,
    pub rewriter_name: &'static str,
    pub category: RewriteCategory,
    pub description: &'static str,
    /// How many places in the object changed
    pub sites: usize,
    /// The statement as it stands
    pub before_sql: String,
    /// The statement after the rewrite
    pub after_sql: String,
    pub diff: String,
}

impl ProposedRewrite {
    /// A stable hash of the statement before the rewrite, recorded in the
    /// audit chain so a later reader can prove what was rewritten
    pub fn before_hash(&self) -> u32 {
        zyron_common::hash32(self.before_sql.as_bytes())
    }

    /// The same for the statement after
    pub fn after_hash(&self) -> u32 {
        zyron_common::hash32(self.after_sql.as_bytes())
    }

    /// What happens to this rewrite under a policy
    pub fn disposition(&self, policy: UserObjectRewritePolicy) -> RewriteDisposition {
        policy.disposition(self.category)
    }
}

/// Why an object could not be classified
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteError {
    /// The object's stored SQL does not parse with this binary
    Unparseable { object_name: String, reason: String },
    /// The stored SQL holds more than one statement
    NotOneStatement { object_name: String, count: usize },
}

impl std::fmt::Display for RewriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RewriteError::Unparseable {
                object_name,
                reason,
            } => write!(
                f,
                "`{object_name}` does not parse with this binary, {reason}. Rewrite it by \
                 hand or drop it before upgrading"
            ),
            RewriteError::NotOneStatement { object_name, count } => write!(
                f,
                "`{object_name}` holds {count} statements, a user object holds one"
            ),
        }
    }
}

impl std::error::Error for RewriteError {}

/// Every rewriter this binary carries
pub fn registered() -> Vec<UserObjectRewrite> {
    inventory::iter::<UserObjectRewrite>
        .into_iter()
        .copied()
        .collect()
}

/// The rewriters that look at one kind of object
pub fn for_kind(kind: ObjectKind) -> Vec<UserObjectRewrite> {
    registered()
        .into_iter()
        .filter(|r| r.target.contains(&kind))
        .collect()
}

/// Runs every registered rewriter against one object without applying
/// anything, returning what each one would do.
///
/// Pure: the statement is parsed fresh each time and every rewriter works on
/// its own clone, so two runs return the same thing and neither leaves a
/// mark
pub fn dry_run(
    object_name: &str,
    kind: ObjectKind,
    sql: &str,
) -> Result<Vec<ProposedRewrite>, RewriteError> {
    let original = parse_one(object_name, sql)?;
    let before_sql = render(&original);
    let mut proposals = Vec::new();
    for rewrite in for_kind(kind) {
        let mut candidate = original.clone();
        let sites = (rewrite.rewriter)(&mut candidate);
        if sites == 0 {
            continue;
        }
        let after_sql = render(&candidate);
        proposals.push(ProposedRewrite {
            object_name: object_name.to_string(),
            object_kind: kind,
            rewriter_name: rewrite.name,
            category: rewrite.category,
            description: rewrite.description,
            sites,
            before_sql: before_sql.clone(),
            after_sql,
            diff: (rewrite.dry_run_diff_generator)(&original, &candidate),
        });
    }
    proposals.sort_by(|a, b| a.rewriter_name.cmp(b.rewriter_name));
    Ok(proposals)
}

/// Applies the rewrites a policy allows, returning the rewritten SQL and the
/// proposals that were applied.
///
/// A rewrite the policy does not allow is left in the returned list with its
/// disposition, so the caller can queue it rather than silently drop it
pub fn apply(
    object_name: &str,
    kind: ObjectKind,
    sql: &str,
    policy: UserObjectRewritePolicy,
) -> Result<AppliedRewrites, RewriteError> {
    let mut statement = parse_one(object_name, sql)?;
    let proposals = dry_run(object_name, kind, sql)?;
    let mut applied = Vec::new();
    let mut deferred = Vec::new();
    for proposal in proposals {
        match proposal.disposition(policy) {
            RewriteDisposition::AutoApply => {
                let rewrite = for_kind(kind)
                    .into_iter()
                    .find(|r| r.name == proposal.rewriter_name);
                if let Some(rewrite) = rewrite {
                    (rewrite.rewriter)(&mut statement);
                }
                applied.push(proposal);
            }
            _ => deferred.push(proposal),
        }
    }
    Ok(AppliedRewrites {
        sql: render(&statement),
        applied,
        deferred,
    })
}

/// The result of applying rewrites to one object
#[derive(Debug, Clone)]
pub struct AppliedRewrites {
    /// The object's SQL after everything the policy allowed
    pub sql: String,
    pub applied: Vec<ProposedRewrite>,
    /// Rewrites the policy left for an admin, with their disposition
    pub deferred: Vec<ProposedRewrite>,
}

impl AppliedRewrites {
    /// Whether anything here stops an upgrade
    pub fn blocks_upgrade(&self, policy: UserObjectRewritePolicy) -> bool {
        self.deferred
            .iter()
            .any(|p| p.disposition(policy).blocks_upgrade())
    }

    /// Whether anything here needs an acknowledgment
    pub fn needs_ack(&self, policy: UserObjectRewritePolicy) -> bool {
        self.deferred
            .iter()
            .any(|p| p.disposition(policy).needs_ack())
    }
}

/// Parses one user object's stored SQL
fn parse_one(object_name: &str, sql: &str) -> Result<Statement, RewriteError> {
    let mut statements = crate::parse(sql).map_err(|e| RewriteError::Unparseable {
        object_name: object_name.to_string(),
        reason: e.to_string(),
    })?;
    match statements.len() {
        1 => Ok(statements.remove(0)),
        count => Err(RewriteError::NotOneStatement {
            object_name: object_name.to_string(),
            count,
        }),
    }
}

/// Renders a statement back to a comparable form.
///
/// The debug rendering is the AST itself, which is what a diff has to
/// compare: two statements that differ only in whitespace render the same,
/// and two that differ in structure render differently
pub fn render(statement: &Statement) -> String {
    format!("{statement:?}")
}

/// The default diff generator, a line-oriented comparison of the two
/// renderings.
///
/// Registered rewriters that can say something more specific supply their
/// own; this is what a mechanical rename uses
pub fn default_diff(before: &Statement, after: &Statement) -> String {
    let before = render(before);
    let after = render(after);
    if before == after {
        return "no change".to_string();
    }
    format!("- {before}\n+ {after}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::rewrite::RewriteStatus;

    /// A rewriter registered only for this test module, so the registry has
    /// something to classify without a release having to deprecate anything
    fn warehouse_to_compute(statement: &mut Statement) -> usize {
        rename(
            statement,
            RenameTarget::Relation,
            "warehouse_x",
            "compute_x",
        )
    }

    fn widen_signature(statement: &mut Statement) -> usize {
        rename(statement, RenameTarget::Function, "old_agg", "new_agg")
    }

    fn removed_feature(statement: &mut Statement) -> usize {
        rename(statement, RenameTarget::Function, "gone_fn", "gone_fn")
    }

    inventory::submit! {
        UserObjectRewrite {
            name: "test_warehouse_to_compute",
            from_version: "0.11.0",
            to_version: "0.12.0",
            target: &[ObjectKind::View, ObjectKind::MaterializedView],
            rewriter: warehouse_to_compute,
            category: RewriteCategory::Safe,
            description: "renames the warehouse_x relation to compute_x",
            dry_run_diff_generator: default_diff,
        }
    }

    inventory::submit! {
        UserObjectRewrite {
            name: "test_widen_signature",
            from_version: "0.11.0",
            to_version: "0.12.0",
            target: &[ObjectKind::View],
            rewriter: widen_signature,
            category: RewriteCategory::Ambiguous,
            description: "old_agg widened its return type, the call becomes new_agg",
            dry_run_diff_generator: default_diff,
        }
    }

    inventory::submit! {
        UserObjectRewrite {
            name: "test_removed_feature",
            from_version: "0.11.0",
            to_version: "0.12.0",
            target: &[ObjectKind::View],
            rewriter: removed_feature,
            category: RewriteCategory::Unsafe,
            description: "gone_fn was removed with no replacement",
            dry_run_diff_generator: default_diff,
        }
    }

    #[test]
    fn test_a_safe_rewrite_is_proposed_and_applied() {
        let sql = "CREATE VIEW v AS SELECT a FROM warehouse_x";
        let proposals = dry_run("v", ObjectKind::View, sql).expect("classifies");
        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].category, RewriteCategory::Safe);
        assert_eq!(proposals[0].sites, 1);
        assert!(proposals[0].after_sql.contains("compute_x"));

        let applied = apply(
            "v",
            ObjectKind::View,
            sql,
            UserObjectRewritePolicy::AutoSafe,
        )
        .expect("applies");
        assert_eq!(applied.applied.len(), 1);
        assert!(applied.deferred.is_empty());
        assert!(applied.sql.contains("compute_x"));
        assert!(!applied.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
    }

    #[test]
    fn test_an_ambiguous_rewrite_waits_for_an_acknowledgment() {
        let sql = "CREATE VIEW v AS SELECT old_agg(a) FROM t";
        let applied = apply(
            "v",
            ObjectKind::View,
            sql,
            UserObjectRewritePolicy::AutoSafe,
        )
        .expect("applies");
        assert!(applied.applied.is_empty());
        assert_eq!(applied.deferred.len(), 1);
        assert_eq!(applied.deferred[0].category, RewriteCategory::Ambiguous);
        assert!(applied.needs_ack(UserObjectRewritePolicy::AutoSafe));
        assert!(!applied.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
        assert!(
            !applied.sql.contains("new_agg"),
            "nothing is applied before the acknowledgment"
        );
    }

    #[test]
    fn test_an_unsafe_rewrite_blocks_the_upgrade() {
        let sql = "CREATE VIEW v AS SELECT gone_fn(a) FROM t";
        let applied = apply(
            "v",
            ObjectKind::View,
            sql,
            UserObjectRewritePolicy::AutoSafe,
        )
        .expect("applies");
        assert_eq!(applied.deferred.len(), 1);
        assert_eq!(applied.deferred[0].category, RewriteCategory::Unsafe);
        assert!(applied.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
    }

    #[test]
    fn test_notify_all_holds_even_a_safe_rewrite() {
        let sql = "CREATE VIEW v AS SELECT a FROM warehouse_x";
        let applied = apply(
            "v",
            ObjectKind::View,
            sql,
            UserObjectRewritePolicy::NotifyAll,
        )
        .expect("applies");
        assert!(applied.applied.is_empty());
        assert_eq!(applied.deferred.len(), 1);
        assert!(applied.needs_ack(UserObjectRewritePolicy::NotifyAll));
        assert!(!applied.sql.contains("compute_x"));
    }

    #[test]
    fn test_manual_only_applies_nothing() {
        let sql = "CREATE VIEW v AS SELECT a FROM warehouse_x";
        let applied = apply(
            "v",
            ObjectKind::View,
            sql,
            UserObjectRewritePolicy::ManualOnly,
        )
        .expect("applies");
        assert!(applied.applied.is_empty());
        assert_eq!(applied.deferred.len(), 1);
        assert!(!applied.blocks_upgrade(UserObjectRewritePolicy::ManualOnly));
        assert!(!applied.needs_ack(UserObjectRewritePolicy::ManualOnly));
    }

    #[test]
    fn test_a_dry_run_is_pure() {
        let sql = "CREATE VIEW v AS SELECT a FROM warehouse_x";
        let first = dry_run("v", ObjectKind::View, sql).expect("classifies");
        let second = dry_run("v", ObjectKind::View, sql).expect("classifies");
        assert_eq!(first, second);
        // The stored SQL is untouched by a dry run
        assert!(sql.contains("warehouse_x"));
    }

    #[test]
    fn test_an_object_no_rewriter_touches_is_unaffected() {
        let sql = "CREATE VIEW v AS SELECT a FROM untouched";
        assert!(
            dry_run("v", ObjectKind::View, sql)
                .expect("classifies")
                .is_empty()
        );
    }

    #[test]
    fn test_an_object_that_does_not_parse_is_named() {
        let err = dry_run("v", ObjectKind::View, "NOT SQL AT ALL").expect_err("refused");
        assert!(err.to_string().contains("`v`"), "{err}");
        assert!(err.to_string().contains("Rewrite it by hand"), "{err}");
    }

    #[test]
    fn test_rewriters_are_filtered_by_object_kind() {
        // The materialized-view rewriter set is a subset of the view one
        let views = for_kind(ObjectKind::View).len();
        let mviews = for_kind(ObjectKind::MaterializedView).len();
        assert!(views >= mviews);
        assert!(for_kind(ObjectKind::Dashboard).len() <= views);
    }

    #[test]
    fn test_hashes_change_only_when_the_statement_does() {
        let sql = "CREATE VIEW v AS SELECT a FROM warehouse_x";
        let proposals = dry_run("v", ObjectKind::View, sql).expect("classifies");
        let proposal = &proposals[0];
        assert_ne!(proposal.before_hash(), proposal.after_hash());
        assert_eq!(proposal.before_hash(), proposal.before_hash());
    }

    #[test]
    fn test_status_labels_are_stable() {
        assert_eq!(RewriteStatus::Pending.label(), "pending");
        assert_eq!(RewriteStatus::AcceptedBroken.label(), "accepted_broken");
    }
}
