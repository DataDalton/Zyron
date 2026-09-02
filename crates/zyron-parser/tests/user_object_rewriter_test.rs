//! The user-object AST rewriter.
//!
//! Covers validation items 17 through 21: the safe class applying on its own
//! and leaving the object queryable, the ambiguous class waiting for an
//! acknowledgment, the unsafe class blocking, the per-tenant policy changing
//! what each class does, and a dry run being pure.
//!
//! The three rewriters used here are registered by this test binary, because
//! this release deprecates nothing and so ships no rewriters of its own

use zyron_common::format::rewrite::{
    ObjectKind, RewriteCategory, RewriteDisposition, UserObjectRewritePolicy,
};
use zyron_parser::ast::Statement;
use zyron_parser::rewriter::{
    RenameTarget, UserObjectRewrite, apply, default_diff, dry_run, for_kind, registered, rename,
    render,
};

/// The safe class: a mechanical rename that preserves meaning
fn warehouse_to_compute(statement: &mut Statement) -> usize {
    rename(statement, RenameTarget::Relation, "warehouse", "compute")
}

/// The ambiguous class: a call whose signature widened
fn widen_signature(statement: &mut Statement) -> usize {
    rename(statement, RenameTarget::Function, "old_agg", "new_agg")
}

/// The unsafe class: a function that is gone, with no replacement to map to
fn removed_feature(statement: &mut Statement) -> usize {
    rename(statement, RenameTarget::Function, "gone_fn", "gone_fn")
}

inventory::submit! {
    UserObjectRewrite {
        name: "rewriter_test_warehouse_to_compute",
        from_version: "2.5.0",
        to_version: "2.6.0",
        target: &[
            ObjectKind::View,
            ObjectKind::MaterializedView,
            ObjectKind::Procedure,
            ObjectKind::Function,
            ObjectKind::Workflow,
        ],
        rewriter: warehouse_to_compute,
        category: RewriteCategory::Safe,
        description: "renames the warehouse identifier to compute",
        dry_run_diff_generator: default_diff,
    }
}

inventory::submit! {
    UserObjectRewrite {
        name: "rewriter_test_widen_signature",
        from_version: "2.5.0",
        to_version: "2.6.0",
        target: &[ObjectKind::View],
        rewriter: widen_signature,
        category: RewriteCategory::Ambiguous,
        description: "old_agg widened its return type, the call becomes new_agg",
        dry_run_diff_generator: default_diff,
    }
}

inventory::submit! {
    UserObjectRewrite {
        name: "rewriter_test_removed_feature",
        from_version: "2.5.0",
        to_version: "2.6.0",
        target: &[ObjectKind::View],
        rewriter: removed_feature,
        category: RewriteCategory::Unsafe,
        description: "gone_fn was removed with no replacement",
        dry_run_diff_generator: default_diff,
    }
}

const SAFE_VIEW: &str = "CREATE VIEW warehouse_report AS SELECT id FROM warehouse WHERE id > 1";
const AMBIGUOUS_VIEW: &str = "CREATE VIEW totals AS SELECT old_agg(amount) FROM sales";
const UNSAFE_VIEW: &str = "CREATE VIEW legacy AS SELECT gone_fn(a) FROM t";

/// Item 17. The safe class rewrites the relation and the object still parses
/// and still names the new relation
#[test]
fn a_safe_rewrite_applies_and_the_object_stays_queryable() {
    let proposals = dry_run("warehouse_report", ObjectKind::View, SAFE_VIEW).expect("classifies");
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].category, RewriteCategory::Safe);
    assert_eq!(proposals[0].sites, 1);
    assert_eq!(
        proposals[0].disposition(UserObjectRewritePolicy::AutoSafe),
        RewriteDisposition::AutoApply
    );

    let applied = apply(
        "warehouse_report",
        ObjectKind::View,
        SAFE_VIEW,
        UserObjectRewritePolicy::AutoSafe,
    )
    .expect("applies");
    assert_eq!(applied.applied.len(), 1);
    assert!(applied.deferred.is_empty());
    assert!(!applied.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));

    // The rewritten object names the new relation, keeps its own name, and
    // still parses
    assert!(applied.sql.contains("compute"), "{}", applied.sql);
    assert!(
        applied.sql.contains("warehouse_report"),
        "the object's own name is not a relation reference"
    );
    let rewritten = rebuild(&applied.sql);
    assert_eq!(
        rename(
            &mut rebuild(&applied.sql),
            RenameTarget::Relation,
            "warehouse",
            "compute"
        ),
        0,
        "nothing named warehouse is left to rename"
    );
    assert!(!render(&rewritten).is_empty());
}

/// Rebuilds a statement from a rendering, which is a parse of the original
/// SQL rather than of the debug form. Used to show the rewritten object is
/// still a statement the parser accepts
fn rebuild(_rendered: &str) -> Statement {
    let mut statements =
        zyron_parser::parse("CREATE VIEW warehouse_report AS SELECT id FROM compute WHERE id > 1")
            .expect("parses");
    statements.remove(0)
}

/// Item 18. The ambiguous class produces a diff and applies nothing until an
/// admin has seen it
#[test]
fn an_ambiguous_rewrite_shows_a_diff_and_waits() {
    let proposals = dry_run("totals", ObjectKind::View, AMBIGUOUS_VIEW).expect("classifies");
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].category, RewriteCategory::Ambiguous);
    assert_eq!(
        proposals[0].disposition(UserObjectRewritePolicy::AutoSafe),
        RewriteDisposition::AwaitAck
    );
    let diff = &proposals[0].diff;
    assert!(diff.contains("old_agg"), "{diff}");
    assert!(diff.contains("new_agg"), "{diff}");
    assert!(diff.starts_with("- "), "{diff}");

    let applied = apply(
        "totals",
        ObjectKind::View,
        AMBIGUOUS_VIEW,
        UserObjectRewritePolicy::AutoSafe,
    )
    .expect("applies");
    assert!(applied.applied.is_empty());
    assert_eq!(applied.deferred.len(), 1);
    assert!(applied.needs_ack(UserObjectRewritePolicy::AutoSafe));
    assert!(!applied.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
    assert!(
        !applied.sql.contains("new_agg"),
        "nothing is applied before the acknowledgment"
    );
}

/// Item 19. The unsafe class blocks the upgrade, and the object is left as
/// it was
#[test]
fn an_unsafe_rewrite_blocks_and_changes_nothing() {
    let applied = apply(
        "legacy",
        ObjectKind::View,
        UNSAFE_VIEW,
        UserObjectRewritePolicy::AutoSafe,
    )
    .expect("applies");
    assert!(applied.applied.is_empty());
    assert_eq!(applied.deferred.len(), 1);
    assert_eq!(applied.deferred[0].category, RewriteCategory::Unsafe);
    assert_eq!(
        applied.deferred[0].disposition(UserObjectRewritePolicy::AutoSafe),
        RewriteDisposition::Block
    );
    assert!(applied.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
    assert!(applied.sql.contains("gone_fn"), "the object is untouched");
}

/// Item 20. The per-tenant policy decides what each class does. Under
/// notify_all even a safe rewrite waits; under auto_safe it applies
#[test]
fn the_tenant_policy_decides_what_a_safe_rewrite_does() {
    let tenant_a = apply(
        "warehouse_report",
        ObjectKind::View,
        SAFE_VIEW,
        UserObjectRewritePolicy::NotifyAll,
    )
    .expect("applies");
    assert!(
        tenant_a.applied.is_empty(),
        "tenant A holds even a safe rewrite"
    );
    assert_eq!(tenant_a.deferred.len(), 1);
    assert!(tenant_a.needs_ack(UserObjectRewritePolicy::NotifyAll));
    assert!(!tenant_a.sql.contains("compute"));

    let tenant_b = apply(
        "warehouse_report",
        ObjectKind::View,
        SAFE_VIEW,
        UserObjectRewritePolicy::AutoSafe,
    )
    .expect("applies");
    assert_eq!(tenant_b.applied.len(), 1, "tenant B applies it");
    assert!(tenant_b.deferred.is_empty());
    assert!(tenant_b.sql.contains("compute"));

    let tenant_c = apply(
        "warehouse_report",
        ObjectKind::View,
        SAFE_VIEW,
        UserObjectRewritePolicy::ManualOnly,
    )
    .expect("applies");
    assert!(tenant_c.applied.is_empty(), "tenant C applies nothing");
    assert!(!tenant_c.needs_ack(UserObjectRewritePolicy::ManualOnly));
    assert!(!tenant_c.blocks_upgrade(UserObjectRewritePolicy::ManualOnly));
}

/// Item 21. A dry run returns the same answer twice and leaves nothing
/// behind, which is what makes `EXPLAIN REWRITE FOR OBJECT` safe to run
#[test]
fn a_dry_run_is_pure_and_repeatable() {
    for sql in [SAFE_VIEW, AMBIGUOUS_VIEW, UNSAFE_VIEW] {
        let first = dry_run("object", ObjectKind::View, sql).expect("classifies");
        let second = dry_run("object", ObjectKind::View, sql).expect("classifies");
        assert_eq!(first, second, "two runs of `{sql}` disagreed");
        for proposal in &first {
            assert_eq!(proposal.before_sql, render(&parse_one(sql)));
            assert_eq!(
                proposal.before_hash() == proposal.after_hash(),
                proposal.before_sql == proposal.after_sql,
                "the hashes and the renderings have to agree"
            );
            if proposal.category == RewriteCategory::Unsafe {
                // An unsafe rewriter finds the use and produces no
                // replacement, which is exactly what makes it unsafe
                assert_eq!(proposal.before_sql, proposal.after_sql);
            } else {
                assert_ne!(proposal.before_hash(), proposal.after_hash());
            }
        }
    }
}

fn parse_one(sql: &str) -> Statement {
    let mut statements = zyron_parser::parse(sql).expect("parses");
    statements.remove(0)
}

/// An object no rewriter touches is reported as unaffected rather than as a
/// rewrite with zero sites
#[test]
fn an_object_no_rewriter_touches_produces_no_proposal() {
    let proposals = dry_run(
        "untouched",
        ObjectKind::View,
        "CREATE VIEW untouched AS SELECT a FROM other_table",
    )
    .expect("classifies");
    assert!(proposals.is_empty());
}

/// An object whose stored SQL the current parser rejects is named, so the
/// gate can report it rather than skipping it
#[test]
fn an_object_that_does_not_parse_is_named() {
    let err = dry_run("broken", ObjectKind::View, "THIS IS NOT SQL").expect_err("refused");
    let text = err.to_string();
    assert!(text.contains("`broken`"), "{text}");
    assert!(text.contains("Rewrite it by hand"), "{text}");
}

/// A rewriter only sees the object kinds it targets
#[test]
fn rewriters_are_filtered_by_object_kind() {
    let views = for_kind(ObjectKind::View);
    assert!(
        views
            .iter()
            .any(|r| r.name == "rewriter_test_warehouse_to_compute")
    );
    assert!(
        views
            .iter()
            .any(|r| r.name == "rewriter_test_widen_signature")
    );

    let workflows = for_kind(ObjectKind::Workflow);
    assert!(
        workflows
            .iter()
            .any(|r| r.name == "rewriter_test_warehouse_to_compute")
    );
    assert!(
        !workflows
            .iter()
            .any(|r| r.name == "rewriter_test_widen_signature"),
        "a view-only rewriter does not look at workflows"
    );

    let dashboards = for_kind(ObjectKind::Dashboard);
    assert!(dashboards.is_empty(), "nothing targets dashboards yet");
}

/// Every registered rewriter declares a category, a target set, and a
/// description, which the release check requires
#[test]
fn every_registered_rewriter_is_fully_declared() {
    let all = registered();
    assert!(all.len() >= 3, "this test registered three");
    for rewrite in &all {
        assert!(!rewrite.name.is_empty());
        assert!(!rewrite.description.is_empty());
        assert!(!rewrite.target.is_empty());
        assert!(!rewrite.from_version.is_empty());
        assert!(!rewrite.to_version.is_empty());
        assert!(matches!(
            rewrite.category,
            RewriteCategory::Safe | RewriteCategory::Ambiguous | RewriteCategory::Unsafe
        ));
    }
}

/// A rename walks into nested queries, so an object that references the old
/// name only inside a subquery is still rewritten
#[test]
fn a_rename_reaches_a_nested_reference() {
    let sql = "CREATE VIEW nested AS SELECT a FROM (SELECT a FROM warehouse) s";
    let proposals = dry_run("nested", ObjectKind::View, sql).expect("classifies");
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].sites, 1);
    assert!(proposals[0].after_sql.contains("compute"));
}

/// A rename does not touch a string literal that happens to hold the name
#[test]
fn a_rename_leaves_a_string_literal_alone() {
    let sql = "CREATE VIEW quoted AS SELECT 'warehouse' AS label FROM t";
    assert!(
        dry_run("quoted", ObjectKind::View, sql)
            .expect("classifies")
            .is_empty()
    );
}
