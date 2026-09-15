//! The `zyron_sys.verify.*` views and the alerts verification raises.
//!
//! `tables` answers what each verified table's chain stands at and when it
//! was last anchored and last verified. `runs` is the history of the
//! verifications this node ran, which is itself the evidence that a
//! verification happened.

use zyron_common::ZyronError;
use zyron_lifecycle::verify;

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_BOOL_OID, PG_INT8_OID, PG_TEXT_OID};

/// Whether this module builds the named view
pub fn owns(schema: &str, object: &str) -> bool {
    matches!((schema, object), ("verify", "tables") | ("verify", "runs"))
}

/// Dispatches one `zyron_sys.verify` view to its builder.
pub fn build(schema: &str, object: &str, server: &ServerState) -> Result<ViewRows, ZyronError> {
    match (schema, object) {
        ("verify", "tables") => Ok(build_tables(server)),
        ("verify", "runs") => Ok(build_runs(server)),
        (schema, object) => Err(ZyronError::Internal(format!(
            "`zyron_sys.{schema}.{object}` is registered but has no builder"
        ))),
    }
}

fn text(value: impl Into<String>) -> Option<Vec<u8>> {
    Some(value.into().into_bytes())
}

fn number(value: impl std::fmt::Display) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

fn flag(value: bool) -> Option<Vec<u8>> {
    Some(if value { b"t".to_vec() } else { b"f".to_vec() })
}

/// The name a chain's scheme tag resolves to, the tag itself when the
/// registry does not hold it
fn algorithm_name(tag: u16) -> String {
    let tag = if tag == 0 {
        verify::DEFAULT_CHAIN_ALGORITHM
    } else {
        tag
    };
    zyron_common::format::substrate()
        .ok()
        .and_then(|substrate| {
            substrate
                .schemes
                .by_id(zyron_common::format::scheme::SchemeId(tag))
                .map(|scheme| scheme.scheme_name.to_string())
        })
        .unwrap_or_else(|| tag.to_string())
}

/// `zyron_sys.verify.tables`, one row per table that carries a chain.
///
/// A table whose chain opens with a genesis entry states so, because the
/// rows that entry covers were already there and are covered as a set
/// rather than one commit at a time
fn build_tables(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table", PG_TEXT_OID, -1),
        make_field("verified", PG_BOOL_OID, 1),
        make_field("algorithm", PG_TEXT_OID, -1),
        make_field("head_version", PG_INT8_OID, 8),
        make_field("head_hash", PG_TEXT_OID, -1),
        make_field("last_anchored_at", PG_INT8_OID, 8),
        make_field("last_verified_at", PG_INT8_OID, 8),
        make_field("last_outcome", PG_TEXT_OID, -1),
        make_field("commits", PG_INT8_OID, 8),
        make_field("chain_bytes", PG_INT8_OID, 8),
        make_field("covers", PG_TEXT_OID, -1),
    ];
    let Some(registry) = server.chain_registry.as_ref() else {
        return (fields, Vec::new());
    };
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    for table in server.catalog.list_all_tables() {
        if !table.lifecycle.verified {
            continue;
        }
        let Ok(chain) = registry.chain(table.id.0) else {
            continue;
        };
        let head = chain.head();
        let anchored = registry.anchors().for_table(table.id.0);
        let run = registry.runs().latest_for(table.id.0);
        // The registry knows where the genesis entry landed, and the entry
        // says how many rows the table already held when it became
        // verifiable, which are covered as a set
        let genesis = registry
            .chained(table.id.0)
            .and_then(|chained| chained.genesis_sequence())
            .and_then(|at| chain.read_range(at, at).ok())
            .and_then(|entries| entries.into_iter().next())
            .filter(|entry| entry.genesis);
        let covers = match genesis {
            Some(entry) => format!(
                "{} row(s) written before the chain began are covered as a set by the genesis \
                 entry, every commit after it individually",
                entry.row_count
            ),
            None => "every commit individually".to_string(),
        };
        rows.push(vec![
            text(table.name.clone()),
            flag(true),
            text(algorithm_name(table.lifecycle.chain_algorithm)),
            number(head.head_version),
            text(verify::hex(&head.head_hash)),
            anchored
                .as_ref()
                .map(|a| a.taken_at)
                .map(number)
                .unwrap_or(None),
            run.as_ref()
                .map(|r| r.started_at)
                .map(number)
                .unwrap_or(None),
            match run.as_ref() {
                Some(run) if run.intact => text(format!("intact, {}", run.mode.label())),
                Some(run) => text(format!("not intact, {}", run.finding)),
                None => text("never verified"),
            },
            number(head.commits),
            number(head.bytes()),
            text(covers),
        ]);
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    (fields, rows)
}

/// `zyron_sys.verify.runs`, newest first.
fn build_runs(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("table", PG_TEXT_OID, -1),
        make_field("actor", PG_TEXT_OID, -1),
        make_field("actor_role", PG_INT8_OID, 8),
        make_field("from_version", PG_INT8_OID, 8),
        make_field("to_version", PG_INT8_OID, 8),
        make_field("mode", PG_TEXT_OID, -1),
        make_field("commits_checked", PG_INT8_OID, 8),
        make_field("rows_checked", PG_INT8_OID, 8),
        make_field("anchors_checked", PG_INT8_OID, 8),
        make_field("intact", PG_BOOL_OID, 1),
        make_field("finding", PG_TEXT_OID, -1),
        make_field("started_at", PG_INT8_OID, 8),
        make_field("duration_micros", PG_INT8_OID, 8),
    ];
    let Some(registry) = server.chain_registry.as_ref() else {
        return (fields, Vec::new());
    };
    let rows = registry
        .runs()
        .all()
        .into_iter()
        .map(|run| {
            vec![
                text(run.table_name),
                text(run.actor_name),
                number(run.actor),
                number(run.from_version),
                number(run.to_version),
                text(run.mode.label()),
                number(run.commits_checked),
                number(run.rows_checked),
                number(run.anchors_checked),
                flag(run.intact),
                text(run.finding),
                number(run.started_at),
                number(run.duration_micros),
            ]
        })
        .collect();
    (fields, rows)
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

/// One alert verification declares, in the shape the alert template view
/// and the contact channels read
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyAlertTemplate {
    pub name: &'static str,
    pub condition: &'static str,
    pub threshold_setting: &'static str,
    pub default_threshold: &'static str,
    pub summary: &'static str,
}

/// Every alert verifiable tables raise
pub const ALERT_TEMPLATES: &[VerifyAlertTemplate] = &[
    VerifyAlertTemplate {
        name: "verify_failed",
        condition: "a VERIFY TABLE run returned not intact",
        threshold_setting: "",
        default_threshold: "",
        summary: "A verified table's chain no longer states what happened to it",
    },
    VerifyAlertTemplate {
        name: "chain_not_anchored",
        condition: "a verified table's head has not been anchored within twice the interval",
        threshold_setting: "verify.anchor_interval_secs",
        default_threshold: "3600",
        summary: "A chain's head is unanchored, so a truncation of it would not be detected",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_each_alert_is_named_once_and_says_what_it_fires_on() {
        let mut seen = std::collections::HashSet::new();
        for template in ALERT_TEMPLATES {
            assert!(
                seen.insert(template.name),
                "{} declared twice",
                template.name
            );
            assert!(!template.condition.is_empty(), "{}", template.name);
            assert!(!template.summary.is_empty(), "{}", template.name);
        }
    }

    #[test]
    fn test_the_anchor_alert_names_the_setting_its_window_comes_from() {
        let anchored = ALERT_TEMPLATES
            .iter()
            .find(|template| template.name == "chain_not_anchored")
            .expect("declared");
        assert_eq!(anchored.threshold_setting, "verify.anchor_interval_secs");
        assert_eq!(
            anchored.default_threshold,
            zyron_lifecycle::verify::anchor::DEFAULT_ANCHOR_INTERVAL_SECS.to_string()
        );
    }

    #[test]
    fn test_this_module_owns_only_its_own_views() {
        assert!(owns("verify", "tables"));
        assert!(owns("verify", "runs"));
        assert!(!owns("verify", "something_else"));
        assert!(!owns("cdc", "feeds"));
    }
}
