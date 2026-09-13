//! APPLY CHANGES, turning a change set into a target table's rows.
//!
//! The statement takes a relation carrying the metadata columns, a key, and a
//! sequence, and settles what the target should hold. Two shapes are served.
//! Type 1 keeps one row per key, the last change winning. Type 2 keeps a row
//! per version of a key, closing the current row and opening a new one.
//!
//! The whole statement is one transaction, so applying from a change stream
//! advances that stream's position in the same commit as the rows it wrote.
//!
//! Idempotent by construction. The plan is a function of the change set and
//! the key alone. Applying one captured range twice therefore settles on the
//! same target, whatever order the changes arrive in, because ordering is
//! decided by the sequence rather than by arrival

use std::collections::HashMap;

use zyron_common::{Result, ZyronError};

/// Which history shape the target keeps
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScdType {
    /// One row per key, the last change winning
    #[default]
    Type1 = 1,
    /// One row per version of a key, with a validity window
    Type2 = 2,
}

impl ScdType {
    pub fn from_number(value: i64) -> Result<Self> {
        match value {
            1 => Ok(ScdType::Type1),
            2 => Ok(ScdType::Type2),
            other => Err(ZyronError::PlanError(format!(
                "STORED AS SCD TYPE accepts 1 or 2, not {other}"
            ))),
        }
    }

    pub fn number(self) -> i64 {
        match self {
            ScdType::Type1 => 1,
            ScdType::Type2 => 2,
        }
    }
}

/// The columns a type 2 target gains
pub const SCD2_START_COLUMN: &str = "__start_at";
pub const SCD2_END_COLUMN: &str = "__end_at";
pub const SCD2_CURRENT_COLUMN: &str = "__is_current";

/// Which columns cause a new version of a key under type 2
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TrackHistory {
    /// Every applied column
    #[default]
    All,
    /// Only these columns. A change touching none of them updates the current
    /// row in place rather than opening a new version
    On(Vec<String>),
    /// Every applied column except these
    Except(Vec<String>),
}

impl TrackHistory {
    /// Whether a change to this column opens a new version
    pub fn versions_on(&self, column: &str) -> bool {
        match self {
            TrackHistory::All => true,
            TrackHistory::On(list) => list.iter().any(|c| c == column),
            TrackHistory::Except(list) => !list.iter().any(|c| c == column),
        }
    }

    /// The columns that open a new version, out of the applied set
    pub fn tracked<'a>(&self, applied: &'a [String]) -> Vec<&'a String> {
        applied
            .iter()
            .filter(|column| self.versions_on(column))
            .collect()
    }
}

/// Everything `APPLY CHANGES` was written with, after binding
#[derive(Debug, Clone, PartialEq)]
pub struct ApplySpec {
    /// Columns that identify a row in the target
    pub keys: Vec<String>,
    /// Expression deciding the order of two changes to one key. None orders
    /// by the commit version and the position inside that commit
    pub sequence_by: Option<String>,
    /// A NULL in a change means "not supplied" rather than "set to NULL"
    pub ignore_null_updates: bool,
    /// Predicate whose truth turns a change into a delete
    pub apply_as_delete_when: Option<String>,
    /// Predicate whose truth turns a change into a target truncate
    pub apply_as_truncate_when: Option<String>,
    /// Columns the apply leaves out, for a target narrower than its source
    pub except_columns: Vec<String>,
    pub scd: ScdType,
    pub track_history: TrackHistory,
}

impl Default for ApplySpec {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            sequence_by: None,
            ignore_null_updates: false,
            apply_as_delete_when: None,
            apply_as_truncate_when: None,
            except_columns: Vec::new(),
            scd: ScdType::Type1,
            track_history: TrackHistory::All,
        }
    }
}

/// What one change asks the target to do
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyAction {
    Upsert,
    Delete,
    Truncate,
}

/// One change, reduced to what the apply needs
#[derive(Debug, Clone, PartialEq)]
pub struct ApplyChange {
    /// Key values as text, in the order the KEYS clause names them
    pub key: Vec<String>,
    /// The sequence value two changes to one key are ordered by
    pub sequence: SequenceValue,
    pub action: ApplyAction,
    /// Column values the change carries. A column absent from the map was not
    /// supplied
    pub values: HashMap<String, Option<String>>,
}

/// What orders two changes to one key
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SequenceValue {
    /// The SEQUENCE BY value when one was given, otherwise the commit version
    pub primary: i64,
    /// Position inside the commit, which breaks a tie deterministically
    pub ordinal: u64,
}

/// A change set reduced to one decision per key
#[derive(Debug, Clone, PartialEq)]
pub struct ApplyPlan {
    /// The winning change per key, ordered by key so two runs write the same
    /// rows in the same order
    pub decisions: Vec<(Vec<String>, ApplyChange)>,
    /// True when a change asked for a target truncate, which runs before any
    /// upsert or delete
    pub truncate_first: bool,
}

/// What one run did, for `zyron_sys.cdc.apply_runs`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ApplyRunCounts {
    pub rows_upserted: u64,
    pub rows_deleted: u64,
    pub rows_versioned: u64,
    pub truncated: bool,
}

/// Reduces a change set to one decision per key.
///
/// The last change per key wins, ordered by the sequence. A source that
/// delivers out of order therefore converges on the same target as one that
/// delivers in order, which is what makes the whole statement replayable
pub fn plan(spec: &ApplySpec, changes: &[ApplyChange]) -> ApplyPlan {
    let mut winners: HashMap<Vec<String>, ApplyChange> = HashMap::new();
    let mut truncate_first = false;
    for change in changes {
        if change.action == ApplyAction::Truncate {
            truncate_first = true;
            continue;
        }
        match winners.get(&change.key) {
            Some(held) if held.sequence >= change.sequence => {}
            _ => {
                winners.insert(
                    change.key.clone(),
                    merged(spec, winners.get(&change.key), change),
                );
            }
        }
    }
    // A truncate clears the target, so the decisions that follow it are the
    // only rows the target keeps
    let mut decisions: Vec<(Vec<String>, ApplyChange)> = winners.into_iter().collect();
    decisions.sort_by(|a, b| a.0.cmp(&b.0));
    ApplyPlan {
        decisions,
        truncate_first,
    }
}

/// Folds a change onto the one it supersedes.
///
/// With IGNORE NULL UPDATES a NULL means the source did not supply the column,
/// so the value the previous change carried stands. Without it a NULL sets the
/// column to NULL
fn merged(spec: &ApplySpec, previous: Option<&ApplyChange>, change: &ApplyChange) -> ApplyChange {
    if !spec.ignore_null_updates {
        return change.clone();
    }
    let mut next = change.clone();
    if let Some(previous) = previous {
        for (column, value) in &previous.values {
            if next.values.get(column).is_some_and(|v| v.is_none()) {
                next.values.insert(column.clone(), value.clone());
            }
        }
    }
    // A column the change carries as NULL and nothing supplied before stays
    // absent rather than writing NULL over a value the target already holds
    next.values.retain(|_, value| value.is_some());
    next
}

/// Whether a change opens a new version of a key under type 2.
///
/// A change touching only untracked columns updates the current row in place,
/// so a churning column does not produce a row per change
pub fn opens_new_version(
    spec: &ApplySpec,
    current: &HashMap<String, Option<String>>,
    change: &ApplyChange,
) -> bool {
    if spec.scd != ScdType::Type2 {
        return false;
    }
    for (column, value) in &change.values {
        if spec.keys.iter().any(|k| k == column) {
            continue;
        }
        if spec.except_columns.iter().any(|c| c == column) {
            continue;
        }
        if !spec.track_history.versions_on(column) {
            continue;
        }
        if current.get(column).unwrap_or(&None) != value {
            return true;
        }
    }
    false
}

/// Checks the target can carry what the apply writes.
///
/// A missing key column, a key whose type does not match, and a type 2 apply
/// onto a target already holding the reserved names at other types all fail
/// here, naming the column, rather than at the first row
pub fn check_target(
    target: &str,
    target_columns: &[(String, String)],
    spec: &ApplySpec,
    source_key_types: &[(String, String)],
) -> Result<()> {
    for key in &spec.keys {
        let Some((_, target_type)) = target_columns.iter().find(|(name, _)| name == key) else {
            return Err(ZyronError::PlanError(format!(
                "APPLY CHANGES names key column '{key}', which target '{target}' does not have"
            )));
        };
        if let Some((_, source_type)) = source_key_types.iter().find(|(name, _)| name == key) {
            if source_type != target_type {
                return Err(ZyronError::PlanError(format!(
                    "APPLY CHANGES key column '{key}' is {source_type} in the change set and \
                     {target_type} in target '{target}'"
                )));
            }
        }
    }
    if spec.scd == ScdType::Type2 {
        for (reserved, expected) in [
            (SCD2_START_COLUMN, "TIMESTAMPTZ"),
            (SCD2_END_COLUMN, "TIMESTAMPTZ"),
            (SCD2_CURRENT_COLUMN, "BOOLEAN"),
        ] {
            if let Some((_, actual)) = target_columns.iter().find(|(name, _)| name == reserved) {
                if actual != expected {
                    return Err(ZyronError::PlanError(format!(
                        "APPLY CHANGES STORED AS SCD TYPE 2 writes column '{reserved}' as \
                         {expected}, and target '{target}' already holds it as {actual}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// The columns an apply writes, out of what the change set carries
pub fn applied_columns(spec: &ApplySpec, source_columns: &[String]) -> Vec<String> {
    source_columns
        .iter()
        .filter(|column| {
            !column.starts_with('_') && !spec.except_columns.iter().any(|c| c == *column)
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(key: &str, seq: i64, ordinal: u64, action: ApplyAction) -> ApplyChange {
        ApplyChange {
            key: vec![key.to_string()],
            sequence: SequenceValue {
                primary: seq,
                ordinal,
            },
            action,
            values: HashMap::new(),
        }
    }

    fn with_values(mut change: ApplyChange, values: &[(&str, Option<&str>)]) -> ApplyChange {
        for (column, value) in values {
            change
                .values
                .insert((*column).to_string(), value.map(|v| v.to_string()));
        }
        change
    }

    #[test]
    fn test_out_of_order_changes_settle_where_ordered_ones_do() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            ..ApplySpec::default()
        };
        let ordered = vec![
            with_values(change("a", 1, 0, ApplyAction::Upsert), &[("v", Some("1"))]),
            with_values(change("a", 2, 0, ApplyAction::Upsert), &[("v", Some("2"))]),
            with_values(change("a", 3, 0, ApplyAction::Upsert), &[("v", Some("3"))]),
        ];
        let shuffled = vec![ordered[2].clone(), ordered[0].clone(), ordered[1].clone()];
        assert_eq!(plan(&spec, &ordered), plan(&spec, &shuffled));
        let settled = plan(&spec, &shuffled);
        assert_eq!(settled.decisions.len(), 1);
        assert_eq!(
            settled.decisions[0].1.values.get("v"),
            Some(&Some("3".to_string()))
        );
    }

    #[test]
    fn test_the_ordinal_breaks_a_sequence_tie() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            ..ApplySpec::default()
        };
        let changes = vec![
            with_values(
                change("a", 5, 1, ApplyAction::Upsert),
                &[("v", Some("second"))],
            ),
            with_values(
                change("a", 5, 0, ApplyAction::Upsert),
                &[("v", Some("first"))],
            ),
        ];
        let settled = plan(&spec, &changes);
        assert_eq!(
            settled.decisions[0].1.values.get("v"),
            Some(&Some("second".to_string())),
            "the later position inside one commit wins"
        );
    }

    #[test]
    fn test_a_delete_wins_when_it_is_last() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            ..ApplySpec::default()
        };
        let changes = vec![
            change("a", 1, 0, ApplyAction::Upsert),
            change("a", 2, 0, ApplyAction::Delete),
        ];
        let settled = plan(&spec, &changes);
        assert_eq!(settled.decisions[0].1.action, ApplyAction::Delete);
    }

    #[test]
    fn test_ignore_null_updates_keeps_what_was_not_supplied() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            ignore_null_updates: true,
            ..ApplySpec::default()
        };
        let changes = vec![
            with_values(
                change("a", 1, 0, ApplyAction::Upsert),
                &[("name", Some("ada")), ("city", Some("london"))],
            ),
            with_values(
                change("a", 2, 0, ApplyAction::Upsert),
                &[("name", Some("ada lovelace")), ("city", None)],
            ),
        ];
        let settled = plan(&spec, &changes);
        let values = &settled.decisions[0].1.values;
        assert_eq!(values.get("name"), Some(&Some("ada lovelace".to_string())));
        assert_eq!(
            values.get("city"),
            Some(&Some("london".to_string())),
            "a NULL means not supplied, so the earlier value stands"
        );
    }

    #[test]
    fn test_without_the_option_a_null_sets_null() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            ..ApplySpec::default()
        };
        let changes = vec![
            with_values(
                change("a", 1, 0, ApplyAction::Upsert),
                &[("city", Some("london"))],
            ),
            with_values(change("a", 2, 0, ApplyAction::Upsert), &[("city", None)]),
        ];
        let settled = plan(&spec, &changes);
        assert_eq!(settled.decisions[0].1.values.get("city"), Some(&None));
    }

    #[test]
    fn test_a_truncate_runs_before_the_rest() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            ..ApplySpec::default()
        };
        let changes = vec![
            change("a", 1, 0, ApplyAction::Truncate),
            change("b", 2, 0, ApplyAction::Upsert),
        ];
        let settled = plan(&spec, &changes);
        assert!(settled.truncate_first);
        assert_eq!(settled.decisions.len(), 1);
        assert_eq!(settled.decisions[0].0, vec!["b".to_string()]);
    }

    #[test]
    fn test_applying_a_plan_twice_produces_the_same_plan() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            scd: ScdType::Type2,
            ..ApplySpec::default()
        };
        let changes: Vec<ApplyChange> = (0..500)
            .map(|i| {
                with_values(
                    change(&format!("k{}", i % 50), i as i64, 0, ApplyAction::Upsert),
                    &[("v", Some(&i.to_string()))],
                )
            })
            .collect();
        assert_eq!(plan(&spec, &changes), plan(&spec, &changes));
    }

    #[test]
    fn test_track_history_limits_which_columns_version() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            scd: ScdType::Type2,
            track_history: TrackHistory::On(vec!["tier".into()]),
            ..ApplySpec::default()
        };
        let mut current = HashMap::new();
        current.insert("tier".to_string(), Some("gold".to_string()));
        current.insert("visits".to_string(), Some("1".to_string()));

        let churn = with_values(
            change("a", 1, 0, ApplyAction::Upsert),
            &[("tier", Some("gold")), ("visits", Some("2"))],
        );
        assert!(
            !opens_new_version(&spec, &current, &churn),
            "an untracked column updates in place"
        );

        let promotion = with_values(
            change("a", 2, 0, ApplyAction::Upsert),
            &[("tier", Some("platinum")), ("visits", Some("3"))],
        );
        assert!(opens_new_version(&spec, &current, &promotion));
    }

    #[test]
    fn test_track_history_except_is_the_complement() {
        let on = TrackHistory::On(vec!["tier".into()]);
        let except = TrackHistory::Except(vec!["visits".into()]);
        for column in ["tier", "visits"] {
            assert_eq!(
                on.versions_on(column),
                except.versions_on(column),
                "{column} disagreed between the two forms"
            );
        }
    }

    #[test]
    fn test_bind_time_refusals_name_the_column() {
        let target = vec![
            ("id".to_string(), "BIGINT".to_string()),
            ("v".to_string(), "TEXT".to_string()),
        ];
        let spec = ApplySpec {
            keys: vec!["customer_id".into()],
            ..ApplySpec::default()
        };
        let text = check_target("silver", &target, &spec, &[])
            .expect_err("refused")
            .to_string();
        assert!(text.contains("'customer_id'"), "{text}");

        let spec = ApplySpec {
            keys: vec!["id".into()],
            ..ApplySpec::default()
        };
        let text = check_target(
            "silver",
            &target,
            &spec,
            &[("id".to_string(), "TEXT".to_string())],
        )
        .expect_err("refused")
        .to_string();
        assert!(text.contains("'id' is TEXT"), "{text}");

        let target2 = vec![
            ("id".to_string(), "BIGINT".to_string()),
            ("__is_current".to_string(), "TEXT".to_string()),
        ];
        let spec = ApplySpec {
            keys: vec!["id".into()],
            scd: ScdType::Type2,
            ..ApplySpec::default()
        };
        let text = check_target("silver", &target2, &spec, &[])
            .expect_err("refused")
            .to_string();
        assert!(text.contains("__is_current"), "{text}");
    }

    #[test]
    fn test_except_columns_drops_them_from_the_apply() {
        let spec = ApplySpec {
            keys: vec!["id".into()],
            except_columns: vec!["internal_note".into()],
            ..ApplySpec::default()
        };
        let source = vec![
            "id".to_string(),
            "v".to_string(),
            "internal_note".to_string(),
            "_change_type".to_string(),
        ];
        assert_eq!(
            applied_columns(&spec, &source),
            vec!["id".to_string(), "v".to_string()]
        );
    }
}
