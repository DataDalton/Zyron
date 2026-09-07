//! DDL dispatch for the format, signature, and upgrade substrate.
//!
//! These statements read and write registries rather than the catalog, so
//! they are answered here instead of going through the planner. Every one of
//! them either changes a registry and reports what it became, or reports what
//! a registry holds, so nothing here can succeed while leaving the caller
//! guessing what happened

use zyron_common::ZyronError;
use zyron_common::format::FormatKind;
use zyron_common::format::rewrite::{ObjectKind, RewriteCategory, UserObjectRewritePolicy};
use zyron_common::format::scheme::ArtifactKind;
use zyron_common::format::upgrade::parse_duration_secs;
use zyron_parser::ast::{
    AcknowledgeRewriteCategory, AcknowledgeUpgradeRewritesStatement, ExplainRewriteStatement,
    ListRegistryStatement, ListRegistryTarget, RotateServicePrincipalKeyStatement,
    RotateSignatureSchemeStatement, SetSignatureSchemeStatement, ShowUpgradeStatement,
    ShowUpgradeTarget, TriggerUpgradeAction, TriggerUpgradeStatement,
};

use crate::ddl_dispatch::DdlResult;
use crate::messages::ProtocolError;
use crate::types::{PG_INT8_OID, PG_TEXT_OID};

/// The settings `ALTER SYSTEM SET` routes to the upgrade board rather than
/// straight to the config file, each paired with the config key it persists
/// under so the next boot reads back what the statement set
pub const UPGRADE_SETTING_KEYS: &[(&str, &str)] = &[
    ("upgrade_channel", "upgrade.channel"),
    ("pinned_version", "upgrade.pinned_version"),
    ("auto_upgrade_enabled", "upgrade.auto_upgrade_enabled"),
    ("auto_upgrade_window", "upgrade.window"),
    ("auto_upgrade_paused", "upgrade.paused"),
    (
        "user_object_rewrite_policy",
        "upgrade.user_object_rewrite_policy",
    ),
    (
        "release_feed_poll_interval",
        "upgrade.release_feed_poll_interval_secs",
    ),
    (
        "federation_coordination_timeout",
        "upgrade.federation_coordination_timeout_secs",
    ),
    (
        "pre_upgrade_backup_snapshot",
        "upgrade.pre_upgrade_backup_snapshot",
    ),
    ("rollback_on_health_fail", "upgrade.rollback_on_health_fail"),
    (
        "deprecation_warning_rate_limit_per_hour",
        "upgrade.deprecation_warning_rate_limit_per_hour",
    ),
];

/// Whether a setting name is one the upgrade board owns
pub fn owns_setting(name: &str) -> bool {
    config_key(name).is_some()
}

/// The config key an upgrade setting persists under, None for a name the
/// board does not own
pub fn config_key(name: &str) -> Option<&'static str> {
    UPGRADE_SETTING_KEYS
        .iter()
        .find(|(setting, _)| setting.eq_ignore_ascii_case(name))
        .map(|(_, key)| *key)
}

/// The setting name a config key under the upgrade section belongs to,
/// None for a key the board does not own. This is what a node applies a
/// replicated setting through, since the log carries the config key
pub fn setting_for_config_key(key: &str) -> Option<&'static str> {
    UPGRADE_SETTING_KEYS
        .iter()
        .find(|(_, config)| *config == key)
        .map(|(setting, _)| *setting)
}

/// What the server's upgrade controller lets the DDL surface ask of it.
///
/// The controller lives above this crate, beside the journal, the release
/// feed, and the cluster it drives. The statements here record an operator's
/// intent and the controller carries it out on its next pass, so every method
/// returns once the intent is written and never waits on the work
pub trait UpgradeControl: Send + Sync {
    /// Records a request to upgrade the cluster to one version
    fn request_upgrade(&self, version: &str, actor: &str, now_secs: u64) -> Result<(), ZyronError>;

    /// Records a request to roll the last upgrade back. Refuses when the
    /// data on disk can no longer be read by the version it would return to
    fn request_rollback(&self, actor: &str, now_secs: u64) -> Result<(), ZyronError>;

    /// Acknowledges every rewrite of one category that waits on an operator
    /// and applies what the acknowledgment unblocks. Returns how many records
    /// moved
    fn acknowledge_rewrites(
        &self,
        category: RewriteCategory,
        actor: &str,
        now_secs: u64,
    ) -> Result<usize, ZyronError>;
}

fn without_controller() -> ProtocolError {
    refused(
        "this node runs without an upgrade controller, so there is nothing to carry the \
         request out",
    )
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn database(e: ZyronError) -> ProtocolError {
    ProtocolError::Database(e)
}

fn refused(message: impl Into<String>) -> ProtocolError {
    ProtocolError::Database(ZyronError::UpgradeRefused(message.into()))
}

/// The config key one artifact kind's binding persists and replicates under.
///
/// The server owns the section and the key shape, this is the spelling the
/// DDL surface hands it so a binding set here reaches the same place a
/// binding restored at boot came from
pub fn binding_config_key(kind: ArtifactKind) -> String {
    format!("crypto.{}", kind.catalog_name().to_ascii_lowercase())
}

/// Persists a binding that has already taken effect on this node.
///
/// The writer is the one `ALTER SYSTEM SET` uses, so the value is rehearsed
/// against the config the next boot will load, written to `zyron.auto.conf`,
/// and handed to the replicated log when this node is in a group. Without a
/// writer there is nothing to persist through, and a binding that lasted
/// only until the process ended would be a worse answer than refusing, so
/// this says so instead of reporting a success
fn persist_binding(
    writer: Option<&SettingWriter>,
    kind: ArtifactKind,
    stored: &str,
) -> Result<(), ProtocolError> {
    let Some(writer) = writer else {
        return Err(refused(
            "this node has nowhere to persist a signature scheme binding, so the change would \
             be lost at the next restart",
        ));
    };
    writer(&binding_config_key(kind), stored).map_err(|message| {
        database(zyron_common::ZyronError::Internal(format!(
            "the signature scheme binding took effect but could not be persisted, {message}"
        )))
    })
}

/// What the DDL surface writes a persisted setting through. The server
/// installs it, and it is the same one `ALTER SYSTEM SET` goes through
pub type SettingWriter =
    std::sync::Arc<dyn Fn(&str, &str) -> std::result::Result<(), String> + Send + Sync>;

/// `SET SIGNATURE SCHEME <scheme> FOR ARTIFACT KIND <kind>`
pub fn handle_set_signature_scheme(
    stmt: &SetSignatureSchemeStatement,
    writer: Option<&SettingWriter>,
) -> Result<DdlResult, ProtocolError> {
    let substrate = zyron_common::format::substrate().map_err(database)?;
    let kind = ArtifactKind::parse(&stmt.artifact_kind).ok_or_else(|| {
        refused(format!(
            "`{}` is not an artifact kind. Read `zyron_sys.crypto.artifact_scheme_map` for \
             the kinds this cluster signs",
            stmt.artifact_kind
        ))
    })?;
    substrate
        .schemes
        .set_scheme(kind, &stmt.scheme)
        .map_err(|e| database(e.into()))?;
    let stored = substrate.schemes.binding_setting(kind).ok_or_else(|| {
        database(zyron_common::ZyronError::Internal(
            "the binding was applied and then could not be read back".to_string(),
        ))
    })?;
    persist_binding(writer, kind, &stored)?;
    Ok(DdlResult::Tag("SET SIGNATURE SCHEME".to_string()))
}

/// `ROTATE SIGNATURE SCHEME <kind> TO <scheme> [OVERLAP <interval>]`
pub fn handle_rotate_signature_scheme(
    stmt: &RotateSignatureSchemeStatement,
    writer: Option<&SettingWriter>,
) -> Result<DdlResult, ProtocolError> {
    let substrate = zyron_common::format::substrate().map_err(database)?;
    let kind = ArtifactKind::parse(&stmt.artifact_kind).ok_or_else(|| {
        refused(format!(
            "`{}` is not an artifact kind. Read `zyron_sys.crypto.artifact_scheme_map` for \
             the kinds this cluster signs",
            stmt.artifact_kind
        ))
    })?;
    let overlap_secs = match &stmt.overlap {
        Some(text) => parse_duration_secs(text).map_err(refused)?,
        None => zyron_auth::signature::DEFAULT_ROTATION_OVERLAP_SECS,
    };
    let now = now_secs();
    let binding = substrate
        .schemes
        .rotate_scheme(kind, &stmt.new_scheme, now + overlap_secs)
        .map_err(|e| database(e.into()))?;
    // The overlap end is an absolute second worked out here, so what the
    // other members store is the instant this node chose rather than one
    // counted again from each member's own clock
    let stored = substrate.schemes.binding_setting(kind).ok_or_else(|| {
        database(zyron_common::ZyronError::Internal(
            "the rotation was applied and then could not be read back".to_string(),
        ))
    })?;
    persist_binding(writer, kind, &stored)?;
    let columns = vec![
        ("artifact_kind".to_string(), PG_TEXT_OID),
        ("current_scheme".to_string(), PG_TEXT_OID),
        ("deprecating_scheme".to_string(), PG_TEXT_OID),
        ("overlap_end_secs".to_string(), PG_INT8_OID),
    ];
    let rows = vec![vec![
        kind.catalog_name().to_string(),
        binding.current_scheme.clone(),
        binding.deprecating_scheme.clone().unwrap_or_default(),
        binding
            .overlap_end_secs
            .map(|s| s.to_string())
            .unwrap_or_default(),
    ]];
    Ok(DdlResult::Rows {
        tag: "ROTATE SIGNATURE SCHEME".to_string(),
        columns,
        rows,
    })
}

/// `ROTATE SERVICE PRINCIPAL KEY <sp> [SCHEME <scheme>] [OVERLAP <interval>]`
///
/// Refused, because there are no service principals to hold a key.
///
/// A key is one half of an identity: something has to be grantable, has to
/// authenticate with the key, and has to be named by the statement. None of
/// that exists. Every privilege in the catalog is granted to a role id, there
/// is no principal kind a service principal could be, and the only JWT
/// verification this engine performs is over a shared secret, so a key pair
/// has nothing to present it to.
///
/// The rotation machinery underneath is complete and does work: it draws a
/// key pair, seals it, retires the outgoing key over an overlap, and sweeps
/// it afterwards. Running it here would write a key for a name that grants
/// nothing and signs nothing, and report success for it, which is worse than
/// saying so. It is refused in the handler rather than by the replication
/// classifier so a single node and a consensus group answer the same way
pub async fn handle_rotate_service_principal_key(
    stmt: &RotateServicePrincipalKeyStatement,
) -> Result<DdlResult, ProtocolError> {
    // The clauses are still checked, so a statement that is wrong in two ways
    // is not reported as wrong in only one of them
    let substrate = zyron_common::format::substrate().map_err(database)?;
    if let Some(text) = &stmt.overlap {
        parse_duration_secs(text).map_err(refused)?;
    }
    if let Some(named) = &stmt.new_scheme {
        substrate
            .schemes
            .by_name(named)
            .ok_or_else(|| refused(format!("`{named}` is not a registered signature scheme")))?;
    }
    Err(ProtocolError::Database(ZyronError::PlanError(format!(
        "there is no service principal `{}` to hold a key, and no service principals at all. \
         Creating one, granting to one, and authenticating as one are not built, so a key \
         rotated here would name nothing and be presented to nothing",
        stmt.principal
    ))))
}

/// `LIST SIGNATURE SCHEMES`, `LIST ARTIFACT SCHEMES`,
/// `LIST UPGRADE HISTORY [LIMIT n]`, `LIST FORMAT REGISTRY`,
/// `LIST DEPRECATIONS`
pub fn handle_list_registry(stmt: &ListRegistryStatement) -> Result<DdlResult, ProtocolError> {
    let (schema, object, tag) = match stmt.target {
        ListRegistryTarget::SignatureSchemes => {
            ("crypto", "scheme_registry", "LIST SIGNATURE SCHEMES")
        }
        ListRegistryTarget::ArtifactSchemes => {
            ("crypto", "artifact_scheme_map", "LIST ARTIFACT SCHEMES")
        }
        ListRegistryTarget::UpgradeHistory => ("upgrade", "history", "LIST UPGRADE HISTORY"),
        ListRegistryTarget::FormatRegistry => {
            ("storage", "format_registry", "LIST FORMAT REGISTRY")
        }
        ListRegistryTarget::Deprecations => ("deprecation", "registry", "LIST DEPRECATIONS"),
    };
    let (fields, rows) = crate::system_format_views::build(schema, object).map_err(database)?;
    let mut rendered = render_rows(rows);
    if let Some(limit) = stmt.limit {
        rendered.truncate(limit as usize);
    }
    Ok(DdlResult::Rows {
        tag: tag.to_string(),
        columns: fields
            .into_iter()
            .map(|field| (field.name, field.type_oid))
            .collect(),
        rows: rendered,
    })
}

/// `TRIGGER MANUAL UPGRADE TO '<version>'` or `TRIGGER MANUAL ROLLBACK`
///
/// Both hand the operator's intent to the upgrade controller, which records
/// it and carries it out on its next pass. The checks that need nothing but
/// the board run here first, so a malformed request is refused before the
/// controller is asked
pub fn handle_trigger_upgrade(
    stmt: &TriggerUpgradeStatement,
    control: Option<&dyn UpgradeControl>,
    actor: &str,
) -> Result<DdlResult, ProtocolError> {
    let board = zyron_common::format::upgrade_board();
    let now = now_secs();
    match &stmt.action {
        TriggerUpgradeAction::UpgradeTo(version) => {
            if zyron_common::format::BinaryVersion::parse(version).is_none() {
                return Err(refused(format!(
                    "`{version}` is not a major.minor.patch version"
                )));
            }
            if board.settings().paused {
                return Err(refused(
                    "auto_upgrade_paused is true. Set it false before triggering an upgrade",
                ));
            }
            let control = control.ok_or_else(without_controller)?;
            control
                .request_upgrade(version, actor, now)
                .map_err(database)?;
            Ok(DdlResult::Tag(format!("TRIGGER MANUAL UPGRADE {version}")))
        }
        TriggerUpgradeAction::Rollback => {
            let control = control.ok_or_else(without_controller)?;
            control.request_rollback(actor, now).map_err(database)?;
            Ok(DdlResult::Tag("TRIGGER MANUAL ROLLBACK".to_string()))
        }
    }
}

/// `ACKNOWLEDGE UPGRADE REWRITES AMBIGUOUS` or `ACKNOWLEDGE UPGRADE REWRITES UNSAFE`
///
/// Answers with how many records the acknowledgment moved, so an operator
/// who acknowledged a category with nothing waiting in it sees zero rather
/// than a tag that looks like something happened
pub fn handle_acknowledge_upgrade_rewrites(
    stmt: &AcknowledgeUpgradeRewritesStatement,
    control: Option<&dyn UpgradeControl>,
    actor: &str,
) -> Result<DdlResult, ProtocolError> {
    let category = match stmt.category {
        AcknowledgeRewriteCategory::Ambiguous => RewriteCategory::Ambiguous,
        AcknowledgeRewriteCategory::Unsafe => RewriteCategory::Unsafe,
    };
    let control = control.ok_or_else(without_controller)?;
    let moved = control
        .acknowledge_rewrites(category, actor, now_secs())
        .map_err(database)?;
    Ok(DdlResult::Rows {
        tag: "ACKNOWLEDGE UPGRADE REWRITES".to_string(),
        columns: vec![
            ("category".to_string(), PG_TEXT_OID),
            ("acknowledged".to_string(), PG_INT8_OID),
        ],
        rows: vec![vec![category.label().to_string(), moved.to_string()]],
    })
}

/// `SHOW UPGRADE STATE` and `SHOW FORMAT MIGRATIONS [FOR FORMAT <kind>]`
pub fn handle_show_upgrade(stmt: &ShowUpgradeStatement) -> Result<DdlResult, ProtocolError> {
    match &stmt.target {
        ShowUpgradeTarget::State => {
            let (fields, rows) =
                crate::system_format_views::build("upgrade", "state").map_err(database)?;
            Ok(DdlResult::Rows {
                tag: "SHOW UPGRADE STATE".to_string(),
                columns: fields
                    .into_iter()
                    .map(|field| (field.name, field.type_oid))
                    .collect(),
                rows: render_rows(rows),
            })
        }
        ShowUpgradeTarget::FormatMigrations { format_kind } => {
            if let Some(named) = format_kind {
                if FormatKind::from_catalog_name(named).is_none() {
                    return Err(refused(format!(
                        "`{named}` is not a format kind. Read \
                         `zyron_sys.storage.format_registry` for the kinds this binary writes"
                    )));
                }
            }
            let (fields, rows) = crate::system_format_views::build("storage", "format_migrations")
                .map_err(database)?;
            let mut rendered = render_rows(rows);
            if let Some(named) = format_kind {
                rendered.retain(|row| {
                    row.first()
                        .map(|value| value.eq_ignore_ascii_case(named))
                        .unwrap_or(false)
                });
            }
            Ok(DdlResult::Rows {
                tag: "SHOW FORMAT MIGRATIONS".to_string(),
                columns: fields
                    .into_iter()
                    .map(|field| (field.name, field.type_oid))
                    .collect(),
                rows: rendered,
            })
        }
    }
}

/// `EXPLAIN REWRITE FOR OBJECT <name>`
///
/// A dry run over every registered rewriter, applying nothing. Running it
/// twice returns the same rows
pub fn handle_explain_rewrite(
    stmt: &ExplainRewriteStatement,
    server: &crate::connection::ServerState,
) -> Result<DdlResult, ProtocolError> {
    let (kind, sql) = lookup_user_object(&stmt.object_name, server)?;
    let proposals = zyron_parser::rewriter::dry_run(&stmt.object_name, kind, &sql)
        .map_err(|e| refused(e.to_string()))?;
    let columns = vec![
        ("object_name".to_string(), PG_TEXT_OID),
        ("object_kind".to_string(), PG_TEXT_OID),
        ("rewriter".to_string(), PG_TEXT_OID),
        ("category".to_string(), PG_TEXT_OID),
        ("sites".to_string(), PG_INT8_OID),
        ("description".to_string(), PG_TEXT_OID),
        ("diff".to_string(), PG_TEXT_OID),
    ];
    let rows = proposals
        .into_iter()
        .map(|proposal| {
            vec![
                proposal.object_name,
                proposal.object_kind.catalog_name().to_string(),
                proposal.rewriter_name.to_string(),
                proposal.category.label().to_string(),
                proposal.sites.to_string(),
                proposal.description.to_string(),
                proposal.diff,
            ]
        })
        .collect();
    Ok(DdlResult::Rows {
        tag: "EXPLAIN REWRITE".to_string(),
        columns,
        rows,
    })
}

/// Finds a user-authored object's kind and stored SQL
fn lookup_user_object(
    name: &str,
    server: &crate::connection::ServerState,
) -> Result<(ObjectKind, String), ProtocolError> {
    let bare = name.rsplit('.').next().unwrap_or(name);
    for view in server.catalog.list_views() {
        if view.name.eq_ignore_ascii_case(bare) {
            return Ok((
                ObjectKind::View,
                format!("CREATE VIEW {} AS {}", view.name, view.definition_sql),
            ));
        }
    }
    for mview in server.catalog.list_mviews() {
        if mview.name.eq_ignore_ascii_case(bare) {
            return Ok((
                ObjectKind::MaterializedView,
                format!(
                    "CREATE MATERIALIZED VIEW {} AS {}",
                    mview.name, mview.definition_sql
                ),
            ));
        }
    }
    Err(refused(format!(
        "`{name}` is not a user-authored object this node holds. \
         `EXPLAIN REWRITE FOR OBJECT` reads views and materialized views"
    )))
}

/// Turns the wire encoding a view builder produces into the text rows the
/// DDL result carries
fn render_rows(rows: Vec<Vec<Option<Vec<u8>>>>) -> Vec<Vec<String>> {
    rows.into_iter()
        .map(|row| {
            row.into_iter()
                .map(|value| match value {
                    Some(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                    None => String::new(),
                })
                .collect()
        })
        .collect()
}

/// Applies one `ALTER SYSTEM SET` of an upgrade setting to the board.
///
/// Returns the value as it was stored, so the caller persists exactly what
/// took effect rather than what was typed
pub fn apply_upgrade_setting(name: &str, value: &str) -> Result<String, ProtocolError> {
    let board = zyron_common::format::upgrade_board();
    let lowered = name.to_ascii_lowercase();
    let trimmed = value.trim().trim_matches('\'').trim_matches('"');
    match lowered.as_str() {
        "upgrade_channel" => {
            let channel =
                zyron_common::format::UpgradeChannel::parse(trimmed).ok_or_else(|| {
                    refused(format!(
                        "`{trimmed}` is not a channel, use stable, beta, canary, or pinned"
                    ))
                })?;
            board.update_settings(|settings| settings.channel = channel.clone());
            Ok(channel.label().to_string())
        }
        "pinned_version" => {
            if zyron_common::format::BinaryVersion::parse(trimmed).is_none() {
                return Err(refused(format!(
                    "`{trimmed}` is not a major.minor.patch version"
                )));
            }
            board.update_settings(|settings| {
                settings.pinned_version = Some(trimmed.to_string());
            });
            Ok(trimmed.to_string())
        }
        "auto_upgrade_enabled"
        | "auto_upgrade_paused"
        | "pre_upgrade_backup_snapshot"
        | "rollback_on_health_fail" => {
            let flag = parse_bool(trimmed)?;
            board.update_settings(|settings| match lowered.as_str() {
                "auto_upgrade_enabled" => settings.auto_upgrade_enabled = flag,
                "auto_upgrade_paused" => settings.paused = flag,
                "pre_upgrade_backup_snapshot" => settings.pre_upgrade_backup_snapshot = flag,
                _ => settings.rollback_on_health_fail = flag,
            });
            Ok(flag.to_string())
        }
        "auto_upgrade_window" => {
            let schedule =
                zyron_common::format::MaintenanceSchedule::parse(trimmed).map_err(refused)?;
            let rendered = schedule.to_string();
            board.update_settings(|settings| settings.window = schedule.clone());
            Ok(rendered)
        }
        "user_object_rewrite_policy" => {
            let policy = UserObjectRewritePolicy::parse(trimmed).ok_or_else(|| {
                refused(format!(
                    "`{trimmed}` is not a rewrite policy, use auto_safe, notify_all, or \
                     manual_only"
                ))
            })?;
            board.update_settings(|settings| settings.user_object_rewrite_policy = policy);
            Ok(policy.label().to_string())
        }
        "release_feed_poll_interval" | "federation_coordination_timeout" => {
            let secs = parse_duration_secs(trimmed).map_err(refused)?;
            board.update_settings(|settings| match lowered.as_str() {
                "release_feed_poll_interval" => settings.release_feed_poll_interval_secs = secs,
                _ => settings.federation_coordination_timeout_secs = secs,
            });
            Ok(secs.to_string())
        }
        "deprecation_warning_rate_limit_per_hour" => {
            let limit: u32 = trimmed
                .parse()
                .map_err(|_| refused(format!("`{trimmed}` is not a whole number of warnings")))?;
            board.update_settings(|settings| {
                settings.deprecation_warning_rate_limit_per_hour = limit;
            });
            if let Ok(substrate) = zyron_common::format::substrate() {
                substrate.warning_limiter.set_limit(limit);
            }
            Ok(limit.to_string())
        }
        other => Err(refused(format!(
            "`{other}` is not an upgrade setting this node owns"
        ))),
    }
}

fn parse_bool(text: &str) -> Result<bool, ProtocolError> {
    match text.to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        other => Err(refused(format!("`{other}` is not true or false"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upgrade_settings_round_trip_through_the_board() {
        let board = zyron_common::format::upgrade_board();
        assert_eq!(
            apply_upgrade_setting("upgrade_channel", "'beta'").expect("sets"),
            "beta"
        );
        assert_eq!(board.settings().channel.label(), "beta");
        assert_eq!(
            apply_upgrade_setting("upgrade_channel", "stable").expect("sets"),
            "stable"
        );
        assert_eq!(board.settings().channel.label(), "stable");

        apply_upgrade_setting("auto_upgrade_paused", "true").expect("sets");
        assert!(board.settings().paused);
        apply_upgrade_setting("auto_upgrade_paused", "false").expect("sets");
        assert!(!board.settings().paused);

        assert_eq!(
            apply_upgrade_setting("auto_upgrade_window", "'02:00-04:00 UTC'").expect("sets"),
            "02:00-04:00 UTC"
        );
        assert!(!board.settings().window.is_open(15 * 3_600));
        apply_upgrade_setting("auto_upgrade_window", "").expect("clears");
        assert!(board.settings().window.is_open(15 * 3_600));
    }

    /// The statement is refused, and it writes no key while refusing.
    ///
    /// A key rotated for a name that grants nothing and authenticates nothing
    /// would report success for an identity that does not exist. The refusal
    /// names what is missing, and the key store is left as it was, so nothing
    /// accumulates keys for principals that were never created
    #[tokio::test]
    async fn test_rotate_service_principal_key_is_refused_and_writes_nothing() {
        let principal = "sp_rotate_exec_test";
        let store = zyron_auth::signature::principal_keys();
        assert!(
            store.current(principal).is_none(),
            "the test principal has to start with no key"
        );

        let statement = RotateServicePrincipalKeyStatement {
            principal: principal.to_string(),
            new_scheme: None,
            overlap: None,
        };
        let err = handle_rotate_service_principal_key(&statement)
            .await
            .err()
            .expect("refused");
        let text = err.to_string();
        assert!(text.contains(principal), "{text}");
        assert!(text.contains("service principal"), "{text}");
        assert!(
            store.current(principal).is_none(),
            "a refused rotation still wrote a key"
        );
    }

    /// A clause that is wrong is reported as wrong, rather than the statement
    /// answering only that service principals do not exist. Someone fixing
    /// the interval should not have to fix it twice
    #[tokio::test]
    async fn test_rotate_service_principal_key_still_checks_its_clauses() {
        let bad_interval = RotateServicePrincipalKeyStatement {
            principal: "sp_clause_test".to_string(),
            new_scheme: None,
            overlap: Some("a fortnight".to_string()),
        };
        let err = handle_rotate_service_principal_key(&bad_interval)
            .await
            .err()
            .expect("refused");
        assert!(
            !err.to_string().contains("service principal"),
            "the interval is what is wrong here, got {err}"
        );

        let bad_scheme = RotateServicePrincipalKeyStatement {
            principal: "sp_clause_test".to_string(),
            new_scheme: Some("NotAScheme".to_string()),
            overlap: None,
        };
        let err = handle_rotate_service_principal_key(&bad_scheme)
            .await
            .err()
            .expect("refused");
        assert!(err.to_string().contains("NotAScheme"), "{err}");
    }

    #[test]
    fn test_a_bad_setting_value_is_refused_with_guidance() {
        let err = apply_upgrade_setting("upgrade_channel", "nightly").expect_err("refused");
        assert!(err.to_string().contains("stable, beta, canary"), "{err}");
        let err = apply_upgrade_setting("pinned_version", "latest").expect_err("refused");
        assert!(err.to_string().contains("major.minor.patch"), "{err}");
        let err =
            apply_upgrade_setting("user_object_rewrite_policy", "whatever").expect_err("refused");
        assert!(err.to_string().contains("auto_safe"), "{err}");
        let err = apply_upgrade_setting("nothing_here", "1").expect_err("refused");
        assert!(err.to_string().contains("not an upgrade setting"), "{err}");
    }

    #[test]
    fn test_owns_setting_covers_the_documented_keys() {
        for (setting, _) in UPGRADE_SETTING_KEYS {
            assert!(owns_setting(setting));
            assert!(owns_setting(&setting.to_uppercase()));
        }
        assert!(!owns_setting("shared_buffers"));
    }

    #[test]
    fn test_list_and_show_answer_from_the_registries() {
        let listed = handle_list_registry(&ListRegistryStatement {
            target: ListRegistryTarget::FormatRegistry,
            limit: Some(3),
        })
        .expect("lists");
        match listed {
            DdlResult::Rows { rows, columns, .. } => {
                assert!(rows.len() <= 3);
                assert!(columns.iter().any(|(name, _)| name == "format_kind"));
            }
            other => panic!("expected rows, got {other:?}"),
        }

        let shown = handle_show_upgrade(&ShowUpgradeStatement {
            target: ShowUpgradeTarget::State,
        })
        .expect("shows");
        match shown {
            DdlResult::Rows { rows, .. } => assert!(!rows.is_empty()),
            other => panic!("expected rows, got {other:?}"),
        }
    }

    #[test]
    fn test_show_format_migrations_refuses_an_unknown_kind() {
        let err = handle_show_upgrade(&ShowUpgradeStatement {
            target: ShowUpgradeTarget::FormatMigrations {
                format_kind: Some("nothing_here".to_string()),
            },
        })
        .expect_err("refused");
        assert!(err.to_string().contains("not a format kind"), "{err}");
    }

    #[test]
    fn test_setting_a_signature_scheme_for_an_unknown_artifact_kind_is_refused() {
        let err = handle_set_signature_scheme(
            &SetSignatureSchemeStatement {
                scheme: "Ed25519".to_string(),
                artifact_kind: "NotAKind".to_string(),
            },
            None,
        )
        .expect_err("refused");
        assert!(err.to_string().contains("is not an artifact kind"), "{err}");
    }

    /// A binding that took effect and could not be written down would be gone
    /// at the next restart and would never reach another member, so the
    /// statement says so rather than reporting a success that does not last
    #[test]
    fn test_a_binding_with_nowhere_to_persist_is_refused() {
        let err = handle_set_signature_scheme(
            &SetSignatureSchemeStatement {
                scheme: "Ed25519".to_string(),
                artifact_kind: "JWT".to_string(),
            },
            None,
        )
        .expect_err("refused");
        assert!(err.to_string().contains("nowhere to persist"), "{err}");
    }

    /// The binding a node writes down is the one it applied, and it comes
    /// back from the same string, so a restart lands on the state the
    /// statement left rather than on the binary's default
    #[test]
    fn test_a_binding_persists_under_the_key_the_next_boot_reads() {
        let written: std::sync::Arc<parking_lot::Mutex<Vec<(String, String)>>> =
            std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&written);
        let writer: SettingWriter = std::sync::Arc::new(move |key: &str, value: &str| {
            sink.lock().push((key.to_string(), value.to_string()));
            Ok(())
        });
        let substrate = zyron_common::format::substrate().expect("substrate");
        let before = substrate
            .schemes
            .binding_setting(ArtifactKind::Jwt)
            .expect("JWT is bound");

        handle_set_signature_scheme(
            &SetSignatureSchemeStatement {
                scheme: "Ed25519".to_string(),
                artifact_kind: "JWT".to_string(),
            },
            Some(&writer),
        )
        .expect("applies");

        let entries = written.lock().clone();
        assert_eq!(entries.len(), 1, "{entries:?}");
        assert_eq!(entries[0].0, binding_config_key(ArtifactKind::Jwt));
        assert_eq!(entries[0].1, "Ed25519");

        handle_rotate_signature_scheme(
            &RotateSignatureSchemeStatement {
                artifact_kind: "JWT".to_string(),
                new_scheme: "RS256".to_string(),
                overlap: Some("1h".to_string()),
            },
            Some(&writer),
        )
        .expect("rotates");

        let entries = written.lock().clone();
        assert_eq!(entries.len(), 2, "{entries:?}");
        let rotated = &entries[1].1;
        let parts: Vec<&str> = rotated.split('|').collect();
        assert_eq!(parts.len(), 3, "{rotated}");
        assert_eq!(parts[0], "RS256");
        assert_eq!(parts[1], "Ed25519");
        assert!(
            parts[2].parse::<u64>().is_ok(),
            "the overlap end has to be an absolute second, it was `{}`",
            parts[2]
        );

        substrate
            .schemes
            .apply_binding_setting(ArtifactKind::Jwt, &before)
            .expect("restores");
    }

    #[test]
    fn test_a_manual_upgrade_to_a_bad_version_is_refused() {
        let err = handle_trigger_upgrade(
            &TriggerUpgradeStatement {
                action: TriggerUpgradeAction::UpgradeTo("latest".to_string()),
            },
            None,
            "operator",
        )
        .expect_err("refused");
        assert!(err.to_string().contains("major.minor.patch"), "{err}");
    }

    /// A controller that records what it was asked, so the statements can
    /// be checked for what they hand over
    struct Recording {
        asked: std::sync::Mutex<Vec<String>>,
    }

    impl UpgradeControl for Recording {
        fn request_upgrade(
            &self,
            version: &str,
            actor: &str,
            _now_secs: u64,
        ) -> Result<(), ZyronError> {
            self.asked
                .lock()
                .expect("test lock")
                .push(format!("upgrade {version} by {actor}"));
            Ok(())
        }
        fn request_rollback(&self, actor: &str, _now_secs: u64) -> Result<(), ZyronError> {
            self.asked
                .lock()
                .expect("test lock")
                .push(format!("rollback by {actor}"));
            Err(ZyronError::UpgradeRefused(
                "there is no upgrade to roll back".into(),
            ))
        }
        fn acknowledge_rewrites(
            &self,
            category: RewriteCategory,
            actor: &str,
            _now_secs: u64,
        ) -> Result<usize, ZyronError> {
            self.asked
                .lock()
                .expect("test lock")
                .push(format!("acknowledge {category} by {actor}"));
            Ok(2)
        }
    }

    #[test]
    fn test_manual_statements_reach_the_controller_with_the_operator_named() {
        let control = Recording {
            asked: std::sync::Mutex::new(Vec::new()),
        };
        let board = zyron_common::format::upgrade_board();
        board.update_settings(|settings| settings.paused = false);
        handle_trigger_upgrade(
            &TriggerUpgradeStatement {
                action: TriggerUpgradeAction::UpgradeTo("9.9.9".to_string()),
            },
            Some(&control),
            "ana",
        )
        .expect("recorded");
        let err = handle_trigger_upgrade(
            &TriggerUpgradeStatement {
                action: TriggerUpgradeAction::Rollback,
            },
            Some(&control),
            "ana",
        )
        .expect_err("the controller refused");
        assert!(err.to_string().contains("no upgrade to roll back"), "{err}");
        match handle_acknowledge_upgrade_rewrites(
            &AcknowledgeUpgradeRewritesStatement {
                category: AcknowledgeRewriteCategory::Unsafe,
            },
            Some(&control),
            "ana",
        )
        .expect("acknowledged")
        {
            DdlResult::Rows { rows, .. } => assert_eq!(rows, vec![vec!["unsafe", "2"]]),
            other => panic!("expected rows, got {other:?}"),
        }
        assert_eq!(
            *control.asked.lock().expect("test lock"),
            vec![
                "upgrade 9.9.9 by ana",
                "rollback by ana",
                "acknowledge unsafe by ana"
            ]
        );
    }

    #[test]
    fn test_without_a_controller_a_manual_statement_is_refused_not_dropped() {
        let err = handle_acknowledge_upgrade_rewrites(
            &AcknowledgeUpgradeRewritesStatement {
                category: AcknowledgeRewriteCategory::Ambiguous,
            },
            None,
            "ana",
        )
        .expect_err("refused");
        assert!(
            err.to_string().contains("without an upgrade controller"),
            "{err}"
        );
    }

    #[test]
    fn test_every_upgrade_setting_persists_under_the_upgrade_section() {
        for (setting, key) in UPGRADE_SETTING_KEYS {
            assert!(owns_setting(setting));
            assert_eq!(config_key(setting), Some(*key));
            assert!(key.starts_with("upgrade."), "{key}");
        }
        assert_eq!(config_key("max_connections"), None);
        for (setting, key) in UPGRADE_SETTING_KEYS {
            assert_eq!(setting_for_config_key(key), Some(*setting));
        }
        assert_eq!(setting_for_config_key("server.port"), None);
    }
}
