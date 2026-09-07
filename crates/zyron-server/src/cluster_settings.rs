//! Upgrade settings as cluster state.
//!
//! An upgrade setting is policy for the whole group, a pause, a pinned
//! version, a channel, so it rides the consensus log as a key-value entry and
//! every node applies it at the same point in the sequence. A leadership
//! change cannot lose it, and a node that was down applies it when it
//! catches up. This is what applies one on a node, the board first so the
//! upgrade service sees it at once and `zyron.auto.conf` second so the next
//! boot seeds the board from it.
//!
//! The node the operator ran `ALTER SYSTEM SET` on applies the value locally
//! before it reaches the log, so the statement takes effect there at once,
//! and applies it again when the entry commits, which is the same value

use std::path::Path;

use zyron_common::format::BinaryVersion;
use zyron_common::{Result, ZyronError};

use crate::config::ZyronConfig;

/// The release whose apply loop takes a cluster setting from the log. The
/// leader proposes one only once every member of the group runs this or
/// later, because a member on an earlier release refuses the entry as
/// corruption and stops applying
pub const INTRODUCED_IN: BinaryVersion = BinaryVersion::new(0, 12, 0);

/// Whether a config key is one the cluster replicates
pub fn is_cluster_setting(key: &str) -> bool {
    zyron_wire::format_dispatch::setting_for_config_key(key).is_some()
        || crate::crypto_settings::is_crypto_setting(key)
}

/// Applies one replicated setting on this node. The value written to the
/// config file is the one the applier stored, so the file and the live state
/// agree.
///
/// Two families share the carrier. Upgrade settings land on the board, scheme
/// bindings land on the signature registry, and both are policy for the whole
/// group rather than for a node, which is what puts them in the log rather
/// than in each node's own config
pub fn apply(data_dir: &Path, key: &str, value: &str) -> Result<()> {
    if crate::crypto_settings::is_crypto_setting(key) {
        return crate::crypto_settings::apply(data_dir, key, value);
    }
    let setting = zyron_wire::format_dispatch::setting_for_config_key(key).ok_or_else(|| {
        ZyronError::Internal(format!("`{key}` is not a setting the cluster replicates"))
    })?;
    let stored = zyron_wire::format_dispatch::apply_upgrade_setting(setting, value)
        .map_err(|e| ZyronError::Internal(format!("{key} = {value} was refused, {e}")))?;
    ZyronConfig::write_auto_conf(data_dir, key, &stored)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_replicated_setting_lands_on_the_board_and_in_the_config_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let board = zyron_common::format::upgrade_board();
        let before = board.settings().release_feed_poll_interval_secs;

        apply(
            dir.path(),
            "upgrade.release_feed_poll_interval_secs",
            "7200",
        )
        .expect("applies");
        assert_eq!(board.settings().release_feed_poll_interval_secs, 7200);
        let conf = std::fs::read_to_string(dir.path().join("zyron.auto.conf")).expect("reads");
        assert!(conf.contains("release_feed_poll_interval_secs"), "{conf}");
        assert!(conf.contains("7200"), "{conf}");
        board.update_settings(|settings| settings.release_feed_poll_interval_secs = before);

        assert!(is_cluster_setting("upgrade.paused"));
        assert!(!is_cluster_setting("server.port"));
        let err = apply(dir.path(), "server.port", "1").expect_err("not replicated");
        assert!(
            err.to_string()
                .contains("not a setting the cluster replicates"),
            "{err}"
        );
    }
}
