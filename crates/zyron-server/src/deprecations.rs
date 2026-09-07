//! Server deprecation registrations.
//!
//! One record per item this server took away, submitted into the
//! deprecation registry the same way `format.rs` submits this crate's
//! formats. A record carries metadata and no code. It is what
//! `zyron_sys.deprecation.registry` reads and what the guide in
//! `zyron_sys.deprecation.migration_guides` is generated from.
//!
//! Records are grouped by the release that removed the item, newest last.
//! Each names its release in every lifecycle field, so retiring one is
//! deleting its block

use zyron_common::format::deprecation::{DeprecatedItemKind, DeprecationRecord};

// 0.14.0 removed the pagerduty contact channel kind. Upgrade notifications
// reach a person through a Discord, Slack, or plain webhook channel
inventory::submit! {
    DeprecationRecord {
        item_kind: DeprecatedItemKind::ConfigKey,
        item_id: "upgrade contact channel kind pagerduty",
        deprecated_since_version: "0.14.0",
        warn_until_version: "0.14.0",
        error_since_version: "0.14.0",
        removed_since_version: "0.14.0",
        replacement_ref: Some("upgrade.notify_discord_webhook_url"),
        migration_guide_url: None,
        no_guide_required: true,
        summary: "The pagerduty contact channel kind was removed. Upgrade notifications go to \
                  a Discord, Slack, or plain webhook channel",
        before_example: "ALTER SYSTEM SET upgrade.notify_pagerduty_routing_key = 'RK-XXXX'",
        after_example: "ALTER SYSTEM SET upgrade.notify_discord_webhook_url = \
                        'https://discord.com/api/webhooks/123/abc'",
        migration_snippet: "SHOW ALL\n-- the upgrade.notify_ keys name every contact channel \
                            this node delivers to",
        common_pitfall: "A PagerDuty routing key is not a URL. A Discord channel takes the \
                         webhook address the Discord server's channel integrations page \
                         issues, and a routing key put in its place is refused when the \
                         channel is built",
    }
}

#[cfg(test)]
mod tests {
    use zyron_common::format::deprecation::{BinaryVersion, DeprecationStage};

    /// Every record this crate submits reaches the registry, and each one is
    /// past the release that removed it so a running node reports it gone
    #[test]
    fn test_server_deprecations_reach_the_registry() {
        let substrate = zyron_common::format::substrate().expect("loads");
        let record = substrate
            .deprecations
            .find("upgrade contact channel kind pagerduty")
            .expect("the removed contact channel kind is registered");
        assert_eq!(
            record.replacement_ref,
            Some("upgrade.notify_discord_webhook_url")
        );
        assert_eq!(
            record.stage(BinaryVersion::parse("0.14.0").expect("parses")),
            DeprecationStage::Removed
        );
    }
}
