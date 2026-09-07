# Auto-Upgrade

Zyron keeps itself up to date. In the common case you leave `auto_upgrade_enabled = true` on the default `stable` channel and the cluster upgrades itself with zero admin intervention.

## What Zyron guarantees during upgrade

- No downtime. Nodes drain and restart one at a time.
- No data loss. Every migration is transactional and rollback-safe.
- No manual migration scripts. All format and catalog migrations are applied automatically.
- No surprise behavior changes. Any deprecated syntax or feature warns before it errors, and errors before it is removed.
- Admins are notified when an upgrade starts, when each node completes, and when the sequence finishes.

Admin acknowledgment is required only when the upgrade would rewrite user-authored SQL objects in a way that could change behavior, or when a change removes a feature and existing objects still reference it.

## Release channels

Set the channel with `ALTER SYSTEM SET upgrade_channel = 'stable'`.

- `stable` (default), production-tested releases.
- `beta`, pre-release for early feedback.
- `canary`, bleeding edge.
- `pinned`, locked to a specific version. Set the version with `ALTER SYSTEM SET pinned_version = '<version>'`.

## Where releases come from

Zyron publishes every release on the project's GitHub releases page, and that page is the feed each node reads. Each release carries a signed manifest per platform listing the versions available on each channel, and the server binary each version names. A node reads the manifest for its platform from the latest release, checks the manifest and the binary against the release signing key carried in every Zyron binary, and installs nothing the key did not sign.

Two settings cover deployments that read releases from somewhere else. `release_feed_url` points a node at an internal mirror of the release files, for servers that cannot reach GitHub. `release_signing_key` is the public half of the key a self-built Zyron is signed with, from `zyron-ctl release keygen` and `zyron-ctl release sign`.

An air-gapped node takes a release by hand. `zyron-ctl release stage --manifest <file> --binary <file> --data-dir <dir>` checks the binary against the digest the manifest declares and places both in the node's `releases` directory, which every node reads ahead of the feed.

## Which node drives

The leader of the consensus group drives an upgrade, or the only node when there is no group. `TRIGGER MANUAL UPGRADE` and `TRIGGER MANUAL ROLLBACK` are refused on any other node with the leader's name. Every other node stages what it is asked to stage, restarts when asked, and answers its status over the mesh, so each cluster member needs a mesh address on the leader, created with `CREATE PEER <name> ADDRESS '<host:port>'`.

The upgrade settings are cluster settings. `ALTER SYSTEM SET` of one on any node takes effect on that node at once and reaches the leader, which writes it to the replicated log, and every node applies it in the same order. A pause, a pinned version, or a channel holds across a leadership change and reaches a node that was down when it comes back. The leader writes a setting to the log only once every node runs a version that applies it. While a node still runs an older binary, during a rolling upgrade or after a rollback, the setting holds on the node that took it and the leader tries again each second, so no node is ever handed an entry it cannot apply.

## Compatibility gate

Before an upgrade proceeds, Zyron checks that your cluster is safe to advance. The gate blocks the upgrade if any of these conditions hold and reports which ones.

- The new binary cannot read one of your currently-persisted format versions.
- A user-authored SQL object cannot be safely rewritten and needs manual attention.
- A federated peer cluster is on an incompatible version.
- A Zyron App deployed on the cluster declares incompatibility with the target version.

For user-authored objects, the gate classifies each proposed rewrite:

- **Safe.** Mechanical rename or syntactic reshuffle. Applied automatically.
- **Ambiguous.** Change could affect behavior in narrow cases. Requires admin acknowledgment before it applies.
- **Unsafe.** Feature removed or behavior change with no safe automatic rewrite. Blocks the upgrade until you rewrite the object or explicitly accept it will break.

## Rolling upgrade behavior

Every node fetches and verifies the release before any node restarts, so a download that fails on the last node fails before the first one has moved. Then nodes upgrade one at a time. Each node stops taking new connections, lets the queries and transactions it has in flight finish, restarts with the new binary, and is monitored against a pre-upgrade health baseline of latency, throughput, and error rate. If a node fails its health check for longer than the recovery threshold, that node is rolled back automatically and the whole upgrade sequence is paused for admin review. The pause is a cluster setting written to every node's config, so it survives restarts and a change of leader until you clear it. The cluster is never cascade-rolled to the new version without health confirmation at each step.

For the length of the sequence the cluster runs two adjacent versions. A node on the new binary uses nothing the old binary does not read, log entries, calls between nodes, and fields in their messages included, until every node runs the new one, and a rollback of any node holds that line again on its own.

Leadership transfer happens after all followers are upgraded successfully. The former leader is the last node to receive the new binary. It hands leadership to the most current follower, drains, records what it is doing in its upgrade journal, and restarts itself. The new process reads the journal, watches its own health against the baseline, and either finishes the upgrade or puts the previous binary back and pauses. The outgoing binary is kept beside the new one for exactly this, so a rollback is a rename, not a download.

A node that finishes on the new binary then moves its own persisted state forward: eager format migrations within the configured time, disk, and memory budgets, catalog rows whose schema changed, and the user-object rewrites the policy applies on its own.

## Downgrade

Downgrade is available when every format, catalog, and user-object migration in the upgrade was reversible. If any one-way migration ran, downgrade is blocked and the specific migration is named in the error message. An optional pre-upgrade backup snapshot (default on for major-version bumps) provides a restore-from-snapshot path in that case. `TRIGGER MANUAL ROLLBACK` checks eligibility against the last completed upgrade recorded in the node's journal, then puts every node back on the previous binary, followers first and the leader last. The cluster comes back with automatic upgrades paused, because the release it left is still on the feed and would otherwise be applied again at the next poll. `ALTER SYSTEM SET auto_upgrade_paused = false` resumes them.

## Emergency controls

```sql
-- Pause all in-progress and queued upgrades cluster-wide
ALTER SYSTEM SET auto_upgrade_paused = true;

-- Set maintenance window. Upgrades queued but not triggered outside the window.
ALTER SYSTEM SET auto_upgrade_window = '02:00-04:00 UTC';

-- Trigger an upgrade manually
TRIGGER MANUAL UPGRADE TO '<version>';

-- Trigger a rollback when eligible
TRIGGER MANUAL ROLLBACK;

-- Let the rewrites that waited for a person go ahead. AMBIGUOUS covers
-- every pending rewrite that is not unsafe, UNSAFE accepts that those
-- objects will break
ACKNOWLEDGE UPGRADE REWRITES AMBIGUOUS;
ACKNOWLEDGE UPGRADE REWRITES UNSAFE;

-- Show current state
SHOW UPGRADE STATE;

-- List past upgrades
LIST UPGRADE HISTORY LIMIT 10;

-- Show in-progress format migrations
SHOW FORMAT MIGRATIONS;
SHOW FORMAT MIGRATIONS FOR FORMAT heap_page;
```

## Configuration

The settings live in the `[upgrade]` section of `zyron.toml`. Each one can also be set at runtime with `ALTER SYSTEM SET`, which takes effect at once and writes the value to `zyron.auto.conf` under its config key, so the next start reads back what was set.

| `ALTER SYSTEM SET` name | Config key | Default | Description |
| ------------------------- | ------------ | --------- | ------------- |
| `auto_upgrade_enabled` | `upgrade.auto_upgrade_enabled` | `true` | Master toggle. A manual trigger goes ahead with this off. |
| `upgrade_channel` | `upgrade.channel` | `'stable'` | Release channel. |
| `pinned_version` | `upgrade.pinned_version` | (unset) | Applies when channel is `pinned`. |
| `auto_upgrade_window` | `upgrade.window` | any time | Maintenance windows as `HH:MM-HH:MM UTC`, comma separated. A manual trigger goes ahead outside the window. |
| `auto_upgrade_paused` | `upgrade.paused` | `false` | Emergency pause. Also set by any rollback, cleared only by you. |
| `user_object_rewrite_policy` | `upgrade.user_object_rewrite_policy` | `'auto_safe'` | `auto_safe`, `notify_all`, or `manual_only`. |
| | `upgrade.format_migration_budget_memory_fraction` | `0.25` | Fraction of node memory an eager migration may hold. |
| | `upgrade.format_migration_budget_time_secs` | 6 hours | Max time per format migration. |
| | `upgrade.format_migration_budget_disk_multiple` | `2.0` | Free disk a migration needs, as a multiple of the file it moves. |
| `rollback_on_health_fail` | `upgrade.rollback_on_health_fail` | `true` | Auto-rollback the current node and auto-pause the sequence. |
| `federation_coordination_timeout` | `upgrade.federation_coordination_timeout_secs` | 30 min | Federation compat gate timeout. |
| `pre_upgrade_backup_snapshot` | `upgrade.pre_upgrade_backup_snapshot` | `true` | Take a pre-upgrade snapshot before a major version bump. |
| `deprecation_warning_rate_limit_per_hour` | `upgrade.deprecation_warning_rate_limit_per_hour` | 10 per item per tenant | Warning rate limit. |
| `release_feed_poll_interval` | `upgrade.release_feed_poll_interval_secs` | 4h | How often to check for new releases. |
| | `upgrade.health_recovery_timeout_secs` | 5 min | How long a restarted node has to reach the baseline. |
| | `upgrade.health_poll_interval_secs` | 5 | Seconds between health observations. |
| | `upgrade.health_latency_multiplier` | `2.0` | p99 may rise to this multiple of the baseline. |
| | `upgrade.health_throughput_floor` | `0.5` | Throughput may fall to this fraction of the baseline. |
| | `upgrade.health_error_rate_ceiling` | `0.01` | Error rate above which a node is unhealthy. |
| | `upgrade.drain_timeout_secs` | 5 min | How long a node has to finish in-flight work before it restarts. |
| | `upgrade.stage_timeout_secs` | 10 min | How long a node has to fetch and verify a release. |
| | `upgrade.release_feed_url` | the project's GitHub releases | An internal mirror of the release files, as a URL, or a directory read with no remote feed. |
| | `upgrade.release_signing_key` | the key carried in the binary | The public half of the key a self-built Zyron is signed with, as hex. |
| | `upgrade.release_signing_scheme` | `Ed25519` | The scheme that key belongs to. |
| | `upgrade.notify_webhook_url` | (unset) | Webhook that receives upgrade notifications. |
| | `upgrade.notify_slack_webhook_url` | (unset) | Slack incoming webhook that receives upgrade notifications. |
| | `upgrade.notify_discord_webhook_url` | (unset) | Discord channel webhook that receives upgrade notifications as an embed. |

Each node reads its own `[upgrade]` section at start, and a value set with `ALTER SYSTEM SET` reaches every node through the replicated log and is written to each node's `zyron.auto.conf`, so keep the `[upgrade]` section the same across a cluster. The overrides converge on their own.

## Observability

Query these views to see upgrade state:

- `zyron_sys.upgrade.state`, current upgrade phase per node.
- `zyron_sys.upgrade.history`, audit trail of past upgrades.
- `zyron_sys.upgrade.format_migrations`, in-progress format migrations with progress.
- `zyron_sys.upgrade.user_object_rewrites`, rewrite queue and status.
- `zyron_sys.upgrade.deprecation_warnings`, warnings emitted in the trailing window.

Live progress is also available via WebSocket subscription:

- `/api/upgrade/state`
- `/api/upgrade/format_migrations`
- `/api/upgrade/rewrites`
- `/api/upgrade/deprecation_warnings`

## CLI

- `zyron-ctl upgrade check`, reports current upgrade state, the format registry, in-progress format migrations, and the user-object rewrite queue. Add `--verbose` for deprecation warnings.
- `zyron-ctl upgrade trigger --version X.Y.Z`, admin explicit trigger.
- `zyron-ctl upgrade rollback`, trigger rollback when eligible.
- `zyron-ctl upgrade pause` and `zyron-ctl upgrade resume`.
- `zyron-ctl upgrade show-state`.
- `zyron-ctl upgrade acknowledge --category ambiguous|unsafe`, runs `ACKNOWLEDGE UPGRADE REWRITES`.
- `zyron-ctl release stage --manifest <file> --binary <file> --data-dir <dir>`, delivers a release to an air-gapped node's feed directory, checking the binary against the manifest's digest first.
- `zyron-ctl release keygen --out <file>` and `zyron-ctl release sign ...`, sign a self-built Zyron into a manifest of your own.
- `zyron-ctl deprecation report`, lists deprecation warnings in the trailing window per tenant.
- `zyron-server --capabilities`, prints the formats and config keys a binary reads, which is what the gate asks a staged binary before trusting it with the data.

## Notification

Every upgrade step is written to the audit hash chain and delivered to the channels configured with `upgrade.notify_webhook_url`, `upgrade.notify_slack_webhook_url`, and `upgrade.notify_discord_webhook_url`, all HTTP posts of the event made by the node directly. Notifications fire when an upgrade is pending, when it starts, when each node completes, on any rollback or pause event, and when the sequence finishes. A Discord channel takes the webhook address a Discord server's channel integrations page issues, described in [Discord's own webhook guide](https://support.discord.com/hc/en-us/articles/228383668), and an address that is not one is refused where it is set. Discord answers a burst with a rate limit. The node waits it out once for up to 60 seconds, and an interval past that reports the event undelivered so one channel never holds an upgrade step.

## Troubleshooting

### Upgrade is stuck at "Awaiting admin acknowledgment"

Run `SHOW UPGRADE STATE`. If the state is `AwaitingAck`, the compat gate found ambiguous or unsafe rewrites requiring admin sign-off. Query `zyron_sys.upgrade.user_object_rewrites` to see the affected objects and their diffs, then `ACKNOWLEDGE UPGRADE REWRITES AMBIGUOUS` to let them apply, or `ACKNOWLEDGE UPGRADE REWRITES UNSAFE` to accept that those objects break. The next pass picks the upgrade up.

### A rewrite reads "computed but not written"

The rewrite was worked out but the rewritten statement could not be rendered as SQL this release parses, so the object was left as it was. The queue holds the diff. Apply it by hand.

### Upgrade is paused after a rollback

Every rollback pauses automatic upgrades, whether a node failed its health check after restart or an operator ran `TRIGGER MANUAL ROLLBACK`. Check `zyron_sys.upgrade.history` for the reason. Common causes are configuration incompatibility, a drain that did not complete in time, or a regression in the new version. The pause is a cluster setting, written through the replicated log to every node's `zyron.auto.conf`, so clear it once, on any node, with `ALTER SYSTEM SET auto_upgrade_paused = false` once the root cause is resolved. The coordinator polls the feed as soon as the pause clears, and `TRIGGER MANUAL UPGRADE` names a version when the feed's newest is not the one wanted.

### A node did not stage the release

`SHOW UPGRADE STATE` names the node and why: the feed did not carry the version, the download failed, or the binary did not match the manifest's digest. The coordinator asks again on the next pass. For an air-gapped cluster, every node needs the release in its own `releases` directory.

### Downgrade is blocked

A one-way migration ran during the upgrade. `SHOW FORMAT MIGRATIONS` lists which format is affected. Restore from the pre-upgrade backup snapshot if one was taken. Otherwise downgrade is not possible.

### Federation peer at incompatible version

The compat gate reports the peer and its version. Coordinate the peer's upgrade first, then retry.

### A Zyron App is blocking upgrade

The App declared incompatibility with the target version. Check the App's compatibility manifest. Update the App or accept it will need to be updated before this upgrade.

## Related

- [../storage/format-agility.md](../storage/format-agility.md), guarantees for on-disk file format upgrades.
- [../security/signature-agility.md](../security/signature-agility.md), guarantees for cryptographic scheme rotation.
