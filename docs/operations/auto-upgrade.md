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

Tenant admins can pin their compute reservation to a specific version for stability.

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

Nodes upgrade one at a time. Each node drains gracefully, restarts with the new binary, and is monitored against a pre-upgrade health baseline. If a node fails its health check for longer than the recovery threshold, that node is rolled back automatically and the whole upgrade sequence is paused for admin review. The cluster is never cascade-rolled to the new version without health confirmation at each step.

Leadership transfer happens after all followers are upgraded successfully. The former leader is the last node to receive the new binary.

## Downgrade

Downgrade is available when every format, catalog, and user-object migration in the upgrade was reversible. If any one-way migration ran, downgrade is blocked and the specific migration is named in the error message. An optional pre-upgrade backup snapshot (default on for major-version bumps) provides a restore-from-snapshot path in that case.

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

-- Show current state
SHOW UPGRADE STATE;

-- List past upgrades
LIST UPGRADE HISTORY LIMIT 10;

-- Show in-progress format migrations
SHOW FORMAT MIGRATIONS;
SHOW FORMAT MIGRATIONS FOR FORMAT heap_page;
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `auto_upgrade_enabled` | `true` | Master toggle. |
| `upgrade_channel` | `'stable'` | Release channel. |
| `pinned_version` | (unset) | Applies when channel is `pinned`. |
| `auto_upgrade_window` | any time | Maintenance window. Multiple windows configurable. |
| `auto_upgrade_paused` | `false` | Emergency pause. |
| `user_object_rewrite_policy` | `'auto_safe'` | `auto_safe`, `notify_all`, or `manual_only`. |
| `format_migration_budget_memory` | 25% of node | Max memory for eager migration. |
| `format_migration_budget_time` | 6 hours | Max time per format migration. |
| `format_migration_budget_disk` | 2x source size | Max additional disk during migration. |
| `rollback_on_health_fail` | per-node yes | Auto-rollback current node and auto-pause sequence. |
| `federation_coordination_timeout` | 30 min | Federation compat gate timeout. |
| `pre_upgrade_backup_snapshot` | on for major | Optional pre-upgrade snapshot. |
| `deprecation_warning_rate_limit_per_hour` | 10 per item per tenant | Warning rate limit. |
| `release_feed_poll_interval` | 4h | How often to check for new releases. |

Cascade order is cluster then tenant. A tenant admin can pin their compute reservation to a version for stability.

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

- `zyron-ctl upgrade check`, dry-run compatibility gate. Shows what would happen.
- `zyron-ctl upgrade trigger --version X.Y.Z`, admin explicit trigger.
- `zyron-ctl upgrade rollback`, trigger rollback when eligible.
- `zyron-ctl upgrade pause` and `zyron-ctl upgrade resume`.
- `zyron-ctl upgrade show-state`.
- `zyron-ctl deprecation report`, lists deprecation warnings in the trailing window per tenant.

## Notification

Every upgrade step is written to the audit hash chain and delivered via the contact channels you have configured (email, webhook, Slack, PagerDuty). Notifications fire when an upgrade is pending, when it starts, when each node completes, on any rollback or pause event, and when the sequence finishes.

## Troubleshooting

### Upgrade is stuck at "Awaiting admin acknowledgment"

Run `SHOW UPGRADE STATE`. If the state is `AwaitingAck`, the compat gate found ambiguous or unsafe rewrites requiring admin sign-off. Run `zyron-ctl upgrade check --verbose` to see the affected objects.

### Upgrade was auto-paused after a rollback

A node failed its health check after restart. Check `zyron_sys.upgrade.history` for the specific failure. Common causes are configuration incompatibility, a drain that did not complete in time, or a regression in the new version. Once the root cause is resolved, `TRIGGER MANUAL UPGRADE` to resume.

### Downgrade is blocked

A one-way migration ran during the upgrade. `SHOW FORMAT MIGRATIONS` lists which format is affected. Restore from the pre-upgrade backup snapshot if one was taken. Otherwise downgrade is not possible.

### Federation peer at incompatible version

The compat gate reports the peer and its version. Coordinate the peer's upgrade first, then retry.

### A Zyron App is blocking upgrade

The App declared incompatibility with the target version. Check the App's compatibility manifest. Update the App or accept it will need to be updated before this upgrade.

## Related

- [../storage/format-agility.md](../storage/format-agility.md), guarantees for on-disk file format upgrades.
- [../security/signature-agility.md](../security/signature-agility.md), guarantees for cryptographic scheme rotation.
