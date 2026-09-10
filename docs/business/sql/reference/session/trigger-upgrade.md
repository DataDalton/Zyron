# TRIGGER UPGRADE

Starts an upgrade now instead of waiting for the configured window, to the version named. The version is written out rather than inferred from the channel, so the statement says what it will do to the cluster and whoever reads the audit log afterwards can tell which build was intended. On a cluster the upgrade proceeds member by member, so this is one statement rather than one per node.

## Syntax

```sql
TRIGGER MANUAL UPGRADE TO version
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `TO version` | The version to upgrade to, written out rather than taken from the channel. | Not applicable. |

## Examples

```sql
TRIGGER MANUAL UPGRADE TO '2.3.1'
```

The upgrade to that version begins now rather than in its window.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SHOW UPGRADE](show-upgrade.md)
- [ACKNOWLEDGE UPGRADE REWRITES](acknowledge-upgrade-rewrites.md)
- [ALTER SYSTEM SET](alter-system-set.md)
