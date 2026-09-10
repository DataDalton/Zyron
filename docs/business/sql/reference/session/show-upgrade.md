# SHOW UPGRADE

Reports an upgrade's progress: which members have taken the new version, what the upgrade is waiting on, and whether it is waiting for a rewrite to be acknowledged.

## Syntax

```sql
SHOW UPGRADE STATE
```

## Examples

```sql
SHOW UPGRADE STATE
```

Which members have upgraded and what the upgrade is waiting on.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [TRIGGER UPGRADE](trigger-upgrade.md)
- [LIST REGISTRY](list-registry.md)
