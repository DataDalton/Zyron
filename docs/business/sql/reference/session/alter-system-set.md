# ALTER SYSTEM SET

Changes a setting and records it where the node reads its configuration, so it survives a restart. SET changes a setting for one session only. On a cluster the change goes through the consensus log and reaches every member.

## Syntax

```sql
ALTER SYSTEM SET name = value
```

## Examples

```sql
ALTER SYSTEM SET max_connections = 500
```

The setting is recorded and survives a restart.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SET](set.md)
- [SHOW](show.md)
- [ALTER CLUSTER](alter-cluster.md)
