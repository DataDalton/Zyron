# DROP REPLICATION SLOT

Releases the position a slot held, allowing the log behind it to be pruned. Changes the slot had not yet delivered are discarded.

## Syntax

```sql
DROP REPLICATION SLOT name
```

## Examples

```sql
DROP REPLICATION SLOT s1
```

The held position is released and the log behind it can be pruned.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE REPLICATION SLOT](create-replication-slot.md)
