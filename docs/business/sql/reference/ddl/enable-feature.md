# ENABLE FEATURE

Turns on a capability for one table. These are per-table rather than node-wide because each costs something to maintain, in write throughput or in space, and that cost is only worth paying on the tables that use it. Turning one on may require a pass over the table to build what it needs.

## Syntax

```sql
ALTER TABLE name ENABLE feature
```

## Examples

```sql
ALTER TABLE orders ENABLE soft_delete
```

The table maintains what that capability needs from now on.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DISABLE FEATURE](disable-feature.md)
- [ALTER TABLE](alter-table.md)
