# ALTER TABLE CLUSTERING SCHEDULE

Sets when the clustering pass runs for one table, overriding the engine's own schedule.

## Syntax

```sql
ALTER TABLE name CLUSTERING SCHEDULE ...
```

## Examples

```sql
ALTER TABLE events CLUSTER BY AUTO
```

The engine chooses both the keys and when to apply them.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER TABLE CLUSTER BY](alter-table-cluster-by.md)
