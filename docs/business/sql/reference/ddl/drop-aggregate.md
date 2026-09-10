# DROP AGGREGATE

Removes an aggregate. The functions it was built from stay, because they are ordinary functions that the aggregate named.

## Syntax

```sql
DROP AGGREGATE [IF EXISTS] name [(type, ...)]
```

## Examples

```sql
DROP AGGREGATE total
```

The aggregate is gone and the functions it used remain.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE AGGREGATE](create-aggregate.md)
