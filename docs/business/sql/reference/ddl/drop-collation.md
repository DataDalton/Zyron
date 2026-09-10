# DROP COLLATION

Removes the rule. Columns declared with it fall back to the default ordering, which changes what ORDER BY produces for them.

## Syntax

```sql
DROP COLLATION [IF EXISTS] name
```

## Examples

```sql
DROP COLLATION german
```

Columns declared with it fall back to the default ordering.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE COLLATION](create-collation.md)
