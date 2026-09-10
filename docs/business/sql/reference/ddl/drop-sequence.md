# DROP SEQUENCE

Removes a sequence. A column whose default took numbers from it keeps the numbers it was already given and has no generator for new rows.

## Syntax

```sql
DROP SEQUENCE [IF EXISTS] name
```

## Examples

```sql
DROP SEQUENCE order_ids
```

The generator is gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE SEQUENCE](create-sequence.md)
