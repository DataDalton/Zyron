# DROP TRIGGER

Removes the trigger, so writes to the table no longer run the function. The function itself stays and can still be called.

## Syntax

```sql
DROP TRIGGER [IF EXISTS] name ON table
```

## Examples

```sql
DROP TRIGGER audit ON orders
```

Writes no longer run the function, which still exists.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TRIGGER](create-trigger.md)
