# DROP FOREIGN TABLE

Removes the declaration. The rows on the other server are untouched, because this side never held them.

## Syntax

```sql
DROP FOREIGN TABLE [IF EXISTS] name
```

## Examples

```sql
DROP FOREIGN TABLE remote_orders
```

The declaration is gone and the remote rows are untouched.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE FOREIGN TABLE](create-foreign-table.md)
