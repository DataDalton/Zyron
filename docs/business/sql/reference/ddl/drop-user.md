# DROP USER

Removes a principal, so it can no longer connect. The objects it created stay, because they belong to the database rather than to the principal that made them.

## Syntax

```sql
DROP USER [IF EXISTS] name
```

## Examples

```sql
DROP USER alice
```

The principal can no longer connect.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE USER](create-user.md)
- [FORGET USER](forget-user.md)
