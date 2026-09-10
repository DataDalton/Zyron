# DROP PROCEDURE

Removes a procedure, so CALL on it fails. Anything it called is untouched.

## Syntax

```sql
DROP PROCEDURE [IF EXISTS] name [(type, ...)]
```

## Examples

```sql
DROP PROCEDURE reset
```

CALL on it fails and what it called remains.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE PROCEDURE](create-procedure.md)
- [CALL](../dml/call.md)
