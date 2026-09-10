# CREATE PROCEDURE

Names a body of statements run by CALL. A procedure is not callable in an expression and may control transactions, which a function may not. SECURITY DEFINER runs it with the creator's privileges instead of the caller's, so a caller can run an operation without holding the privileges it requires.

## Syntax

```sql
CREATE PROCEDURE name(arg type, ...) AS 'body' LANGUAGE PLSQL [SECURITY DEFINER | INVOKER]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `SECURITY DEFINER` | Runs with the creator's privileges rather than the caller's. | It runs with the caller's privileges. |

## Examples

```sql
CREATE PROCEDURE reset(a INT) AS 'DELETE FROM staging' LANGUAGE PLSQL SECURITY DEFINER
```

A procedure CALL runs, with the creator's privileges.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP PROCEDURE](drop-procedure.md)
- [CALL](../dml/call.md)
- [CREATE FUNCTION](create-function.md)
