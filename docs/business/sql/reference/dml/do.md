# DO

Runs a body once without creating a procedure for it. It is for a one-off operation that needs several statements, where naming it would leave an object behind that nothing would call again.

## Syntax

```sql
DO 'body'
```

## Examples

```sql
DO 'INSERT INTO staging (id) VALUES (1)'
```

The body runs once and nothing is created.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [CALL](call.md)
- [CREATE PROCEDURE](../ddl/create-procedure.md)
