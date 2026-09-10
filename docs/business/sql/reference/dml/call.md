# CALL

Runs a procedure with the arguments given. CALL is a statement, not an expression, so the procedure may control transactions and run several statements in turn.

## Syntax

```sql
CALL name(arg, ...)
```

## Examples

```sql
CALL reset(1)
```

The procedure's body runs with that argument.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [CREATE PROCEDURE](../ddl/create-procedure.md)
- [DO](do.md)
