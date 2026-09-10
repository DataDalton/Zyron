# CREATE FUNCTION

Names a SQL body so it can be called in an expression. The language is SQL, which is deliberate rather than a limitation: a SQL function is inlined into the statement that calls it, so the planner still sees through it to choose an index, push a predicate down and decompose an aggregate. A function the planner could not see into would forfeit all three. The volatility tells the planner whether the result can be reused for the same arguments.

## Syntax

```sql
CREATE FUNCTION name(arg type [DEFAULT expr], ...) RETURNS type AS 'body' LANGUAGE SQL [IMMUTABLE | STABLE | VOLATILE]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `IMMUTABLE` | Promises the same arguments always give the same answer, so the call can be folded. | The function is treated as volatile and called per row. |
| `STABLE` | Promises the answer does not change within one statement. | Not applicable. |
| `DEFAULT expr` | Lets a caller leave that argument out. | Every argument must be given. |

## Examples

```sql
CREATE FUNCTION double(a INT) RETURNS BIGINT AS 'SELECT a * 2' LANGUAGE SQL IMMUTABLE
```

A function callable in an expression, inlined where it is called.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP FUNCTION](drop-function.md)
- [CREATE PROCEDURE](create-procedure.md)
- [CREATE AGGREGATE](create-aggregate.md)
