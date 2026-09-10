# CREATE TRIGGER

Runs a function when a table is written. BEFORE runs ahead of the write and may change or refuse the row. AFTER runs once the write has happened. FOR EACH ROW runs the function per affected row, and without it the function runs once per statement.

## Syntax

```sql
CREATE TRIGGER name BEFORE | AFTER INSERT | UPDATE | DELETE ON table [FOR EACH ROW | STATEMENT] EXECUTE FUNCTION name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `FOR EACH ROW` | Runs once per row the statement wrote. | It runs once for the statement. |
| `BEFORE` | Runs ahead of the write, so the function may change or refuse the row. | Not applicable. |

## Examples

```sql
CREATE TRIGGER audit AFTER INSERT ON orders FOR EACH ROW EXECUTE FUNCTION log_order
```

That function runs for each row inserted.

## Refused

- The table is a temporary one.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP TRIGGER](drop-trigger.md)
- [CREATE FUNCTION](create-function.md)
