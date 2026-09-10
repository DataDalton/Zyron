# PREPARE

Parses and plans a statement once under a name. EXECUTE supplies the parameters and reuses the plan, so neither parsing nor planning is repeated. The prepared statement belongs to the session and is discarded when the session ends.

## Syntax

```sql
PREPARE name [(type, ...)] AS statement
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `(type, ...)` | Declares the parameter types, rather than letting them be inferred from the statement. | The types are inferred from where the parameters are used. |

## Examples

```sql
PREPARE by_id AS SELECT total FROM orders WHERE id = $1
```

A named plan that EXECUTE runs with an id.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [EXECUTE](execute.md)
- [DEALLOCATE](deallocate.md)
