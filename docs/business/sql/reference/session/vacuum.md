# VACUUM

Reclaims the space held by rows no transaction can still see. A row deleted or updated is kept until every transaction that might read the old version has ended, so space is released by this pass rather than by the statement that made the row dead. Without a table it runs over all of them.

## Syntax

```sql
VACUUM [table]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `table` | Runs over that table alone. | Every table is vacuumed. |

## Examples

```sql
VACUUM orders
```

Space held by rows nobody can see is released.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ANALYZE](analyze.md)
- [OPTIMIZE TABLE](../lake/optimize-table.md)
