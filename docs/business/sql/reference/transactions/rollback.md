# ROLLBACK

Discards the open transaction's writes as though it had not run. Naming a savepoint discards only the work after that savepoint and leaves the transaction open, which is how one failed step is undone without losing the steps before it.

## Syntax

```sql
ROLLBACK [TRANSACTION] [TO [SAVEPOINT] name]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `TO [SAVEPOINT] name` | Discards only the work done after that savepoint, leaving the transaction open. | The whole transaction is discarded and ends. |

## Examples

```sql
ROLLBACK
```

Everything the transaction wrote is discarded.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [BEGIN](begin.md)
- [COMMIT](commit.md)
- [SAVEPOINT](savepoint.md)
