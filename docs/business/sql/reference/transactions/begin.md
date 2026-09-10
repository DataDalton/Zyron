# BEGIN

Opens a transaction. Until it commits or rolls back, every statement the session runs is part of it and nothing it writes is visible to another session. Without one, each statement runs in a transaction of its own that commits when it finishes. The isolation level says what the transaction sees of other transactions' work while it runs.

## Syntax

```sql
BEGIN [TRANSACTION] [ISOLATION LEVEL READ COMMITTED | REPEATABLE READ | SERIALIZABLE] [READ ONLY | READ WRITE]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ISOLATION LEVEL READ COMMITTED` | Each statement sees what was committed when that statement started. | The session's default isolation level is used. |
| `ISOLATION LEVEL REPEATABLE READ` | Every statement sees what was committed when the transaction started. | Not applicable. |
| `ISOLATION LEVEL SERIALIZABLE` | The transaction commits only if the result could have been reached by running transactions one at a time. | Not applicable. |
| `READ ONLY` | Refuses any write in this transaction, which lets it take a lighter path. | The transaction may write. |

## Examples

```sql
BEGIN
```

A transaction that the statements after it belong to.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [COMMIT](commit.md)
- [ROLLBACK](rollback.md)
- [SAVEPOINT](savepoint.md)
