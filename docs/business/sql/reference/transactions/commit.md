# COMMIT

Ends the open transaction and keeps its writes, which become visible to every other session at that moment. On a table declared ON COMMIT DELETE ROWS or ON COMMIT DROP, this is also when that clause acts.

## Syntax

```sql
COMMIT [TRANSACTION]
```

## Examples

```sql
COMMIT
```

The transaction's writes become visible to everyone.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [BEGIN](begin.md)
- [ROLLBACK](rollback.md)
- [ON COMMIT](../session/on-commit.md)
