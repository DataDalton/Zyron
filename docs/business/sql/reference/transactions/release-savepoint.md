# RELEASE SAVEPOINT

Removes a savepoint without undoing anything. Work done since it remains part of the transaction. Resources held to make a rollback to that point possible are released. Rolling back to a released savepoint is an error.

## Syntax

```sql
RELEASE [SAVEPOINT] name
```

## Examples

```sql
RELEASE SAVEPOINT before_load
```

The savepoint is gone and its work remains.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [SAVEPOINT](savepoint.md)
- [ROLLBACK](rollback.md)
