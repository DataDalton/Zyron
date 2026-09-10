# SAVEPOINT

Marks a point inside the open transaction. Rolling back to it undoes the work done since without ending the transaction, so a step that failed can be undone and retried while the steps before it stand. Naming a savepoint that already exists replaces it.

## Syntax

```sql
SAVEPOINT name
```

## Examples

```sql
SAVEPOINT before_load
```

A point the transaction can return to without ending.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ROLLBACK](rollback.md)
- [RELEASE SAVEPOINT](release-savepoint.md)
