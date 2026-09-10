# DEALLOCATE

Forgets a prepared statement and the plan it held. A session that prepares many statements under generated names releases them this way rather than waiting for the connection to end.

## Syntax

```sql
DEALLOCATE name | DEALLOCATE ALL
```

## Examples

```sql
DEALLOCATE by_id
```

The prepared statement is gone.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [PREPARE](prepare.md)
- [EXECUTE](execute.md)
