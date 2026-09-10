# EXECUTE

Runs a statement PREPARE already planned, supplying its parameters. The plan is reused rather than made again, so the cost is execution alone.

## Syntax

```sql
EXECUTE name [(value, ...)]
```

## Examples

```sql
EXECUTE by_id (1)
```

The prepared statement's result for that parameter.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [PREPARE](prepare.md)
- [DEALLOCATE](deallocate.md)
