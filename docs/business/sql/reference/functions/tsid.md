# tsid

Names the same function as gen_tsid.

## Syntax

```sql
tsid()
```

## Returns

BIGINT. Never NULL.

## Examples

```sql
SELECT tsid() > 0
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_tsid](gen-tsid.md)
