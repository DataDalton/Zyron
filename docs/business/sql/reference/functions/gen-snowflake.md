# gen_snowflake

Packs a 41-bit millisecond timestamp counted from 2020-01-01, a 10-bit machine number and a 12-bit per-millisecond sequence. When the sequence fills within one millisecond the call waits for the next, so values are always distinct and increasing within one process. Distinctness across nodes rests on each node passing a different machine number, which this function does not check.

## Syntax

```sql
gen_snowflake([machine_id])
```

## Returns

BIGINT. Never NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `machine_id` | Node number from 0 to 1023, taken from the first row and applied to every row. | 0. |

## Examples

```sql
SELECT gen_snowflake(7) > 0
```

true.

## Refused

- Called with more than one argument.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [snowflake](snowflake.md)
- [gen_tsid](gen-tsid.md)
- [gen_uuid_v7](gen-uuid-v7.md)
