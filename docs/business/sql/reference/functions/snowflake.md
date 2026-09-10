# snowflake

Names the same function as gen_snowflake and shares its process-wide sequence state.

## Syntax

```sql
snowflake([machine_id])
```

## Returns

BIGINT. Never NULL.

## Examples

```sql
SELECT snowflake(0) > 0
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_snowflake](gen-snowflake.md)
- [tsid](tsid.md)
