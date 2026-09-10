# gen_tsid

Packs a 42-bit millisecond timestamp and 22 random bits. Unlike a snowflake it needs no machine number, trading the guarantee of distinctness for the chance of a collision between two values created in the same millisecond.

## Syntax

```sql
gen_tsid()
```

## Returns

BIGINT. Never NULL.

## Examples

```sql
SELECT gen_tsid() > 0
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_snowflake](gen-snowflake.md)
- [tsid](tsid.md)
- [gen_uuid_v7](gen-uuid-v7.md)
