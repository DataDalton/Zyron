# uuid_v7

Names the same function as gen_uuid_v7.

## Syntax

```sql
uuid_v7()
```

## Returns

UUID, 16 bytes. Never NULL.

## Examples

```sql
SELECT uuid_to_string(uuid_v7())
```

A time-ordered UUID in the hyphenated form.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_uuid_v7](gen-uuid-v7.md)
