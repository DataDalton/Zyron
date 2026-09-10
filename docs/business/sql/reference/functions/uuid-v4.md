# uuid_v4

Names the same function as gen_uuid_v4.

## Syntax

```sql
uuid_v4()
```

## Returns

UUID, 16 bytes. Never NULL.

## Examples

```sql
SELECT uuid_to_string(uuid_v4())
```

A random UUID in the hyphenated form.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_uuid_v4](gen-uuid-v4.md)
