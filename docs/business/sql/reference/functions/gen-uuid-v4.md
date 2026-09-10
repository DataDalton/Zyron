# gen_uuid_v4

Returns 16 bytes of randomness with the version and variant bits set per RFC 9562. The value carries no time component, so successive calls are unordered and an index on the column scatters writes across its whole key space. Use gen_uuid_v7 where ordered keys matter. A fresh value is produced per row.

## Syntax

```sql
gen_uuid_v4()
```

## Returns

UUID, 16 bytes. Never NULL.

## Examples

```sql
SELECT uuid_to_string(gen_uuid_v4())
```

A random UUID in the hyphenated form.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_uuid_v7](gen-uuid-v7.md)
- [uuid_to_string](uuid-to-string.md)
- [uuid_v4](uuid-v4.md)
