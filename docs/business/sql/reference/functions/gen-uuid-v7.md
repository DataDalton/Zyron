# gen_uuid_v7

Holds a 48-bit millisecond timestamp followed by the version bits and 74 bits of randomness, per RFC 9562. Values generated in order sort in order, so an index on the column appends rather than scattering, and the creation time is recoverable from the first six bytes. The random bits keep values distinct within one millisecond.

## Syntax

```sql
gen_uuid_v7()
```

## Returns

UUID, 16 bytes. Never NULL.

## Examples

```sql
SELECT uuid_to_string(gen_uuid_v7())
```

A time-ordered UUID in the hyphenated form.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_uuid_v4](gen-uuid-v4.md)
- [gen_ulid](gen-ulid.md)
- [uuid_to_string](uuid-to-string.md)
- [uuid_v7](uuid-v7.md)
