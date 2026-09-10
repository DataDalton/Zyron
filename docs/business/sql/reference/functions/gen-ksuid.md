# gen_ksuid

Writes a four-byte second-resolution timestamp and 16 random bytes in base 62, giving 27 characters that sort in creation order. The timestamp is coarser than a ULID's and the random payload larger, so identifiers created in the same second are unordered among themselves.

## Syntax

```sql
gen_ksuid()
```

## Returns

VARCHAR of 27 characters. Never NULL.

## Examples

```sql
SELECT length(gen_ksuid())
```

27.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_ulid](gen-ulid.md)
- [gen_uuid_v7](gen-uuid-v7.md)
- [ksuid](ksuid.md)
