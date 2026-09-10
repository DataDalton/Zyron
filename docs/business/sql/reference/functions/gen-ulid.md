# gen_ulid

Writes a 48-bit millisecond timestamp and 80 bits of randomness in Crockford base 32, giving 26 characters that sort in creation order as text. The alphabet omits the letters I, L, O and U, so a transcribed identifier cannot be confused with a digit. Text ordering matching time ordering is what distinguishes this from a v4 UUID.

## Syntax

```sql
gen_ulid()
```

## Returns

VARCHAR of 26 characters. Never NULL.

## Examples

```sql
SELECT length(gen_ulid())
```

26.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [gen_uuid_v7](gen-uuid-v7.md)
- [gen_ksuid](gen-ksuid.md)
- [ulid](ulid.md)
