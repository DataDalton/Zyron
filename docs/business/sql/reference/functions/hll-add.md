# hll_add

Hashes the value's bytes and raises one register to the leading-zero count of that hash, leaving the register alone when it already holds more. Text hashes its UTF-8 bytes, integers their 64-bit little-endian form, floats their f64 bytes, and booleans a single byte. Adding a value already counted leaves the sketch unchanged. Sketch bytes that do not parse give NULL rather than an error.

## Syntax

```sql
hll_add(sketch, value)
```

## Returns

HYPERLOGLOG. NULL when the sketch or the value is NULL, or the sketch bytes do not parse.

## Examples

```sql
SELECT hll_add(hll_create(12), 'alice')
```

The sketch with one register raised.

## Refused

- The sketch argument is not text or binary.
- The value is an interval, which has no byte form to hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hll_create](hll-create.md)
- [hll_count](hll-count.md)
- [hll_merge](hll-merge.md)
