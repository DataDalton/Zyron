# bloom_add

Sets the bits the value's hashes address, one per hash function. Byte forms match hll_add, with text hashing its UTF-8 bytes and numbers their little-endian patterns. The filter's size never changes, so every addition past the expected item count raises the false positive rate instead.

## Syntax

```sql
bloom_add(filter, value)
```

## Returns

BLOOMFILTER. NULL when the filter or the value is NULL, or the filter bytes do not parse.

## Examples

```sql
SELECT bloom_contains(bloom_add(bloom_create(100, 0.01), 'alice'), 'alice')
```

true.

## Refused

- The filter argument is not text or binary.
- The value is an interval, which has no byte form to hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bloom_create](bloom-create.md)
- [bloom_contains](bloom-contains.md)
- [bloom_false_positive_rate](bloom-false-positive-rate.md)
