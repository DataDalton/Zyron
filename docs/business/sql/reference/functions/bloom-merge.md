# bloom_merge

Takes the bitwise OR of the two bit arrays, which holds every value either filter holds. Both filters must carry the same bit count and hash count, which means both must come from one pair of bloom_create arguments. Filters built with differing arguments give NULL.

## Syntax

```sql
bloom_merge(a, b)
```

## Returns

BLOOMFILTER. NULL when either filter is NULL, their parameters differ, or either fails to parse.

## Examples

```sql
SELECT bloom_contains(bloom_merge(bloom_add(bloom_create(100, 0.01), 'a'), bloom_add(bloom_create(100, 0.01), 'b')), 'b')
```

true.

## Refused

- Either argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bloom_create](bloom-create.md)
- [bloom_add](bloom-add.md)
- [hll_merge](hll-merge.md)
