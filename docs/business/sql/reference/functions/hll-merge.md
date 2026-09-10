# hll_merge

Takes the larger of the two values at each register, which counts the union of the two input sets with no access to the values themselves. Both sketches must carry the same precision, and the result carries it too. Sketches of differing precision give NULL.

## Syntax

```sql
hll_merge(a, b)
```

## Returns

HYPERLOGLOG. NULL when either sketch is NULL, their precisions differ, or either fails to parse.

## Examples

```sql
SELECT hll_count(hll_merge(hll_add(hll_create(12), 'a'), hll_add(hll_create(12), 'b')))
```

2.

## Refused

- Either argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hll_create](hll-create.md)
- [hll_count](hll-count.md)
- [bloom_merge](bloom-merge.md)
- [cms_merge](cms-merge.md)
