# cms_merge

Adds the two sketches counter by counter, giving the counts of both input streams combined. Both sketches must carry the same width and depth. Sketches of differing dimensions give NULL.

## Syntax

```sql
cms_merge(a, b)
```

## Returns

COUNTMINSKETCH. NULL when either sketch is NULL, their dimensions differ, or either fails to parse.

## Examples

```sql
SELECT cms_estimate(cms_merge(cms_add(cms_create(256, 4), 'a'), cms_add(cms_create(256, 4), 'a')), 'a')
```

2.

## Refused

- Either argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cms_create](cms-create.md)
- [cms_add](cms-add.md)
- [hll_merge](hll-merge.md)
- [tdigest_merge](tdigest-merge.md)
