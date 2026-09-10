# tdigest_merge

Concatenates the centroids of both digests, sorts them by mean, and compresses at the larger of the two compression settings. Unlike the other sketch merges this accepts digests built at different settings, because compression is a budget rather than a shape the bytes depend on.

## Syntax

```sql
tdigest_merge(a, b)
```

## Returns

TDIGEST. NULL when either digest is NULL or fails to parse.

## Examples

```sql
SELECT tdigest_cdf(tdigest_merge(tdigest_add(tdigest_create(100), 1), tdigest_add(tdigest_create(100), 9)), 5)
```

0.5.

## Refused

- Either argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [tdigest_create](tdigest-create.md)
- [tdigest_add](tdigest-add.md)
- [hll_merge](hll-merge.md)
- [cms_merge](cms-merge.md)
