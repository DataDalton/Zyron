# cosine_similarity

Divides the dot product by the product of the two vector lengths, measuring direction without regard to magnitude, so doubling one vector does not change the result. Both vectors must hold the same number of elements. Two empty vectors give 0.0.

## Syntax

```sql
cosine_similarity(a, b)
```

## Returns

DOUBLE PRECISION between -1.0 and 1.0. NULL when either vector is NULL.

## Examples

```sql
SELECT cosine_similarity('[1,0]', '[1,0]')
```

1, because the vectors point the same way.

## Refused

- The two vectors hold different numbers of elements.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [dot_product](dot-product.md)
- [jaccard_similarity](jaccard-similarity.md)
