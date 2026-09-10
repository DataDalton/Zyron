# sorensen_dice

Divides twice the number of tokens in both sets by the total size of the two sets, after removing duplicates from each side. It gives a higher score than Jaccard for the same pair, because shared tokens count twice in the numerator. Two empty sets give 1.0.

## Syntax

```sql
sorensen_dice(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either set is NULL.

## Examples

```sql
SELECT sorensen_dice('["a","b"]', '["b","c"]')
```

0.5, above the Jaccard score for the same pair.

## Refused

- An argument is not a text, binary or array column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jaccard_similarity](jaccard-similarity.md)
- [overlap_coefficient](overlap-coefficient.md)
