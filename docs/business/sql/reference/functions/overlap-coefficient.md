# overlap_coefficient

Divides the number of tokens in both sets by the size of the smaller set, so a set fully contained in a larger one scores 1.0 whatever the size difference. Either set being empty gives 0.0. Use this to test containment and Jaccard to test agreement.

## Syntax

```sql
overlap_coefficient(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either set is NULL.

## Examples

```sql
SELECT overlap_coefficient('["a"]', '["a","b","c"]')
```

1, because the smaller set is fully held by the larger.

## Refused

- An argument is not a text, binary or array column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jaccard_similarity](jaccard-similarity.md)
- [sorensen_dice](sorensen-dice.md)
