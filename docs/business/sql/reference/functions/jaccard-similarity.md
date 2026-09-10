# jaccard_similarity

Divides the number of tokens in both sets by the number in either, after removing duplicates from each side. Two empty sets give 1.0. The measure falls as either set grows, so comparing a short set against a long one scores low even when the short one is fully contained, which overlap_coefficient does not.

## Syntax

```sql
jaccard_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either set is NULL.

## Examples

```sql
SELECT jaccard_similarity('["a","b"]', '["b","c"]')
```

0.3333333333333333, one shared token out of three.

## Refused

- An argument is not a text, binary or array column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sorensen_dice](sorensen-dice.md)
- [overlap_coefficient](overlap-coefficient.md)
- [minhash_similarity](minhash-similarity.md)
