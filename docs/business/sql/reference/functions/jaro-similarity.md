# jaro_similarity

Scores two strings on the characters they share within a sliding window and the transpositions among those matches. Identical strings give 1.0 and strings sharing no characters give 0.0. It weights matches near the start no more than matches near the end, which jaro_winkler changes.

## Syntax

```sql
jaro_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either argument is NULL.

## Examples

```sql
SELECT jaro_similarity('martha', 'marhta')
```

Approximately 0.944.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jaro_winkler](jaro-winkler.md)
- [levenshtein_similarity](levenshtein-similarity.md)
