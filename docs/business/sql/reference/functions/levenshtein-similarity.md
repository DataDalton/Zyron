# levenshtein_similarity

Scales the edit distance by the length of the longer string, so the result does not grow with string length the way the raw distance does. Identical strings give 1.0 and strings sharing nothing give 0.0. Use this rather than levenshtein when comparing pairs of differing lengths against one threshold.

## Syntax

```sql
levenshtein_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either argument is NULL.

## Examples

```sql
SELECT levenshtein_similarity('kitten', 'sitting')
```

Approximately 0.571.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [levenshtein](levenshtein.md)
- [jaro_similarity](jaro-similarity.md)
