# levenshtein

Counts the single-character insertions, deletions and substitutions needed to turn one string into the other. A transposition counts as two edits, because it is a deletion and an insertion. Identical strings give 0. The cost is proportional to the product of the two lengths.

## Syntax

```sql
levenshtein(a, b)
```

## Returns

INTEGER, the number of edits. NULL when either argument is NULL.

## Examples

```sql
SELECT levenshtein('kitten', 'sitting')
```

3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [damerau_levenshtein](damerau-levenshtein.md)
- [levenshtein_similarity](levenshtein-similarity.md)
- [jaro_winkler](jaro-winkler.md)
