# damerau_levenshtein

Counts insertions, deletions, substitutions and transpositions of adjacent characters, each as one edit. Suits typed input, where swapped adjacent keys are a common error that levenshtein charges as two edits.

## Syntax

```sql
damerau_levenshtein(a, b)
```

## Returns

INTEGER, the number of edits. NULL when either argument is NULL.

## Examples

```sql
SELECT damerau_levenshtein('teh', 'the')
```

1, where levenshtein gives 2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [levenshtein](levenshtein.md)
