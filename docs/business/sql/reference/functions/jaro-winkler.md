# jaro_winkler

Raises the Jaro similarity in proportion to the length of the common prefix, up to four characters. It suits personal and place names, where the opening characters are rarely mistyped and a difference there usually means a different name.

## Syntax

```sql
jaro_winkler(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either argument is NULL.

## Examples

```sql
SELECT jaro_winkler('martha', 'marhta')
```

Approximately 0.961, above the plain Jaro score for the same pair.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jaro_similarity](jaro-similarity.md)
- [levenshtein_similarity](levenshtein-similarity.md)
