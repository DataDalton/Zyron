# phonetic_score

Scores 1.0 where the codes are equal and otherwise the normalised edit similarity between them, so a near miss scores above an unrelated pair where phonetic_match only answers false. Under double metaphone the best of the code pairings is taken.

## Syntax

```sql
phonetic_score(a, b, algorithm)
```

## Returns

REAL between 0.0 and 1.0. NULL when either string is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `algorithm` | One of soundex, metaphone and double_metaphone. | Not applicable. |

## Examples

```sql
SELECT phonetic_score('Robert', 'Rupert', 'soundex')
```

1, because the codes are equal.

## Refused

- The algorithm is not one of the three names.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [phonetic_match](phonetic-match.md)
- [levenshtein_similarity](levenshtein-similarity.md)
- [jaro_winkler](jaro-winkler.md)
