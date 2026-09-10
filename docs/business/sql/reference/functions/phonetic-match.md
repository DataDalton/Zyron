# phonetic_match

Encodes both strings under the named algorithm and compares the codes. Under double metaphone a match counts when any of the four pairings of the two codes from each side agree, so it matches more pairs than the single-code algorithms.

## Syntax

```sql
phonetic_match(a, b, algorithm)
```

## Returns

BOOLEAN. NULL when either string is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `algorithm` | One of soundex, metaphone and double_metaphone. | Not applicable. |

## Examples

```sql
SELECT phonetic_match('Robert', 'Rupert', 'soundex')
```

true, because both encode to R163.

## Refused

- The algorithm is not one of the three names.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [phonetic_score](phonetic-score.md)
- [soundex](soundex.md)
- [metaphone](metaphone.md)
- [double_metaphone](double-metaphone.md)
