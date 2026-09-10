# qgram_distance

Counts how often each run of q characters appears in each string and sums the absolute differences. Unlike ngram_similarity this keeps repeats, so a string stating a pattern twice is distinguished from one stating it once. The result grows with string length and is not a fraction, so compare it against a length-aware threshold rather than a fixed one.

## Syntax

```sql
qgram_distance(a, b, q)
```

## Returns

INTEGER. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `q` | Characters per q-gram. | Not applicable. |

## Examples

```sql
SELECT qgram_distance('abab', 'ab', 2)
```

2, because ab appears twice in one string and once in the other, and ba only in the first.

## Refused

- A string argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ngram_similarity](ngram-similarity.md)
- [levenshtein](levenshtein.md)
- [hamming](hamming.md)
