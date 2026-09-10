# ngram_similarity

Cuts both strings into overlapping runs of n characters and takes the Jaccard similarity of the two resulting sets. Repeated n-grams collapse, so a string repeating a pattern scores the same as one stating it once. Two strings both shorter than n give 1.0, because both n-gram sets are empty.

## Syntax

```sql
ngram_similarity(a, b, n)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `n` | Characters per n-gram. 2 or 3 suits short strings such as names. | Not applicable. |

## Examples

```sql
SELECT ngram_similarity('night', 'nacht', 2)
```

0.14285714285714285, one shared pair out of seven.

## Refused

- A string argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [qgram_distance](qgram-distance.md)
- [jaccard_similarity](jaccard-similarity.md)
- [shingle](shingle.md)
