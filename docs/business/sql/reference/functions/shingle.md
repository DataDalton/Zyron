# shingle

Slides a window of k characters one character at a time, producing one fewer than the character count plus k shingles. Text shorter than k, and a k of 0, give an empty array. Shingles are the usual input to minhash_signature, because they turn a string into a set whose overlap tracks how similar two strings are.

## Syntax

```sql
shingle(text, k)
```

## Returns

ARRAY of strings as JSON text. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `k` | Characters per shingle. Larger values are more specific and match fewer near-misses. | Not applicable. |

## Examples

```sql
SELECT shingle('abcd', 2)
```

["ab","bc","cd"].

## Refused

- The text argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [word_shingle](word-shingle.md)
- [minhash_signature](minhash-signature.md)
- [ngram_similarity](ngram-similarity.md)
