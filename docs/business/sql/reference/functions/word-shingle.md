# word_shingle

Splits on whitespace and slides a window of k words, joining each window with single spaces. Runs of whitespace collapse, so the output is normalised rather than matching the input's spacing. Text holding fewer than k words, and a k of 0, give an empty array.

## Syntax

```sql
word_shingle(text, k)
```

## Returns

ARRAY of strings as JSON text. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `k` | Words per shingle. | Not applicable. |

## Examples

```sql
SELECT word_shingle('the quick brown fox', 2)
```

["the quick","quick brown","brown fox"].

## Refused

- The text argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [shingle](shingle.md)
- [minhash_signature](minhash-signature.md)
