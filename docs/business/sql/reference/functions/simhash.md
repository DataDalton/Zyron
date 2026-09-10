# simhash

Splits the text on whitespace, hashes each word, and sets each of the 64 output bits according to whether that bit was set in more word hashes than it was clear. Two texts sharing most of their words differ in few bits, and simhash_distance counts those bits. Empty text gives 0. Word order does not affect the result, so two documents with the same words in a different order fingerprint identically.

## Syntax

```sql
simhash(text)
```

## Returns

BIGINT holding the 64-bit fingerprint. NULL when the text is NULL.

## Examples

```sql
SELECT simhash_distance(simhash('the quick brown fox'), simhash('the quick brown fox'))
```

0.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [simhash_distance](simhash-distance.md)
- [simhash_similar](simhash-similar.md)
- [minhash_signature](minhash-signature.md)
