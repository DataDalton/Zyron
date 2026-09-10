# minhash_similarity

Counts the slots where the two signatures agree and divides by the slot count. The result estimates the Jaccard similarity of the original token sets without either set being available. Both signatures must hold the same number of slots. Two empty signatures give 0.0.

## Syntax

```sql
minhash_similarity(a, b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either signature is NULL.

## Examples

```sql
SELECT minhash_similarity(minhash_signature('a b c d', 64), minhash_signature('a b c e', 64))
```

0.53125 at 64 slots, estimating a true Jaccard similarity of 0.6.

## Refused

- The two signatures hold different numbers of slots.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [minhash_signature](minhash-signature.md)
- [jaccard_similarity](jaccard-similarity.md)
- [simhash_similar](simhash-similar.md)
