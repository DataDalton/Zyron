# minhash_signature

Hashes every token under num_hashes independent permutations and keeps the smallest result from each, giving a fixed-size signature whatever the token count. Two signatures agree at a slot with probability equal to the Jaccard similarity of the two token sets, and minhash_similarity reads that agreement rate. The coefficients are derived from the slot number rather than drawn at random, so signatures built on different nodes and at different times compare directly.

## Syntax

```sql
minhash_signature(tokens, num_hashes)
```

## Returns

BYTEA holding one little-endian 64-bit value per slot. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `tokens` | A JSON array of strings, or text that is split on whitespace when it does not parse as one. | Not applicable. |
| `num_hashes` | Signature length, between 0 and 65536. More slots narrow the error in the estimate and widen the signature by 8 bytes each. | Not applicable. |

## Examples

```sql
SELECT minhash_similarity(minhash_signature('["a","b","c"]', 128), minhash_signature('["a","b","c"]', 128))
```

1, because the two token sets are the same.

## Refused

- The hash count is negative or above 65536.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [minhash_similarity](minhash-similarity.md)
- [minhash_encode](minhash-encode.md)
- [shingle](shingle.md)
- [jaccard_similarity](jaccard-similarity.md)
