# minhash_encode

Writes each slot as a little-endian 64-bit value, which is the form minhash_signature already returns. Use this to convert a signature held as a JSON array of numbers into the compact form, which is eight bytes per slot rather than the decimal digits.

## Syntax

```sql
minhash_encode(signature)
```

## Returns

BYTEA of 8 bytes per slot. NULL when the signature is NULL.

## Examples

```sql
SELECT minhash_decode(minhash_encode('[1,2,3]'))
```

[1,2,3].

## Refused

- The argument is not a binary or array column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [minhash_decode](minhash-decode.md)
- [minhash_signature](minhash-signature.md)
