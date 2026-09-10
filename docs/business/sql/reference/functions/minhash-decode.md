# minhash_decode

Reads each group of eight bytes as one little-endian slot value and returns them as a JSON array. A byte length that is not a multiple of eight gives NULL for that row rather than a partial signature.

## Syntax

```sql
minhash_decode(bytes)
```

## Returns

ARRAY of numbers as JSON text. NULL when the bytes are NULL or their length is not a multiple of eight.

## Examples

```sql
SELECT minhash_decode(minhash_encode('[7,8]'))
```

[7,8].

## Refused

- The argument is not a binary column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [minhash_encode](minhash-encode.md)
- [minhash_signature](minhash-signature.md)
