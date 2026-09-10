# blake3

Returns the BLAKE3 of the input. It is a cryptographic digest like SHA-256 and considerably faster on long inputs, because its tree structure lets one input be hashed in parallel.

## Syntax

```sql
blake3(bytes)
```

## Returns

BYTEA of 32 bytes. NULL when the input is NULL.

## Examples

```sql
SELECT length(blake3('abc'))
```

32.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sha256](sha256.md)
- [xxhash64](xxhash64.md)
- [hmac_sha256](hmac-sha256.md)
