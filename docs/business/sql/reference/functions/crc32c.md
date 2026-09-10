# crc32c

Computes CRC-32C, which uses a different polynomial from crc32 and detects some burst errors that crc32 misses. Most processors compute it in hardware, so it is faster than crc32 on long inputs. The two produce different values for the same input and are not interchangeable.

## Syntax

```sql
crc32c(bytes)
```

## Returns

INTEGER. NULL when the argument is NULL.

## Examples

```sql
SELECT crc32c('hello'::BYTEA)
```

An INTEGER checksum, different from crc32 of the same input.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [crc32](crc32.md)
- [xxhash64](xxhash64.md)
