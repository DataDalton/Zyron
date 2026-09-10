# crc32

Computes the CRC-32 checksum used by zip, gzip and PNG. It detects accidental corruption such as a flipped bit or a truncated transfer. It is not a cryptographic hash: a value with a chosen checksum is cheap to construct, so it must not be used to verify that data was not tampered with.

## Syntax

```sql
crc32(bytes)
```

## Returns

INTEGER. NULL when the argument is NULL.

## Examples

```sql
SELECT crc32('hello'::BYTEA)
```

907060870.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [crc32c](crc32c.md)
- [adler32](adler32.md)
- [xxhash64](xxhash64.md)
