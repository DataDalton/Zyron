# adler32

Computes the Adler-32 checksum used by zlib. It costs less than crc32 and detects fewer errors, markedly so on inputs of a few bytes. Use crc32 or crc32c unless matching an existing zlib value.

## Syntax

```sql
adler32(bytes)
```

## Returns

INTEGER. NULL when the argument is NULL.

## Examples

```sql
SELECT adler32('hello'::BYTEA)
```

103547413.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [crc32](crc32.md)
- [crc32c](crc32c.md)
