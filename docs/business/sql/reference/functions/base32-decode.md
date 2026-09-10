# base32_decode

Reads base32 text back into bytes. A character outside the alphabet is an error. Padding is accepted and is not required.

## Syntax

```sql
base32_decode(text)
```

## Returns

BYTEA. NULL when the argument is NULL.

## Examples

```sql
SELECT base32_decode('IFBA====')
```

The two bytes 0x41 and 0x42.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [base32_encode](base32-encode.md)
