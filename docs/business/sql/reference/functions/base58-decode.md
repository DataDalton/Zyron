# base58_decode

Reads base58 text back into bytes. A character outside the alphabet is an error, including the four the alphabet omits.

## Syntax

```sql
base58_decode(text)
```

## Returns

BYTEA. NULL when the argument is NULL.

## Examples

```sql
SELECT base58_decode('5y3')
```

The two bytes 0x41 and 0x42.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [base58_encode](base58-encode.md)
