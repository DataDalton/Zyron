# base58_encode

Encodes bytes using an alphabet that omits 0, O, I and l, so a transcribed value cannot be misread between them. It has no padding and no fixed expansion ratio, which makes it suited to identifiers a person copies and unsuited to streaming.

## Syntax

```sql
base58_encode(bytes)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT base58_encode('AB'::BYTEA)
```

5y3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [base58_decode](base58-decode.md)
- [base32_encode](base32-encode.md)
