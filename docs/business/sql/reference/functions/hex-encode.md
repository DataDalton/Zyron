# hex_encode

Writes each byte as two lower-case hexadecimal digits. The result is twice the input length and contains only 0 to 9 and a to f, so it is safe in any text context. hex_decode reverses it.

## Syntax

```sql
hex_encode(bytes)
```

## Returns

TEXT, twice the input length. NULL when the argument is NULL.

## Examples

```sql
SELECT hex_encode('AB'::BYTEA)
```

4142.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hex_decode](hex-decode.md)
- [base32_encode](base32-encode.md)
- [base64url_encode](base64url-encode.md)
