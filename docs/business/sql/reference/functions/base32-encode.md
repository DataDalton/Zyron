# base32_encode

Encodes five bytes into eight characters drawn from A to Z and 2 to 7. The alphabet excludes the digits that look like letters, so the result survives being read aloud or typed from a screen. It is about 60 percent longer than the input, against base64's 33 percent.

## Syntax

```sql
base32_encode(bytes)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT base32_encode('AB'::BYTEA)
```

IFBA====.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [base32_decode](base32-decode.md)
- [base64url_encode](base64url-encode.md)
- [base58_encode](base58-encode.md)
