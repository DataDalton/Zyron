# base64url_encode

Encodes three bytes into four characters, using hyphen and underscore in place of plus and slash. The result needs no escaping in a URL path, a query string or a filename. Plain base64 output is not interchangeable with it.

## Syntax

```sql
base64url_encode(bytes)
```

## Returns

TEXT, about a third longer than the input. NULL when the argument is NULL.

## Examples

```sql
SELECT base64url_encode('AB'::BYTEA)
```

QUI.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [base64url_decode](base64url-decode.md)
- [base32_encode](base32-encode.md)
- [hex_encode](hex-encode.md)
