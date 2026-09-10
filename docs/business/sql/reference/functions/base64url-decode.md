# base64url_decode

Reads URL-safe base64 text back into bytes. Padding is accepted and is not required. A plus or slash, which the URL-safe alphabet does not use, is an error.

## Syntax

```sql
base64url_decode(text)
```

## Returns

BYTEA. NULL when the argument is NULL.

## Examples

```sql
SELECT base64url_decode('QUI')
```

The two bytes 0x41 and 0x42.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [base64url_encode](base64url-encode.md)
