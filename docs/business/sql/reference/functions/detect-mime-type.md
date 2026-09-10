# detect_mime_type

Reads the signature at the head of the payload and names the media type. The answer comes from the content rather than from a file name, so a mislabelled upload is identified correctly. A payload matching no known signature is reported as the generic byte stream type.

## Syntax

```sql
detect_mime_type(bytes)
```

## Returns

VARCHAR naming a media type. NULL when the payload is NULL.

## Examples

```sql
SELECT detect_mime_type(qr_encode('hello'))
```

image/png.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [detect_encoding](detect-encoding.md)
- [is_binary](is-binary.md)
- [file_extension](file-extension.md)
