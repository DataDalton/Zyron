# file_extension

Maps a media type to the extension a file of that type usually carries, so it pairs with detect_mime_type to name a payload. The argument is a media type, not a file name, and a type the table does not hold gives an empty string.

## Syntax

```sql
file_extension(mime_type)
```

## Returns

VARCHAR, empty for a type the table does not hold. NULL when the type is NULL.

## Examples

```sql
SELECT file_extension('text/csv')
```

csv.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [detect_mime_type](detect-mime-type.md)
- [is_binary](is-binary.md)
- [detect_encoding](detect-encoding.md)
