# is_binary

Judges by the presence of bytes that do not occur in text, such as a null byte. Use it to decide whether a payload can be shown as text at all, before choosing an encoding.

## Syntax

```sql
is_binary(bytes)
```

## Returns

BOOLEAN. NULL when the payload is NULL.

## Examples

```sql
SELECT is_binary('plain text')
```

false.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [detect_mime_type](detect-mime-type.md)
- [detect_encoding](detect-encoding.md)
