# validate_uuid

Requires exactly 36 characters with hyphens at the four standard positions and hex digits elsewhere. The compact 32-character form without hyphens is rejected, as is a form wrapped in braces.

## Syntax

```sql
validate_uuid(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_uuid('00000000-0000-0000-0000-000000000000')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [uuid_to_string](uuid-to-string.md)
- [gen_uuid_v7](gen-uuid-v7.md)
