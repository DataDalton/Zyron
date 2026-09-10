# jsonb_exists

Tests for the member's presence without reading it, so a member present and holding null answers true where json_get would answer NULL.

## Syntax

```sql
jsonb_exists(json, key)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Examples

```sql
SELECT jsonb_exists('{"a":null}', 'a')
```

true, because the member is present even though its value is null.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jsonb_exists_any](jsonb-exists-any.md)
- [jsonb_exists_all](jsonb-exists-all.md)
- [json_get](json-get.md)
