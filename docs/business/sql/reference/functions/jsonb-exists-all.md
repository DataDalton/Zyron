# jsonb_exists_all

True when every named member is present. An empty key list is satisfied by any object and answers true, the opposite of jsonb_exists_any.

## Syntax

```sql
jsonb_exists_all(json, keys)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Examples

```sql
SELECT jsonb_exists_all('{"a":1}', '["a","b"]')
```

false, because b is absent.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jsonb_exists_any](jsonb-exists-any.md)
- [jsonb_exists](jsonb-exists.md)
