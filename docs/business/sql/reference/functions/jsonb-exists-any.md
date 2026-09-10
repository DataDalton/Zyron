# jsonb_exists_any

True when at least one of the named members is present. An empty key list has nothing to find and answers false.

## Syntax

```sql
jsonb_exists_any(json, keys)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `keys` | JSON array of member names. | Not applicable. |

## Examples

```sql
SELECT jsonb_exists_any('{"a":1}', '["b","a"]')
```

true, because a is present.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jsonb_exists_all](jsonb-exists-all.md)
- [jsonb_exists](jsonb-exists.md)
