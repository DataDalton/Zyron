# json_get_path

Walks the path through nested objects and arrays in one call, where json_get reads a single level. The result stays JSON, so a string comes back quoted.

## Syntax

```sql
json_get_path(json, path)
```

## Returns

The value at the path, as JSON. NULL when the value is NULL or the path reaches nothing.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `path` | Path naming each step, member names and element positions. | Not applicable. |

## Examples

```sql
SELECT json_get_path('{"a":{"b":1}}', 'a', 'b')
```

1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_get_path_text](json-get-path-text.md)
- [json_get](json-get.md)
- [variant_extract](variant-extract.md)
