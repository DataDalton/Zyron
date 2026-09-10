# json_get

Reads a member by name from an object or an element by position from an array, and returns it still as JSON, so a string member comes back with its quotes. Use json_get_text to get the value without them.

## Syntax

```sql
json_get(json, key)
```

## Returns

The member as JSON. NULL when the value is NULL or holds no such member.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `key` | Member name for an object, or element position for an array. | Not applicable. |

## Examples

```sql
SELECT json_get('{"a":1}', 'a')
```

1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_get_text](json-get-text.md)
- [json_get_path](json-get-path.md)
- [variant_extract](variant-extract.md)
