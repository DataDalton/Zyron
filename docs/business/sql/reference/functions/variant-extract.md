# variant_extract

Walks the path into the stored value and returns what it finds. A VARIANT carries its own type per row, so the result type is decided by what the path reaches rather than by the column's declaration.

## Syntax

```sql
variant_extract(value, path)
```

## Returns

Whatever type the path reaches. NULL when the value is NULL or the path reaches nothing.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `path` | Path into the value. | Not applicable. |

## Examples

```sql
SELECT variant_extract(payload, 'user.id') FROM zyron_test.events
```

The id inside the user member of each payload.

## Refused

- Called with any count of arguments other than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_get_path](json-get-path.md)
- [struct_field](struct-field.md)
- [map_value](map-value.md)
