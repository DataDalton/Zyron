# nested_json

Renders a STRUCT, MAP or array value as JSON text using the shape the column declares, which names the fields the stored bytes do not. The shape is read from the first row and applies to every row.

## Syntax

```sql
nested_json(value, shape)
```

## Returns

TEXT holding JSON. NULL when the value is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `shape` | The declared shape of the value, as the encoded form the schema holds. | Not applicable. |

## Examples

```sql
SELECT nested_json(address, shape) FROM zyron_test.people
```

Each address as a JSON object with its declared field names.

## Refused

- Called with any count of arguments other than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [struct_field](struct-field.md)
- [map_value](map-value.md)
- [json_get](json-get.md)
