# struct_field

Reads the field at the position the column's declaration fixes, which is two loads rather than a search for a name inside the value. The planner writes this when a query names a struct field, so the position and the result type come from the schema rather than from the query.

## Syntax

```sql
struct_field(value, ordinal, result_type)
```

## Returns

The declared field type. NULL when the value is NULL or holds no such field.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ordinal` | Position of the field in the declaration, counting from 0. | Not applicable. |
| `result_type` | Type id the field carries, taken from the declaration. | Not applicable. |

## Examples

```sql
SELECT struct_field(address, 0, 25) FROM zyron_test.people
```

The first declared field of each address.

## Refused

- Called with any count of arguments other than three.
- The result type argument is not a type id.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [map_value](map-value.md)
- [variant_extract](variant-extract.md)
- [nested_json](nested-json.md)
