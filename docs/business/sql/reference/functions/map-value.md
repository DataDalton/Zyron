# map_value

Finds the entry whose key equals the given key and returns its value. A key the declared key type cannot hold matches no entry, giving the same NULL an absent key gives, so a type mismatch is not reported separately.

## Syntax

```sql
map_value(value, key, key_type, result_type)
```

## Returns

The declared value type. NULL when the value is NULL or the key is absent.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `key_type` | Type id the map's keys carry. | Not applicable. |
| `result_type` | Type id the map's values carry. | Not applicable. |

## Examples

```sql
SELECT map_value(labels, 'env', 25, 25) FROM zyron_test.services
```

The value stored under env in each label map.

## Refused

- Called with any count of arguments other than four.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [struct_field](struct-field.md)
- [variant_extract](variant-extract.md)
- [json_get](json-get.md)
