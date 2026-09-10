# json_schema_validate

Checks the value against the schema and answers only whether it passed. Use json_schema_errors where the reason matters, because this discards it.

## Syntax

```sql
json_schema_validate(json, schema)
```

## Returns

BOOLEAN. NULL when either argument is NULL or is not JSON.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `schema` | JSON Schema document as text. | Not applicable. |

## Examples

```sql
SELECT json_schema_validate('{"a":1}', '{"type":"object"}')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_schema_errors](json-schema-errors.md)
- [validate_json_schema](validate-json-schema.md)
- [validate_json](validate-json.md)
