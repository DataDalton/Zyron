# validate_json_schema

Names the same function as json_schema_validate.

## Syntax

```sql
validate_json_schema(json, schema)
```

## Returns

BOOLEAN. NULL when either argument is NULL or is not JSON.

## Examples

```sql
SELECT validate_json_schema('{"a":1}', '{"type":"object"}')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_schema_validate](json-schema-validate.md)
- [json_schema_errors](json-schema-errors.md)
