# validate_json

Parses the text and requires that nothing but whitespace follows the value, so two JSON values written back to back are rejected. It checks syntax alone, where json_schema_validate checks the shape against a schema.

## Syntax

```sql
validate_json(text)
```

## Returns

BOOLEAN. NULL when the text is NULL.

## Examples

```sql
SELECT validate_json('{"a":1}')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_schema_validate](json-schema-validate.md)
- [json_schema_errors](json-schema-errors.md)
