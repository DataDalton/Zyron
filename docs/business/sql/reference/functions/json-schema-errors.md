# json_schema_errors

Returns one message per rule the value breaks, each naming the path it was found at. A value that satisfies the schema gives an empty array, which is the same answer json_schema_validate reports as true.

## Syntax

```sql
json_schema_errors(json, schema)
```

## Returns

ARRAY of messages as JSON text. NULL when either argument is NULL or is not JSON.

## Examples

```sql
SELECT json_schema_errors('{"a":1}', '{"type":"array"}')
```

One message saying the value is not an array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_schema_validate](json-schema-validate.md)
- [validate_json](validate-json.md)
