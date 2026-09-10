# json_equals

Compares the parsed values, so member order and whitespace do not matter where a text comparison would report a difference. Two numbers written differently but equal in value compare as equal.

## Syntax

```sql
json_equals(a, b)
```

## Returns

BOOLEAN. NULL when either argument is NULL or is not JSON.

## Examples

```sql
SELECT json_equals('{"a":1,"b":2}', '{"b":2,"a":1}')
```

true, because member order does not matter.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_diff](json-diff.md)
- [json_diff_table](json-diff-table.md)
- [validate_json](validate-json.md)
