# jsonb_contains

True when every member and element of the second value appears in the first, at the same place and with the same value. Containment is structural rather than textual, so member order does not matter, and an object holding extra members still contains a smaller one.

## Syntax

```sql
jsonb_contains(a, b)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Examples

```sql
SELECT jsonb_contains('{"a":1,"b":2}', '{"a":1}')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jsonb_contained_by](jsonb-contained-by.md)
- [json_equals](json-equals.md)
- [jsonb_exists](jsonb-exists.md)
