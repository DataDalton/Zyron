# jsonb_contained_by

The same test jsonb_contains makes, with the arguments the other way round, so the value being tested comes first.

## Syntax

```sql
jsonb_contained_by(a, b)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Examples

```sql
SELECT jsonb_contained_by('{"a":1}', '{"a":1,"b":2}')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [jsonb_contains](jsonb-contains.md)
- [json_equals](json-equals.md)
