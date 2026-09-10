# json_diff_table

Returns one entry per differing path, naming the path and the value on each side. A path present on one side only reports the absent side as null. Use json_diff for the patch form instead.

## Syntax

```sql
json_diff_table(a, b)
```

## Returns

ARRAY of path and value entries as JSON text. NULL when either argument is NULL or is not JSON.

## Examples

```sql
SELECT json_diff_table('{"a":1}', '{"a":2}')
```

One entry naming a, with 1 on the left and 2 on the right.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_diff](json-diff.md)
- [json_equals](json-equals.md)
- [row_diff](row-diff.md)
