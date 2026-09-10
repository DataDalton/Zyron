# row_diff

Reads both arguments as JSON objects whose members are columns, and names the members whose values differ, with the value on each side. A member present on one side only is reported as a change. row_diff_ordinal matches by position instead.

## Syntax

```sql
row_diff(old, new)
```

## Returns

ARRAY of change entries as JSON text. NULL when either argument is NULL.

## Examples

```sql
SELECT row_diff('{"a":1,"b":2}', '{"a":1,"b":3}')
```

One entry naming b, with 2 before and 3 after.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [row_diff_ordinal](row-diff-ordinal.md)
- [json_diff_table](json-diff-table.md)
- [json_equals](json-equals.md)
