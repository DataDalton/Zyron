# json_diff

Returns the changes as a patch json_patch can apply, where json_diff_table returns them as rows for reading. A member removed on the right side appears in the patch as a removal rather than being left out.

## Syntax

```sql
json_diff(a, b)
```

## Returns

The patch as JSON. NULL when either argument is NULL or is not JSON.

## Examples

```sql
SELECT json_equals(json_patch('{"a":1}', json_diff('{"a":1}', '{"a":2}')), '{"a":2}')
```

true, because applying the patch reaches the second value.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_patch](json-patch.md)
- [json_diff_table](json-diff-table.md)
- [json_merge_patch](json-merge-patch.md)
