# json_patch

Applies the operations the patch names, in order, and returns the result. An operation whose path does not exist is an error rather than a skip, so a patch cannot be half applied unnoticed. json_merge_patch takes the simpler merge form instead.

## Syntax

```sql
json_patch(json, patch)
```

## Returns

The patched value as JSON. NULL when either argument is NULL or the patch does not apply.

## Examples

```sql
SELECT json_patch('{"a":1}', json_diff('{"a":1}', '{"a":2}'))
```

The value with a set to 2.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_diff](json-diff.md)
- [json_merge_patch](json-merge-patch.md)
- [text_patch](text-patch.md)
