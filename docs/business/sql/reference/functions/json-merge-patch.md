# json_merge_patch

Copies the patch's members over the target's, recursing into nested objects, and removes a member the patch sets to null. Because null means removal, this form cannot set a member to null, which json_patch can.

## Syntax

```sql
json_merge_patch(target, patch)
```

## Returns

The merged value as JSON. NULL when either argument is NULL or is not JSON.

## Examples

```sql
SELECT json_merge_patch('{"a":1,"b":2}', '{"b":null}')
```

The object with b removed.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [json_patch](json-patch.md)
- [json_diff](json-diff.md)
