# ltree_index

Returns the position of the first place the subpath appears as a consecutive run of labels, counting from 0, and -1 when it does not appear. The match is on whole labels, so top does not match topic.

## Syntax

```sql
ltree_index(path, subpath)
```

## Returns

BIGINT, -1 when absent. NULL when either path is NULL or invalid.

## Examples

```sql
SELECT ltree_index('top.middle.leaf', 'middle')
```

1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [subpath](subpath.md)
- [ltree_matches](ltree-matches.md)
- [nlevel](nlevel.md)
