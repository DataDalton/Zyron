# subpath

Takes labels from the offset onward, counting from 0. A negative offset counts back from the end, and a negative length drops that many labels from the end instead of taking that many. An offset or length reaching outside the path gives NULL.

## Syntax

```sql
subpath(path, offset [, len])
```

## Returns

VARCHAR. NULL when the path is NULL, invalid, or the range falls outside it.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `offset` | First label to take, counting from 0, or from the end when negative. | Not applicable. |
| `len` | How many labels to take, or how many to drop from the end when negative. | Every label from the offset to the end. |

## Examples

```sql
SELECT subpath('top.middle.leaf', 1)
```

middle.leaf.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [nlevel](nlevel.md)
- [ltree_index](ltree-index.md)
- [build_path](build-path.md)
