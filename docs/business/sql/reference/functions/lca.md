# lca

Returns the longest path that is a proper prefix of all the arguments. Because the ancestor must be a proper prefix, two identical single-label paths share nothing and give NULL rather than themselves.

## Syntax

```sql
lca(path, path [, ...])
```

## Returns

VARCHAR. NULL when a path is NULL or invalid, or no ancestor is shared.

## Examples

```sql
SELECT lca('top.a.x', 'top.a.y')
```

top.a.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ltree_is_ancestor](ltree-is-ancestor.md)
- [build_path](build-path.md)
- [nlevel](nlevel.md)
