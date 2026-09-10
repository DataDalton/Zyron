# ltree_is_ancestor

True when the first path is a label prefix of the second. Two equal paths count, so a path is its own ancestor. Use it to select a subtree, because every path under a node has that node as an ancestor.

## Syntax

```sql
ltree_is_ancestor(ancestor, descendant)
```

## Returns

BOOLEAN. NULL when either path is NULL or invalid.

## Examples

```sql
SELECT ltree_is_ancestor('top.a', 'top.a.x')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ltree_is_descendant](ltree-is-descendant.md)
- [lca](lca.md)
- [ltree_matches](ltree-matches.md)
