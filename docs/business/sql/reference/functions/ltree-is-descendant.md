# ltree_is_descendant

The same test as ltree_is_ancestor with the arguments the other way round, so the path being tested comes first. Two equal paths count here too.

## Syntax

```sql
ltree_is_descendant(descendant, ancestor)
```

## Returns

BOOLEAN. NULL when either path is NULL or invalid.

## Examples

```sql
SELECT ltree_is_descendant('top.a.x', 'top.a')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ltree_is_ancestor](ltree-is-ancestor.md)
- [ltree_matches](ltree-matches.md)
