# path_compare

Splits both paths on forward and back slashes and compares the components in turn with natural ordering, so a shorter path sorts before a longer one that starts with it. Comparing component by component keeps a separator from outranking a character, which a plain string comparison does not.

## Syntax

```sql
path_compare(a, b)
```

## Returns

INTEGER, one of -1, 0 or 1. NULL when either path is NULL.

## Examples

```sql
SELECT path_compare('/a/file2', '/a/file10')
```

-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [natural_compare](natural-compare.md)
- [build_path](build-path.md)
