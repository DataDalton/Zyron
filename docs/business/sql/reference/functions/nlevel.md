# nlevel

Counts the dot-separated labels, so a root path holds 1. The path is validated first, and one holding an invalid label gives NULL.

## Syntax

```sql
nlevel(path)
```

## Returns

BIGINT. NULL when the path is NULL or invalid.

## Examples

```sql
SELECT nlevel('top.middle.leaf')
```

3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [subpath](subpath.md)
- [build_path](build-path.md)
- [ltree_index](ltree-index.md)
