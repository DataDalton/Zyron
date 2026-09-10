# build_path

Joins the labels with dots, checking each one, so a label holding a dot or another character a label may not hold gives NULL rather than a path that will not parse back.

## Syntax

```sql
build_path(label [, ...])
```

## Returns

VARCHAR. NULL when a label is NULL or invalid.

## Examples

```sql
SELECT nlevel(build_path('top', 'middle', 'leaf'))
```

3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [nlevel](nlevel.md)
- [subpath](subpath.md)
- [lca](lca.md)
