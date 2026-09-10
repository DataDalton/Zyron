# version_compare

Returns -1, 0 or 1. The parts before any pre-release suffix compare numerically, and a version carrying a pre-release suffix sorts below the same version without one. It accepts any dotted number of components, where semver_parse requires exactly three.

## Syntax

```sql
version_compare(a, b)
```

## Returns

INTEGER, one of -1, 0 or 1. NULL when either string is NULL.

## Examples

```sql
SELECT version_compare('1.2.3', '1.10.0')
```

-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_parse](semver-parse.md)
- [semver_compare](semver-compare.md)
- [natural_compare](natural-compare.md)
