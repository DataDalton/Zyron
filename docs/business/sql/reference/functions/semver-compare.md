# semver_compare

Returns -1, 0 or 1. Both strings are parsed first, and a string that does not parse is treated as version 0 rather than failing, so two unparseable strings compare as equal. The comparison keeps only the three numbers and the pre-release flag, so two different pre-release tags on one version compare as equal. Use semver_sort where the tags must order among themselves.

## Syntax

```sql
semver_compare(a, b)
```

## Returns

INTEGER, one of -1, 0 or 1. NULL when either string is NULL.

## Examples

```sql
SELECT semver_compare('1.2.3', '1.10.0')
```

-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_sort](semver-sort.md)
- [semver_parse](semver-parse.md)
- [version_compare](version-compare.md)
