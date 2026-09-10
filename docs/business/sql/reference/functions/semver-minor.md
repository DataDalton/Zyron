# semver_minor

Returns the second of the three numbers, the one that changes when a release adds compatible behaviour.

## Syntax

```sql
semver_minor(version)
```

## Returns

INTEGER. NULL when the version is NULL.

## Examples

```sql
SELECT semver_minor(semver_parse('1.2.3'))
```

2.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_major](semver-major.md)
- [semver_patch](semver-patch.md)
