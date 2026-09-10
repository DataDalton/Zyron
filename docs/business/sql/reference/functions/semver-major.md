# semver_major

Returns the first of the three numbers, the one that changes when a release breaks compatibility.

## Syntax

```sql
semver_major(version)
```

## Returns

INTEGER. NULL when the version is NULL.

## Examples

```sql
SELECT semver_major(semver_parse('1.2.3'))
```

1.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_minor](semver-minor.md)
- [semver_patch](semver-patch.md)
- [semver_parse](semver-parse.md)
