# semver_patch

Returns the third of the three numbers, the one that changes for a fix carrying no behaviour change.

## Syntax

```sql
semver_patch(version)
```

## Returns

INTEGER. NULL when the version is NULL.

## Examples

```sql
SELECT semver_patch(semver_parse('1.2.3'))
```

3.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_major](semver-major.md)
- [semver_minor](semver-minor.md)
