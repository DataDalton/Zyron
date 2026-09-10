# semver_increment_patch

Adds one to the patch component and leaves the major and minor alone, which is the version a fix release takes.

## Syntax

```sql
semver_increment_patch(version)
```

## Returns

SEMVER. NULL when the version is NULL.

## Examples

```sql
SELECT semver_format(semver_increment_patch(semver_parse('1.2.3')))
```

1.2.4.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_increment_minor](semver-increment-minor.md)
- [semver_increment_major](semver-increment-major.md)
