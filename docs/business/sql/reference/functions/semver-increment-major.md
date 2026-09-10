# semver_increment_major

Adds one to the major component and resets the minor and patch to 0, which is the version a compatibility-breaking release takes. The result is a release rather than a pre-release.

## Syntax

```sql
semver_increment_major(version)
```

## Returns

SEMVER. NULL when the version is NULL.

## Examples

```sql
SELECT semver_format(semver_increment_major(semver_parse('1.2.3')))
```

2.0.0.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_increment_minor](semver-increment-minor.md)
- [semver_increment_patch](semver-increment-patch.md)
