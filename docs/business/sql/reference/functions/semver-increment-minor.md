# semver_increment_minor

Adds one to the minor component, resets the patch to 0 and leaves the major alone, which is the version a compatible feature release takes.

## Syntax

```sql
semver_increment_minor(version)
```

## Returns

SEMVER. NULL when the version is NULL.

## Examples

```sql
SELECT semver_format(semver_increment_minor(semver_parse('1.2.3')))
```

1.3.0.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_increment_major](semver-increment-major.md)
- [semver_increment_patch](semver-increment-patch.md)
