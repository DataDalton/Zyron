# semver_format

Writes major.minor.patch, adding the suffix -pre when the packed value records a pre-release. The original pre-release tag is not recoverable, so a version parsed as 1.0.0-rc.1 is written back as 1.0.0-pre.

## Syntax

```sql
semver_format(version)
```

## Returns

VARCHAR. NULL when the version is NULL.

## Examples

```sql
SELECT semver_format(semver_parse('2.0.0-rc.1'))
```

2.0.0-pre.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_parse](semver-parse.md)
- [semver_is_prerelease](semver-is-prerelease.md)
