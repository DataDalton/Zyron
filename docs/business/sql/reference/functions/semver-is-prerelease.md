# semver_is_prerelease

Reads the release flag the packing carries. A pre-release sorts below the release sharing its three numbers, so ordering by the packed value puts 1.0.0-rc before 1.0.0.

## Syntax

```sql
semver_is_prerelease(version)
```

## Returns

BOOLEAN. NULL when the version is NULL.

## Examples

```sql
SELECT semver_is_prerelease(semver_parse('1.0.0-rc.1'))
```

true.

## Refused

- The argument is not a packed version column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_parse](semver-parse.md)
- [semver_format](semver-format.md)
