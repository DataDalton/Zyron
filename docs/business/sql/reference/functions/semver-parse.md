# semver_parse

Reads major.minor.patch with an optional leading v and an optional pre-release suffix, and packs the three numbers with a release flag into one 64-bit value that compares in version order. The pre-release tag itself is not kept, only the fact that one was present, so 1.0.0-rc.1 and 1.0.0-rc.2 pack identically. Each component must be at most 2097151.

## Syntax

```sql
semver_parse(text)
```

## Returns

SEMVER, a packed 64-bit value. NULL when the text is NULL or is not a version.

## Examples

```sql
SELECT semver_format(semver_parse('v1.2.3'))
```

1.2.3, with the leading v dropped.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_format](semver-format.md)
- [semver_satisfies](semver-satisfies.md)
- [semver_major](semver-major.md)
