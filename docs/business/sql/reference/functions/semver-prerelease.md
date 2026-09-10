# semver_prerelease

Returns what follows the hyphen, so 1.0.0-rc.1 gives rc.1. A version carrying no pre-release gives NULL. The tag is read from the text, because the packed form semver_parse returns keeps only the fact that one was present.

## Syntax

```sql
semver_prerelease(version)
```

## Returns

TEXT. NULL when the version is NULL or carries no pre-release tag.

## Examples

```sql
SELECT semver_prerelease('1.0.0-rc.1')
```

rc.1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_is_prerelease](semver-is-prerelease.md)
- [semver_parse](semver-parse.md)
- [semver_sort](semver-sort.md)
