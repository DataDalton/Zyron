# semver_sort

Sorts by the specification's own rules, including the ordering among pre-release identifiers, so 1.0.0-alpha comes before 1.0.0-alpha.1, then 1.0.0-beta, then 1.0.0. This is finer than sorting on the packed form, which treats every pre-release of one version as equal. Any version in the array that does not parse fails the call and names the offending value.

## Syntax

```sql
semver_sort(versions)
```

## Returns

ARRAY of version strings as JSON text. NULL when the array is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `versions` | JSON array of version strings. | Not applicable. |

## Examples

```sql
SELECT semver_sort('["1.0.0","1.0.0-beta","1.0.0-alpha"]')
```

["1.0.0-alpha","1.0.0-beta","1.0.0"].

## Refused

- The argument is not an array.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_compare](semver-compare.md)
- [semver_prerelease](semver-prerelease.md)
- [version_compare](version-compare.md)
