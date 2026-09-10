# semver_satisfies

Tests the packed version against a constraint string. A constraint holding several space-separated terms is met only when every term is met.

## Syntax

```sql
semver_satisfies(version, constraint)
```

## Returns

BOOLEAN. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `constraint` | A caret for the same major version at or above the stated minor and patch, a tilde for the same major and minor at or above the stated patch, >= <= > < for the obvious comparisons, an equals sign or a bare version for an exact match, and several terms separated by spaces for a range. | Not applicable. |

## Examples

```sql
SELECT semver_satisfies(semver_parse('1.4.2'), '^1.2.0')
```

true.

## Refused

- A version inside the constraint is not a version.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [semver_parse](semver-parse.md)
- [semver_major](semver-major.md)
