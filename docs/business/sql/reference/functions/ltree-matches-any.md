# ltree_matches_any

True when at least one of the patterns matches, which tests a path against a set of subtrees in one call rather than joining several ltree_matches with OR.

## Syntax

```sql
ltree_matches_any(path, patterns)
```

## Returns

BOOLEAN. NULL when the path is NULL or invalid, or the patterns do not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `patterns` | JSON array of lquery patterns. | Not applicable. |

## Examples

```sql
SELECT ltree_matches_any('top.a.x', '["other.*","top.*"]')
```

true, because the second pattern matches.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ltree_matches](ltree-matches.md)
- [ltree_is_ancestor](ltree-is-ancestor.md)
