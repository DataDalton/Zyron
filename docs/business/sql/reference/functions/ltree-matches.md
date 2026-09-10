# ltree_matches

Tests the path against an lquery pattern, which matches by label with wildcards for one label and for any number of them. A pattern is matched against the whole path, so a pattern naming fewer labels than the path holds does not match unless it ends in a wildcard.

## Syntax

```sql
ltree_matches(path, pattern)
```

## Returns

BOOLEAN. NULL when the path is NULL or invalid, or the pattern does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `pattern` | An lquery pattern. | Not applicable. |

## Examples

```sql
SELECT ltree_matches('top.a.x', 'top.*')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ltree_matches_any](ltree-matches-any.md)
- [ltree_is_ancestor](ltree-is-ancestor.md)
- [ltree_index](ltree-index.md)
