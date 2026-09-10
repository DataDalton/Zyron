# regex_replace_all

Replaces each non-overlapping match and returns the whole string. Text holding no match is returned unchanged.

## Syntax

```sql
regex_replace_all(text, pattern, replacement)
```

## Returns

VARCHAR. NULL when an argument is NULL or the pattern does not compile.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `replacement` | Text to put in place of each match. | Not applicable. |

## Examples

```sql
SELECT regex_replace_all('a1b2', '[0-9]', 'x')
```

axbx.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_replace](regex-replace.md)
- [regex_split](regex-split.md)
