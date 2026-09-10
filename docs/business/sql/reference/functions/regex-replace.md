# regex_replace

Replaces only the first match and returns the whole string. Text holding no match is returned unchanged rather than as NULL.

## Syntax

```sql
regex_replace(text, pattern, replacement)
```

## Returns

VARCHAR. NULL when an argument is NULL or the pattern does not compile.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `replacement` | Text to put in place of the match. | Not applicable. |

## Examples

```sql
SELECT regex_replace('a1b2', '[0-9]', 'x')
```

axb2, with only the first digit replaced.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_replace_all](regex-replace-all.md)
- [regex_capture](regex-capture.md)
