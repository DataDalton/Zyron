# truncate_chars

Keeps the first count characters and appends the suffix. Text already at or below the count is returned unchanged, with no suffix added. The count is in characters rather than bytes, so a multi-byte character counts as one, and the result can exceed the count by the length of the suffix.

## Syntax

```sql
truncate_chars(text, count, suffix)
```

## Returns

VARCHAR. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `count` | Characters to keep. | Not applicable. |
| `suffix` | Text appended when the input was cut. | Not applicable. |

## Examples

```sql
SELECT truncate_chars('abcdefgh', 3, '...')
```

abc...

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [truncate_words](truncate-words.md)
