# regex_match

Searches the whole string, so the pattern need not be anchored to match. The engine covers literals, the dot, star, plus, question mark, alternation, character classes and their negation, the digit, word and space classes, the start and end anchors, counted repetition and capture groups. Lookaround, backreferences, non-greedy repetition and named groups are not supported. Matching runs as an NFA simulation with a cost proportional to the text length times the pattern length, so no pattern can be made to run away.

## Syntax

```sql
regex_match(text, pattern)
```

## Returns

BOOLEAN. NULL when either argument is NULL or the pattern does not compile.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `pattern` | The expression to search for. | Not applicable. |

## Examples

```sql
SELECT regex_match('abc123', '[0-9]+')
```

true.

## Refused

- A backslash ends the pattern with nothing after it.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_find](regex-find.md)
- [regex_count](regex-count.md)
- [regex_compile](regex-compile.md)
- [regex_match_compiled](regex-match-compiled.md)
