# regex_match_compiled

Takes the value regex_compile returns rather than pattern text. The pattern is checked once where regex_compile is called, so a pattern that does not compile gives NULL there instead of once per row.

## Syntax

```sql
regex_match_compiled(text, compiled)
```

## Returns

BOOLEAN. NULL when either argument is NULL or the compiled value does not hold a pattern.

## Examples

```sql
SELECT regex_match_compiled('abc123', regex_compile('[0-9]+'))
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_compile](regex-compile.md)
- [regex_match](regex-match.md)
- [regex_find_compiled](regex-find-compiled.md)
