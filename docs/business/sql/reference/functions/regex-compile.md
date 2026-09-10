# regex_compile

Parses the pattern and returns it as a value the functions ending in _compiled take. A pattern that does not parse gives NULL, so a bad pattern is caught once at the point it is written rather than per row. The value carries the pattern text, so the expression is reparsed on each row that uses it.

## Syntax

```sql
regex_compile(pattern)
```

## Returns

BYTEA holding the pattern. NULL when the pattern is NULL or does not parse.

## Examples

```sql
SELECT regex_compile('[0-9') IS NULL
```

true, because the class is never closed.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_match_compiled](regex-match-compiled.md)
- [regex_find_compiled](regex-find-compiled.md)
- [regex_match](regex-match.md)
