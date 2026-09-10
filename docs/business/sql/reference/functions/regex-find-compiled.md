# regex_find_compiled

Behaves as regex_find does, taking the value regex_compile returns in place of pattern text.

## Syntax

```sql
regex_find_compiled(text, compiled)
```

## Returns

COMPOSITE holding the start and end positions, as JSON text. NULL when an argument is NULL or nothing matches.

## Examples

```sql
SELECT regex_find_compiled('abc123', regex_compile('[0-9]+'))
```

The start and end positions of 123.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_find](regex-find.md)
- [regex_compile](regex-compile.md)
