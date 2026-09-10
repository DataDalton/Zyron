# regex_find

Returns the start and end of the first match, counted in characters rather than bytes, so a multi-byte character counts as one. Text holding no match gives NULL.

## Syntax

```sql
regex_find(text, pattern)
```

## Returns

COMPOSITE holding the start and end positions, as JSON text. NULL when an argument is NULL, the pattern does not compile, or nothing matches.

## Examples

```sql
SELECT regex_find('abc123', '[0-9]+')
```

The start and end positions of 123.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_find_all](regex-find-all.md)
- [regex_capture](regex-capture.md)
- [regex_match](regex-match.md)
