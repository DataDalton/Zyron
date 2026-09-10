# regex_capture

Returns the whole match first, then one entry per capture group in the order the groups open. A group that took part in no match is null in the array, so the positions stay fixed whatever matched. Groups are numbered rather than named.

## Syntax

```sql
regex_capture(text, pattern)
```

## Returns

ARRAY as JSON text, the whole match then each group. NULL when an argument is NULL, the pattern does not compile, or nothing matches.

## Examples

```sql
SELECT regex_capture('2026-09-09', '([0-9]{4})-([0-9]{2})-([0-9]{2})')
```

The whole date followed by the year, month and day.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_find](regex-find.md)
- [regex_replace](regex-replace.md)
- [regex_match](regex-match.md)
