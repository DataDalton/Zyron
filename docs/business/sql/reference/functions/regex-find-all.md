# regex_find_all

Returns one start and end pair per match, in the order they appear. Matches do not overlap, so after a match the search resumes at its end. Text holding no match gives an empty array rather than NULL.

## Syntax

```sql
regex_find_all(text, pattern)
```

## Returns

ARRAY of position pairs as JSON text. NULL when an argument is NULL or the pattern does not compile.

## Examples

```sql
SELECT regex_count('a1b2c3', '[0-9]') = 3
```

true, and regex_find_all gives the three positions.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_find](regex-find.md)
- [regex_count](regex-count.md)
- [regex_capture](regex-capture.md)
