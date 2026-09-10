# regex_count

Counts the non-overlapping matches. A pattern able to match an empty string still advances, so the count is bounded by the text length rather than running forever.

## Syntax

```sql
regex_count(text, pattern)
```

## Returns

INTEGER. NULL when an argument is NULL or the pattern does not compile.

## Examples

```sql
SELECT regex_count('a1b2c3', '[0-9]')
```

3.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [regex_find_all](regex-find-all.md)
- [regex_match](regex-match.md)
