# regex_split

Splits on each match and returns the parts between them, dropping the matched text itself. A match at the start or the end leaves an empty part there rather than dropping it.

## Syntax

```sql
regex_split(text, pattern)
```

## Returns

ARRAY of strings as JSON text. NULL when an argument is NULL or the pattern does not compile.

## Examples

```sql
SELECT regex_split('a1b2c', '[0-9]')
```

["a","b","c"].

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [string_to_array](string-to-array.md)
- [regex_find_all](regex-find-all.md)
