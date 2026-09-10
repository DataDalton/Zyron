# double_metaphone

Returns a primary code and an alternate one, where the alternate differs for spellings that are pronounced two ways in English, and equals the primary otherwise. Two names match when any of their four code pairings agree, which catches variants a single code misses. Non-letter characters are dropped and the comparison is case-insensitive.

## Syntax

```sql
double_metaphone(text)
```

## Returns

ARRAY of two strings as JSON text, primary then alternate. NULL when the text is NULL.

## Examples

```sql
SELECT double_metaphone('Smith')
```

["SMT","SM0"], where the alternate spells the th sound differently.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [soundex](soundex.md)
- [metaphone](metaphone.md)
- [nysiis](nysiis.md)
