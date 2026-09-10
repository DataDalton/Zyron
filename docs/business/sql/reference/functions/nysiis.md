# nysiis

Encodes a name by the New York State Identification and Intelligence System rules, which handle surname patterns soundex mishandles, including several of non-English origin. Use it over soundex when matching surnames across populations.

## Syntax

```sql
nysiis(text)
```

## Returns

TEXT, variable length. NULL when the argument is NULL.

## Examples

```sql
SELECT nysiis('Knuth')
```

NNAT.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [soundex](soundex.md)
- [metaphone](metaphone.md)
