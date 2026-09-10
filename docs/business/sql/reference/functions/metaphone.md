# metaphone

Encodes English pronunciation with more of the consonant structure than soundex keeps, and without truncating to four characters. It distinguishes names soundex collapses together, at the cost of matching fewer spelling variants.

## Syntax

```sql
metaphone(text)
```

## Returns

TEXT, variable length. NULL when the argument is NULL.

## Examples

```sql
SELECT metaphone('Thompson')
```

TMSN.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [soundex](soundex.md)
- [nysiis](nysiis.md)
