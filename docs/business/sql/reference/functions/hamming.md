# hamming

Compares the two strings character by character and counts the mismatches. Both must hold the same number of characters, and a length mismatch gives NULL for that row rather than failing the statement. Comparison is by character rather than by byte, so a multi-byte character counts as one position. Use levenshtein where the strings may differ in length.

## Syntax

```sql
hamming(a, b)
```

## Returns

INTEGER. NULL when either string is NULL or the two differ in length.

## Examples

```sql
SELECT hamming('karolin', 'kathrin')
```

3.

## Refused

- An argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [levenshtein](levenshtein.md)
- [simhash_distance](simhash-distance.md)
- [qgram_distance](qgram-distance.md)
