# simhash_distance

Counts the bits set in the exclusive or of the two fingerprints, giving 0 for identical fingerprints and up to 64 for opposite ones. Near-duplicate documents usually land within a few bits, so a threshold in the single digits separates them from unrelated text.

## Syntax

```sql
simhash_distance(a, b)
```

## Returns

INTEGER between 0 and 64. NULL when either fingerprint is NULL.

## Examples

```sql
SELECT simhash_distance(0, 7)
```

3, because three bits differ.

## Refused

- An argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [simhash](simhash.md)
- [simhash_similar](simhash-similar.md)
- [hamming](hamming.md)
