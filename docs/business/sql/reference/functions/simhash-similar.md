# simhash_similar

Compares simhash_distance against the threshold, inclusive at exactly the threshold. A threshold of 3 is the usual near-duplicate test for 64-bit fingerprints.

## Syntax

```sql
simhash_similar(a, b, threshold)
```

## Returns

BOOLEAN. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `threshold` | Largest differing bit count still counted as similar. | Not applicable. |

## Examples

```sql
SELECT simhash_similar(0, 7, 3)
```

true, because three bits differ and the threshold is 3.

## Refused

- An argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [simhash_distance](simhash-distance.md)
- [simhash](simhash.md)
