# bloom_create

Sizes the bit array at the ceiling of minus the item count times the natural log of the rate, divided by the square of the natural log of 2, with a floor of 64 bits. Hash count is that bit count over the item count times the natural log of 2, rounded and clamped to between 1 and 30. The rate is clamped to between 1e-10 and 0.5 before sizing, so a rate outside those bounds sizes as if it were the nearest bound.

## Syntax

```sql
bloom_create(expected_items, false_positive_rate)
```

## Returns

BLOOMFILTER. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `expected_items` | How many distinct values the filter is sized to hold. Adding more than this raises the false positive rate above the target. | Not applicable. |
| `false_positive_rate` | Target rate at the expected item count, as a fraction. | Not applicable. |

## Examples

```sql
SELECT bloom_create(1000, 0.01)
```

An empty filter of 9586 bits with 7 hash functions.

## Refused

- The expected item count is below one.
- Called with any count of arguments other than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bloom_add](bloom-add.md)
- [bloom_contains](bloom-contains.md)
- [bloom_merge](bloom-merge.md)
- [bloom_false_positive_rate](bloom-false-positive-rate.md)
