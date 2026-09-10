# hll_error

Returns 1.04 divided by the square root of the register count. The value follows from the precision alone and does not move as values are added, so it states the sketch's accuracy bound rather than the error of any one estimate.

## Syntax

```sql
hll_error(sketch)
```

## Returns

DOUBLE PRECISION. NULL when the sketch is NULL or its bytes do not parse.

## Examples

```sql
SELECT hll_error(hll_create(14))
```

Approximately 0.00813.

## Refused

- The sketch argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hll_create](hll-create.md)
- [hll_count](hll-count.md)
- [bloom_false_positive_rate](bloom-false-positive-rate.md)
