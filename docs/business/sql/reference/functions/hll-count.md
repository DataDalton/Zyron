# hll_count

Takes the bias-corrected harmonic mean of the registers, rounded to a whole number. When that estimate falls at or below 2.5 times the register count and at least one register is still zero, linear counting over the empty registers replaces it, which is the accurate form at low cardinality. Sketch bytes that do not parse give 0.

## Syntax

```sql
hll_count(sketch)
```

## Returns

BIGINT. NULL when the sketch is NULL, 0 when its bytes do not parse.

## Examples

```sql
SELECT hll_count(hll_add(hll_add(hll_create(12), 'a'), 'b'))
```

2.

## Refused

- The sketch argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hll_create](hll-create.md)
- [hll_add](hll-add.md)
- [hll_error](hll-error.md)
- [bloom_contains](bloom-contains.md)
