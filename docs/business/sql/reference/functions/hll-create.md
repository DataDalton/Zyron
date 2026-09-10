# hll_create

Allocates a sketch with 2^precision one-byte registers and a two-byte header. Relative standard error is 1.04 divided by the square root of the register count, so precision 14 holds 16386 bytes and estimates to within about 0.8 percent. A NULL precision gives a NULL sketch.

## Syntax

```sql
hll_create(precision)
```

## Returns

HYPERLOGLOG. NULL when the precision is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `precision` | Register exponent, between 4 and 16. Each step doubles the size and cuts the error by about 30 percent. | Not applicable. |

## Examples

```sql
SELECT hll_create(14)
```

An empty sketch of 16386 bytes.

## Refused

- The precision is outside 4 to 16.
- Called with any count of arguments other than one.
- The precision argument is text, binary or an interval.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [hll_add](hll-add.md)
- [hll_count](hll-count.md)
- [hll_merge](hll-merge.md)
- [hll_error](hll-error.md)
