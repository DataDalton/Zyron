# fv

Compounds the present value and the stream of level payments forward to the end of the last period. Money paid in is negative and money received is positive, so a savings plan takes a negative payment and returns a positive future value.

## Syntax

```sql
fv(rate, nper, pmt, pv)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `pmt` | Amount paid each period, negative for money in. | Not applicable. |

## Examples

```sql
SELECT fv(0.01, 12, -100, 0)
```

Approximately 1268.25, the balance after twelve monthly deposits of 100.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [pv](pv.md)
- [pmt](pmt.md)
- [compound_interest](compound-interest.md)
