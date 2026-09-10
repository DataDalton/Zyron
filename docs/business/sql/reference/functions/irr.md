# irr

Solves for the rate that makes the net present value zero, by Newton-Raphson. The flows must include at least one negative and one positive amount, or no rate exists. A series crossing zero more than once has several valid answers and the one found depends on where the search starts. A series that does not converge gives NULL.

## Syntax

```sql
irr(cashflows)
```

## Returns

DOUBLE PRECISION. NULL when the array is NULL, holds fewer than two flows, holds no sign change, or the search does not converge.

## Examples

```sql
SELECT irr('[-100,60,60]')
```

Approximately 0.1306, the break-even rate.

## Refused

- The argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [npv](npv.md)
- [xirr](xirr.md)
