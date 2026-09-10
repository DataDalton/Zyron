# xirr

Solves for the rate that makes xnpv zero, giving an annual rate whatever the spacing of the flows. The flows must change sign at least once. A series that does not converge gives NULL.

## Syntax

```sql
xirr(dates, cashflows)
```

## Returns

DOUBLE PRECISION. NULL when either array is NULL, the two differ in length, or the search does not converge.

## Examples

```sql
SELECT xirr('[0,365]', '[-100,110]')
```

Approximately 0.1, a 10 percent annual return.

## Refused

- The flows never change sign.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [xnpv](xnpv.md)
- [irr](irr.md)
