# bond_yield

Solves for the yield that makes bond_price return the given price, by bisection. A price far below face implies a high yield, and a yield beyond the search range gives NULL rather than an unbounded figure.

## Syntax

```sql
bond_yield(face, coupon_rate, price, periods)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL or the search fails.

## Examples

```sql
SELECT bond_yield(1000, 0.05, 1000, 10)
```

Approximately 0.05, matching the coupon rate at par.

## Refused

- The implied yield is beyond the search range.
- A parameter is zero or negative.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [bond_price](bond-price.md)
- [irr](irr.md)
