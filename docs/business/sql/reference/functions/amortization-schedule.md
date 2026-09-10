# amortization_schedule

Returns one row per period, each holding the payment, the part of it going to principal, the part going to interest, and the balance left after it. The payment is level, so the interest share falls and the principal share rises as the balance drops.

## Syntax

```sql
amortization_schedule(principal, rate, periods)
```

## Returns

ARRAY of four-number entries as JSON text. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `periods` | Number of payments in the schedule. | Not applicable. |

## Examples

```sql
SELECT amortization_schedule(1000, 0.01, 12)
```

Twelve entries, the last leaving a balance of effectively zero.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [pmt](pmt.md)
- [pv](pv.md)
