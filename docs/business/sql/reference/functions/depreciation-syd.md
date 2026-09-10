# depreciation_syd

Weights each period by the years remaining over the sum of the year numbers, so the charge falls by a fixed step each period rather than by a fixed proportion as declining balance does. The charges over the full life add up to the cost less the salvage value exactly.

## Syntax

```sql
depreciation_syd(cost, salvage, life, period)
```

## Returns

DOUBLE PRECISION, that period's charge. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `period` | Which period to charge, counting from 1. | Not applicable. |

## Examples

```sql
SELECT depreciation_syd(10000, 1000, 5, 1)
```

3000, five fifteenths of the 9000 to be written down.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [depreciation_sl](depreciation-sl.md)
- [depreciation_db](depreciation-db.md)
