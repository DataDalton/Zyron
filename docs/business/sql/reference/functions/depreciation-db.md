# depreciation_db

Applies a fixed rate to the balance remaining at the start of the period, so the charge is largest in the first period and falls thereafter. The charge is capped so the book value never drops below the salvage value.

## Syntax

```sql
depreciation_db(cost, salvage, life, period)
```

## Returns

DOUBLE PRECISION, that period's charge. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `period` | Which period to charge, counting from 1. | Not applicable. |

## Examples

```sql
SELECT depreciation_db(10000, 1000, 5, 1) > depreciation_db(10000, 1000, 5, 5)
```

true, because the charge falls over the life.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [depreciation_sl](depreciation-sl.md)
- [depreciation_syd](depreciation-syd.md)
