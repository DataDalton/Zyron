# depreciation_sl

Spreads the cost less the salvage value evenly over the life, so every period takes the same charge. The result is one period's charge rather than the total.

## Syntax

```sql
depreciation_sl(cost, salvage, life)
```

## Returns

DOUBLE PRECISION, one period's charge. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `salvage` | Value remaining at the end of the life. | Not applicable. |
| `life` | Number of periods the asset is written down over. | Not applicable. |

## Examples

```sql
SELECT depreciation_sl(10000, 1000, 9)
```

1000 per period.

## Refused

- An argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [depreciation_db](depreciation-db.md)
- [depreciation_syd](depreciation-syd.md)
