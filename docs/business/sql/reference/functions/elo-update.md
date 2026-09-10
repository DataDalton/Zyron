# elo_update

Adds the k factor times the gap between the actual and expected score to the rating. The expected score comes from elo_expected, and the actual score is 1.0 for a win, 0.5 for a draw and 0.0 for a loss. Ratings move by at most the k factor, so 32 suits new players, 24 established ones and 16 masters.

## Syntax

```sql
elo_update(rating, expected, actual, k_factor)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `actual` | 1.0 for a win, 0.5 for a draw, 0.0 for a loss. | Not applicable. |
| `k_factor` | Largest rating change one result can cause. | Not applicable. |

## Examples

```sql
SELECT elo_update(1500, 0.5, 1, 32)
```

1516, a win against an equal opponent at k factor 32.

## Refused

- An argument is not a numeric column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [elo_expected](elo-expected.md)
- [glicko2_update](glicko2-update.md)
