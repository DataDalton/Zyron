# bayesian_average

Weights the item's own average by its vote count and the global average by the minimum vote count, then divides by the total weight. An item with few votes therefore sits near the global average rather than at an extreme, and the item's own average takes over as its vote count passes the minimum. Counts are clamped at 0 and truncated to whole numbers.

## Syntax

```sql
bayesian_average(item_avg, item_count, global_avg, min_votes)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `min_votes` | Vote count at which the item's own average and the global average carry equal weight. | Not applicable. |

## Examples

```sql
SELECT bayesian_average(5, 1, 3, 9)
```

3.2, close to the global average because one vote carries little weight.

## Refused

- An argument is not a numeric column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [wilson_score](wilson-score.md)
- [win_rate](win-rate.md)
