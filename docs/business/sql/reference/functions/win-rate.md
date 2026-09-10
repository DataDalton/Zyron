# win_rate

Divides wins by total, giving 0.0 when the total is 0. The figure carries no weight for sample size, so one win from one game scores 1.0. Use wilson_score to rank across unequal sample sizes.

## Syntax

```sql
win_rate(wins, total)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either argument is NULL.

## Examples

```sql
SELECT win_rate(3, 4)
```

0.75.

## Refused

- An argument is not a numeric column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [wilson_score](wilson-score.md)
- [bayesian_average](bayesian-average.md)
- [elo_expected](elo-expected.md)
