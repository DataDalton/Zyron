# elo_expected

Evaluates the Elo logistic curve, where a 400-point lead gives about a 0.909 chance and equal ratings give 0.5. The two probabilities for a pair sum to 1, so the second player's expectation is one minus this result.

## Syntax

```sql
elo_expected(rating_a, rating_b)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when either rating is NULL.

## Examples

```sql
SELECT elo_expected(1500, 1500)
```

0.5.

## Refused

- An argument is not a numeric column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [elo_update](elo-update.md)
- [glicko2_update](glicko2-update.md)
- [trueskill_update](trueskill-update.md)
