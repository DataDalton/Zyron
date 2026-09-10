# glicko2_update

Takes every result from one rating period at once and returns the updated rating, rating deviation and volatility. Unlike Elo this tracks how certain the rating is, so a player with a high deviation moves further on the same result, and the deviation falls as games accumulate. Opponents arrive as a JSON array of three-element arrays holding the opponent's rating, the opponent's deviation and the score.

## Syntax

```sql
glicko2_update(rating, rd, volatility, opponents)
```

## Returns

ARRAY of three numbers as JSON text. NULL when any argument is NULL or the opponent list does not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `rd` | Rating deviation, the uncertainty in the rating. A new player starts high. | Not applicable. |
| `volatility` | Expected swing in performance, around 0.06 for a settled player. | Not applicable. |
| `opponents` | JSON array of [rating, rd, score] triples, one per game in the period, with the score 1.0, 0.5 or 0.0. | Not applicable. |

## Examples

```sql
SELECT glicko2_update(1500, 200, 0.06, '[[1400,30,1.0]]')
```

[1563.5641943063383,175.402655938555,0.059998657304847616], a rating raised by the win and a deviation narrowed by the game.

## Refused

- A numeric argument is not a numeric column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [elo_update](elo-update.md)
- [trueskill_update](trueskill-update.md)
