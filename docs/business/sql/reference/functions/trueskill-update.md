# trueskill_update

Updates each team's skill estimate from the finishing order, returning one mean and deviation per team in the order given. Teams are processed in pairs by rank. A single team is returned unchanged, and two teams with the same rank are treated as a draw.

## Syntax

```sql
trueskill_update(team_ratings, ranks)
```

## Returns

ARRAY of [mu, sigma] pairs as JSON text. NULL when either argument is NULL, does not parse, or the two lengths differ.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `team_ratings` | JSON array of [mu, sigma] pairs, one per team. | Not applicable. |
| `ranks` | JSON array of finishing positions, 0 for first place. Must hold one entry per team. | Not applicable. |

## Examples

```sql
SELECT trueskill_update('[[25,8.333],[25,8.333]]', '[0,1]')
```

[[29.205269869118478,7.1945492403853],[20.794730130881522,7.1945492403853]], the winner's mean raised and both deviations narrowed.

## Refused

- An argument is not a binary, array or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [glicko2_update](glicko2-update.md)
- [elo_update](elo-update.md)
