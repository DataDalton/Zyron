# wilson_score

Returns the lowest proportion consistent with the observed counts at the given confidence, which ranks items by positive share while penalising small samples. One positive vote out of one scores far below 50 positive out of 50, where a plain ratio would tie them. A total of 0 gives 0.0.

## Syntax

```sql
wilson_score(positive, total, confidence)
```

## Returns

DOUBLE PRECISION between 0.0 and 1.0. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `confidence` | Confidence level as a fraction above 0 and below 1. 0.95 is the usual choice. | Not applicable. |

## Examples

```sql
SELECT wilson_score(1, 1, 0.95)
```

0.20654931411298352, well below the observed ratio of 1.

## Refused

- The confidence is not above 0 and below 1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [win_rate](win-rate.md)
- [bayesian_average](bayesian-average.md)
