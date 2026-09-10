# ASOF JOIN

Joins each left row to the nearest right row along an ordered column, within the equality group the ON clause names. One left row matches at most one right row. Both inputs are read in order of their equality keys then their match column. A lake table clustered on the match column already provides that order and its sort is elided. Any other input is sorted. EXPLAIN reports the match direction and whether each side's sort was elided.

## Syntax

```sql
left ASOF [LEFT] JOIN right MATCH_CONDITION (left.ts >= right.ts) [ON left.key = right.key [AND ...]]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `LEFT` | Keeps a left row that found no match, with the right columns null. | A left row with no match is dropped. |
| `ON left.key = right.key` | Names the equality group a match must fall inside. Only equalities are read here. | Every left row may match any right row. |

## Examples

```sql
SELECT * FROM trades ASOF LEFT JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol
```

Each trade with the latest quote for its symbol at or before it, and nulls where none exists.

## Refused

- A RIGHT or FULL form is written.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [MATCH_CONDITION](match-condition.md)
