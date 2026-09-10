# MATCH_CONDITION

Says which way an ASOF JOIN reaches and how far. The inequality holds one column of each side, both of the same orderable type: any numeric type, DATE, TIMESTAMP, TIMESTAMPTZ or INTERVAL. Writing >= or > reaches backwards and takes the greatest right value not above the left one, strictly below it for >. Writing <= or < reaches forwards and takes the least right value not below the left one, strictly above it for <. A second conjunct bounds the reach, so a match further away than the bound leaves the left row unmatched.

## Syntax

```sql
MATCH_CONDITION (left.col >= right.col [AND left.col - right.col <= INTERVAL '5 minutes'])
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `AND left.col - right.col <= INTERVAL '5 minutes'` | Bounds how far a match may reach, leaving a left row unmatched when the nearest right row is further away than this. | The reach is unbounded. |

## Examples

```sql
SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts)
```

Each trade with the latest quote at or before it, with no equality group.

## Refused

- The two columns are not the same orderable type.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ASOF JOIN](asof-join.md)
