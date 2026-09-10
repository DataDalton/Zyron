# ASOF JOIN

An `ASOF JOIN` matches each left row to the nearest right row along an ordered column, rather than to every right row that satisfies a predicate. It is the join a time series asks for: the quote in force when a trade happened, the price in force when an order was placed, the reading nearest a sample.

## What this means for you

- One statement instead of a correlated subquery per row.
- One pass over each input, so the cost is the two sorts and a merge, not a lookup per left row.
- Memory is one batch per side plus the candidate row being held, whatever the inputs' size.
- A side that already arrives in the right order is not sorted again, and `EXPLAIN` says which side that was.

## Writing one

```sql
left ASOF [LEFT] JOIN right
    MATCH_CONDITION (left.ts >= right.ts)
    [ON left.key = right.key [AND ...]]
```

```sql
SELECT t.ts, t.price, q.bid
FROM trades AS t
ASOF JOIN quotes AS q
    MATCH_CONDITION (t.ts >= q.ts)
    ON t.symbol = q.symbol;
```

Each trade takes the latest quote on its own symbol at or before its own timestamp.

`ON` carries equalities only. They divide both inputs into groups that are matched independently, the way a partition does, and a trade is never matched against another symbol's quote. `ON` is optional; without it the whole input is one group. An inequality written into `ON` is refused, naming `MATCH_CONDITION` as where it belongs.

## The match condition

`MATCH_CONDITION` holds one inequality between one column of the left relation and one column of the right. Both are the same orderable type: any numeric type, `DATE`, `TIMESTAMP`, `TIMESTAMPTZ` or `INTERVAL`.

The operator says which way the match reaches.

| Written | Which right row matches |
| --- | --- |
| `left.ts >= right.ts` | The greatest right value at or below the left value |
| `left.ts > right.ts` | The greatest right value strictly below it |
| `left.ts <= right.ts` | The least right value at or above the left value |
| `left.ts < right.ts` | The least right value strictly above it |

Given a left value of 100 and right values 90, 100 and 110, `>=` matches 100, `>` matches 90, `<=` matches 100 and `<` matches 110.

Writing the right relation's column first is the same condition read the other way: `MATCH_CONDITION (q.ts <= t.ts)` and `MATCH_CONDITION (t.ts >= q.ts)` mean the same match.

A condition comparing two columns of one relation is refused, naming that it spans both.

## Unmatched left rows

`ASOF JOIN` drops a left row that found no right row. `ASOF LEFT JOIN` keeps it, with the right columns NULL.

```sql
SELECT t.ts, q.bid
FROM trades AS t
ASOF LEFT JOIN quotes AS q MATCH_CONDITION (t.ts >= q.ts) ON t.symbol = q.symbol;
```

A trade before the first quote on its symbol has no earlier quote to take. Under the inner form it does not appear; under the LEFT form it appears with a NULL bid.

There is no `RIGHT` or `FULL` form. `ASOF RIGHT JOIN` does not parse. The join asks a question about each left row, and there is no reading of it that runs the other way as well.

## Tolerance

A second conjunct bounds how far back a match may reach:

```sql
MATCH_CONDITION (t.ts >= q.ts AND t.ts - q.ts <= INTERVAL '5 minutes')
```

A left row whose nearest right row lies further away than the bound is unmatched: dropped under the inner form, NULL-extended under the LEFT form. It is the difference between the two match columns that is bounded, written as their subtraction in either order, with `<=` or `<`. `<=` matches a distance exactly equal to the bound and `<` does not.

Without a tolerance the reach is unbounded, and a trade takes the last quote on its symbol however old it is.

## What it runs as

Both inputs are read in `(equality keys, match column)` order. Inside each equality group the merge steps through both once, holding the nearest right row it has passed; that held row is the answer for every left row until the right side passes it.

Nothing is materialized. The operator holds one batch per side, the held row, and the output batch it is filling, whatever the inputs' size.

A sort is planted over an input only when it does not already hold that order. Three shapes already do:

- A lake table laid out on those columns.
- An index scan walking those columns forward.
- A sort that already ran, from an `ORDER BY` or an earlier operator.

`EXPLAIN` names the join, which way the match reaches, whether the reach is bounded, and what happened to each side's sort:

```text
AsofJoin  join_type=Inner  match=left >= right, nearest at or before
          tolerance=unbounded  left_sort=sorted  right_sort=sorted
  SeqScan table_id=7
  SeqScan table_id=8
```

Over a lake table clustered on the match column, the same query reports the sort elided:

```text
AsofJoin  join_type=Inner  match=left >= right, nearest at or before
          tolerance=unbounded  left_sort=elided, input already ordered
          right_sort=sorted
  LakeScan table_id=12
  SeqScan table_id=8
```

Clustering the right side on `(symbol, ts)` removes the other sort too, which is where a large ASOF join spends most of its time.

## Cost

The planner estimates the result at the left cardinality for `ASOF LEFT JOIN`, since every left row survives, and at the left cardinality times the match probability for the inner form, taken from how far the two match columns' recorded ranges overlap. `ANALYZE` on both tables is what makes that estimate a measurement rather than an assumption.
