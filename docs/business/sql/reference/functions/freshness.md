# freshness

Takes the largest timestamp in the batch and compares its age against the limit, so one recent row carries the batch. An empty batch passes, and a batch whose column is entirely NULL fails, because nothing can be shown to be fresh.

## Syntax

```sql
freshness(timestamp_column, max_age)
```

## Returns

BOOLEAN, the same value for every row of the batch.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `max_age` | Oldest the newest row may be, as a duration string or as a whole number of seconds. | Not applicable. |

## Examples

```sql
ALTER TABLE readings ADD EXPECTATION recent EXPECT FRESHNESS(taken_at, '1h') ON VIOLATION WARN
```

An expectation that fails a batch whose newest reading is over an hour old.

## Refused

- The limit is neither a duration string nor a number.
- The column does not hold timestamps.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [null_rate](null-rate.md)
- [distinct_rate](distinct-rate.md)
