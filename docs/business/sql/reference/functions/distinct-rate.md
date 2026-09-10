# distinct_rate

Counts the distinct non-NULL values in the batch and compares the fraction of the non-NULL rows against the threshold, passing when it reaches it. NULLs are left out of both counts. An empty batch, and a batch whose column is entirely NULL, both pass. A threshold near 1.0 asserts that values are close to unique.

## Syntax

```sql
distinct_rate(column, threshold)
```

## Returns

BOOLEAN, the same value for every row of the batch.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `threshold` | Smallest distinct fraction that still passes, from 0.0 to 1.0. | Not applicable. |

## Examples

```sql
ALTER TABLE events ADD EXPECTATION ids_vary EXPECT DISTINCT_RATE(session_id, 0.9) ON VIOLATION WARN
```

An expectation that fails a batch repeating session ids heavily.

## Refused

- The threshold is not a number.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [null_rate](null-rate.md)
- [freshness](freshness.md)
