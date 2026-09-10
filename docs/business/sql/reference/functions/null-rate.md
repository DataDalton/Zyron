# null_rate

Counts the NULLs in the batch being written and compares the fraction against the threshold, then answers the same verdict for every row, so the rows of one batch pass or fail together. An empty batch passes, because there is nothing to judge. Built for a table expectation rather than for reading a rate, which this does not return.

## Syntax

```sql
null_rate(column, threshold)
```

## Returns

BOOLEAN, the same value for every row of the batch.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `threshold` | Largest NULL fraction that still passes, from 0.0 to 1.0. | Not applicable. |

## Examples

```sql
ALTER TABLE people ADD EXPECTATION emails_present EXPECT NULL_RATE(email, 0.05) ON VIOLATION FAIL
```

An expectation that fails a batch where more than one row in 20 has no email.

## Refused

- The threshold is not a number.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [distinct_rate](distinct-rate.md)
- [freshness](freshness.md)
