# nth_value

Reads the frame's nth row, counting from 1. A frame holding fewer than n rows gives NULL.

## Syntax

```sql
nth_value(value, n) OVER (...)
```

## Returns

The type of the value argument. NULL when the frame holds fewer than n rows.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `n` | Which row of the frame to read, counting from 1. | Not applicable. |

## Examples

```sql
SELECT nth_value(score, 2) OVER (PARTITION BY team ORDER BY taken_at) FROM zyron_test.scores
```

Each team's second score, NULL until two rows are in the frame.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [first_value](first-value.md)
- [last_value](last-value.md)
