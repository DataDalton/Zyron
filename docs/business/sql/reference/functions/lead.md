# lead

Reads the value the given number of rows forward in the window's order. A row with no such successor gives the default, or NULL when none is given.

## Syntax

```sql
lead(value [, offset [, default]]) OVER (...)
```

## Returns

The type of the value argument. NULL at the end of a partition unless a default is given.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `offset` | How many rows forward to read. | 1. |

## Examples

```sql
SELECT lead(taken_at) OVER (ORDER BY taken_at) FROM zyron_test.scores
```

The next reading's time, NULL on the last row.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [lag](lag.md)
- [last_value](last-value.md)
