# first_value

Reads the frame's first row, which is the partition's first row when no frame is stated. Adding a frame changes what counts as first, so a frame of the preceding row alone makes this behave as lag does.

## Syntax

```sql
first_value(value) OVER (...)
```

## Returns

The type of the value argument. NULL when that row's value is NULL.

## Examples

```sql
SELECT first_value(score) OVER (PARTITION BY team ORDER BY taken_at) FROM zyron_test.scores
```

Each team's earliest score, repeated on every row of that team.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [last_value](last-value.md)
- [nth_value](nth-value.md)
- [lag](lag.md)
