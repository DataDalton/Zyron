# last_value

Reads the frame's last row. The default frame ends at the current row, so without an explicit frame this returns the current row's own value rather than the partition's final one. State a frame ending at the partition's end to get the final value.

## Syntax

```sql
last_value(value) OVER (...)
```

## Returns

The type of the value argument. NULL when that row's value is NULL.

## Examples

```sql
SELECT last_value(score) OVER (PARTITION BY team ORDER BY taken_at ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM zyron_test.scores
```

Each team's latest score, repeated on every row of that team.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [first_value](first-value.md)
- [nth_value](nth-value.md)
- [lead](lead.md)
