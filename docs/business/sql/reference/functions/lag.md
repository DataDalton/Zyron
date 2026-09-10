# lag

Reads the value the given number of rows back in the window's order. A row with no such predecessor gives the default, or NULL when none is given. The offset counts rows rather than ordering values, so a tie is stepped over one row at a time.

## Syntax

```sql
lag(value [, offset [, default]]) OVER (...)
```

## Returns

The type of the value argument. NULL at the start of a partition unless a default is given.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `offset` | How many rows back to read. | 1. |
| `default` | Value for a row with no predecessor at that offset. | NULL. |

## Examples

```sql
SELECT score - lag(score) OVER (ORDER BY taken_at) FROM zyron_test.scores
```

The change from the previous row, NULL on the first.

## Refused

- Called without an OVER clause.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [lead](lead.md)
- [first_value](first-value.md)
- [delta](delta.md)
