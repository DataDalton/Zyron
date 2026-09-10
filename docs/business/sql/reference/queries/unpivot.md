# UNPIVOT

Turns columns into rows. Each column named in the IN list becomes a row carrying that column's value and a label naming where it came from. The columns the groups consumed do not travel, and every other column of the input is carried through. Several value columns are written as parenthesized tuple groups, which must all have the same arity, so a pair of columns unpivots into a pair of value columns per row.

## Syntax

```sql
rel UNPIVOT [INCLUDE NULLS | EXCLUDE NULLS] (value_col FOR name_col IN (col [AS literal] [, ...])) [AS alias]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `INCLUDE NULLS` | Keeps a row whose value column is null. | A row whose value is null is dropped, which is EXCLUDE NULLS. |
| `EXCLUDE NULLS` | Drops a row whose value column is null. This is what happens without either word. | Not applicable. |
| `IN (col AS literal)` | Sets the label a column takes in the name column, rather than its own name. | The label is the column's name. |

## Examples

```sql
SELECT * FROM budget UNPIVOT (amount FOR month IN (jan, feb)) AS u
```

Two rows per input row, one for each month named, carrying the other columns.

## Refused

- A named column is not one the input produces.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [PIVOT](pivot.md)
