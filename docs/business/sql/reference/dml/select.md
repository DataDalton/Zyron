# SELECT

Reads rows and shapes them. The clauses run in the order the result is built rather than the order they are written: FROM assembles the relations, WHERE drops rows before grouping, GROUP BY folds them, HAVING drops groups, QUALIFY drops rows by a window function's value, ORDER BY arranges what is left, and LIMIT and OFFSET take a slice of it. A SELECT with no FROM evaluates its items once and yields one row.

## Syntax

```sql
SELECT [DISTINCT [ON (expr, ...)]] items [INTO [TEMP] name] [FROM ...] [WHERE ...] [GROUP BY ...] [HAVING ...] [QUALIFY ...] [ORDER BY ...] [LIMIT n] [OFFSET n] [FOR UPDATE | FOR SHARE]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `DISTINCT` | Keeps one row per distinct combination of the selected items. | Every row is kept, duplicates included. |
| `DISTINCT ON (expr, ...)` | Keeps the first row of each group these expressions form, which ORDER BY decides. | Not applicable. |
| `WHERE predicate` | Drops rows the predicate does not hold for, before any grouping. | Every row from FROM is considered. |
| `GROUP BY expr, ...` | Folds rows agreeing on these expressions into one row each. | Rows are not folded, unless an aggregate with no GROUP BY folds them all into one. |
| `HAVING predicate` | Drops groups the predicate does not hold for, after grouping. | Not applicable. |
| `QUALIFY predicate` | Drops rows by the value of a window function, which WHERE cannot read because windows run after it. | Not applicable. |
| `ORDER BY expr [ASC | DESC] [NULLS FIRST | LAST]` | Arranges the result. Without it no order is promised. | The order is whatever the plan produced. |
| `LIMIT n` | Takes at most this many rows. | Every row is returned. |
| `OFFSET n` | Skips this many rows before taking any. | Nothing is skipped. |
| `FOR UPDATE | FOR SHARE` | Locks the rows the query read, so another transaction cannot change them until this one ends. | No rows are locked. |
| `INTO [TEMP] name` | Creates a table shaped by this query's output and fills it, instead of returning rows. | The rows are returned to the caller. |

## Examples

```sql
SELECT region, SUM(amount) FROM sales WHERE amount > 0 GROUP BY region ORDER BY region
```

One row per region, holding the total of its positive amounts.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [VALUES](values.md)
- [SELECT INTO](../session/select-into.md)
- [PIVOT](../queries/pivot.md)
- [UNNEST](../queries/unnest.md)
