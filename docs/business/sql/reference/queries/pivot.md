# PIVOT

Produces one output column per literal in the IN list, holding the aggregate over rows whose pivot column equals that literal. Input columns the pivot does not consume become grouping keys, producing one row per remaining combination. Several aggregates produce one column per aggregate per literal, named by their aliases. The statement is rewritten to a grouped aggregate with one conditional aggregate per output column, and EXPLAIN shows the rewritten plan.

## Syntax

```sql
rel PIVOT (agg_fn(value_col) [AS name] [, ...] FOR pivot_col IN (literal [AS alias] [, ...])) [AS alias]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `AS name` | Names an aggregate, which becomes part of each column name it produces. | A single aggregate's columns are named by their value alone. |
| `IN (literal AS alias)` | Names the column a value produces, rather than taking the literal's own text. | The column is named for the literal. |
| `AS alias` | Names the pivoted relation. | The relation is named pivot. |

## Examples

```sql
SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS p
```

One row per remaining column combination, with a Q1 and a Q2 column.

## Refused

- The IN list is a subquery rather than written-out values.
- The IN list is empty.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [UNPIVOT](unpivot.md)
