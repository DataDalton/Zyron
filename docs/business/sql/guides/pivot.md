# PIVOT and UNPIVOT

`PIVOT` turns values of one column into columns of their own. `UNPIVOT` turns columns into rows. Both are written after a relation in `FROM`.

## What this means for you

- A cross-tabulation is one statement instead of one conditional aggregate typed per value.
- The value list is written out, so the result's columns are known before the query runs and a client can describe the result without executing it.
- `EXPLAIN` shows the plan each one becomes, so nothing about the cost is hidden by the shorthand.
- Stored SQL keeps the `PIVOT` or `UNPIVOT` spelling. What is written is what comes back.

## PIVOT

```sql
SELECT ... FROM rel
PIVOT (agg_fn(value_col) [AS name] [, ...]
       FOR pivot_col IN (literal [AS alias] [, ...]))
[AS alias]
```

Each listed value of the pivot column becomes an output column, holding the aggregate over the rows that carry that value.

```sql
SELECT * FROM sales
PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')) AS p;
```

```text
 region | Q1  | Q2  | Q3  | Q4
--------+-----+-----+-----+-----
 east   |  10 |  20 |  30 |  40
 north  |  20 |  40 |  60 |  80
 south  |  40 |  80 | 120 | 160
 west   |  30 |  60 |  90 | 120
```

Every column of the input that the pivot does not consume becomes a grouping key. `region` is there because `sales` has it and the pivot named neither it nor an aggregate over it. Adding a column to `sales` adds a grouping key, which is worth knowing before one is added.

An output column takes its alias, or the literal's own text when it has none:

```sql
PIVOT (SUM(amount) FOR quarter IN ('Q1' AS q1, 'Q2' AS q2))
```

Every aggregate the engine has is allowed, and more than one may be listed. Two aggregates over four values make eight output columns, each named for the value and the aggregate:

```sql
SELECT * FROM sales
PIVOT (SUM(amount) AS total, COUNT(amount) AS n
       FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')) AS p;
```

## The value list is static

A `PIVOT` whose `IN` list is a subquery is refused:

```text
a PIVOT value list must be static, because the output columns are fixed before
the query runs; run SELECT DISTINCT over the pivot column first, then write the
values it returned into the IN list
```

The output columns are part of the statement's shape. A client binds a result set to them, a view records them, and a prepared statement describes them before any row is read. A list that came from a subquery would make the shape depend on data, and the same statement would describe differently from one run to the next.

The two statements the refusal names are the way to do it:

```sql
SELECT DISTINCT quarter FROM sales ORDER BY quarter;
-- then, with what that returned
SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS p;
```

## What PIVOT runs as

One conditional aggregate per (aggregate, value) pair, grouped by every remaining column. `EXPLAIN` shows that plan rather than a pivot operator, so the cost is read the way any grouped aggregate's is:

```text
HashAggregate  group_by=1  aggregates=4
  SeqScan table_id=7
```

Written by hand the same query is the same plan, and runs in the same time.

## UNPIVOT

```sql
SELECT ... FROM rel
UNPIVOT [INCLUDE NULLS | EXCLUDE NULLS]
        (value_col FOR name_col IN (col [AS literal] [, ...]))
[AS alias]
```

Each listed column becomes a row carrying its name and its value.

```sql
SELECT * FROM budget
UNPIVOT (amount FOR month IN (jan, feb, mar, apr)) AS u;
```

```text
 dept | month | amount
------+-------+--------
 ops  | jan   |      1
 ops  | feb   |      2
 ops  | mar   |      3
 ops  | apr   |      4
```

The name column holds the source column's name, or the literal written after `AS`:

```sql
UNPIVOT (amount FOR month IN (jan AS 'January', feb AS 'February'))
```

The columns the `IN` list consumed do not travel to the result. `dept` is there because the list did not name it, and `jan` through `apr` are not, because it did.

`EXCLUDE NULLS` drops a group whose value is NULL, and is what an unwritten modifier means. `INCLUDE NULLS` keeps it.

## Unpivoting several columns together

More than one value column unpivots several columns per group. Every `IN` group is then a parenthesized tuple of the same arity:

```sql
SELECT * FROM budget
UNPIVOT ((amount, tax) FOR month IN ((jan_amt, jan_tax) AS 'Jan',
                                     (feb_amt, feb_tax) AS 'Feb')) AS u;
```

```text
 dept | month | amount | tax
------+-------+--------+-----
 ops  | Jan   |      1 |   2
 ops  | Feb   |      3 |   4
```

A group listing a different number of columns from the value list is refused, and the refusal states both counts.

Within a group, each position's columns share a type: the first group's columns set what each value column is, and a later group whose column disagrees is refused naming both types.

## What UNPIVOT runs as

One expansion of each input row into one row per group, with the columns that travel gathered once per output batch rather than copied per row. A twelve-column unpivot reads the table once and writes twelve rows per input row, which is what `EXPLAIN` shows:

```text
Unpivot  source=12 group(s)
  SeqScan table_id=7
```
