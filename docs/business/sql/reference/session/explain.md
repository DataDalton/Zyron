# EXPLAIN

Shows the plan chosen for a statement: which operators run, in what order, and what the planner expected each to cost and produce. ANALYZE runs the statement as well and reports what actually happened, which is how an estimate is compared against the truth.

## Syntax

```sql
EXPLAIN [ANALYZE] [VERBOSE] statement
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ANALYZE` | Runs the statement and reports the real row counts and timings beside the estimates. | The statement is planned but not run. |
| `VERBOSE` | Shows the columns and expressions each operator carries. | Only the operators and their costs are shown. |

## Examples

```sql
EXPLAIN SELECT id FROM orders WHERE total > 0
```

The operator tree the query would run as, with estimated costs.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [EXPLAIN REWRITE](explain-rewrite.md)
- [ANALYZE](analyze.md)
