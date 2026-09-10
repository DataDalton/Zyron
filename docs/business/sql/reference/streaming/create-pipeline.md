# CREATE PIPELINE

Names an ordered set of stages that run as one unit. A stage names its source, its target, and the query between them. The mode determines whether the target is rebuilt or added to. A stage does not begin before the stage producing what it reads has finished.

## Syntax

```sql
CREATE PIPELINE name AS (STAGE name (SOURCE src, TARGET dst, MODE full | incremental, TRANSFORM AS (query)), ...)
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `MODE full` | Rebuilds the target from the source each run. | Not applicable. |
| `MODE incremental` | Adds to the target rather than rebuilding it. | Not applicable. |
| `TRANSFORM AS (query)` | The query that turns the source into what the target holds. | The source's rows are written to the target unchanged. |

## Examples

```sql
CREATE PIPELINE etl AS (STAGE s (SOURCE raw, TARGET staging, MODE full, TRANSFORM AS (SELECT id FROM raw)))
```

A named unit whose stages run in order, each reading a source and writing a target.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [RUN PIPELINE](run-pipeline.md)
- [DROP PIPELINE](drop-pipeline.md)
- [CREATE SCHEDULE](../session/create-schedule.md)
