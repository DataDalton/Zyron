# CREATE PIPELINE

Names an ordered set of stages that run as one unit. A stage names its source, its target, and the query between them. The mode determines whether the target is rebuilt or added to. A stage does not begin before the stage producing what it reads has finished. A stage over a change stream runs as one transaction that moves the stream's position, so a stage that fails leaves the position where it was and the next run reads the same changes. A pipeline created ON CHANGE DATA runs on its own when the stream holds enough pending changes.

## Syntax

```sql
CREATE PIPELINE name [ON CHANGE DATA FROM stream [MIN ROWS n] [MAX WAIT duration]] AS (STAGE name (SOURCE src, TARGET dst, MODE full | incremental, TRANSFORM AS (query)) | STAGE name (CONSUME CHANGES FROM stream [MAX ROWS n] INTO relation | AS (statement)) | STAGE name (APPLY CHANGES ...), ...)
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `MODE full` | Rebuilds the target from the source each run. | Not applicable. |
| `MODE incremental` | Adds to the target rather than rebuilding it. | Not applicable. |
| `TRANSFORM AS (query)` | The query that turns the source into what the target holds. | The source's rows are written to the target unchanged. |
| `STAGE name (CONSUME CHANGES FROM stream [MAX ROWS n] INTO relation)` | Reads the stream's pending changes into the relation, writing the columns the relation and the change set share by name. A relation that does not exist is created from the change set, metadata columns included. MAX ROWS ends a run after about n changes at a boundary no transaction writes across, so a backlog drains over several runs. | Not applicable. |
| `STAGE name (CONSUME CHANGES FROM stream [MAX ROWS n] AS (statement))` | Runs the statement with the stream's pending changes bound as the relation named changes, in the transaction that moves the position. | Not applicable. |
| `STAGE name (APPLY CHANGES INTO target FROM stream KEYS (col, ...) ...)` | Maintains the target from the stream the way the APPLY CHANGES statement does, in a transaction of its own. | Not applicable. |
| `ON CHANGE DATA FROM stream [MIN ROWS n] [MAX WAIT duration]` | Runs the pipeline when the stream holds at least n pending changes, or when the duration has passed with at least one pending. The pending count is read without moving the position. A run starts within a second of the condition being met, on the node standing alone or leading its group. | The pipeline runs when RUN PIPELINE is issued. MIN ROWS is one, and there is no wait. |

## Examples

```sql
CREATE PIPELINE etl AS (STAGE s (SOURCE raw, TARGET staging, MODE full, TRANSFORM AS (SELECT id FROM raw)))
```

A named unit whose stages run in order, each reading a source and writing a target.

```sql
CREATE PIPELINE cdc ON CHANGE DATA FROM order_changes MIN ROWS 100 MAX WAIT 5 MINUTES AS (STAGE land (CONSUME CHANGES FROM order_changes MAX ROWS 10000 INTO bronze_orders), STAGE apply (APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id)))
```

A pipeline that runs once a hundred changes pend or five minutes pass with one, landing the changes and maintaining the target, each stage moving the position in its own commit.

## Refused

- ON CHANGE DATA names something that is not a change stream.
- A CONSUME CHANGES statement never reads the relation named changes.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [RUN PIPELINE](run-pipeline.md)
- [DROP PIPELINE](drop-pipeline.md)
- [CREATE SCHEDULE](../session/create-schedule.md)
- [CREATE CHANGE STREAM](create-change-stream.md)
- [APPLY CHANGES](apply-changes.md)
