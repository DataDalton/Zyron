# RUN PIPELINE

Runs the pipeline's stages in order. A stage that fails stops the stages after it, and the run reports a failure.

## Syntax

```sql
RUN PIPELINE name
```

## Examples

```sql
RUN PIPELINE etl
```

The steps run in order, stopping at the first failure.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [CREATE PIPELINE](create-pipeline.md)
