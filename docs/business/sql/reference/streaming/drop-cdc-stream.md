# DROP CDC STREAM

Stops the stream. Changes committed after it is dropped are not sent, and the table is otherwise untouched.

## Syntax

```sql
DROP CDC STREAM name
```

## Examples

```sql
DROP CDC STREAM s
```

Changes stop arriving at the sink.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE CDC STREAM](create-cdc-stream.md)
