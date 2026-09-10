# DROP EXTERNAL SINK

Removes the sink and the credentials held against it. What was already written at the URI stays.

## Syntax

```sql
DROP EXTERNAL SINK [IF EXISTS] name
```

## Examples

```sql
DROP EXTERNAL SINK out
```

The name and its credentials are gone, what was written stays.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE EXTERNAL SINK](create-external-sink.md)
