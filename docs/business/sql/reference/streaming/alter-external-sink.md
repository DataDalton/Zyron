# ALTER EXTERNAL SINK

Renames a sink or changes the options and credentials held against it. Credentials can be replaced in place, so rotating a secret requires no change to the statements that name the sink. A sink has no pause action.

## Syntax

```sql
ALTER EXTERNAL SINK name RENAME TO name | SET OPTIONS (...) | SET CREDENTIALS (...) | SET CREDENTIAL PROVIDER ...
```

## Examples

```sql
ALTER EXTERNAL SINK out RENAME TO archive
```

The sink answers to a new name and keeps its configuration.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE EXTERNAL SINK](create-external-sink.md)
