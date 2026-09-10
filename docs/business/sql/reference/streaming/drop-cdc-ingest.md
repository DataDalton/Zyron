# DROP CDC INGEST

Stops the ingest. Rows already applied remain. Changes arriving afterwards are not read.

## Syntax

```sql
DROP CDC INGEST name
```

## Examples

```sql
DROP CDC INGEST i
```

The feed stops being applied and the rows already applied stay.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE CDC INGEST](create-cdc-ingest.md)
