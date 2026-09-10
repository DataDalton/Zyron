# DROP VERSION

Removes the name. The version it resolved to remains reachable by number while retention keeps it.

## Syntax

```sql
DROP VERSION name
```

## Examples

```sql
DROP VERSION v1
```

The name is gone and the version it pointed at is not.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE VERSION](create-version.md)
