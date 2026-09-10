# DROP EXTERNAL SOURCE

Removes the source and the credentials held against it. The data at the URI is untouched, because the source was a name for it rather than a copy of it.

## Syntax

```sql
DROP EXTERNAL SOURCE [IF EXISTS] name
```

## Examples

```sql
DROP EXTERNAL SOURCE src
```

The name and its credentials are gone, the data is not.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE EXTERNAL SOURCE](create-external-source.md)
