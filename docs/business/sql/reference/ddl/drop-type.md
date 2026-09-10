# DROP TYPE

Removes the type. A column declared with it keeps the storage type underneath, so the rows are readable and the meaning the name carried is gone.

## Syntax

```sql
DROP TYPE [IF EXISTS] name
```

## Examples

```sql
DROP TYPE zip
```

The name is gone and columns keep their storage type.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TYPE](create-type.md)
