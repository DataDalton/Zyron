# DROP FEATURE GROUP

Removes the declaration and the values it stored. A model that read it can no longer be served or retrained against it, so dropping one is a statement about the models that use it as much as about the group.

## Syntax

```sql
DROP FEATURE GROUP [IF EXISTS] name
```

## Examples

```sql
DROP FEATURE GROUP user_features
```

The declaration and its stored values are gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE FEATURE GROUP](create-feature-group.md)
