# ALTER VIEW

Renames a view or changes the options recorded against it. The query it names is unchanged, so what the view reads does not move.

## Syntax

```sql
ALTER VIEW name RENAME TO name | SET ...
```

## Examples

```sql
ALTER VIEW paid RENAME TO settled
```

The view answers to a new name.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE VIEW](create-view.md)
