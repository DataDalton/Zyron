# ALTER ROLE

Renames a role or changes the options recorded against it. The privileges granted to it and the members holding it follow the role rather than its name.

## Syntax

```sql
ALTER ROLE name [RENAME TO name] [options ...]
```

## Examples

```sql
ALTER ROLE analyst RENAME TO reader
```

The role answers to a new name and keeps its privileges.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ROLE](create-role.md)
