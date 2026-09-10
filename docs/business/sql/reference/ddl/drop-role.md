# DROP ROLE

Removes a role. Every privilege granted to it goes with it, so a principal that reached a privilege only through this role no longer has it.

## Syntax

```sql
DROP ROLE [IF EXISTS] name
```

## Examples

```sql
DROP ROLE analyst
```

The role and the privileges held through it are gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ROLE](create-role.md)
- [REVOKE](revoke.md)
