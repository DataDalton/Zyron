# CREATE ROLE

Creates a role. Privileges are granted to roles and users reach them by holding the role, so a role is the unit a privilege review reasons about. A role may be granted to another role, which is how a hierarchy is built.

## Syntax

```sql
CREATE ROLE name [options ...]
```

## Examples

```sql
CREATE ROLE analyst
```

A role that privileges can be granted to.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [GRANT](grant.md)
- [ALTER ROLE](alter-role.md)
- [DROP ROLE](drop-role.md)
