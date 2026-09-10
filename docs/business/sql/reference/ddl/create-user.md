# CREATE USER

Creates a principal that can open a connection. A user holds no privileges directly. Privileges reach it through the roles it is granted.

## Syntax

```sql
CREATE USER name [WITH PASSWORD 'text'] [options ...]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH PASSWORD 'text'` | Sets the password the principal authenticates with. | The principal has no password and authenticates another way. |

## Examples

```sql
CREATE USER alice
```

A principal that can connect and holds no privileges yet.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER USER](alter-user.md)
- [DROP USER](drop-user.md)
- [CREATE ROLE](create-role.md)
