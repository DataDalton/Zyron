# GRANT

Gives a role a privilege on an object. Privileges are held by roles rather than by users, and a user reaches a privilege by being granted the role that holds it. WITH GRANT OPTION lets the role pass the privilege on. A grant is durable: it survives a restart, and it is forgotten when the object it names is dropped.

## Syntax

```sql
GRANT privilege [, ...] ON object TO role [, ...] [WITH GRANT OPTION]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH GRANT OPTION` | Lets the role grant this privilege to others. | The role holds the privilege and cannot pass it on. |

## Examples

```sql
GRANT SELECT ON orders TO analyst
```

Members of that role may read the table.

## Refused

- The object is a temporary table.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [REVOKE](revoke.md)
- [CREATE ROLE](create-role.md)
