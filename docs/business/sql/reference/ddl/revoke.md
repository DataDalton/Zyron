# REVOKE

Takes a privilege back from a role. Revoking only the grant option leaves the privilege in place and removes the ability to pass it on.

## Syntax

```sql
REVOKE [GRANT OPTION FOR] privilege [, ...] ON object FROM role [, ...]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `GRANT OPTION FOR` | Removes only the ability to pass the privilege on, leaving the privilege itself. | The privilege itself is removed. |

## Examples

```sql
REVOKE SELECT ON orders FROM analyst
```

Members of that role can no longer read the table.

## Refused

- The object is a temporary table.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [GRANT](grant.md)
