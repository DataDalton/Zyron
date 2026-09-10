# ALTER SECURITY MAP

Maps an issuer and subject pair from an external token onto a role in this database. No user object is created per identity, so the identity provider can add and remove people without any statement running here.

## Syntax

```sql
ALTER SECURITY MAP JWT ISSUER 'issuer' SUBJECT 'subject' TO ROLE 'role' | REMOVE
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `REMOVE` | Removes the mapping, so that identity no longer resolves to a role. | Not applicable. |

## Examples

```sql
ALTER SECURITY MAP JWT ISSUER 'https://idp' SUBJECT 'alice' TO ROLE 'analyst'
```

A token from that issuer for that subject acts as the role.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP SECURITY MAP](drop-security-map.md)
- [CREATE ROLE](create-role.md)
- [GRANT](grant.md)
