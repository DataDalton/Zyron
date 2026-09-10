# DROP SECURITY MAP

Removes a mapping. A token for that identity still authenticates and resolves to no role, so it can connect and hold no privileges. Access is revoked here without involving the identity provider.

## Syntax

```sql
DROP SECURITY MAP JWT ISSUER 'issuer' SUBJECT 'subject'
```

## Examples

```sql
DROP SECURITY MAP JWT ISSUER 'https://idp' SUBJECT 'alice'
```

That identity resolves to no role and can do nothing.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER SECURITY MAP](alter-security-map.md)
