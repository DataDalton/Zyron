# DROP PEER

Removes the peer. Tables following one of its tables stop receiving changes, and the rows that already arrived stay.

## Syntax

```sql
DROP PEER name
```

## Examples

```sql
DROP PEER west
```

Tables following that peer stop receiving changes.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE PEER](create-peer.md)
