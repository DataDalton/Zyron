# CREATE FOREIGN TABLE

Declares a table whose rows live somewhere else and are read through a peer when queried. Nothing is copied here, so a query pays the remote read each time and sees the other side's current rows rather than a snapshot.

## Syntax

```sql
CREATE FOREIGN TABLE name (col type, ...) SERVER name [OPTIONS (...)]
```

## Examples

```sql
CREATE FOREIGN TABLE remote_orders (id BIGINT) SERVER west
```

A table read through that peer when queried.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP FOREIGN TABLE](drop-foreign-table.md)
- [CREATE PEER](../session/create-peer.md)
