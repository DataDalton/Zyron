# ALTER TABLE FOLLOW

Applies changes from a table on another Zyron cluster into this one. The followed table is the only writer, and this table is read-only while it follows. UNFOLLOW stops the flow and keeps the rows that arrived.

## Syntax

```sql
ALTER TABLE name FOLLOW peer.table | UNFOLLOW
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `UNFOLLOW` | Stops following and keeps the rows that arrived. | Not applicable. |

## Examples

```sql
ALTER TABLE orders FOLLOW west.orders
```

Changes from that peer's table arrive here and are applied.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE PEER](../session/create-peer.md)
- [ALTER TABLE](../ddl/alter-table.md)
