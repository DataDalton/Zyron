# CREATE PEER

Names another Zyron cluster, so a table can follow one of its tables or publish to it. A peer is not a member of this cluster's consensus group and participates in no quorum. Data moves between them only where a statement says so.

## Syntax

```sql
CREATE PEER [IF NOT EXISTS] name ADDRESS 'host:port' [MODE mode]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `MODE mode` | How this cluster talks to the peer. | The default mode for a peer applies. |

## Examples

```sql
CREATE PEER west ADDRESS 'west.example:5432'
```

A named cluster that tables can follow or publish to.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP PEER](drop-peer.md)
- [ALTER TABLE FOLLOW](../lake/alter-table-follow.md)
