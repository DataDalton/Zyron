# ALTER CLUSTER

Changes the membership of the consensus group. A node added receives the log and catches up before counting towards a quorum. A node removed stops counting immediately. The membership change itself goes through the log.

## Syntax

```sql
ALTER CLUSTER ADD NODE 'id' AT 'address' | REMOVE NODE 'id'
```

## Examples

```sql
ALTER CLUSTER ADD NODE 'node-4' AT 'host:5432'
```

The node joins the group and catches up before counting towards a quorum.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ALTER SYSTEM SET](alter-system-set.md)
- [CREATE PEER](create-peer.md)
