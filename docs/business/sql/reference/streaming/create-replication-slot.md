# CREATE REPLICATION SLOT

Opens a named position in the change log and holds it, preventing changes at or after that position from being pruned. A slot that is never read holds log space indefinitely. The plugin determines the form changes are delivered in.

## Syntax

```sql
CREATE REPLICATION SLOT name PLUGIN 'plugin' [FOR TABLE name [, ...]]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `FOR TABLE name [, ...]` | Carries changes for those tables alone. | Every table's changes are carried. |

## Examples

```sql
CREATE REPLICATION SLOT s1 PLUGIN 'zyron_cdc' FOR TABLE orders
```

A held position in the log carrying that table's changes.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP REPLICATION SLOT](drop-replication-slot.md)
- [CREATE PUBLICATION](create-publication.md)
