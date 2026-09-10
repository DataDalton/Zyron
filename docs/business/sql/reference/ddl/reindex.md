# REINDEX

Rebuilds an index from the table's current rows. It is what recovers an index whose file was damaged, and what compacts one whose tree has been left sparse by a long history of deletes.

## Syntax

```sql
REINDEX TABLE name | INDEX name
```

## Examples

```sql
REINDEX TABLE orders
```

Every index on the table rebuilt from its rows.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [CREATE INDEX](create-index.md)
- [VACUUM](../session/vacuum.md)
