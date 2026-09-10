# OPTIMIZE TABLE

Rewrites a table's storage. CLUSTER arranges rows by the table's clustering keys, so a query reading one key range touches fewer files. DELETE applies recorded delete predicates and compacts the space they freed. Writing both, or neither, runs both passes.

## Syntax

```sql
OPTIMIZE TABLE name [CLUSTER] [, DELETE]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `CLUSTER` | Runs the layout pass alone, arranging rows by the clustering keys. | Both passes run. |
| `DELETE` | Applies recorded deletes and compacts the freed space, without the layout pass. | Both passes run. |

## Examples

```sql
OPTIMIZE TABLE orders CLUSTER, DELETE
```

The rows are arranged by the clustering keys and recorded deletes are applied.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ALTER TABLE CLUSTER BY](alter-table-cluster-by.md)
- [VACUUM](../session/vacuum.md)
