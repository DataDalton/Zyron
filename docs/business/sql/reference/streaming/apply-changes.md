# APPLY CHANGES

Applies a set of changes into a target table in one transaction. The source is a change stream or any relation carrying the change metadata columns, such as table_changes. The changes are grouped by KEYS, ordered within a key by SEQUENCE BY, and the last change to each key is what the target ends up holding, so applying the same changes twice leaves the target as one application did. An insert or an update upserts the row, a delete removes it, and a change matching APPLY AS DELETE WHEN or APPLY AS TRUNCATE WHEN is treated as that. Read from a stream, the stream's position moves in the same transaction, so a failure leaves both the target and the position where they were. SCD TYPE 2 keeps one row per version of a key with __start_at, __end_at and __is_current columns, closing the current row and opening a new one on each change to the tracked columns.

## Syntax

```sql
APPLY CHANGES INTO target FROM stream | relation KEYS (col, ...) [SEQUENCE BY expr] [IGNORE NULL UPDATES] [APPLY AS DELETE WHEN predicate] [APPLY AS TRUNCATE WHEN predicate] [EXCEPT COLUMNS (col, ...)] [STORED AS SCD TYPE 1 | 2] [TRACK HISTORY ON (col, ...) | EXCEPT (col, ...)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `KEYS (col, ...)` | The columns that identify a row, which the target must hold. | Not applicable. |
| `SEQUENCE BY expr` | Orders the changes to one key, so the highest value is the one applied. | Changes to one key apply in the order they were recorded. |
| `IGNORE NULL UPDATES` | Treats a NULL in an update as a column not supplied, keeping the target's value. | A NULL in an update writes NULL. |
| `APPLY AS DELETE WHEN predicate` | Treats a change matching the predicate as a delete of its key. | Not applicable. |
| `APPLY AS TRUNCATE WHEN predicate` | Treats a change matching the predicate as a truncation of the target, applied before the other changes in the set. | Not applicable. |
| `EXCEPT COLUMNS (col, ...)` | Leaves those columns out of what is written to the target. | Every column the source and the target share is written. |
| `STORED AS SCD TYPE 2` | Keeps one row per version of a key, with __start_at, __end_at and __is_current marking each version's validity. | TYPE 1, one row per key holding the last change. |
| `TRACK HISTORY ON (col, ...) | EXCEPT (col, ...)` | Which columns open a new version under SCD TYPE 2. A change to an untracked column alone updates the current row in place. | Every column opens a new version. |

## Examples

```sql
APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id) SEQUENCE BY _commit_version
```

The target holds one row per id reflecting the last change to it, and the stream's position moves in the same commit.

```sql
APPLY CHANGES INTO dim_customer FROM customer_changes KEYS (id) SEQUENCE BY updated_at STORED AS SCD TYPE 2 TRACK HISTORY EXCEPT (last_seen)
```

One row per version of each customer, a change to last_seen alone updating the current row in place.

## Refused

- KEYS is left out.
- The target is a lake table.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [CREATE CHANGE STREAM](create-change-stream.md)
- [table_changes](table-changes.md)
- [MERGE](../dml/merge.md)
- [CREATE PIPELINE](create-pipeline.md)
