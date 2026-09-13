# ALTER TABLE SET OPTIONS

Changes the storage options a table was created with, such as how it compresses or how large a file it writes. The rows already written keep the options they were written under, so a change takes effect as the table is next written rather than retroactively. The change data feed options turn the table's feed on or off and set what it records and for how long. Turning the feed off marks every change stream over the table stale.

## Syntax

```sql
ALTER TABLE name SET (option = value, ...)
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `change_data_feed = true | false` | Records the table's changes in a feed that table_changes and change streams read. Off marks every change stream over the table stale. | Off. |
| `cdf_retention = 'interval'` | How long a recorded change is held before retention reclaims it. A stream positioned below the reclaimed changes is stale. | Seven days. |
| `cdf_columns = 'col, col'` | Records only the named columns and the table's primary key in each change. Every other column reads NULL in a change. | Every column is recorded. |
| `cdf_before_image = true | false` | Records the row an update replaced as well as the row it wrote. Off records the row after the update alone, so an update-heavy feed is half the size and no preimage is readable. | On. |
| `cdf_compression = 'none' | 'lz4' | 'zstd'` | The codec a sealed feed segment is written with. | lz4. |

## Examples

```sql
ALTER TABLE events SET (compression = 'zstd')
```

Later writes use that option and the rows already written keep theirs.

```sql
ALTER TABLE orders SET (change_data_feed = true, cdf_retention = '3 days')
```

The table records its changes and holds each one for three days.

## Refused

- cdf_columns names a column the table does not have.
- cdf_retention is longer than the node's cdc.cdf_max_retention_secs setting.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER TABLE SET USING](alter-table-set-using.md)
- [table_changes](../streaming/table-changes.md)
- [CREATE CHANGE STREAM](../streaming/create-change-stream.md)
