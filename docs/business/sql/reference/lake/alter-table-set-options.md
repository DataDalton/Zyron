# ALTER TABLE SET OPTIONS

Changes the storage options a table was created with, such as how it compresses or how large a file it writes. The rows already written keep the options they were written under, so a change takes effect as the table is next written rather than retroactively.

## Syntax

```sql
ALTER TABLE name SET (option = value, ...)
```

## Examples

```sql
ALTER TABLE events SET (compression = 'zstd')
```

Later writes use that option and the rows already written keep theirs.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [ALTER TABLE SET USING](alter-table-set-using.md)
