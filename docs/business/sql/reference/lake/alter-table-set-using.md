# ALTER TABLE SET USING

Converts a table between the row store and the lake, rewriting it in full. A lake table accepts branching, time travel and clustering. A heap table has lower per-row write cost. Converting to a heap is refused while the table has version history, unless `drop_history` is set.

## Syntax

```sql
ALTER TABLE name SET USING ZYRONLAKE | HEAP [WITH (drop_history = bool)]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH (drop_history = true)` | Discards the version history on the way to a heap, which has nowhere to keep it. | Moving to a heap with history to discard is refused. |

## Examples

```sql
ALTER TABLE events SET USING ZYRONLAKE
```

The table is rewritten as a lake table and accepts the lake operations.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE TABLE](../ddl/create-table.md)
- [ALTER TABLE SET OPTIONS](alter-table-set-options.md)
