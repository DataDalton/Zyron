# ALTER TABLE CLUSTER BY

Sets the keys a lake table arranges its rows by, which decides how few files a query reading one key range has to touch. A lake table clusters automatically and continuously by default, so this statement is how the keys are chosen rather than how clustering is turned on. A multi-column key may interleave its columns, which keeps a range on either column readable rather than only on the first.

## Syntax

```sql
ALTER TABLE name CLUSTER BY (col [USING method] [, ...]) [AUTO | FORCE] | CLUSTER BY AUTO
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `USING BitInterleave` | Interleaves that column's bits with the others, so a range on any of them reads few files. | The columns order lexicographically, which favours the first. |
| `AUTO` | Lets the engine decide when to run the clustering pass. | Not applicable. |
| `CLUSTER BY AUTO` | Lets the engine choose the keys as well as the schedule. | Not applicable. |

## Examples

```sql
ALTER TABLE events CLUSTER BY (id) AUTO
```

The table arranges its rows by that column, on the engine's own schedule.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [OPTIMIZE TABLE](optimize-table.md)
- [ALTER TABLE CLUSTERING SCHEDULE](alter-table-clustering-schedule.md)
