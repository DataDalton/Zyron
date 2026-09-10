# CREATE SPATIAL INDEX

Indexes a geometry column so a query can find the rows inside a region, near a point or overlapping a shape without testing every row. A B-tree cannot answer those, because a position in two dimensions has no single order that keeps neighbours together.

## Syntax

```sql
CREATE SPATIAL INDEX name ON table (col)
```

## Examples

```sql
CREATE SPATIAL INDEX loc_idx ON places (position)
```

An index answering containment and proximity queries on that column.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE INDEX](create-index.md)
