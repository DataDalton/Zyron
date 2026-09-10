# ALTER PUBLICATION

Adds or removes a table from a publication, or changes its options. A subscriber sees the change from the point it takes effect rather than retroactively, so a table added now does not deliver the changes it had before.

## Syntax

```sql
ALTER PUBLICATION name ADD TABLE name | DROP TABLE name | SET (option = value, ...)
```

## Examples

```sql
ALTER PUBLICATION pub ADD TABLE items
```

Subscribers begin receiving that table's changes too.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE PUBLICATION](create-publication.md)
