# CREATE COLLATION

Names a rule for ordering text, so a comparison follows a language's own alphabet rather than byte order. It decides what ORDER BY produces and what a unique index treats as equal, which is why two columns that compare differently are not interchangeable even when both hold text.

## Syntax

```sql
CREATE COLLATION name (locale = 'tag', ...)
```

## Examples

```sql
CREATE COLLATION german (locale = 'de_DE')
```

A named rule columns and comparisons can be declared with.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP COLLATION](drop-collation.md)
- [CREATE TYPE](create-type.md)
