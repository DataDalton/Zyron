# ALTER COLUMN CLASSIFICATION

Records a sensitivity level against a column. The classification grants and denies nothing by itself. A rule written against a classification applies to every column carrying it, including columns classified later. FORGET USER and EXPORT USER read classifications to determine which columns to act on.

## Syntax

```sql
ALTER TABLE name ALTER [COLUMN] col SET CLASSIFICATION level
```

## Examples

```sql
ALTER TABLE customers ALTER COLUMN email SET CLASSIFICATION 'pii'
```

The column carries that level, and rules written against it reach this column.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE ABAC POLICY](create-abac-policy.md)
- [EXPORT USER](export-user.md)
- [FORGET USER](forget-user.md)
