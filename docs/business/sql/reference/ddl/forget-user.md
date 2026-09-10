# FORGET USER

Erases the rows belonging to one data subject in every table holding them. Tables are found through column classifications, so a table classified after the statement was first used is still covered. Rows under a legal hold are reported rather than erased, because the hold and the erasure cannot both be satisfied.

## Syntax

```sql
FORGET USER 'subject'
```

## Examples

```sql
FORGET USER 'u-1'
```

That subject's rows are erased wherever classification says they are held.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [EXPORT USER](export-user.md)
- [ALTER COLUMN CLASSIFICATION](alter-column-classification.md)
- [LEGAL HOLD](legal-hold.md)
