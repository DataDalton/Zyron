# ON COMMIT

Says what every commit does to a temporary table, whether the transaction was opened with BEGIN or was the implicit one a single statement runs in. PRESERVE ROWS leaves the rows alone and is what happens when the clause is left out. DELETE ROWS empties the table and keeps its definition. DROP removes the table at the first commit after it was created. The clause reads only on a temporary table: a permanent table carrying it is refused rather than having it ignored, because a commit does nothing to a permanent table's rows and accepting the words would say otherwise.

## Syntax

```sql
ON COMMIT PRESERVE ROWS | DELETE ROWS | DROP
```

## Examples

```sql
CREATE TEMPORARY TABLE staging (id INT) ON COMMIT DROP
```

A table that lasts until the first commit.

## Refused

- It is written on a permanent table.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [CREATE TEMPORARY TABLE](create-temporary-table.md)
