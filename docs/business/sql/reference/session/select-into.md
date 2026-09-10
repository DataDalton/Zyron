# SELECT INTO

Creates a table shaped by a query's output and fills it with that output, in one statement. Writing TEMP or TEMPORARY makes it a temporary table belonging to this session. It is the same statement as CREATE TABLE AS SELECT written the other way round, and the two produce the same table.

## Syntax

```sql
SELECT ... INTO [TEMP | TEMPORARY] name FROM ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `TEMP | TEMPORARY` | Makes the created table belong to this session alone. | The table is permanent. |

## Examples

```sql
SELECT id, total INTO TEMP recent FROM orders WHERE total > 100
```

A temporary table named recent holding the rows the query returned.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [CREATE TEMPORARY TABLE](create-temporary-table.md)
