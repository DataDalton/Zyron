# ARCHIVE TABLE

Writes rows to a destination outside the table and removes them from it, which is how a table is kept to a working size without the rows being lost. A predicate chooses what is archived, and DRY RUN reports what would be written and removed without doing either, so the selection can be checked before it is acted on.

## Syntax

```sql
ARCHIVE TABLE name [WHERE predicate] TO 'destination' [DRY RUN]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WHERE predicate` | Archives only the rows the predicate holds for. | Every row is archived. |
| `DRY RUN` | Reports what would be archived without writing or removing anything. | The rows are written out and removed. |

## Examples

```sql
ARCHIVE TABLE orders WHERE id < 100 TO 's3://bucket/arch/' DRY RUN
```

A report of which rows would be written out and removed.

## Refused

- The table is a temporary one.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [RESTORE TABLE](restore-table.md)
- [RUN RETENTION JOB](run-retention-job.md)
