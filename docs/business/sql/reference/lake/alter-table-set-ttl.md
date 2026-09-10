# ALTER TABLE SET TTL

Sets how long a row lives, measured from a timestamp column the row carries. A retention run acts on rows that have passed it. The action determines whether the row is deleted or anonymized, which clears the identifying columns and keeps the row.

## Syntax

```sql
ALTER TABLE name SET TTL [ARCHIVE] duration ON column [ACTION DELETE | ANONYMIZE]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ARCHIVE` | Writes the expired rows out before removing them, rather than only removing them. | The rows are acted on in place. |
| `ACTION ANONYMIZE` | Keeps the expired row and clears what identified it. | The expired row is deleted. |

## Examples

```sql
ALTER TABLE events SET TTL 30 DAYS ON created_at
```

Rows older than thirty days by that column are removed by a retention run.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [RUN RETENTION JOB](run-retention-job.md)
- [ALTER TABLE](../ddl/alter-table.md)
