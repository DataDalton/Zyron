# LEGAL HOLD

Holds rows against removal. While a hold stands, retention does not act on the rows it covers and a DELETE against them is refused. The reason is recorded with the hold. RELEASE returns the rows to retention and deletion, and the hold's history is kept.

## Syntax

```sql
LEGAL HOLD CREATE name ON table [WHERE predicate] REASON 'text' | LEGAL HOLD RELEASE name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WHERE predicate` | Holds only the rows the predicate selects. | The whole table is held. |
| `RELEASE name` | Lifts the hold, returning the rows to retention and deletion. | Not applicable. |

## Examples

```sql
LEGAL HOLD CREATE h1 ON orders WHERE id = 5 REASON 'litigation'
```

Those rows cannot be deleted or expired until the hold is released.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [RUN RETENTION JOB](../lake/run-retention-job.md)
- [ALTER TABLE SET TTL](../lake/alter-table-set-ttl.md)
- [FORGET USER](forget-user.md)
