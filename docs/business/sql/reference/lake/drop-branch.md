# DROP BRANCH

Removes a branch and discards writes made on it that were never merged. The branch it was forked from is unaffected.

## Syntax

```sql
DROP BRANCH name [ON table]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ON table` | Drops that table's branch rather than the database-wide one. | The database-wide branch is dropped. |

## Examples

```sql
DROP BRANCH dev
```

The branch and its unmerged writes are gone.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE BRANCH](create-branch.md)
- [MERGE BRANCH](merge-branch.md)
