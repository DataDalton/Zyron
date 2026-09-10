# MERGE BRANCH

Replays the writes made on one branch onto another, which is how work staged on a branch reaches main. FOR TABLE merges one table's branch rather than the database-wide one. A conflict between the two branches is reported rather than resolved, because which side wins is not a decision the engine can make.

## Syntax

```sql
MERGE BRANCH source INTO target [FOR TABLE name]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `FOR TABLE name` | Merges that table's branch rather than the database-wide branch. | The database-wide branch is merged. |

## Examples

```sql
MERGE BRANCH dev INTO main
```

The writes made on the branch are replayed onto main.

## On a cluster

The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.

## See also

- [CREATE BRANCH](create-branch.md)
- [DROP BRANCH](drop-branch.md)
