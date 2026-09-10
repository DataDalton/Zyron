# CREATE BRANCH

Forks the data at a version. Writes to the branch are not visible on the branch it was forked from until MERGE BRANCH replays them. A branch covers the database unless ON names one table, which forks that table's log alone.

## Syntax

```sql
CREATE BRANCH name [FROM branch] [AT VERSION n] [ON table]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `FROM branch` | Forks from that branch rather than from main. | The fork is taken from main. |
| `AT VERSION n` | Forks from that version rather than from the branch's current head. | The fork is taken from the current head. |
| `ON table` | Forks one table's log rather than the whole database. | The branch covers the database. |

## Examples

```sql
CREATE BRANCH dev FROM main AT VERSION 10
```

A branch holding the data as it was at that version.

## Refused

- AT VERSION is given something that is not a non-negative integer.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [MERGE BRANCH](merge-branch.md)
- [DROP BRANCH](drop-branch.md)
- [USE BRANCH](use-branch.md)
