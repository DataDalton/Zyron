# USE BRANCH

Directs this session's reads and writes to a branch. The setting is per session, and no other session is affected.

## Syntax

```sql
USE BRANCH name
```

## Examples

```sql
USE BRANCH dev
```

This session reads and writes that branch.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [CREATE BRANCH](create-branch.md)
- [MERGE BRANCH](merge-branch.md)
