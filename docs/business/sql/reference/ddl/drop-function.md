# DROP FUNCTION

Removes a function. An argument list names which one when several share a name. A view or trigger that called it fails at its next use rather than being rewritten.

## Syntax

```sql
DROP FUNCTION [IF EXISTS] name [(type, ...)]
```

## Examples

```sql
DROP FUNCTION double
```

The function is gone and callers fail at their next use.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE FUNCTION](create-function.md)
