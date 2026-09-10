# DROP EVENT HANDLER

Removes the handler. The event is still raised and nothing runs on it.

## Syntax

```sql
DROP EVENT HANDLER [IF EXISTS] name
```

## Examples

```sql
DROP EVENT HANDLER on_violation
```

The event is still raised and nothing runs on it.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE EVENT HANDLER](create-event-handler.md)
