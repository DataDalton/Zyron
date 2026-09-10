# CREATE EVENT HANDLER

Runs a function when the engine raises a named event, such as a job finishing or an expectation being violated. It is how a reaction is kept inside the database rather than needing something outside to watch for the event and act.

## Syntax

```sql
CREATE EVENT HANDLER name WHEN event EXECUTE FUNCTION name
```

## Examples

```sql
CREATE EVENT HANDLER on_violation WHEN expectation_violated EXECUTE FUNCTION notify_team
```

That function runs whenever the event is raised.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP EVENT HANDLER](drop-event-handler.md)
- [CREATE FUNCTION](create-function.md)
