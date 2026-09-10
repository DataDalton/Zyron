# sm_is_terminal

True when no transition leaves the state, which makes it a final state of the machine. A state not named in the definition has no outgoing transitions either, so it also reads as terminal.

## Syntax

```sql
sm_is_terminal(statemachine, state)
```

## Returns

BOOLEAN. NULL when an argument is NULL or the definition does not parse.

## Examples

```sql
SELECT sm_is_terminal('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}', 'a')
```

false, because a can go to b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_available_events](sm-available-events.md)
- [sm_reachable_states](sm-reachable-states.md)
