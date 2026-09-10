# sm_transition

Finds the transition leaving the given state on the given event and returns the state it arrives at. An unknown state, and an event the state has no transition for, both give NULL, so use sm_can_transition to tell a refused move from a NULL input.

## Syntax

```sql
sm_transition(statemachine, state, event)
```

## Returns

VARCHAR naming the new state. NULL when an argument is NULL, the state is unknown, or the event has no transition from it.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `state` | State the machine is in now. | Not applicable. |
| `event` | Event being applied. | Not applicable. |

## Examples

```sql
SELECT sm_transition('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}', 'a', 'go')
```

b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_can_transition](sm-can-transition.md)
- [sm_available_events](sm-available-events.md)
- [sm_parse](sm-parse.md)
