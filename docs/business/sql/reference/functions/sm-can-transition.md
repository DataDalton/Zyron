# sm_can_transition

Answers whether a transition exists, without performing it. A move refused here answers false, where sm_transition answers NULL for both a refused move and a NULL input.

## Syntax

```sql
sm_can_transition(statemachine, state, event)
```

## Returns

BOOLEAN. NULL when an argument is NULL or the definition does not parse.

## Examples

```sql
SELECT sm_can_transition('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}', 'b', 'go')
```

false, because b has no go transition.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_transition](sm-transition.md)
- [sm_available_events](sm-available-events.md)
