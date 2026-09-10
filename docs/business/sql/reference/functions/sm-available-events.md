# sm_available_events

Lists the events of every transition leaving the state. A terminal state gives an empty array, which is the same answer sm_is_terminal reports as true.

## Syntax

```sql
sm_available_events(statemachine, state)
```

## Returns

ARRAY of event names as JSON text. NULL when an argument is NULL or the definition does not parse.

## Examples

```sql
SELECT sm_available_events('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}', 'a')
```

["go"].

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_can_transition](sm-can-transition.md)
- [sm_is_terminal](sm-is-terminal.md)
