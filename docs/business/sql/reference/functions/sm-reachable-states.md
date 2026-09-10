# sm_reachable_states

Walks the transitions breadth first and lists the states it can arrive at, however many steps away. A state the definition names but no path reaches is left out, which makes this the check for an unreachable state.

## Syntax

```sql
sm_reachable_states(statemachine, state)
```

## Returns

ARRAY of state names as JSON text. NULL when an argument is NULL or the definition does not parse.

## Examples

```sql
SELECT sm_reachable_states('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}', 'a')
```

The states reachable from a, which includes b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_shortest_path](sm-shortest-path.md)
- [sm_is_terminal](sm-is-terminal.md)
