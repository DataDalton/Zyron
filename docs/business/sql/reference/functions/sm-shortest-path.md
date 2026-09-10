# sm_shortest_path

Searches breadth first and returns the events of the shortest route, so the result names the sequence to apply rather than the states passed through. A pair with no route gives NULL.

## Syntax

```sql
sm_shortest_path(statemachine, from, to)
```

## Returns

ARRAY as JSON text. NULL when an argument is NULL, the definition does not parse, or no route exists.

## Examples

```sql
SELECT sm_shortest_path('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}', 'a', 'b')
```

The single step that reaches b.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_reachable_states](sm-reachable-states.md)
- [sm_transition](sm-transition.md)
