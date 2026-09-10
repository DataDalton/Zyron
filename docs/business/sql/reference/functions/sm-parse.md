# sm_parse

Takes a JSON object holding a states array, an initial state and a transitions array whose entries name a from state, an event and a to state. Every other state machine function accepts either this value or the JSON text, so parsing once avoids reparsing per row. A definition that does not parse gives NULL.

## Syntax

```sql
sm_parse(text)
```

## Returns

BYTEA holding the compiled definition. NULL when the text is NULL or does not parse.

## Examples

```sql
SELECT sm_is_terminal(sm_parse('{"states":["a","b"],"initial":"a","transitions":[{"from":"a","event":"go","to":"b"}]}'), 'b')
```

true, because b has no outgoing transition.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [sm_transition](sm-transition.md)
- [sm_can_transition](sm-can-transition.md)
- [sm_available_events](sm-available-events.md)
