# CREATE AGGREGATE

Builds an aggregate from a state type, a function folding one row into the state, and optionally a function converting the final state into the result. Separating the fold from the finish allows the fold to run over partitions with the states combined afterwards, so the aggregate parallelizes.

## Syntax

```sql
CREATE AGGREGATE name(arg type) (SFUNC = name, STYPE = type [, INITCOND = 'value'] [, FINALFUNC = name])
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `INITCOND = 'value'` | The state before any row is folded in. | The state starts null and the fold is given a null first. |
| `FINALFUNC = name` | Turns the final state into the answer, for an aggregate whose state is not its result. | The final state is the answer. |

## Examples

```sql
CREATE AGGREGATE total(val INT) (SFUNC = add_int, STYPE = INT, INITCOND = '0')
```

An aggregate usable wherever SUM is.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP AGGREGATE](drop-aggregate.md)
- [CREATE FUNCTION](create-function.md)
