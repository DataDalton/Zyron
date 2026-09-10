# CREATE SEQUENCE

Creates a generator that hands out numbers, each one different from the last. Numbers taken from it are not returned when a transaction rolls back, so a sequence has gaps and is a source of unique values rather than of a contiguous count.

## Syntax

```sql
CREATE SEQUENCE [IF NOT EXISTS] name [START WITH n] [INCREMENT BY n] [MINVALUE n] [MAXVALUE n] [CYCLE | NO CYCLE]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `START WITH n` | The first number the sequence hands out. | It starts at 1. |
| `INCREMENT BY n` | How far each number is from the one before it. A negative value counts down. | Each number is one more than the last. |
| `CYCLE` | Returns to the minimum after the maximum, rather than failing. | Passing the maximum fails. |

## Examples

```sql
CREATE SEQUENCE order_ids START WITH 1000
```

A generator whose first number is 1000.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP SEQUENCE](drop-sequence.md)
- [ALTER SEQUENCE](alter-sequence.md)
