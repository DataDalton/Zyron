# FETCH

Reads rows from an open cursor and moves it. The direction words move it backwards or to a position, which a cursor declared SCROLL allows and one declared without it does not.

## Syntax

```sql
FETCH NEXT | PRIOR | FIRST | LAST | ALL | ABSOLUTE n | RELATIVE n | FORWARD [n | ALL] | BACKWARD [n | ALL] [FROM | IN] name
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ALL` | Reads every remaining row. | One row is read. |
| `BACKWARD [count]` | Reads rows before the cursor's position, on a cursor declared SCROLL. | Not applicable. |

## Examples

```sql
FETCH FORWARD 100 FROM c
```

The next hundred rows, and the cursor moved past them.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [DECLARE](declare.md)
- [CLOSE](close.md)
