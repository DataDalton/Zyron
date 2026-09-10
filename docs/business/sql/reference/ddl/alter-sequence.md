# ALTER SEQUENCE

Changes a sequence's bounds or where it continues from. Restarting it at a number already handed out is allowed, so a restart is a statement about intent rather than a guarantee of uniqueness.

## Syntax

```sql
ALTER SEQUENCE name [RESTART [WITH n]] [INCREMENT BY n] [MINVALUE n] [MAXVALUE n]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `RESTART [WITH n]` | Continues from this number rather than from where the sequence had reached. | It continues from its start value. |

## Examples

```sql
ALTER SEQUENCE order_ids RESTART WITH 5000
```

The next number handed out is 5000.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [CREATE SEQUENCE](create-sequence.md)
