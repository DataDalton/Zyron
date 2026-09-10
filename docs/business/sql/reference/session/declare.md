# DECLARE

Opens a cursor over a query. FETCH reads its rows a batch at a time, so a result larger than memory can be read. Without WITH HOLD, the cursor closes when its transaction ends.

## Syntax

```sql
DECLARE name [NO SCROLL | SCROLL] CURSOR [WITH HOLD] FOR SELECT ...
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `WITH HOLD` | Keeps the cursor readable after the transaction that declared it commits. | The cursor closes when its transaction ends. |
| `SCROLL` | Allows fetching backwards as well as forwards. | The cursor reads forwards. |

## Examples

```sql
DECLARE c CURSOR FOR SELECT id FROM orders
```

A cursor that FETCH reads a batch at a time.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [FETCH](fetch.md)
- [CLOSE](close.md)
