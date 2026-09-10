# ADD EXPECTATION

States a property a table's rows must have, and the action taken on a write that violates it. A constraint can only refuse the write. An expectation can warn, refuse, drop the violating rows, or quarantine them, so a batch containing some violating rows need not be refused in full.

## Syntax

```sql
ALTER TABLE name ADD EXPECTATION name EXPECT predicate | NULL_RATE(col, r) | FRESHNESS(col, 'interval') | DISTINCT_RATE(col, r) | ROW_COUNT_CHANGE(pct) ON VIOLATION WARN | FAIL | DROP | QUARANTINE
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ON VIOLATION WARN` | Records the violation and lets the write through. | Not applicable. |
| `ON VIOLATION DROP` | Drops the offending rows and keeps the rest of the write. | Not applicable. |
| `ON VIOLATION QUARANTINE` | Sets the offending rows aside rather than dropping or keeping them. | Not applicable. |
| `ON VIOLATION FAIL` | Refuses the write, the way a constraint would. | Not applicable. |

## Examples

```sql
ALTER TABLE orders ADD EXPECTATION pos EXPECT total > 0 ON VIOLATION WARN
```

A write with a non-positive total is recorded as a violation and let through.

## On a cluster

The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.

## See also

- [DROP EXPECTATION](drop-expectation.md)
- [ALTER TABLE](alter-table.md)
