# CREATE TEMPORARY TABLE

Creates a table that belongs to the session that created it and to the node that session is connected to. No other session sees it, no other node has it, and it is gone when the session ends. It is a heap table in every other respect: rows follow the session's own transaction snapshot, indexes may be created on it, and a rolled-back insert is invisible afterwards. Nothing about it is written to the write-ahead log or proposed to the consensus log, so creating one on a cluster costs no agreement round. It takes a bare name and shadows a permanent table of that name for the creating session alone.

## Syntax

```sql
CREATE [OR REPLACE] TEMPORARY | TEMP TABLE name (...) [ON COMMIT PRESERVE ROWS | DELETE ROWS | DROP]
```

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `OR REPLACE` | Drops a temporary table of the same name in this session and creates the new one in its place. | A table of that name in this session is an error. |
| `TEMP` | The same word as TEMPORARY. | Not applicable. |
| `ON COMMIT PRESERVE ROWS` | Leaves the rows in place at every commit. This is what happens without the clause. | Not applicable. |
| `ON COMMIT DELETE ROWS` | Empties the table at every commit, keeping its definition. | Not applicable. |
| `ON COMMIT DROP` | Drops the table at the first commit after it is created. | Not applicable. |
| `AS SELECT ...` | Takes the columns from the query's own output and fills the table in the same statement. | The columns are written out and the table starts empty. |

## Examples

```sql
CREATE TEMPORARY TABLE scratch (a INT) ON COMMIT DELETE ROWS
```

A table only this session sees, emptied at every commit.

## Refused

- The name carries a schema qualifier.
- The session already holds temp_table_max_count of them.
- The session's temporary tables already hold temp_table_max_bytes.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ON COMMIT](on-commit.md)
- [SELECT INTO](select-into.md)
