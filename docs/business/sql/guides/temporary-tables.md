# Temporary Tables

A temporary table belongs to the session that created it and to the node that session is connected to. No other session sees it, no other node has it, and it is gone when the session ends.

## What this means for you

- Any principal that may connect may create one. There is no privilege to grant.
- A temporary table is invisible to every other session, so two sessions may hold one of the same name and neither sees the other's.
- It shadows a permanent table of the same bare name for the creating session only, and a qualified name always reaches the permanent one.
- Nothing about it reaches the consensus log or the catalog, so creating one on a cluster costs no agreement round.
- It is dropped when the session ends, when the connection drops, and when an HTTP session expires. A node restart clears whatever a crash left behind.

## Creating one

```sql
CREATE [OR REPLACE] TEMPORARY | TEMP TABLE name (...)
    [ON COMMIT PRESERVE ROWS | DELETE ROWS | DROP]

CREATE TEMPORARY | TEMP TABLE name AS SELECT ...

SELECT ... INTO TEMP name FROM ...
```

`TEMPORARY` and `TEMP` mean the same thing.

The `AS SELECT` form takes its columns from the query's own output, so the table is declared with exactly what the query produces and filled in the same statement. `SELECT ... INTO TEMP name` is the same statement written the other way round.

A temporary table takes a bare name. A schema qualifier is refused:

```text
a temporary table takes a bare name, 'app.scratch' names a schema; a temporary
table lives in the session's own namespace and a qualified name always reaches
a permanent table
```

That refusal is what makes the shadowing rule usable. Inside a session that has created a temporary `orders`, `SELECT * FROM orders` reads the temporary one and `SELECT * FROM sales.orders` reads the permanent one, always.

## What a commit does

`ON COMMIT` says what happens to the table's rows when a transaction commits. Every commit acts on it, whether the transaction was opened with `BEGIN` or was the implicit one a single statement runs in.

| Clause | What a commit does |
| --- | --- |
| `ON COMMIT PRESERVE ROWS` | Nothing. The rows outlive the transaction. This is the default. |
| `ON COMMIT DELETE ROWS` | Empties the table. The definition stays. |
| `ON COMMIT DROP` | Drops the table at the first commit. |

`ON COMMIT` reads only on a temporary table. A permanent table carrying the clause is refused rather than having it ignored, because a commit does nothing to a permanent table's rows and accepting the words would say otherwise.

## Where it lives

The table is a heap table. Its heap file, its free-space map and any index on it sit under `<data_dir>/tmp/<session_key>/` on the node the session is connected to. The session key names the directory and appears in the view below.

- Pages come out of the session's memory budget first and spill to that directory under the pressure substrate, the same way any other working set does.
- Nothing is written to the write-ahead log and nothing is checkpointed. A temporary table has no state worth recovering: the session that could read it is gone by the time recovery runs.
- Row visibility is the session's own transaction snapshot, exactly as for a permanent heap table. An `INSERT` that is rolled back is invisible afterwards.
- Indexes may be created on it. An index on a temporary table is session-local too, and its file sits in the same directory.

A node clears `<data_dir>/tmp/` before it accepts a connection, so files a crash left behind are gone by the time anything could read them.

## What cannot name one

Nothing durable may reference a temporary table, because a durable definition outlives the session and would be left naming something no session can resolve. Each of these is refused, naming the table and the reason:

- A view or a materialized view whose query reads it.
- A trigger on it.
- A row security policy on it.
- A grant or revoke on it.
- A comment on it or on one of its columns.
- A foreign key on a permanent table pointing at it.

`ARCHIVE` and `RESTORE` are refused for one, because both move stored versions to and from a destination and a temporary table has neither. `CREATE TEMPORARY TABLE` is refused inside a branch operation, because a branch is a version of stored data that other sessions read.

A temporary table may reference a permanent one. It is the other direction that cannot hold.

`USING ZYRONLAKE`, `CLUSTER BY`, `CLONE OF` and `TTL` are refused on one, each naming what it does and why a session-local heap table has no place for it.

## Limits

Two per-session limits bound what one connection may hold. Both are cascading configuration, so they can be set for the node, the database or the session.

| Setting | Default | What it bounds |
| --- | --- | --- |
| `temp_table_max_bytes` | 10% of the node's memory budget | The bytes one session's temporary tables hold in total |
| `temp_table_max_count` | 256 | How many temporary tables one session holds |

Reaching either refuses further creation and names the limit that refused it:

```text
this session already holds 256 temporary tables, which is the
temp_table_max_count limit of 256; drop one before creating another or raise
temp_table_max_count
```

## Watching them

`zyron_sys.stat.temp_tables` lists every temporary table on the node, per session, with what each holds.

```sql
SELECT session_id, name, rows, bytes, on_commit FROM zyron_sys.stat.temp_tables;
```

| Column | What it holds |
| --- | --- |
| `session_id` | The backend process id, which joins to `zyron_sys.stat.sessions` |
| `session_key` | Names the session's directory under `<data_dir>/tmp/` |
| `name` | The bare name it was created with |
| `table_id` | The node-local id it is addressed by |
| `bytes` | Bytes this table's own files hold, its indexes included, as of the last collection |
| `rows` | Live rows, as of the last collection |
| `on_commit` | What a commit does to it |
| `stale` | True when rows have changed since the figures were collected |

Statistics are collected after a write, which is what gives the planner a cardinality for the table. Collecting one costs no pass over the table: the session that owns a temporary table is its only writer, so each statement carries the live row count forward by the rows it reported, and the byte total is the page count of the table's own files. A MERGE is the exception, because its count does not separate the rows it added from the rows it removed, so the next collection after one counts the rows once.

`zyron_sys.catalog.tables` never lists a temporary table. It is not in the catalog, so the operator view above is where to look.

## On a cluster

A statement that creates, fills or drops a temporary table runs on the node it arrives at and reaches no other member. Its rows never enter a changeset and its definition is never proposed, so a follower's catalog and its consensus log are unchanged by the whole life of one.

A session holding a temporary table cannot be moved to another node during a drain, because the table's files are on this node's disk. `zyron_sys.stat.sessions` names `temp_tables` as the pin reason, and the drain lets the session finish rather than relocating it.

## Who uses them

A notebook cell that materializes an intermediate result, and the Python driver's DataFrame materialization, both use exactly this feature. Neither has a private mechanism of its own.
