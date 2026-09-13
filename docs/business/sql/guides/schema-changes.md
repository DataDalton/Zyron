# Schema Changes

A schema change on a Zyron table runs beside the traffic on that table. No session waits for one except the session that issued it.

## What this means for you

- `ADD COLUMN`, `DROP COLUMN` and most type changes finish in the time one catalog record takes to write, whatever the table's size.
- `CREATE INDEX` reads the table once, in bounded batches, while inserts, updates and deletes carry on against it.
- `ADD CONSTRAINT` starts enforcing the rule immediately and checks the existing rows afterwards.
- Nothing takes a table-wide lock, and no statement is queued behind one.

## Column changes that write no row

Three changes are a catalog record and nothing else, because the rows on disk can be read correctly without moving them.

| Change | What happens to the rows |
| --- | --- |
| `ADD COLUMN c INT DEFAULT 7` | Rows that predate the column read 7. Rows written afterwards store what they were given. |
| `ADD COLUMN c TEXT` | Rows that predate the column read NULL. |
| `DROP COLUMN c` | The column disappears from every listing. Its bytes stay in the rows already written and come back as those rows are rewritten in the ordinary course. |
| `ALTER COLUMN c TYPE <wider>` | Rows read at the new width. |

A `DEFAULT` on an added column is evaluated once, when the column is added. `ADD COLUMN c TIMESTAMP DEFAULT now()` gives every existing row the same instant, which is the instant the column was added.

`ADD COLUMN ... NOT NULL` with no `DEFAULT` is refused: the rows that predate the column have no value to read.

`DROP COLUMN` is refused when the column is the table's last one, when a constraint is declared over it, or when an index keys on it. The refusal names the index or constraint. Drop those first, so the decision to give up the access path or the rule is yours.

## Type changes that write no row

A type change completes without touching a row when every value the old type permits is a valid value of the new type. Widening is the whole of it.

| From | To | Result |
| --- | --- | --- |
| `INT` | `BIGINT` | Catalog only |
| `SMALLINT` | `INT`, `BIGINT` | Catalog only |
| `VARCHAR(10)` | `VARCHAR(40)`, `TEXT` | Catalog only |
| `TIMESTAMP(3)` | `TIMESTAMP(6)`, `TIMESTAMP(9)` | Catalog only |
| `DECIMAL(10,2)` | `DECIMAL(14,2)` | Catalog only |
| `NOT NULL` column | nullable | Catalog only |

The judgement is made on the declared types, not on the values the table happens to hold. A `BIGINT` column whose values all fit in an `INT` still rewrites, because the declaration permits values that do not.

## Type changes that rewrite

Everything else re-encodes every row: `BIGINT` to `INT`, `TEXT` to `INT`, `DECIMAL(14,2)` to `DECIMAL(10,2)`, `TIMESTAMP(6)` to `TIMESTAMP(3)`.

The rewrite runs beside the live table. A second copy of the table is filled from the first while every write to the table lands in both, its indexes are built, and the catalog is pointed at it in one step. Statements already reading the old copy finish on it.

Two things end a rewrite:

- **A value the new type cannot hold.** The rewrite stops, names the row and the value, and changes nothing. `ALTER COLUMN price TYPE INT` on a `TEXT` column holding `'abc'` reports that row and leaves the table as it was.
- **A writer supplying such a value while the rewrite runs.** That writer's statement fails with the cast error. The rewrite carries on.

## CREATE INDEX

`CREATE INDEX` returns when the index is complete and the planner may choose it. Until then the planner runs the query without it, and `EXPLAIN` says so on the scan it chose instead:

```text
SeqScan table_id=7 columns=3 filter=yes index_building=ix_amount is still building, this scan runs without it
```

Writes against the table are maintained into the index from the moment the statement starts, so nothing written during the build is missing from it afterwards.

`CREATE UNIQUE INDEX` over a table that already holds two rows with the same key is refused, naming the key and both rows, and leaves no index behind.

`REINDEX` builds the new tree beside the old one. Queries keep being answered through the old tree until the new one is installed.

## ADD CONSTRAINT

A constraint is enforced on every write from the moment the statement starts. The rows that predate it are checked afterwards. While that check runs, `zyron_sys.core.constraints` reports the constraint's `state` as `validating`, and it becomes `valid` when the check completes.

A row that already breaks the rule ends the statement and the constraint is not added, so a rule the data does not satisfy never sits on a table unenforced.

## Watching progress

`zyron_sys.storage.ddl_progress` holds one row per schema change running now, and the row disappears when the change ends.

```sql
SELECT table_name, operation, object_name, phase, rows_done, rows_total_estimate, pause_signal
FROM zyron_sys.storage.ddl_progress;
```

| Column | Meaning |
| --- | --- |
| `operation` | `create_index`, `reindex`, `validate_constraint` or `shadow_rewrite` |
| `phase` | `publishing`, `waiting_old_txns`, `scanning`, `loading`, `catching_up`, `swapping`, `paused_on_pressure` |
| `rows_done` | Rows read so far, which only rises |
| `rows_total_estimate` | From table statistics, so it is an estimate and can be wrong in either direction |
| `bytes_spilled` | Bytes the build's sort has written to disk |
| `pause_signal` | What the node was short of when the build stood down, empty while it is running |

An empty result means no schema change is running on this node.

`zyron_sys.storage.indexes` reports each index's `state` as `building` or `ready`, and points a building one at its `build_progress_id` in the progress view.

## Under pressure

A build is background work. When the node is short of memory or IO it yields between batches, and when the shortage persists it stands down entirely and `ddl_progress` shows `paused_on_pressure` with what it is waiting on. It resumes on its own when the node has room. Foreground queries are never queued behind one.

## Lake tables

A ZyronLake table's schema change is a commit in its own transaction log rather than a change to heap rows, and its index is a lake artifact versioned with the data. None of the above applies to one: the commit is atomic and there is no build to watch.
