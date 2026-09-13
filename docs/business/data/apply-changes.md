# Apply Changes

`APPLY CHANGES` maintains a target table from a set of changes. The changes are grouped by key and ordered within a key. Under type 1 the last change to each key is what the target ends up holding. Under type 2 every change to a key is a version of its row. Applying the same set twice leaves the target as one application did.

## What this means for you

- The source is a change stream or any relation carrying the change metadata columns, such as `table_changes(...)`.
- `KEYS` names what identifies a row. `SEQUENCE BY` says which of two changes to one key is later.
- One statement upserts, deletes and, under SCD type 2, versions. There is no separate delete pass.
- Read from a stream, the position moves in the same transaction as the target's rows. A failure leaves both where they were.
- Every row reaches the target through the ordinary write path, so constraints hold and triggers fire the way they do for any statement.

## Keys and sequencing

```sql
APPLY CHANGES INTO dim_customer
FROM customer_changes
KEYS (id)
SEQUENCE BY updated_at
```

Within one apply, the changes to a key are ordered by `SEQUENCE BY` and the highest wins. A source that delivers changes out of order produces the same target as one that delivers them in order, because the order they arrive in is not the order they are applied in. Without `SEQUENCE BY`, changes to a key apply in the order they were recorded, which for a change stream is commit order.

The winning change for a key is applied as one write. An insert or an update upserts the row. A delete removes it. A row already in the target that no change names is left alone.

`IGNORE NULL UPDATES` treats a NULL in a change as a column not supplied, so a source that sends only the columns that changed does not blank the rest. `EXCEPT COLUMNS (...)` leaves columns out of what is written, which is where the metadata columns and a source's own bookkeeping columns go.

## Deletes and truncates

A delete in the change set removes its key from the target. A source that marks deletes in a column rather than sending them as deletes says so.

```sql
APPLY CHANGES INTO dim_customer
FROM customer_changes
KEYS (id)
SEQUENCE BY updated_at
APPLY AS DELETE WHEN op = 'D'
APPLY AS TRUNCATE WHEN op = 'T'
```

A change matching `APPLY AS DELETE WHEN` is treated as a delete of its key. A change matching `APPLY AS TRUNCATE WHEN` empties the target before the rest of the set is applied, so a source that sends a full reload as a truncate followed by inserts lands as one.

## Type 1 and type 2

Type 1 keeps one row per key holding the last change. It is the default.

Type 2 keeps one row per version of a key, with three columns the apply maintains.

| Column | What it holds |
| --- | --- |
| `__start_at` | The sequence value the version became current at. |
| `__end_at` | The sequence value the version stopped being current at. NULL while current. |
| `__is_current` | True for the one current version of a key. |

```sql
APPLY CHANGES INTO dim_customer
FROM customer_changes
KEYS (id)
SEQUENCE BY updated_at
STORED AS SCD TYPE 2
TRACK HISTORY EXCEPT (last_seen)
```

A change to a tracked column closes the current version at the change's sequence value and opens a new one. A change that touches untracked columns alone updates the current version in place. `TRACK HISTORY ON (...)` names the columns that open a version, and `TRACK HISTORY EXCEPT (...)` is its complement. A delete closes the current version and opens none. At every instant there is exactly one current row per key.

One change set may carry several changes to a key. They are applied in sequence order, so three changes to a key land three versions, each earlier one closed at the next change's sequence value and the last one current. A change no newer than the newest sequence value the target already holds for the key, over its current and closed rows alike, has already been absorbed and is passed over.

The target must hold the three columns with the types the apply writes. A target that holds one of the names at another type is refused at bind naming the column.

## What is refused at bind

- A `KEYS` column the target does not have.
- A source column whose type cannot be written into the target column of the same name.
- Under type 2, a target holding `__start_at`, `__end_at` or `__is_current` at a type the apply does not write.
- A lake table as the target. Apply into a heap table and load the lake table from it.

Each refusal names the column, before any row is read.

## A medallion, bronze to silver to gold

Bronze holds every change as it arrived. Silver holds one row per key. Gold holds the history of each key.

```sql
ALTER TABLE orders SET (change_data_feed = true);

CREATE CHANGE STREAM orders_to_bronze ON TABLE orders SHOW INITIAL ROWS;
CREATE CHANGE STREAM orders_to_silver ON TABLE orders SHOW INITIAL ROWS;
CREATE CHANGE STREAM orders_to_gold ON TABLE orders SHOW INITIAL ROWS;

CREATE TABLE silver_orders (id BIGINT PRIMARY KEY, customer_id BIGINT, status TEXT, total BIGINT);
CREATE TABLE gold_orders (
    id BIGINT, customer_id BIGINT, status TEXT, total BIGINT,
    __start_at BIGINT, __end_at BIGINT, __is_current BOOLEAN
);

CREATE PIPELINE medallion ON CHANGE DATA FROM orders_to_bronze MIN ROWS 1000 MAX WAIT 1 MINUTES AS (
    STAGE bronze (CONSUME CHANGES FROM orders_to_bronze INTO bronze_orders),
    STAGE silver (APPLY CHANGES INTO silver_orders FROM orders_to_silver
                  KEYS (id) SEQUENCE BY _commit_version),
    STAGE gold (APPLY CHANGES INTO gold_orders FROM orders_to_gold
                KEYS (id) SEQUENCE BY _commit_version
                STORED AS SCD TYPE 2 TRACK HISTORY ON (status, total))
);
```

Each stream holds its own position, so the three stages are three independent consumers of one feed and a failure in one leaves the others where they were. The first run seeds all three targets from the table's existing rows and every run after it applies what changed. `bronze_orders` is created by the first run from the change set, metadata columns included. Every run is recorded in `zyron_sys.cdc.apply_runs` with its counts and duration, and a run that failed fires `cdc_apply_failed`.

## Idempotence

A range of changes wholly inside retention reads the same every time. Applying it twice yields the same target as applying it once, under type 1 because each key's winner is the same, and under type 2 because a change no newer than what the target holds for the key opens nothing. Reading from a stream, the position moving in the apply's own commit is what keeps a retry from applying a range the first attempt already committed.
