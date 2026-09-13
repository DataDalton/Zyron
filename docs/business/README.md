# Zyron Documentation

Guides for running Zyron: what the database guarantees on your behalf, which of those behaviors you control, and what to check when one of them stops.

Zyron upgrades itself, moves its own on-disk formats forward, and rotates its own signing keys. The three guides below cover what happens without you, what you can steer, and how to see it happening.

## SQL

- [sql/reference/](sql/reference/README.md), one page per statement and construct: what it does, what each clause does and what it defaults to, what is refused, and a runnable example. Written from the parser's own grammar registry by `zyron-ctl docs generate`, so nothing under it is edited by hand.
- [sql/guides/schema-changes.md](sql/guides/schema-changes.md), what ALTER TABLE and CREATE INDEX do to a live table. Which column and type changes complete without touching a row, which run a rewrite, what a constraint enforces while it is still being checked, and how to watch a change progress.
- [sql/guides/arrays-and-variant.md](sql/guides/arrays-and-variant.md), turning an array column or a VARIANT document into rows and back. UNNEST, FLATTEN, the array functions and their lambdas, and when a statement has to say LATERAL.
- [sql/guides/temporary-tables.md](sql/guides/temporary-tables.md), a table that belongs to one session on one node. Where its files live, what a commit does to it, what cannot reference it, the per-session limits, and where to watch what sessions are holding.
- [sql/guides/pivot.md](sql/guides/pivot.md), turning values into columns and columns into rows. PIVOT, UNPIVOT, why the value list is written out rather than queried, and the plan each one runs as.
- [sql/guides/asof-join.md](sql/guides/asof-join.md), joining each row to the nearest row along an ordered column. The match condition and its direction, the tolerance that bounds how far a match reaches, and when a side's sort is elided.

## Data

- [data/change-streams.md](data/change-streams.md), what a table's change data feed records, how a change stream holds a position that moves in the reading transaction's own commit, why that makes reprocessing safe, the batch shape through `table_changes` and the continuous shape through a stream, pipelines that run on change data, and when a stream is stale.
- [data/apply-changes.md](data/apply-changes.md), maintaining a target from a set of changes. Keys and sequencing, type 1 and type 2 with tracked history, sources that deliver out of order, deletes and truncates, what is refused at bind, and a medallion worked from bronze to silver to gold.

## Storage

- [storage/format-agility.md](storage/format-agility.md), what happens to files written by an older version of Zyron. Migration policies per format, how far back the running binary can read, and how to watch a migration progress.

## Operations

- [operations/auto-upgrade.md](operations/auto-upgrade.md), how a release is picked up, gated against compatibility, and rolled across the cluster one node at a time, with automatic rollback for a node that fails its health check. Release channels, emergency controls, configuration, observability, and troubleshooting.

## Security

- [security/signature-agility.md](security/signature-agility.md), how signing schemes rotate without invalidating artifacts already issued. Which schemes are accepted, the DDL to rotate one, and how long an outgoing scheme keeps verifying.
