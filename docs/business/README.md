# Zyron Documentation

Guides for running Zyron: what the database guarantees on your behalf, which of those behaviors you control, and what to check when one of them stops.

Zyron upgrades itself, moves its own on-disk formats forward, and rotates its own signing keys. The three guides below cover what happens without you, what you can steer, and how to see it happening.

## SQL

- [sql/schema-changes.md](sql/schema-changes.md), what ALTER TABLE and CREATE INDEX do to a live table. Which column and type changes complete without touching a row, which run a rewrite, what a constraint enforces while it is still being checked, and how to watch a change progress.

## Storage

- [storage/format-agility.md](storage/format-agility.md), what happens to files written by an older version of Zyron. Migration policies per format, how far back the running binary can read, and how to watch a migration progress.

## Operations

- [operations/auto-upgrade.md](operations/auto-upgrade.md), how a release is picked up, gated against compatibility, and rolled across the cluster one node at a time, with automatic rollback for a node that fails its health check. Release channels, emergency controls, configuration, observability, and troubleshooting.

## Security

- [security/signature-agility.md](security/signature-agility.md), how signing schemes rotate without invalidating artifacts already issued. Which schemes are accepted, the DDL to rotate one, and how long an outgoing scheme keeps verifying.
