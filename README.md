<div align="center">

# ZyronDB

**An HTAP database engine in Rust - row store for transactions, custom columnar format for analytics, one engine.**

MVCC concurrency · lock-free hot paths · time travel · branching · CDC · native search · in-database ML · enterprise security

</div>

---

ZyronDB is a hybrid transactional/analytical database. It implements its own write-ahead log, buffer pool, B+ tree, MVCC engine, SQL parser, cost-based optimizer, vectorized executor, columnar encoding engine, and wire protocol: no embedded SQL engine, no third-party storage layer, no ORM.

Fresh writes land in an MVCC row heap tuned for OLTP. A background thread compacts committed rows into a custom column format (`.zyr`) with per-column encoding, and analytical queries run directly on the encoded data with predicate pushdown and late materialization. The same SQL, the same connection, both workloads.

> **Status:** active development. The single-node engine is feature-complete through data lifecycle management and benchmarked end to end. Next up is the ZyronLake table format and distribution: multi-region, Raft consensus, and sharding.

## Table of Contents

- [ZyronDB](#zyrondb)
  - [Table of Contents](#table-of-contents)
  - [Highlights](#highlights)
  - [Architecture](#architecture)
    - [Storage tiers](#storage-tiers)
  - [Capabilities](#capabilities)
  - [Performance](#performance)
    - [End-to-end](#end-to-end)
    - [Engine internals](#engine-internals)
  - [Getting Started](#getting-started)
  - [SQL Highlights](#sql-highlights)
  - [Project Layout](#project-layout)
  - [Roadmap](#roadmap)
  - [Development](#development)
  - [License](#license)

## Highlights

- **One engine, both workloads.** Row heap for transactional writes, `.zyr` columnar format for analytical scans, with automatic background compaction between them.
- **Lock-free hot paths.** LSN assignment, the WAL ring buffer, MVCC visibility checks, the buffer pool, and the B+ tree avoid mutexes on the query and write path entirely. Locks exist only on single-owner background threads.
- **Full durability, no per-page fsync tax.** Every commit fsyncs the WAL through group commit before acknowledging the client. Dirty heap and index pages are written by a background writer and then fsynced at explicit durability barriers, so an acknowledged transaction survives a power loss without paying a per-page fsync on the OLTP hot path.
- **Query-on-encoded.** Dictionary, RLE, bit-pack, and FastLanes columns are filtered without being decoded; only rows that survive predicates are materialized.
- **Time as a first-class dimension.** `AS OF TIMESTAMP` and `VERSION AS OF` queries, copy-on-write branches, slowly changing dimensions, and bitemporal tables are part of the engine, with picosecond-resolution timestamps and hybrid logical clocks underneath.
- **Security built into the planner.** Three-state privileges, row-level security, column masking, ABAC, and mandatory access control are evaluated inside query planning, not in an application tier.
- **Drop-in client compatibility.** Implements the PostgreSQL wire protocol v3 over both TCP and QUIC, so existing drivers and tooling connect unchanged.

## Architecture

ZyronDB is a Cargo workspace. Each crate owns one layer and depends only on the layers beneath it.

```mermaid
%%{init: {'theme':'base','themeVariables':{'background':'transparent','primaryColor':'#0d1117','primaryBorderColor':'#1f6feb','primaryTextColor':'#e6edf3','secondaryColor':'#0d1117','tertiaryColor':'#0d1117','lineColor':'#58a6ff','clusterBkg':'transparent','clusterBorder':'#1f6feb','titleColor':'#e6edf3','fontFamily':'ui-sans-serif, system-ui, sans-serif'}}}%%
flowchart TB
    subgraph CLIENTS [Clients]
        direction LR
        C1[psql / drivers] ~~~ C2[zyrondb-cli] ~~~ C3[zyrondb-ctl]
    end

    subgraph PROTO [Connectivity]
        direction LR
        W[zyron-wire · PostgreSQL v3 · TCP + QUIC · TLS 1.3]
    end

    subgraph ORCH [Orchestration]
        direction LR
        S[zyron-server · sessions · background workers · backup]
    end

    subgraph QUERY [Query Path]
        direction LR
        P[zyron-parser] ~~~ PL[zyron-planner<br/>cost-based optimizer] ~~~ EX[zyron-executor<br/>vectorized · morsel-parallel] ~~~ CAT[zyron-catalog]
    end

    subgraph FEAT [Native Subsystems]
        direction LR
        AUTH[zyron-auth] ~~~ VER[zyron-versioning] ~~~ CDC[zyron-cdc] ~~~ SRCH[zyron-search] ~~~ AN[zyron-analytics]
        PIPE[zyron-pipeline] ~~~ LIFE[zyron-lifecycle] ~~~ STR[zyron-streaming] ~~~ TYP[zyron-types]
    end

    subgraph STORAGE [Storage Engine]
        direction LR
        ST[zyron-storage<br/>heap · B+tree · .zyr · MVCC] ~~~ BUF[zyron-buffer<br/>clock-sweep pool] ~~~ WAL[zyron-wal<br/>ring buffer · group commit]
    end

    COM[zyron-common · errors · pages · hashing · PRNG]

    CLIENTS --> PROTO --> ORCH --> QUERY --> FEAT --> STORAGE --> COM
```

### Storage tiers

| Tier | Backing | Purpose |
|------|---------|---------|
| B+ tree index | Resident in RAM, persisted via WAL + checkpoint | Point lookups, range scans |
| Row heap (OLTP) | Buffer pool with clock-sweep eviction | Recent and transactional rows |
| Columnar (`.zyr`) | Disk, hot segments cached in the buffer pool | Encoded analytical scans |
| Write-ahead log | Append-only, group commit, fsync on commit | Durability and crash recovery |

## Capabilities

<table>
<tr><td valign="top" width="33%">

**Engine & SQL**

- MVCC, snapshot isolation, savepoints
- Lock-free WAL, group commit, crash recovery
- Unconditional durability: WAL fsync on every commit, background page writes with explicit fsync barriers
- `.zyr` columnar format, query-on-encoded
- HybridScan over row heap + `.zyr` segments in one operator
- Bloom filters and zone maps on segment metadata
- Cost-based optimizer: DP join reorder, predicate/projection pushdown, subquery decorrelation
- Vectorized, morsel-parallel execution
- CTEs, window functions, `MERGE`, `QUALIFY`, `ROLLUP`/`CUBE`/`GROUPING SETS`
- Time-series `GAP FILL` operator
- Prepared statements, cursors, `COPY`
- Wire protocol over TCP and QUIC

**Versioning & change**

- `AS OF TIMESTAMP` / `VERSION AS OF`
- Copy-on-write branches with merge conflict resolution
- SCD types, system/application/bitemporal time
- Picosecond-resolution timestamps with hybrid logical clocks
- Arrow `ps`->`ns` export for downstream tooling
- Diff and patch between versions
- CDC: change feeds, replication slots, Debezium / Avro / Wal2Json / native decoders, publications, snapshots

</td><td valign="top" width="33%">

**Security & governance**

- Three-state privileges (GRANT / DENY / unset); DENY always wins, column-level overrides table-level
- Temporal grants with recurring time windows
- Row-level security and row ownership
- Attribute-based access control (ABAC)
- Mandatory access control and security labels
- Column data masking (email, phone, SSN, card, hash, partial, custom)
- Data classification with clearance enforcement
- Memory-hard password KDF, API keys, JWT, TOTP, WebAuthn, OAuth2
- AWS STS / Secrets Manager and Kubernetes auth
- Break-glass emergency access with audit trail
- Two-person rule for sensitive DDL
- Privilege analytics, delegation lineage, cascade revoke
- Brute-force throttling, session binding (IP lock, query limits)

</td><td valign="top" width="33%">

**Search & analytics**

- Full-text: inverted index, BM25, highlighting, synonyms, autocomplete
- Vector search: HNSW / IVF with self-tuning
- Graph indexing and traversal
- Cohort, funnel, period-over-period
- Single-pass profiler, outlier (IQR / MAD / Z-score), correlation (Pearson / Spearman / Kendall)
- In-database training: linear, logistic, trees, random forest, GBM, k-means, KNN
- Forecasting (ARIMA, Holt-Winters, FFT-ACF), causal inference
- Feature store with point-in-time-correct joins and lineage

**Pipelines & lifecycle**

- Declarative pipelines, triggers, UDFs, stored procedures
- Materialized views with refresh strategies, SLAs, advisor
- Data quality checks and drift detection
- Retention / TTL, tiered storage, archival
- WORM and time-bounded retention locks (immutable to admins)
- Soft delete, recycle bin, `UNDROP`
- GDPR erasure, DSAR export, legal hold, crypto-shred
- `compliance_profile` presets (GDPR / HIPAA / SOX)
- Data-residency enforcement on tier moves
- `DRY RUN` previews for archive / erasure / retention
- Lock-free cleanup governor (rate + time-window)
- Tamper-evident audit chain

</td></tr>
</table>

## Performance

Two views: what a client actually gets from a running server, and the raw subsystem numbers underneath it. Every figure is extracted from the committed result files in [`benchmarks/`](benchmarks/); the tables and charts are regenerated from them by [`scripts/gen_bench_charts.py`](scripts/gen_bench_charts.py).

<!-- BENCH:START -->
_Release build, single machine: Intel(R) Core(TM) i9-14900KS, 32 cores, 31.8 GB RAM, windows/x86_64. Regenerated from `benchmarks/` by `scripts/gen_bench_charts.py`._

### End-to-end

What a client sees from a running server over the wire protocol, from cold start to shutdown:

| Lifecycle / workload | Result |
|----------------------|--------|
| Cold boot to accepting queries | 30 ms |
| First `ReadyForQuery` | 0.39 ms |
| Schema DDL bootstrap | 3.9 ms |
| Seed insert | 470K rows/sec |
| OLTP, 1 client | 15.9K tps, p99 191 us |
| OLTP, 4 clients | 47.7K tps, p99 247 us |
| OLTP, 16 clients | 75.8K tps, p99 471 us |
| OLTP, 64 clients | 77.1K tps, p99 1489 us |
| OLTP, 256 clients | 72.3K tps, p99 6458 us |
| Analytical query (median) | 2.18 ms |
| Graceful shutdown | 55 ms |

```mermaid
%%{init: {'theme':'base','themeVariables':{'xyChart':{'backgroundColor':'transparent','titleColor':'#e6edf3','xAxisLabelColor':'#e6edf3','xAxisTitleColor':'#e6edf3','xAxisTickColor':'#1f6feb','xAxisLineColor':'#1f6feb','yAxisLabelColor':'#e6edf3','yAxisTitleColor':'#e6edf3','yAxisTickColor':'#1f6feb','yAxisLineColor':'#1f6feb','plotColorPalette':'#1f6feb'}},'xyChart':{'width':750,'height':360}}}%%
xychart-beta
    title "OLTP throughput vs. concurrent clients (thousand tps, higher is better)"
    x-axis ["1 client", "4 clients", "16 clients", "64 clients", "256 clients"]
    y-axis "K tps" 0 --> 88
    bar [15.9, 47.7, 75.8, 77.1, 72.3]
```

### Engine internals

Raw subsystem throughput and hot-path latency under microbenchmark:

```mermaid
%%{init: {'theme':'base','themeVariables':{'xyChart':{'backgroundColor':'transparent','titleColor':'#e6edf3','xAxisLabelColor':'#e6edf3','xAxisTitleColor':'#e6edf3','xAxisTickColor':'#1f6feb','xAxisLineColor':'#1f6feb','yAxisLabelColor':'#e6edf3','yAxisTitleColor':'#e6edf3','yAxisTickColor':'#1f6feb','yAxisLineColor':'#1f6feb','plotColorPalette':'#1f6feb'}},'xyChart':{'width':1350,'height':360}}}%%
xychart-beta
    title "Throughput (million ops/sec, higher is better)"
    x-axis ["B+tree insert", "B+tree delete", "Hash join build", "SCD-2 merge", "COPY FROM", "Row serialize", "SQL parse", "TTL purge", "Archive"]
    y-axis "M ops/sec" 0 --> 133
    bar [39.2, 46.3, 116.3, 45.8, 23.2, 18.2, 2.2, 15.2, 83.8]
```

```mermaid
%%{init: {'theme':'base','themeVariables':{'xyChart':{'backgroundColor':'transparent','titleColor':'#e6edf3','xAxisLabelColor':'#e6edf3','xAxisTitleColor':'#e6edf3','xAxisTickColor':'#1f6feb','xAxisLineColor':'#1f6feb','yAxisLabelColor':'#e6edf3','yAxisTitleColor':'#e6edf3','yAxisTickColor':'#1f6feb','yAxisLineColor':'#1f6feb','plotColorPalette':'#1f6feb'}},'xyChart':{'width':1200,'height':360}}}%%
xychart-beta
    title "Hot-path latency (nanoseconds, lower is better)"
    x-axis ["MVCC visible", "Version lookup", "B+tree lookup", "Branch resolve", "Row lock", "Legal-hold check", "Bloom probe", "Selectivity est"]
    y-axis "ns/op" 0 --> 136
    bar [1.4, 10.8, 14.1, 39.0, 76.1, 6.0, 5.9, 118.5]
```

A few more numbers not shown in the charts above:

| Subsystem | Metric | Result |
|-----------|--------|--------|
| MVCC | GC sweep | ~1.9B tuples/sec |
| Columnar | .zyr scan throughput | ~3.0 GB/sec |
| Columnar | Compaction pipeline | ~4.0M rows/sec |
| Columnar | HybridScan overhead vs heap-only | ~-4.7% |
| Columnar | Metadata-aggregate pruning speedup | ~14.2x |
| Temporal | Picosecond timestamp decode | ~528M rows/sec |
| Versioning | Time-travel scan overhead | ~17% |
| Wire | QUIC PostgreSQL handshake | ~4 us |

29 benchmark suites cover storage, executor, optimizer, encoding, wire, search, analytics, CDC, versioning, transactions, temporal, columnar, types, lifecycle, gateway, Zyron-to-Zyron, and end-to-end. Each run writes a timestamped JSON/TXT pair under `benchmarks/<suite>/`.
<!-- BENCH:END -->

## Getting Started

**Prerequisites.** The Rust nightly toolchain pinned in `rust-toolchain.toml` (`rustup` selects it automatically) and a C toolchain for the TLS dependency.

```bash
# Build
cargo build --release

# Run the server
cargo zyron -- --data-dir ./data --port 5432
#   alias for: cargo run --release --bin zyrondb-server --

# Connect with the bundled client
cargo cli -- --host localhost --port 5432

# Or any PostgreSQL client
psql -h localhost -p 5432 -U postgres

# Administer
cargo run --release --bin zyrondb-ctl -- status
cargo run --release --bin zyrondb-ctl -- backup --out ./backup
```

Server flags include `--config`, `--host`, `--log-level`, `--foreground`, `--single-user`, and `--skip-recovery`.

## SQL Highlights

Standard SQL, plus native extensions that go beyond it:

```sql
-- Read a table as it was at a point in time
SELECT * FROM orders AS OF TIMESTAMP '2026-05-06 09:00:00';

-- Branch the database, experiment in isolation, then merge or drop
CREATE BRANCH experiment FROM main;
USE BRANCH experiment;

-- Native vector search
CREATE VECTOR INDEX ON docs (embedding) WITH (metric = 'cosine');
SELECT id FROM docs ORDER BY embedding <=> $1 LIMIT 10;

-- BM25-scored full-text search
CREATE FULLTEXT INDEX ON articles (body);
SELECT id FROM articles
WHERE MATCH(body) AGAINST ('rust database' IN NATURAL LANGUAGE MODE);

-- Feature store and in-database model training
CREATE FEATURE GROUP customer_features AS SELECT ... ;
CREATE MODEL churn AS TRAIN logistic ON customer_features;

-- Security expressed in DDL
ALTER TABLE patients ADD MASKING POLICY ssn USING mask_ssn();
GRANT SELECT ON revenue TO analyst VALID FROM '2026-01-01' UNTIL '2026-12-31';

-- Lifecycle governance
ALTER TABLE events SET TTL '90 days';
ARCHIVE TABLE old_events WHERE created < '2024-01-01' TO 's3://archive/events';
```

## Project Layout

<details>
<summary><b>Workspace crates and binaries</b></summary>

```
crates/
  zyron-common        errors, page constants, FX hash, PRNG, config
  zyron-wal           ring-buffer WAL, LSN sequencer, group commit, recovery
  zyron-buffer        clock-sweep buffer pool, background writer
  zyron-storage       heap, B+ tree, .zyr columnar, encoding, MVCC txn module
  zyron-catalog       databases/schemas/tables/indexes, stats, WAL-logged DDL
  zyron-parser        recursive descent + Pratt SQL parser, typed AST
  zyron-planner       binder, logical/physical plans, cost model, optimizer
  zyron-executor      vectorized, morsel-parallel operators
  zyron-wire          PostgreSQL v3 over TCP + QUIC, COPY, auth, pooling
  zyron-server        orchestration, sessions, background workers, backup
  zyron-auth          RBAC/ABAC, masking, RLS, governance, external auth
  zyron-versioning    time travel, branching, SCD, bitemporal, diff/patch
  zyron-cdc           change feeds, replication slots, logical decoders
  zyron-pipeline      pipelines, triggers, UDFs, materialized views, SLAs
  zyron-search        full-text (BM25), vector (HNSW/IVF), graph
  zyron-analytics     grouping, cohort, funnel, profiling, forecasting, ML
  zyron-types         native data types and operations
  zyron-lifecycle     retention, tiered storage, archival, GDPR, audit chain
  zyron-streaming     windowing, exactly-once, stream joins, backpressure
  zyron-bench-harness shared benchmark harness and result output

binaries/
  zyrondb-server      database server entry point
  zyrondb-cli         psql-like interactive client
  zyrondb-ctl         admin tool: status, backup, restore, vacuum, compact

benchmarks/           timestamped JSON/TXT results, one folder per suite
scripts/              tooling, including the benchmark chart generator
```

</details>

## Roadmap

Each area ships with an optimization review and a validation checkpoint with hard performance budgets before the next begins.

| Area | Scope | State |
|------|-------|-------|
| Storage foundation | WAL, buffer pool, heap, B+ tree, MVCC, encoding engine, parser, catalog, planner, executor, wire, server | ✅ Complete |
| Security & optimization | Authentication, RBAC/ABAC, row/column security, cost model, configuration | ✅ Complete |
| Data operations | Versioning & time travel, CDC, pipelines/triggers/UDFs, streaming, server integration | ✅ Complete |
| Native features | Full-text search, vector & graph search, native data types, utility operations | ✅ Complete |
| Analytics & lifecycle | Analytics engine, feature store & ML, data lifecycle management | ✅ Complete |
| Enterprise & distribution | ZyronLake table format, Raft consensus, sharding, key management, observability, migration & connectors | 🔨 In progress |
| Innovation & serverless | Serverless compute/storage split, innovative query features, client SDKs | ⏳ Planned |

## Development

```bash
# Per-crate tests
cargo test -p zyron-storage

# Whole workspace
cargo test --workspace

# A benchmark suite (release, serialized; results land in benchmarks/<suite>/)
cargo test -p zyron-search --test search_bench --release -- --nocapture

# Regenerate the README performance charts from the latest benchmark files
python scripts/gen_bench_charts.py

# Lint and format
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

## License

Copyright (c) 2025 DataDalton. **All rights reserved.** Proprietary software - no part may be reproduced, distributed, modified, or used in any form without prior written permission of the copyright holder. See [LICENSE](LICENSE) for the full terms.
