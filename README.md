<div align="center">

# ZyronDB

**An HTAP database engine in Rust - row store for transactions, custom columnar format for analytics, one engine.**

MVCC concurrency · lock-free hot paths · time travel · branching · CDC · native search · in-database ML · enterprise security

</div>

---

ZyronDB is a hybrid transactional/analytical database. It implements its own write-ahead log, buffer pool, B+ tree, MVCC engine, SQL parser, cost-based optimizer, vectorized executor, columnar encoding engine, and wire protocol: no embedded SQL engine, no third-party storage layer, no ORM.

Fresh writes land in an MVCC row heap tuned for OLTP. A background thread compacts committed rows into a custom column format (`.zyr`) with per-column encoding, and analytical queries run directly on the encoded data with predicate pushdown and late materialization. The same SQL, the same connection, both workloads.

> **Status:** active development. The single-node engine is feature-complete through data lifecycle management, and the ZyronLake table format runs beside it with cross-format federation. Next up is distribution: Raft consensus, multi-region, and sharding.

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
- **Mesh, not monolith.** A Zyron node can host a database, a ZyronLake, both, or neither (embedded in an app). Nodes peer over the wire and a single `SELECT` reaches across the mesh, joining a local heap table to a remote publication or a shared lake with no external gateway or middleware in front.
- **Shared-storage table format.** ZyronLake stores immutable `.zyr` files against an append-only transaction log versioned per commit, with `File::create_new` as the optimistic-concurrency primitive and periodic manifest checkpoints so readers open one file instead of replaying history. Branches, time travel, secondary indexes, Z-order clustering, constraint enforcement, and change feed all live in the format.
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
        ST[zyron-storage<br/>heap · B+tree · .zyr · MVCC] ~~~ LK[zyron-lake<br/>versioned .zyr · log · manifest] ~~~ BUF[zyron-buffer<br/>clock-sweep pool] ~~~ WAL[zyron-wal<br/>ring buffer · group commit]
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
| ZyronLake (`.zyr` + log) | Object store or shared filesystem, versioned per commit, manifest-checkpointed | Analytical tables on shared storage; branches, time travel, cross-engine reads |
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
<tr><td valign="top" width="50%">

**ZyronLake table format**

- Immutable `.zyr` data files against an append-only transaction log, one numbered version file per commit
- `File::create_new` as the optimistic-concurrency primitive; periodic manifest checkpoints collapse the log
- Copy-on-write branches with merge
- Time travel to any committed version
- Secondary indexes (also versioned as lake artifacts, safe against staleness)
- Bloom filters, zone maps, and a struct-of-arrays prune index (94-100% file skip rates in practice)
- Z-order and Hilbert space-filling-curve clustering with background maintenance
- Unique and foreign-key constraint enforcement with a clustered fast path
- Change feed between two versions
- `OPTIMIZE`, `VACUUM`, `FOLLOW`, `REPAIR` operations
- Runs on local disk, shared filesystem, or object storage

</td><td valign="top" width="50%">

**Federation & mesh**

- Zyron nodes peer over the wire and address each other's data directly
- One node can host a database, a ZyronLake, both, or run embedded in an application
- Publications and subscriptions between instances with exactly-once wire-level push
- Foreign scans across the mesh: a single query can join local heap tables to remote publications or to a shared lake
- HybridScan bridges row heap and `.zyr` segments (local or remote) in one operator
- Cross-format federation: heap and lake tables are first-class in the same SQL, no separate query layer
- Dynamic REST / WebSocket / SSE endpoints (`CREATE ENDPOINT`) surface SQL directly to browsers and apps
- OAuth 2.0, AWS IAM STS, Kubernetes SA tokens, Vault / AWS Secrets Manager / GCP Secret Manager / Azure Key Vault for remote credentials
- Connection routing over host lists with role hints, DNS SRV, health checks, and automatic failover
- mTLS with SPKI pinning and labeled Prometheus families per subscriber, publication, and TLS direction

</td></tr>
</table>

## Performance

Two views: what a client actually gets from a running server, and the raw subsystem numbers underneath it. Every figure is extracted from the committed result files in [`benchmarks/`](benchmarks/); the tables and charts are regenerated from them by [`scripts/gen_bench_charts.py`](scripts/gen_bench_charts.py).

<!-- BENCH:START -->
_Release build, single machine: Intel(R) Core(TM) Ultra 7 270K Plus, 24 cores, 31.4 GB RAM, windows/x86_64. Regenerated from `benchmarks/` by `scripts/gen_bench_charts.py`._

### End-to-end

What a client sees from a running server over the wire protocol, from cold start to shutdown:

| Lifecycle / workload | Result |
|----------------------|--------|
| Cold boot to accepting queries | 40 ms |
| First `ReadyForQuery` | 0.60 ms |
| Schema DDL bootstrap | 7.2 ms |
| Seed insert | 251K rows/sec |
| OLTP, 1 client | 12.0K tps, p99 263 us |
| OLTP, 4 clients | 33.4K tps, p99 360 us |
| OLTP, 16 clients | 65.1K tps, p99 493 us |
| OLTP, 64 clients | 60.5K tps, p99 1824 us |
| OLTP, 256 clients | 55.2K tps, p99 8487 us |
| Analytical query (median) | 1.30 ms |
| Graceful shutdown | 53 ms |

![OLTP throughput vs. concurrent clients (thousand tps, higher is better)](benchmarks/charts/oltp_throughput.svg)

### Engine internals

Raw subsystem throughput and hot-path latency under microbenchmark:

![Throughput (million ops/sec, higher is better)](benchmarks/charts/engine_throughput.svg)

![Hot-path latency (nanoseconds, lower is better)](benchmarks/charts/hot_path_latency.svg)

### Row heap vs ZyronLake

Same workload, same rows, run against both formats. Blue = Row heap, purple = ZyronLake; the shorter bar wins on wall-clock. Write-heavy trade-offs where the Row heap wins (bulk load, trickle load, indexed point lookup) are in the table below so the picture is honest, not cherry-picked.

![Cross-format wall-clock, Row heap vs ZyronLake](benchmarks/charts/cross_format.svg)

A few more numbers not shown in the charts above:

| Subsystem | Metric | Result |
|-----------|--------|--------|
| MVCC | GC sweep | ~1.8B tuples/sec |
| Columnar | .zyr scan throughput | ~3.0 GB/sec |
| Columnar | Compaction pipeline | ~3.4M rows/sec |
| Columnar | HybridScan overhead vs heap-only | ~-2.5% |
| Columnar | Metadata-aggregate pruning speedup | ~3.1x |
| Temporal | Picosecond timestamp decode | ~310M rows/sec |
| Versioning | Time-travel scan overhead | ~20% |
| Wire | QUIC PostgreSQL handshake | ~5 us |
| Transactions | Durable commit floor (device write) | ~80.7 us |
| Lake | Scan throughput | ~154M rows/sec |
| Lake | Point probe through index | ~199 us |
| Lake | Zone-map row rejection | ~100% |
| Transactions | Durable group-commit peak | ~703K txn/sec |
| Transactions | Group-commit amplification (c=1 to c=512) | ~70.9x |
| Cross-format | Point-lookup bytes read, heap vs lake | ~89x less I/O for lake |
| Cross-format | Point lookup with a heap B+tree index, lake vs heap | ~2.0x (heap wins indexed points) |
| Cross-format | Bulk load to queryable, lake vs heap | ~10.8x (heap wins large batches) |
| Cross-format | Trickle load to queryable, lake vs heap | ~32.1x (heap wins tiny commits) |

32 benchmark suites cover storage, executor, optimizer, encoding, wire, search, analytics, CDC, versioning, transactions, temporal, columnar, lake, cross-format, types, lifecycle, gateway, Zyron-to-Zyron, and end-to-end. Each run writes a timestamped JSON/TXT pair under `benchmarks/<suite>/`.
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
| ZyronLake table format | Immutable versioned `.zyr` on a transaction log; branches, time travel, secondary indexes, clustering, constraint enforcement, change feed, cross-format federation | ✅ Complete |
| Distribution | Raft consensus, sharding, multi-region, key management, observability, migration & connectors | 🔨 In progress |
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
