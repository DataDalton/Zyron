<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/zyronBannerCobalt.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/brand/zyronBannerLightCobalt.svg">
  <img src="assets/brand/zyronBannerCobalt.svg" alt="Zyron" width="640">
</picture>

**A unified data platform - HTAP database, shared-storage lake, and federated mesh in one runtime, reachable from any client language.**

MVCC concurrency · lock-free hot paths · shared-storage table format · full-mesh federation · time travel · branching · CDC · native search · in-database ML · enterprise security

</div>

---

Zyron runs three surfaces off one codebase: an **HTAP database** with an MVCC row heap and a custom `.zyr` columnar format, **ZyronLake** as a shared-storage table format on an append-only transaction log with branches, time travel, and secondary indexes, and a **mesh** where any node can host a database, a lake, both, or embed in an application, and nodes peer over the wire so a single query reaches across them. All three are reachable from any client language over the same wire protocol.

Under the hood, Rust throughout, with an in-repo write-ahead log, buffer pool, B+ tree, MVCC engine, SQL parser, cost-based optimizer, vectorized executor, columnar encoding engine, lake transaction log, and wire protocol: no embedded SQL engine, no third-party storage layer, no ORM.

Fresh writes land in the MVCC row heap tuned for OLTP. A background thread compacts committed rows into `.zyr` segments with per-column encoding, and analytical queries run directly on the encoded data with predicate pushdown and late materialization. Heap and lake tables are first-class in the same SQL, joined by the same HybridScan operator, whether they live on the local node or across the mesh.

> **Status:** active development. The single-node engine is feature-complete through data lifecycle management, the ZyronLake table format runs beside it with cross-format federation, Raft consensus replicates every statement in the grammar across a group under one fsync per quorum ack, a format + signature agility substrate carries a versioned envelope on every persistent file and drives automatic migration and rolling upgrade, heap schema changes and index builds run online against live writers, and change streams give every table a transactionally consumed change log with declarative SCD apply. Next up is transport security with hybrid post-quantum key exchange on every connection, then sharding and the enterprise stack around it.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Highlights](#highlights)
- [Architecture](#architecture)
  - [Mesh topology](#mesh-topology)
  - [Replication flow](#replication-flow)
  - [Storage tiers](#storage-tiers)
- [Capabilities](#capabilities)
- [Performance](#performance)
  - [End-to-end](#end-to-end)
  - [Engine internals](#engine-internals)
  - [Row heap vs ZyronLake](#row-heap-vs-zyronlake)
- [Getting Started](#getting-started)
  - [Download](#download)
  - [Build from source](#build-from-source)
- [SQL Highlights](#sql-highlights)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [Development](#development)
- [License](#license)

## Highlights

- **One engine, both workloads.** Row heap for transactional writes, `.zyr` columnar format for analytical scans, with automatic background compaction between them.
- **Mesh, not monolith.** A Zyron node can host a database, a ZyronLake, both, or neither (embedded in an app). Nodes peer over the wire and a single `SELECT` reaches across the mesh, joining a local heap table to a remote publication or a shared lake with no external gateway or middleware in front. Database nodes can also join a Raft group where writes commit through quorum and any node serves linearizable reads.
- **Quorum-committed writes.** Every statement in the grammar rides the same replication channel, proven by a conformance suite that runs each one against a live three-node group over the wire. The leader group-commits under one fsync per quorum ack, followers apply committed entries through the same operator path the leader used, and follower reads take a ReadIndex round trip so they stay linearizable. Cluster settings ride the same log behind a version gate so a rolling upgrade never applies a setting a member cannot read.
- **Online schema changes.** A schema epoch in the tuple slot lets `ALTER TABLE` add, drop, or retype a column without rewriting a row, index builds publish, wait for in-flight writers, scan, and flip without blocking the table, and an incompatible type change runs as a shadow rewrite behind live traffic.
- **Change streams.** Every table carries a change log that a consumer reads and advances inside its own transaction, so the position moves only when the consumer commits. `APPLY CHANGES` lands a stream into a target as SCD type 1 or type 2, pipelines trigger `ON CHANGE DATA`, and on a group the applier writes the feed at the raft index so the consumed position names the same record on every member.
- **Shared-storage table format.** ZyronLake stores immutable `.zyr` files against an append-only transaction log versioned per commit, with `File::create_new` as the optimistic-concurrency primitive and periodic manifest checkpoints so readers open one file instead of replaying history. Branches, time travel, secondary indexes, Z-order clustering, constraint enforcement, and change feed all live in the format.
- **Lock-free hot paths.** LSN assignment, the WAL ring buffer, MVCC visibility checks, the buffer pool, and the B+ tree avoid mutexes on the query and write path entirely. Locks exist only on single-owner background threads.
- **Full durability, no per-page fsync tax.** Every commit fsyncs the WAL through group commit before acknowledging the client. Dirty heap and index pages are written by a background writer and then fsynced at explicit durability barriers, so an acknowledged transaction survives a power loss without paying a per-page fsync on the OLTP hot path.
- **Query-on-encoded.** Dictionary, RLE, bit-pack, and FastLanes columns are filtered without being decoded, and only rows that survive predicates are materialized.
- **Time as a first-class dimension.** `AS OF TIMESTAMP` and `VERSION AS OF` queries, copy-on-write branches, slowly changing dimensions, and bitemporal tables are part of the engine, with picosecond-resolution timestamps and hybrid logical clocks underneath.
- **Security built into the planner.** Three-state privileges, row-level security, column masking, ABAC, and mandatory access control are evaluated inside query planning, not in an application tier.
- **Drop-in client compatibility.** Implements the PostgreSQL wire protocol v3 over both TCP and QUIC, so existing drivers and tooling connect unchanged.
- **Format and signature agility.** Every persistent file carries a versioned envelope, every signed artifact declares its scheme, and every reader dispatches on both. Upgrades ship a migration function per format, catalog rows migrate through a schema evolution registry, user SQL objects rewrite through an AST transformer, and a health-baselined rolling upgrade lands new binaries across a Raft group with automatic rollback. Retired readers are refused by CI past their retirement date so old code never lingers.

## Architecture

Zyron is a Cargo workspace. Each crate owns one layer and depends only on the layers beneath it.

![Zyron crate layering: clients, connectivity, orchestration, distribution, query path, native subsystems, storage engine, and the common foundation, each layer depending only on the ones beneath it](assets/diagrams/architecture.svg)

### Mesh topology

The diagram above is one node's internals. In a deployment, Zyron nodes peer over the wire. Each node can host a database only, a lake reader only, both, or be embedded inside an application. Any client can hit any node, and a single query reaches across the mesh. Lake-holding nodes share the same object-store backing. Database nodes can also be joined into a Raft group where writes commit through quorum and any node serves linearizable reads.

![Zyron mesh deployment topology: clients above, four peered nodes in the middle with dashed peer edges, and a shared object store below](assets/diagrams/mesh.svg)

### Replication flow

On a grouped node a write is sealed into a changeset and proposed to the Raft log. The leader pipelines the entry to every follower, each member fsyncs once, and the leader answers the client after a quorum has acknowledged. Every member then runs the same applier over the committed entry and writes the change feed at the raft index, so a consumed position names the same record everywhere. A read on a follower takes a ReadIndex round trip to the leader and waits for local apply to reach that index before answering.

![Zyron replication sequence: client writes to the leader, AppendEntries fan out to followers, each member fsyncs, acks return, quorum advances the commit index, every member runs the applier, the client is acknowledged, then a follower read takes a ReadIndex round trip before answering](assets/diagrams/replication.svg)

### Storage tiers

A write lands in the write-ahead log, is acknowledged after that one fsync, and lives in the MVCC row heap. A background thread compacts committed rows into `.zyr` segments, and a lake table publishes them as a version on shared storage. Dirty heap and index pages reach disk through a background writer, never on the commit path. HybridScan reads all three tiers in one operator.

![Zyron write path: client to write-ahead log to row heap to .zyr segments to ZyronLake, with a Raft log note beside the WAL and HybridScan reading heap, .zyr, and lake in one operator](assets/diagrams/write_path.svg)

| Tier | Backing | Purpose |
| ------ | --------- | --------- |
| B+ tree index | Resident in RAM, persisted via WAL + checkpoint | Point lookups, range scans |
| Row heap (OLTP) | Buffer pool with clock-sweep eviction | Recent and transactional rows |
| Columnar (`.zyr`) | Disk, hot segments cached in the buffer pool | Encoded analytical scans |
| ZyronLake (`.zyr` + log) | Object store or shared filesystem, versioned per commit, manifest-checkpointed | Analytical tables on shared storage, branches, time travel, cross-engine reads |
| Write-ahead log | Append-only, group commit, fsync on commit | Durability and crash recovery |
| Raft log | Group-committed on the leader, quorum-replicated to followers, byte-aware residency cap with older entries paged from disk | Consensus record of DML and DDL for replication and failover |

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
- `UNNEST`, `FLATTEN`, `UNPIVOT` as one expand-rows operator, `PIVOT`, `ASOF JOIN`
- Node-local temporary tables
- Online heap DDL, schema epoch in the tuple slot, publish-wait-scan-flip index builds, shadow rewrite for incompatible type changes
- Time-series `GAP FILL` operator
- Prepared statements, holdable cursors, `COPY`, `CANCEL BACKEND`
- SQL reference generated from the grammar registry so an undocumented statement fails the build
- Wire protocol over TCP and QUIC

**Versioning & change**

- `AS OF TIMESTAMP` / `VERSION AS OF`
- Copy-on-write branches with merge conflict resolution
- SCD types, system/application/bitemporal time
- Picosecond-resolution timestamps with hybrid logical clocks
- Arrow `ps`->`ns` export for downstream tooling
- Diff and patch between versions
- CDC: change feeds, replication slots, Debezium / Avro / Wal2Json / native decoders, publications, snapshots
- Change streams: `CREATE CHANGE STREAM`, read and advance in the consumer's own transaction, `APPLY CHANGES` as SCD type 1 or 2, per-branch feeds, lake tables as derived sources

</td><td valign="top" width="33%">

**Security & governance**

- Three-state privileges (GRANT / DENY / unset) where DENY always wins and column-level overrides table-level
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

- Declarative pipelines, triggers, SQL UDFs, stored procedures
- Pipelines triggered `ON CHANGE DATA` with `CONSUME CHANGES` and `APPLY CHANGES` stages
- Materialized views with atomic refresh, refresh strategies, SLAs, advisor
- Contact channels for Slack, Discord, email, and webhook delivery
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
<tr><td valign="top" width="33%">

**ZyronLake table format**

- Immutable `.zyr` data files against an append-only transaction log, one numbered version file per commit
- `File::create_new` as the optimistic-concurrency primitive, with periodic manifest checkpoints collapsing the log
- Copy-on-write branches with merge
- Time travel to any committed version
- Secondary indexes (also versioned as lake artifacts, safe against staleness)
- Bloom filters, zone maps, and a struct-of-arrays prune index (94-100% file skip rates in practice)
- Z-order and Hilbert space-filling-curve clustering with background maintenance
- Unique and foreign-key constraint enforcement with a clustered fast path
- Change feed between two versions
- `OPTIMIZE`, `VACUUM`, `FOLLOW`, `REPAIR` operations
- Runs on local disk, shared filesystem, or object storage

</td><td valign="top" width="33%">

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

</td><td valign="top" width="33%">

**Consensus & replication**

- Raft group per cluster with leaders, followers, learners, and single-command membership change
- Group commit under one fsync per quorum ack
- Logical DML replication, heap follows row-level puts and deletes, lake follows agreed version numbers
- No primary key required. Falls back to secondary index probe or full-image match, following `REPLICA IDENTITY FULL` semantics
- DDL runs on the connection so clients see the real command tag, and the applier picks it up if the client drops
- Every statement in the grammar replicates, with an exhaustive classifier and a conformance suite that runs each statement against a live three-node group over the wire
- Explicit `BEGIN` blocks, savepoints, `MERGE`/`CALL`/`DO`, and prepared statements over the extended protocol all replicate
- Savepoints resolved at capture time so rolled-back rows never ship
- Cluster settings ride the consensus log behind a cluster version gate that waits for the lowest member version
- Client, mesh, and consensus wire protocols each registered with an add-only rule and a version stamped in every frame
- Linearizable follower reads through ReadIndex
- WAL commit record carries the raft index so recovery reconciles local durability with the agreed log
- Byte-aware log residency cap with older entries paged from disk on demand
- Whole-cluster snapshots with byte-aware compaction bounded by the slowest group member
- Background writers stand down inside a group so only the replicated log commits data

</td></tr>
<tr><td valign="top" colspan="3">

**Format agility, signature agility, and auto-upgrade**

- Versioned envelope on every persistent file, 28 registered format kinds covering WAL, heap, B+tree, `.zyr`, lake manifest, lake indexes, delete predicates, raft log, snapshot transfer, replication apply log, backup archive, `zyron.toml`, secret store, audit hash chain, statistics files, MVCC CLOG, and more
- Three framings for the same identity, a 20-byte header for standalone files, a 9-byte stamp for records inside a container, and a `[format]` TOML section for hand-editable files
- Per-format migration policy (`eager`, `lazy`, `coexist`) with steady-state overhead of a single integer compare on file open
- Reader window carries the current version plus two older readers, with CI failing the build on reader code past its registered retirement date
- Signature agility for JWTs, X.509 certificates, and custom binary artifacts with per-artifact-kind current and deprecating schemes and an overlap window
- Six signature schemes registered today (Ed25519, ES256, RS256 and family), infrastructure ready for post-quantum additions without wire or code churn
- Catalog schema evolution registry across 33 tables with per-row migration functions run in-transaction at upgrade time
- AST-based user-object rewriter for views, procedures, functions, materialized views, workflows, dashboards, prompts, and data quality rules, with safe, ambiguous, and unsafe categories
- Per-tenant rewrite policy (`auto_safe`, `notify_all`, `manual_only`) and `EXPLAIN REWRITE FOR OBJECT` dry-run
- Deprecation lifecycle (`deprecated` → `warn` → `error` → `removed`) with time-bounded windows, per-tenant rate-limited warnings, and auto-generated migration guides
- Auto-upgrade orchestrator with `stable`, `beta`, `canary`, and `pinned` channels, a signed release manifest, a compatibility gate that plans chained upgrades, rolling upgrade over Raft with drain coordination, health-baselined per-node rollback with cluster-wide auto-pause, and configurable maintenance windows
- Journaled self-restart into a staged binary, one cluster driver elected over the mesh so a rolling pass has a single coordinator, and per-target release archives signed in CI with the public half shipped in the server
- Deprecation registry with a documented home for every removal, first record landed
- Federation-aware compatibility gate blocks upgrades that would leave a peer cluster on an incompatible version
- `zyron-ctl format`, `upgrade`, `deprecation`, and `release` subcommands for inspection, batch migration, admin trigger, and CI release verification

</td></tr>
</table>

## Performance

Two views: what a client actually gets from a running server, and the raw subsystem numbers underneath it. Every figure is extracted from the committed result files in [`benchmarks/`](benchmarks/), and the tables and charts are regenerated from them by [`scripts/gen_bench_charts.py`](scripts/gen_bench_charts.py).

<!-- BENCH:START -->
_Release build, single machine: Intel(R) Core(TM) Ultra 7 270K Plus, 24 cores, 31.4 GB RAM, windows/x86_64. Regenerated from `benchmarks/` by `scripts/gen_bench_charts.py`._

### End-to-end

What a client sees from a running server over the wire protocol, from cold start to shutdown:

| Lifecycle / workload | Result |
|----------------------|--------|
| Cold boot to accepting queries | 35 ms |
| First `ReadyForQuery` | 0.77 ms |
| Schema DDL bootstrap | 11.2 ms |
| Seed insert | 237K rows/sec |
| OLTP, 1 client | 13.6K tps, p99 234 us |
| OLTP, 4 clients | 39.0K tps, p99 283 us |
| OLTP, 16 clients | 65.6K tps, p99 476 us |
| OLTP, 64 clients | 62.9K tps, p99 1706 us |
| OLTP, 256 clients | 55.0K tps, p99 8224 us |
| Analytical query (median) | 0.57 ms |
| Graceful shutdown | 64 ms |

![OLTP throughput vs. concurrent clients (thousand tps, higher is better)](benchmarks/charts/oltp_throughput.svg)

### Engine internals

Raw subsystem throughput and hot-path latency under microbenchmark:

![Throughput (million ops/sec, higher is better)](benchmarks/charts/engine_throughput.svg)

![Hot-path latency (nanoseconds, lower is better)](benchmarks/charts/hot_path_latency.svg)

### Row heap vs ZyronLake

Same workload, same rows, run against both formats. Blue is Row heap and purple is ZyronLake, and the shorter bar wins on wall-clock. Write-heavy trade-offs where the Row heap wins (bulk load, trickle load, indexed point lookup) are in the table below so the picture is honest, not cherry-picked.

![Cross-format wall-clock, Row heap vs ZyronLake](benchmarks/charts/cross_format.svg)

### Consensus and replication

A three-node Raft group with the leader accepting writes, followers applying committed entries through the same operator path the leader used, and reads served through a ReadIndex round trip so they stay linearizable. Throughput first, then hot-path latency:

![Consensus and replication throughput (thousand ops/sec, higher is better)](benchmarks/charts/raft_throughput.svg)

![Consensus hot-path latency (microseconds, lower is better)](benchmarks/charts/raft_latency.svg)

A few more numbers not shown in the charts above:

| Subsystem | Metric | Result |
|-----------|--------|--------|
| MVCC | GC sweep | ~1.8B tuples/sec |
| Columnar | .zyr scan throughput | ~11.9 GB/sec |
| Columnar | Compaction pipeline | ~12.3M rows/sec |
| Columnar | HybridScan overhead vs heap-only | ~1.3% |
| Columnar | Metadata-aggregate pruning speedup | ~36.8x |
| Temporal | Picosecond timestamp decode | ~1816M rows/sec |
| Versioning | Time-travel scan overhead | ~24% |
| Wire | QUIC PostgreSQL handshake | ~5 us |
| Transactions | Durable commit floor (device write) | ~69.9 us |
| Lake | Commit rate (insert) | ~2403 commits/sec |
| Lake | Commit latency (insert) | ~2.40 ms |
| Lake | Commit rate (delete predicate) | ~1972 commits/sec |
| Lake | Derived clustering expression files pruned | ~93% |
| Lake | Load with clustering expression | ~739K rows/sec |
| Consensus | Leader election after a kill | ~216 ms |
| Consensus | Single log append | ~0.19 us |
| Consensus | Snapshot 1GB, create | ~1.13 s |
| Consensus | Snapshot 1GB, transfer | ~2.72 s |
| Replication | Follower keep-up vs leader | ~99.8% |
| Replication | Worst follower lag | ~97 entries |
| Transactions | Durable group-commit peak | ~778K txn/sec |
| Transactions | Group-commit amplification (c=1 to c=512) | ~83.0x |
| Cross-format | Point lookup with a heap B+tree index, lake vs heap | ~0.6x (heap wins indexed points) |
| Cross-format | Bulk load to queryable, lake vs heap | ~1.5x (heap wins large batches) |
| Cross-format | Trickle load to queryable, lake vs heap | ~9.7x (heap wins tiny commits) |

46 benchmark suites cover storage, executor, optimizer, encoding, wire, search, analytics, CDC, versioning, transactions, temporal, columnar, lake, cross-format, raft, replication, types, lifecycle, gateway, Zyron-to-Zyron, and end-to-end. Each run writes a timestamped JSON/TXT pair under `benchmarks/<suite>/`.
<!-- BENCH:END -->

## Getting Started

### Download

Prebuilt binaries for Linux x86_64 and Windows x86_64 are attached to every
[release](https://github.com/DataDalton/Zyron/releases). Each archive contains
`zyron-server`, `zyron-cli`, and `zyron-ctl`.

```bash
tar xzf zyron-<version>-x86_64-unknown-linux-gnu.tar.gz
cd zyron-<version>-x86_64-unknown-linux-gnu
./zyron-server --version
./zyron-server --data-dir ./data
```

Checksums are published as `SHA256SUMS.txt` on the release page.

### Build from source

**Prerequisites.** The Rust nightly toolchain pinned in `rust-toolchain.toml` (`rustup` selects it automatically) and a C toolchain for the TLS dependency.

```bash
# Build
cargo build --release

# Run the server
cargo zyron -- --data-dir ./data --port 5432
#   alias for: cargo run --release --bin zyron-server --

# Connect with the bundled client
cargo cli -- --host localhost --port 5432

# Or any PostgreSQL client
psql -h localhost -p 5432 -U postgres

# Administer
cargo run --release --bin zyron-ctl -- status
cargo run --release --bin zyron-ctl -- backup --out ./backup
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
  zyron-lake          ZyronLake table format, transaction log, manifest, branches, indexes
  zyron-raft          Raft consensus for one replication group, log, snapshot, membership
  zyron-mesh          cross-node coordination, mesh RPC, scheduler drain, warm pool
  zyron-pressure      per-node load adaptation, self-calibration, actuator ladder
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
  zyron-media         media descriptors, content store, versioned binary codec
  zyron-lifecycle     retention, tiered storage, archival, GDPR, audit chain
  zyron-streaming     windowing, exactly-once, stream joins, backpressure
  zyron-tpc           TPC-H and TPC-C schemas, streaming data generation, workloads
  zyron-bench-harness shared benchmark harness and result output

binaries/
  zyron-server      database server entry point
  zyron-cli         psql-like interactive client
  zyron-ctl         admin tool: status, backup, restore, vacuum, compact

benchmarks/           timestamped JSON/TXT results, one folder per suite
scripts/              tooling, including the benchmark chart generator
```

</details>

## Roadmap

Each area ships with a validation checkpoint and hard performance budgets before the next begins. Rows are in build order.

| Area | Scope | State |
| ------ | ------- | ------- |
| Storage foundation | WAL, buffer pool, heap, B+ tree, MVCC, encoding engine, parser, catalog, planner, executor, wire, server | ✅ Complete |
| Security & optimization | Authentication, RBAC/ABAC, row/column security, cost model, configuration | ✅ Complete |
| Data operations | Versioning & time travel, CDC, pipelines/triggers/UDFs, streaming, server integration | ✅ Complete |
| Native features | Full-text search, vector & graph search, native data types, utility operations | ✅ Complete |
| Analytics & lifecycle | Analytics engine, feature store & ML, data lifecycle management | ✅ Complete |
| ZyronLake table format | Immutable versioned `.zyr` on a transaction log, branches, time travel, secondary indexes, clustering, constraint enforcement, change feed, cross-format federation | ✅ Complete |
| Consensus and replication | Raft groups per cluster, quorum-committed writes with one fsync per ack, every statement in the grammar replicated and proven on a live three-node group, linearizable follower reads through ReadIndex, learner promotion, whole-cluster snapshots, cluster settings on the consensus log behind a version gate | ✅ Complete |
| SQL surface, type system and media types | Range and multirange types, interval refinements, native media types, additional function coverage, and a silent-bug hardening pass across the engine | ✅ Complete |
| Format agility and auto-upgrade | Versioned envelope on every persistent file, signature agility for JWTs, X.509 certs, and custom artifacts, catalog schema evolution registry, AST-based user-object rewriter, deprecation lifecycle, a health-baselined rolling upgrade orchestrator over Raft with automatic rollback and federation-aware compat gating, journaled self-restart, signed per-target releases | ✅ Complete |
| Online heap DDL | Schema epoch in the tuple slot so a column change writes no row, publish-wait-scan-flip index builds, online shadow rewrite for an incompatible type change | ✅ Complete |
| SQL surface completions | `UNNEST`, `FLATTEN`, `UNPIVOT`, `PIVOT`, `ASOF JOIN`, node-local temporary tables, schema-qualified names on every statement form, SQL reference generated from the grammar registry | ✅ Complete |
| Change streams | `CREATE CHANGE STREAM`, transactional consumption with the position advanced in the consumer's own transaction, `APPLY CHANGES` as SCD type 1 or 2, pipelines triggered `ON CHANGE DATA`, per-branch feeds, lake tables as derived sources, feed written at the raft index on a group | ✅ Complete |
| Verifiable tables | One chained-append mechanism for tamper-evident tables, audit log converted onto it, proof export | ⏳ Planned |
| Legibility | `HELP` and documentation search inside the binary, `VALIDATE` for a statement without running it, stable error codes on every error, expected-token sets in parse errors, examples rewritten against the caller's schema | ⏳ Planned |
| Transport security and post-quantum cryptography | Hybrid X25519MLKEM768 key exchange required between Zyron components and offered first to third parties with classical accepted and recorded, one stricter-only setting, 256-bit suites only and TLS 1.2 removed, a node listener with mutual TLS to a cluster CA the operator holds so consensus and mesh traffic are authenticated and encrypted, ML-DSA-65, SLH-DSA-SHA2-128s, and hybrid Ed25519+ML-DSA-65 signature schemes active per artifact kind, release manifests signed hybrid, a key-exchange exposure report naming every outbound connection that negotiated classical | ⏳ Planned |
| Memory governance | One accounted pool across the node, class floors, grants instead of caps, spill on denial | ⏳ Planned |
| Sharding | Sharding core on heap tables with top-bit hash placement so a split is local, distributed query execution, cross-shard transactions with prepare in each participant's own Raft log, split and rebalance | ⏳ Planned |
| Enterprise & distribution | Secret store and KMS with HYOK adapters, high availability and DR, observability and compliance, semantic views, schema registry, data contracts, data governance, multi-tenancy and cost tracking, migration tools and ecosystem connectors, external tables, Volumes (arbitrary-file storage as catalog objects), enterprise type-system extensions | ⏳ Planned |
| Serverless mesh and autonomous operations | Base compute mesh, user meshes with compute reservations, workload-aware scheduling, continuous self-tuning and self-healing across every node in the mesh, query engine advances, model monitoring | ⏳ Planned |
| Wire protocols, API and drivers | ZWP native wire protocol alongside PostgreSQL wire compatibility, Zyron API as a route registry that drives the router, OpenAPI, and generated reference docs with SQL at `/api/sql`, native drivers across every supported client language, Zyron Embedded (`libzyron`) for in-process use | ⏳ Planned |
| Application and workflow hosting | Zyron Apps (container hosting on the mesh substrate), Workflows (task orchestration subsuming pipelines and schedules), Queues and Topics as heap tables through Raft with transactional enqueue, deployment artifacts, SQL function library expansion | ⏳ Planned |
| Dashboards, workspace, and metrics | Zyron Dashboards (first-party visualization with live query overlays, formatting with data, branch-native and time-travel-aware), Zyron Workspace with editors, language runtimes (Python, JS/TS, Java/Scala, Go, Rust), data & discovery, ops & integrations, a Zyron-native metric store | ⏳ Planned |
| Forms, Sheets, and Monitors | Forms as catalog objects whose controls derive from the target's columns, Sheets as a `.zysheet` workspace file that stores query, formulas, and typed cells but never rows, an `EXPORT` privilege enforced by an authorization token, Monitors that read metadata only and learn their bands | ⏳ Planned |
| Enterprise identity, organization, cross-cluster and Git | Enterprise auth (OIDC, SAML, TOTP, WebAuthn, universal PATs, tenant security policies), an organization directory with grantable units alongside groups, cross-cluster federation with mutual TLS and primary/secondary/peer roles, shared Git backend registry per workspace with item versioning | ⏳ Planned |
| AI Gateway, AI-native assist, and AI-derived columns | BYO-credentials gateway with prompt versioning and branching, semantic caching, continuous evaluation and cost tracking, opt-in per-workspace SQL, chart, and notebook assist that routes through the gateway, and columns whose values a model derives through the same gateway | ⏳ Planned |
| Zyron Web | React + Vite + Tailwind webapp shipping as a first-party static Zyron App, path-based URLs, admin surfaces for every backend area above | ⏳ Planned |

## Development

```bash
# Per-crate tests
cargo test -p zyron-storage

# Whole workspace
cargo test --workspace

# A benchmark suite, release build and serialized, results land in benchmarks/<suite>/
cargo test -p zyron-search --test search_bench --release -- --nocapture

# Regenerate the README performance charts from the latest benchmark files
python scripts/gen_bench_charts.py

# Lint and format
cargo fmt --check
cargo clippy --workspace -- -D warnings
```

## License

Copyright (c) 2026 Zyron LLC. **All rights reserved.** Zyron is proprietary software. Personal use is free, commercial use requires a license. See [LICENSE](LICENSE.md) for the full terms.

Zyron LLC · Licensing inquiries: [licensing@zyrondb.com](mailto:licensing@zyrondb.com)
