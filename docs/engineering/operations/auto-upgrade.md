# Auto-Upgrade, Developer Notes

Implementation reference for Zyron's auto-upgrade orchestration. Customer-facing documentation lives in [`business/operations/auto-upgrade.md`](../../business/operations/auto-upgrade.md).

## Composition

Auto-upgrade composes existing subsystems rather than adding coordination of its own:

- Raft consensus for leadership transfer and rolling coordination.
- The drain coordinator for a per-node restart that hands its hot set to survivors first.
- The audit hash chain for a step-by-step record.
- Contact channels for notification delivery.
- Signature agility for release manifest signing and verification.
- Federation coordination for the cross-cluster compatibility gate.
- App compatibility declarations for matching an App against the target version.

`UpgradeController` in `crates/zyron-server/src/upgrade/mod.rs` is the entry point that runs a gate, then a rolling pass, and records state on the `UpgradeBoard`. `UpgradeService` in `service.rs` is what runs it on a node.

## The service on a node

One `UpgradeService` per process, spawned once the wire listener accepts. At boot it opens the upgrade journal, puts the journal's board rows, history, and rewrite queue on the in-memory board, then seeds the board's settings from the `[upgrade]` config section. The config wins over the journal for settings because `ALTER SYSTEM SET` of an upgrade setting persists under its `upgrade.*` key in `zyron.auto.conf`, so the config already holds the latest value.

## Cluster settings

An upgrade setting is policy for the whole group, so it is carried by the consensus log as a `RaftCommand::Put` whose key is the setting's config key. `ALTER SYSTEM SET` of one applies it to the node's board and `zyron.auto.conf` at once, then leaves the request on `NodeControl`. The service loop carries it each second. The leader proposes the entry and waits for it to commit, any other node hands it to the leader over the `set_cluster_setting` mesh call, and a node with no group has nothing further to do. A request the leader cannot be reached with stays queued, ahead of anything asked since, unless a newer value for the same key has arrived. Every node applies the committed entry through `cluster_settings::apply`, board first and config file second, so a leadership change cannot lose a pause and a node that was down applies it when it catches up. A value a node's binary refuses is logged and skipped rather than stopping its apply loop, because being behind on one setting is recoverable and being behind on every entry is not.

The leader proposes the entry only once every member of the group runs a release that applies one. The members are the group's live consensus membership, each matched to a configured peer by the id its name hashes to, this node naming itself, and a member no configured peer names holds the request by id and address. `ClusterDriver::cluster_allows` asks every other member for its version over `node_status`, all at once, this node answering for itself, takes the lowest answer as the group's version floor, and holds the request while the floor is below `cluster_settings::INTRODUCED_IN` or a member has not answered. A held request stays queued, the loop asks again each second, and a reading is shared for five seconds so a burst of requests costs one round of probes. The driver drops the reading whenever it restarts or rolls back a member. The floor is the general gate for a rolling upgrade: anything a release adds to what one member puts in front of the others, a kind of log entry, a mesh call, a field in a message, is used once the floor reaches the release that introduced it, and a rollback lowers the floor and closes the gate on its own. A member the leader cannot reach keeps the floor unknown until it answers or is removed from the group.

The pause after a rollback goes to the log from the coordinator before it restarts, while it still leads, and the process that comes up on the previous binary writes it locally as well.

The service loops once a second, carrying out intents left on `NodeControl`: releases the mesh asked it to stage, restarts a coordinator asked for, an operator's `TRIGGER MANUAL` request, and acknowledged rewrites to apply. On the coordinator it also polls the feed every `release_feed_poll_interval_secs` and runs the controller's pass when the channel carries something newer than what runs.

The coordinator is the leader of the consensus group, or the only node without one. `TRIGGER MANUAL UPGRADE` and `TRIGGER MANUAL ROLLBACK` reach the service through the `UpgradeControl` trait the wire crate defines, and are refused on a non-leader with the leader's name.

A pass on the coordinator goes: poll the manifest, pick the target (the manual version, the pinned version, or the newest release after the running one), stage the release on this node, run the staged binary with `--capabilities` to read its reader floors and config keys, build the plan from the cluster members (each probed for its version, which has to match), take a physical backup when the target is a new major and `pre_upgrade_backup_snapshot` is on, and hand everything to `UpgradeController::run_pass`. The pass runs the gate, stages the release on every other node through the driver, captures the baseline, and walks the nodes.

## The journal

`upgrade.journal` in the data directory is the durable half of the board, an envelope of kind `UpgradeJournal` with a JSON body, rewritten whole through a temp file and a rename. It holds the board snapshot, the restart the node is in the middle of (`PendingRestart`), the last completed upgrade a rollback is measured against, the schema version each catalog table's rows are stored at, and the rows of a catalog table mid-replacement. `Journal::update` applies a change and writes it, keeping the change in memory only when the write succeeded.

## Restarting a node

`ClusterDriver` in `cluster_driver.rs` implements `NodeDriver` over this node and its mesh peers. Draining this node is `Admission::begin_drain` followed by a wait for `is_quiescent`, the shared admission the wire accept loop refuses on and the readiness probe reads. Draining a peer is `MeshScheduler::drain_for_restart`, the drain half of the scale-in path. Restarting a peer is the `restart_into_staged` mesh call, after which `observe` reads `node_status` and reports an unreachable node, or one still answering with the old version, as fully failing so the health watch keeps polling until the recovery window closes.

Restarting this node activates the staged binary beside the outgoing one, writes a `PendingRestart` with the baseline and the sequence's plan into the journal, arms the restart on `NodeControl`, and parks. `Server::run` selects between the shutdown signal and that restart, stops the server the way it does for a signal, and returns `RunOutcome::Restart`. Stopping wakes every maintenance loop through the server's shutdown notify, so a loop on an hourly interval exits at once rather than at its next tick. The binary's `main` replaces the process image on Unix and starts the new binary detached on Windows.

The new process's service runs `finish_restart` once the node serves. A self-driven upgrade watches the node's own health against the journaled baseline and either finishes or puts the previous binary back, arms another restart, and pauses. A coordinator-driven restart records the outcome and moves on, because the coordinator is watching. Finishing runs the post-upgrade migrations, pushes the history entry with what they did, records the completed upgrade for a later rollback, and clears the journal's restart.

Leadership moves through `RaftNode::transfer_leadership`, a TimeoutNow to the most current voter after it has caught up.

## Mesh calls

Five paths under `/internal/mesh/v1/` carry the driver's traffic beside the six the scheduler uses. `node_status` reports the running and staged versions, admission state, in-flight counts, and the connections' latency and rate sample. `stage_release`, `restart_into_staged`, `rollback_to_previous`, and `set_cluster_setting` leave intents on `NodeControl` and answer at once, so a handler never holds a connection for the length of a download. The coordinator reads `node_status` to learn when a stage landed or a restart came back.

## Peer protocols

Three protocols cross a node boundary, and `zyron_sys.wire.protocol_versions` lists the version this binary speaks of each. The client protocol is at 3, the mesh protocol at 1, and the consensus protocol at 1. Each registers a `WireProtocolVersion` row in its own crate, `zyron-wire`, `zyron-mesh`, and `zyron-raft`, the startup gate logs all three, and `zyron-ctl release verify` refuses a build where a protocol has no current version or two, two rows of one protocol share a number, a row names a release that is not `major.minor.patch`, or a retired row names no release.

The members of one group run two adjacent releases for the length of a rolling upgrade, so a peer protocol changes inside a version by adding, never by removing. Every mesh body carries `#[serde(default)]` at the struct level, decodes with any field absent, and ignores a field its reader does not know, so a release adds a field and the release beside it reads the default. A call added later is a new path, which an older node answers with a 404 the caller reads as `MeshRpcError::Unknown`. A consensus message appends a new field after the ones that shipped, and its decoder reads an absent trailing field as the default. Every consensus frame carries `CONSENSUS_PROTOCOL_VERSION` in the last two bytes of its header, a reply carries the version of the request it answers, a frame from a newer version is refused with both numbers in the reason, and a header carrying zero reads as version one. A change that fits neither rule, a removed or renamed field, a new variant of an enum a body carries, a different encoding, is a new version of that protocol, registered beside the old one and spoken only once the group's version floor reaches the release that introduced it.

## Capabilities

`zyron-server --capabilities` prints a JSON document of the binary's version, per-format reader floors, and accepted config keys. The gate's `TargetCapabilities` come from running the staged binary with that flag, so the check is against what the target actually reads rather than what a version number implies. Config keys the running binary accepts and the target does not are the gate's removed keys.

## Admission and metrics

`zyron_common::Admission` is the one place the accept loop, every connection, the mesh drain handler, the readiness probe, and the driver read the node's admission state. Each connection holds a session guard, each statement a query guard, and each open transaction a transaction guard, all released on drop. `zyron_common::QueryMetrics` is what the connections record into, read by the Prometheus exposition and sampled for the health baseline. A statement counts as failed only when its error is a fault of the server, SQLSTATE classes 08, 58, and XX, and class 53 apart from 53400, which is a query the node shed on purpose, so neither a client's mistakes nor the node's own load shedding read as ill health after a restart.

## Compatibility gate

`compat_gate::run` returns a `GateReport` that either passes, needs acknowledgment, or lists blockers. Its checks:

1. Formats. Every persisted format's reader window must cover what is on disk. The check reads the in-process format registry, taking each format's oldest supported version as the on-disk floor.
2. Config. A blocking config key removed between the current and target versions fails the gate. The removed-key set is supplied by the caller.
3. User objects. Every registered user-object rewrite runs in dry-run mode against each object and the per-object classification is collected. An object that cannot be parsed blocks. The object list is supplied by the caller, and the resolver handles views and materialized views today.
4. Federation. A peer cluster that cannot be reached inside the timeout, or that is more than one major version behind, blocks. The peer probe is supplied by the caller.
5. Apps. A deployed App that declares incompatibility with the target blocks, read from a precomputed compatibility flag.
6. Chained planning. When the target is more than one version ahead, the release manifest's chain is filtered to the steps ahead of the running version and reported.

The report is consumed by the notifier and the board.

## Rolling upgrade coordination

`rolling::run` upgrades nodes one at a time through the `NodeDriver` trait, after `run_pass` has staged the release on every node and told the driver the sequence is beginning:

1. Order the nodes followers first, the leader last.
2. Drain the next node and wait for the drain to complete.
3. Restart it with the new binary.
4. Watch it against a pre-upgrade health baseline until it recovers or the recovery window elapses. The baseline captures p50 and p99 latency, throughput, error rate, and connection count. The verdict is taken on p99 against twice the baseline, throughput against half the baseline, and error rate against a ceiling. Latency and throughput are compared only when the baseline carries thirty queries in its sixty second window, and latency only when the observation does too, so a node nobody is querying is never read as starved and a few cold queries are never read as slow. A quieter node is judged on its error rate alone.
5. If it recovers, mark it upgraded and move on.
6. If it fails past the recovery window, roll that node back to the previous binary when `rollback_on_health_fail` is set, and pause the whole sequence for admin review.
7. Transfer leadership during the leader's own step, after its drain.

The sequence never cascade-rolls the cluster on a single health failure. One node's failure pauses the sequence, and the pause is a cluster setting, so it survives the restarts that follow and a change of leader.

## Post-upgrade migration

Three independent operations move persisted state forward, run by each node once it is healthy on the new binary:

- The eager format sweep walks the data directory for files of a kind, migrates each in memory, and writes it back through a temp file and an atomic rename. It is bounded by the time budget and, per file, by the disk and memory budgets measured against the data volume's free space and the node's memory. A lazy or coexist format is a no-op.
- Catalog schema migration rolls each table whose schema version is behind forward. `HeapCatalogTableStore` reads and replaces the raw rows of a registered catalog table through the catalog storage, with each table's stored version kept in the journal. The rows as they were go into the journal before the heap is rewritten, so a crash mid-replacement is undone on the next start.
- User-object rewrites classified as safe are applied, and acknowledged ones on the pass after the acknowledgment. The rewritten statement tree is rendered back to SQL by `zyron_parser::unparse`, and the rendering is accepted only when parsing it yields the same tree, so the catalog never stores text that means something other than what the rewrite produced. A statement the unparser does not render, or whose rendering parses to a different tree, is reported as computed but not written with its diff kept for a person to apply. A materialized view or a streaming job has no write path, because each owns state fixed at creation, so its rewrite is reported the same way.

Progress is published to `zyron_sys.upgrade.format_migrations` and `zyron_sys.upgrade.user_object_rewrites`.

## Release feed and staging

The release poller fetches a signed release manifest from the configured feed URL. The manifest signature is verified through the signature agility substrate, and the poller honors a not-modified response.

If the manifest advertises a newer version on the cluster's channel, the stager fetches the binary into memory, verifies the manifest signature and the SHA-256 against the in-memory bytes, and only then writes it, through a partial file and an atomic rename. Nothing that fails verification reaches disk. Activation keeps the outgoing binary alongside as the previous one, so a rollback is a rename rather than a re-download. A rollback moves the live binary aside as the rolled-back one before the previous one takes its place, because the live binary is the running image and Windows lets a rename move it but never replace it. The next activation removes what a rollback left behind.

A node reads two feed sources in order. The `releases` directory under the data directory comes first, which is where `zyron-ctl release stage` places a release delivered by hand. The remote feed comes second, the project's GitHub releases unless `upgrade.release_feed_url` names an internal mirror, and no remote at all when that setting names a directory. Manifests are published per target as `<channel>.<target triple>.manifest`, because a node downloads the binary for the platform it runs on, and each binary knows its target from the build script. A feed with no manifest for the channel answers that nothing is new.

The verifying key is the public half of the vendor's Ed25519 release key, read from `crates/zyron-server/release-signing.pub` at build time and compiled into every binary. `upgrade.release_signing_key` names another public key for a self-built Zyron signed with `zyron-ctl release sign`.

The release workflow signs. `zyron-ctl release keygen` draws the key, its private half is the `ZYRON_RELEASE_SIGNING_KEY` secret of the workflow, and the release job runs `zyron-ctl release sign` over each platform's server binary, producing the per-target manifests that ship as release assets beside the bare binaries. A release without the secret fails rather than publishing something no node would install. Rotating the key is a new keygen, a new secret, a new `release-signing.pub`, and a release carrying it.

## Notification dispatch

Every upgrade state transition emits an audit event and a notification. Targets are the webhook and Slack channels configured with `upgrade.notify_webhook_url` and `upgrade.notify_slack_webhook_url`, delivered by an HTTP POST of the event JSON. `rolling::run` announces each node that completes with how many remain.

## Downgrade eligibility check

`downgrade::evaluate` reports downgrade eligible when every format migration and catalog schema migration applied during the upgrade was reversible, and no user object was accepted in a knowingly broken state. If a one-way migration ran, downgrade is blocked and the specific migration is named, with a restore-from-snapshot hint when a pre-upgrade snapshot is available. `TRIGGER MANUAL ROLLBACK` evaluates the last completed upgrade in the journal, which records the formats and tables that actually moved and where the snapshot was written, and refuses when the previous binary is no longer beside the live one. The coordinator writes `upgrade.paused` to the replicated log before it restarts, and the rollback's `PendingRestart` carries `pause_on_return` so the process that comes up on the previous binary writes it locally as well before it serves. Without the pause the coordinator's next poll would find the same release on the feed and stage it again.

## Related

- [../storage/format-agility.md](../storage/format-agility.md), format registry and migration mechanics.
- [../security/signature-agility.md](../security/signature-agility.md), release manifest signing.
- Business-facing counterpart: [`business/operations/auto-upgrade.md`](../../business/operations/auto-upgrade.md).
