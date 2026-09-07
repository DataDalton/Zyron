#![allow(non_snake_case)]

//! Format Substrate Benchmark Suite
//!
//! Measures the substrate's hot paths and its bulk operations against the
//! phase's Minimum Threshold table. Every measurement goes through the same
//! production code a server runs: the envelope encoder and decoder, the
//! registry the server loads at startup, the real signature verifiers, the
//! real rewriter, and the real migration runner.
//!
//! Performance Targets (Minimum Threshold):
//! | Operation                                | Metric   | Minimum      |
//! |------------------------------------------|----------|--------------|
//! | Envelope parse on hot path               | latency  | 100ns        |
//! | Reader dispatch per version              | latency  | 50ns         |
//! | Format registry lookup                   | latency  | 500ns        |
//! | Migration function overhead per file     | overhead | 20%          |
//! | Background eager sweep overhead          | overhead | 20% of I/O   |
//! | Signature scheme dispatch and verify     | latency  | 1us          |
//! | Signature rotation, single SP key        | latency  | 1s           |
//! | Catalog schema migration, 100K rows      | latency  | 30s          |
//! | User-object AST rewriter, 1000 objects   | latency  | 60s          |
//! | Deprecation warning rate-limit check     | latency  | 2us          |
//! | Compat gate, 10K user objects            | latency  | 120s         |
//! | Format documentation query               | latency  | 20ms         |
//! | Startup with 30 formats registered       | latency  | 100ms        |
//! | System catalog registration at startup   | latency  | 30ms         |
//! | Discord embed encode per event           | latency  | 10us         |
//! | Discord webhook address check            | latency  | 5us          |
//! | Discord delivery through a 1s rate limit | latency  | 1.2s         |
//!
//! Validation Requirements:
//! - Each measurement runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by the average
//! - A single run more than 2x worse than target fails the suite

// No `#[global_allocator]` here. zyron-wire installs mimalloc for the whole
// binary, and a second declaration in a test that links it is a conflict

use std::hint::black_box;
use std::sync::Mutex;

use ed25519_dalek::{Signer, SigningKey};
use zyron_auth::signature::{PrincipalKeyStore, VerifyingMaterial, verify_artifact};
use zyron_bench_harness::*;
use zyron_common::format::catalog_evolution::{
    CatalogSchemaEvolution, CatalogSchemaRegistry, CatalogTableRegistration,
};
use zyron_common::format::deprecation::{
    DeprecatedItemKind, DeprecationRecord, DeprecationRegistry, WarningLog, WarningRateLimiter,
};
use zyron_common::format::envelope;
use zyron_common::format::migration::{self, MigrationBoard};
use zyron_common::format::registry::{
    DeprecationStatus, FormatFixture, FormatMigrator, FormatRegistration, MigrationPolicy,
};
use zyron_common::format::rewrite::{ObjectKind, RewriteCategory, UserObjectRewritePolicy};
use zyron_common::format::scheme::{
    ArtifactKind, SchemeCategory, SchemeId, SchemeRegistry, SchemeStatus,
    SignatureSchemeRegistration, default_artifact_bindings,
};
use zyron_common::format::version::VersionWindow;
use zyron_common::format::wire_version::WireVersionRegistry;
use zyron_common::format::{
    ALL_FORMAT_KINDS, FormatKind, FormatRegistry, FormatSubstrate, FormatVersion, UpgradeBoard,
    UpgradeOutcome,
};
use zyron_parser::ast::Statement;
use zyron_parser::rewriter::{self, RenameTarget, UserObjectRewrite, default_diff, rename};
use zyron_server::upgrade::compat_gate::{self, GateInput, UserObject};
use zyron_server::upgrade::migrations::{self, InMemoryCatalogStore};
use zyron_server::upgrade::notification::{self, NotificationSink, UpgradeEvent};

// =============================================================================
// Performance Target Constants
// =============================================================================

const ENVELOPE_PARSE_TARGET_NS: f64 = 100.0;
const READER_DISPATCH_TARGET_NS: f64 = 50.0;
const REGISTRY_LOOKUP_TARGET_NS: f64 = 500.0;
const MIGRATION_OVERHEAD_TARGET_PCT: f64 = 20.0;
const SWEEP_OVERHEAD_TARGET_PCT: f64 = 20.0;
const SIGNATURE_DISPATCH_TARGET_US: f64 = 1.0;
const SP_ROTATION_TARGET_MS: f64 = 1_000.0;
const CATALOG_MIGRATION_TARGET_MS: f64 = 30_000.0;
const REWRITER_1K_TARGET_MS: f64 = 60_000.0;
const RATE_LIMIT_CHECK_TARGET_NS: f64 = 2_000.0;
const COMPAT_GATE_10K_TARGET_MS: f64 = 120_000.0;
const FORMAT_DOC_QUERY_TARGET_MS: f64 = 20.0;
const SUBSTRATE_LOAD_TARGET_MS: f64 = 100.0;
const CATALOG_REGISTRATION_TARGET_MS: f64 = 30.0;
const DISCORD_EMBED_ENCODE_TARGET_US: f64 = 10.0;
const DISCORD_URL_VALIDATION_TARGET_US: f64 = 5.0;
const DISCORD_RATE_LIMITED_DELIVERY_TARGET_MS: f64 = 1_200.0;

/// The suite's measurements are serialized so one test's allocation and
/// cache pressure does not land in another's numbers
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Rewriters the rewrite and gate measurements classify against
// =============================================================================

fn warehouse_to_compute(statement: &mut Statement) -> usize {
    rename(
        statement,
        RenameTarget::Relation,
        "warehouse_x",
        "compute_x",
    )
}

fn widen_signature(statement: &mut Statement) -> usize {
    rename(statement, RenameTarget::Function, "old_agg", "new_agg")
}

inventory::submit! {
    UserObjectRewrite {
        name: "bench_warehouse_to_compute",
        from_version: "0.11.0",
        to_version: "0.12.0",
        target: &[ObjectKind::View, ObjectKind::MaterializedView],
        rewriter: warehouse_to_compute,
        category: RewriteCategory::Safe,
        description: "renames the warehouse_x relation to compute_x",
        dry_run_diff_generator: default_diff,
    }
}

inventory::submit! {
    UserObjectRewrite {
        name: "bench_widen_signature",
        from_version: "0.11.0",
        to_version: "0.12.0",
        target: &[ObjectKind::View],
        rewriter: widen_signature,
        category: RewriteCategory::Ambiguous,
        description: "old_agg widened its return type, the call becomes new_agg",
        dry_run_diff_generator: default_diff,
    }
}

// =============================================================================
// Fixtures
// =============================================================================

fn append_marker(body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(body.len() + 3);
    out.extend_from_slice(body);
    out.extend_from_slice(b"-v2");
    Ok(out)
}

fn strip_marker(body: &[u8]) -> Result<Vec<u8>, String> {
    match body.strip_suffix(b"-v2") {
        Some(rest) => Ok(rest.to_vec()),
        None => Err("no marker".to_string()),
    }
}

/// A registry where one format has two versions, so both the current reader
/// path and the migrating reader path can be measured
fn bumped_registry(policy: MigrationPolicy) -> FormatRegistry {
    let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
        .iter()
        .copied()
        .map(|kind| {
            let bumped = kind == FormatKind::StatisticsFile;
            FormatRegistration {
                kind,
                writer_current_version: if bumped {
                    FormatVersion::new(1, 1)
                } else {
                    FormatVersion::V1
                },
                reader_supported_versions: if bumped {
                    VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                } else {
                    VersionWindow::single(FormatVersion::V1)
                },
                migration_policy: if bumped {
                    policy
                } else {
                    MigrationPolicy::Lazy
                },
                migration_reversible: true,
                binary_version_gate: "0.11.0",
                deprecation_status: DeprecationStatus::Active,
                retirement_date: if bumped { Some("2027-01-01") } else { None },
                downgrade_write_supported: false,
                notes: "bench registry",
            }
        })
        .collect();
    let migrators = [FormatMigrator {
        kind: FormatKind::StatisticsFile,
        from: FormatVersion::V1,
        to: FormatVersion::new(1, 1),
        reversible: true,
        forward: append_marker,
        backward: Some(strip_marker),
        no_body_change: false,
        description: "appends the v2 marker",
    }];
    let fixtures = [FormatFixture {
        kind: FormatKind::StatisticsFile,
        version: FormatVersion::V1,
        bytes: b"",
        path: "fixtures/v1.bin",
    }];
    FormatRegistry::from_parts(&registrations, &migrators, &fixtures).expect("loads")
}

fn scheme_registry() -> SchemeRegistry {
    SchemeRegistry::from_parts(
        vec![
            SignatureSchemeRegistration {
                scheme_name: "Ed25519",
                scheme_id: SchemeId(1),
                category: SchemeCategory::Signature,
                status: SchemeStatus::Active,
                first_available_version: "0.1.0",
                retirement_date: None,
                notes: "bench",
            },
            SignatureSchemeRegistration {
                scheme_name: "ES256",
                scheme_id: SchemeId(2),
                category: SchemeCategory::Signature,
                status: SchemeStatus::Active,
                first_available_version: "0.1.0",
                retirement_date: None,
                notes: "bench",
            },
        ],
        default_artifact_bindings(),
    )
}

// =============================================================================
// 1. Envelope parse on the hot path
// =============================================================================

/// The peek an open pays: read the magic and the version, compare, done.
///
/// The inputs vary across format kinds and versions so the compare cannot be
/// folded away and the branch predictor cannot learn one answer
#[tokio::test]
async fn bench_envelope_parse() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Envelope Parse (hot path) ===");

    // One encoded file per format kind, so the peek reads a different magic
    // and a different version each iteration
    let encoded: Vec<Vec<u8>> = ALL_FORMAT_KINDS
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            envelope::encode(
                *kind,
                FormatVersion::new(1, i as u16),
                b"a body the peek never touches",
            )
        })
        .collect();

    let iterations = 2_000_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0u64;
        for i in 0..iterations {
            let bytes = &encoded[i % encoded.len()];
            let (kind, version) = envelope::peek(black_box(bytes)).expect("peeks");
            sink = sink
                .wrapping_add(kind as u64)
                .wrapping_add(version.as_u32() as u64);
        }
        black_box(sink);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }

    let v = validate_metric_with_unit(
        "Envelope Parse",
        "peek",
        "ns",
        runs,
        ENVELOPE_PARSE_TARGET_NS,
        false,
    );
    assert!(v.passed, "envelope peek exceeded its target");

    // The full decode, which verifies both checksums, is the other half of
    // what an open can cost and is reported beside the peek
    let body: Vec<u8> = (0u8..=255).cycle().take(4_096).collect();
    let full = envelope::encode(FormatKind::ZyrColumnar, FormatVersion::V1, &body);
    let iterations = 200_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iterations {
            let parsed = envelope::decode(black_box(&full)).expect("decodes");
            sink = sink.wrapping_add(parsed.body.len());
        }
        black_box(sink);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    record_metric("Envelope Parse", "full decode, 4 KiB body", "ns", runs);

    let after = take_util_snapshot();
    record_test_util("Envelope Parse", before, after);
}

// =============================================================================
// 2. Reader dispatch per version
// =============================================================================

/// Resolving which reader a version takes. The current version returns after
/// one compare; a version behind plans the chain
#[tokio::test]
async fn bench_reader_dispatch() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Reader Dispatch ===");

    let registry = bumped_registry(MigrationPolicy::Lazy);
    let entry = registry
        .get(FormatKind::StatisticsFile)
        .expect("registered");
    let versions = [FormatVersion::new(1, 1), FormatVersion::V1];

    let iterations = 1_000_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0usize;
        for i in 0..iterations {
            let path = migration::reader_path(black_box(entry), versions[i % versions.len()])
                .expect("dispatches");
            sink += usize::from(path.needs_migration());
        }
        black_box(sink);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }

    let v = validate_metric_with_unit(
        "Reader Dispatch",
        "resolve",
        "ns",
        runs,
        READER_DISPATCH_TARGET_NS,
        false,
    );
    assert!(v.passed, "reader dispatch exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Reader Dispatch", before, after);
}

// =============================================================================
// 3. Format registry lookup
// =============================================================================

/// The dense-slot lookup a read path pays to reach a format's registration
#[tokio::test]
async fn bench_registry_lookup() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Format Registry Lookup ===");

    let registry = bumped_registry(MigrationPolicy::Lazy);
    let iterations = 2_000_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0u32;
        for i in 0..iterations {
            let kind = ALL_FORMAT_KINDS[i % ALL_FORMAT_KINDS.len()];
            let entry = registry.get(black_box(kind)).expect("registered");
            sink = sink.wrapping_add(entry.registration.writer_current_version.as_u32());
        }
        black_box(sink);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }

    let v = validate_metric_with_unit(
        "Registry Lookup",
        "get",
        "ns",
        runs,
        REGISTRY_LOOKUP_TARGET_NS,
        false,
    );
    assert!(v.passed, "registry lookup exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Registry Lookup", before, after);
}

// =============================================================================
// 4. Migration overhead per file
// =============================================================================

/// What migrating a file costs over reading one that is already current,
/// measured on the same bytes through the same open path
#[tokio::test]
async fn bench_migration_overhead() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Migration Overhead Per File ===");

    let registry = bumped_registry(MigrationPolicy::Lazy);
    let body: Vec<u8> = (0u8..=255).cycle().take(16_384).collect();
    let current = envelope::encode(FormatKind::StatisticsFile, FormatVersion::new(1, 1), &body);
    let behind = envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, &body);

    let iterations = 20_000usize;
    let mut baseline_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut migrating_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iterations {
            let opened =
                migration::open_as(&registry, black_box(&current), FormatKind::StatisticsFile)
                    .expect("opens");
            sink += opened.body.len();
        }
        black_box(sink);
        baseline_runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);

        let start = Instant::now();
        let mut sink = 0usize;
        for _ in 0..iterations {
            let opened =
                migration::open_as(&registry, black_box(&behind), FormatKind::StatisticsFile)
                    .expect("opens");
            sink += opened.body.len();
        }
        black_box(sink);
        migrating_runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }

    let baseline = record_metric(
        "Migration Overhead",
        "open a current file, 16 KiB",
        "ns",
        baseline_runs,
    );
    let migrating = record_metric(
        "Migration Overhead",
        "open a file one version behind, 16 KiB",
        "ns",
        migrating_runs,
    );
    let overhead_pct = if baseline > 0.0 {
        ((migrating - baseline) / baseline) * 100.0
    } else {
        0.0
    };
    let passed = check_performance_with_unit(
        "Migration Overhead",
        "overhead over a current read",
        "%",
        overhead_pct,
        MIGRATION_OVERHEAD_TARGET_PCT,
        false,
    );
    assert!(passed, "migration overhead exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Migration Overhead", before, after);
}

// =============================================================================
// 5. Background eager sweep overhead
// =============================================================================

/// What an eager sweep costs over the raw I/O of reading and rewriting the
/// same files, which is the floor any sweep has to pay
#[tokio::test]
async fn bench_sweep_overhead() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_storage();
    let before = take_util_snapshot();

    tprintln!("\n=== Eager Sweep Overhead ===");

    let files = 500usize;
    let body: Vec<u8> = (0u8..=255).cycle().take(8_192).collect();

    let mut io_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut sweep_runs = Vec::with_capacity(VALIDATION_RUNS);
    let registry = bumped_registry(MigrationPolicy::Eager);

    for _ in 0..VALIDATION_RUNS {
        // The I/O floor: read every file and write it back untouched
        let io_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..files {
            std::fs::write(
                io_dir.path().join(format!("{i}.zysts")),
                envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, &body),
            )
            .expect("writes");
        }
        let start = Instant::now();
        for i in 0..files {
            let path = io_dir.path().join(format!("{i}.zysts"));
            let bytes = std::fs::read(&path).expect("reads");
            std::fs::write(&path, &bytes).expect("writes");
        }
        io_runs.push(start.elapsed().as_secs_f64() * 1_000.0);

        // The sweep, which reads, migrates, and writes back
        let sweep_dir = tempfile::tempdir().expect("tempdir");
        for i in 0..files {
            std::fs::write(
                sweep_dir.path().join(format!("{i}.zysts")),
                envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, &body),
            )
            .expect("writes");
        }
        let board = MigrationBoard::new();
        let start = Instant::now();
        let result = migrations::sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            sweep_dir.path(),
            migrations::MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        sweep_runs.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(result.files_migrated, files as u64);
    }

    let io = record_metric("Eager Sweep", "read and rewrite 500 files", "ms", io_runs);
    let sweep = record_metric("Eager Sweep", "sweep 500 files", "ms", sweep_runs);
    let overhead_pct = if io > 0.0 {
        ((sweep - io) / io) * 100.0
    } else {
        0.0
    };
    let passed = check_performance_with_unit(
        "Eager Sweep",
        "overhead over raw I/O",
        "%",
        overhead_pct,
        SWEEP_OVERHEAD_TARGET_PCT,
        false,
    );
    assert!(passed, "sweep overhead exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Eager Sweep", before, after);
}

// =============================================================================
// 6. Signature dispatch and verify
// =============================================================================

/// Resolving the scheme an artifact declares and running its verifier, which
/// is what every signed artifact pays
#[tokio::test]
async fn bench_signature_dispatch() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Signature Dispatch and Verify ===");

    let registry = scheme_registry();
    let signing = SigningKey::from_bytes(&[23u8; 32]);
    let material = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());

    // Distinct messages so the verifier cannot reuse a cached result and the
    // measurement is of real signature work
    let messages: Vec<Vec<u8>> = (0..64)
        .map(|i| format!("the artifact signing input {i}").into_bytes())
        .collect();
    let signatures: Vec<Vec<u8>> = messages
        .iter()
        .map(|m| signing.sign(m).to_bytes().to_vec())
        .collect();

    let iterations = 20_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut verified = 0usize;
        for i in 0..iterations {
            let slot = i % messages.len();
            let ok = verify_artifact(
                black_box(&registry),
                ArtifactKind::Jwt,
                "Ed25519",
                &material,
                &messages[slot],
                &signatures[slot],
                0,
            )
            .expect("verifies");
            verified += usize::from(ok);
        }
        assert_eq!(verified, iterations);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64 / 1_000.0);
    }

    let v = validate_metric_with_unit(
        "Signature Dispatch",
        "resolve and verify",
        "us",
        runs,
        SIGNATURE_DISPATCH_TARGET_US,
        false,
    );

    // The dispatch on its own, without the verifier, is what a scheme
    // rotation changes and is reported beside the whole operation. It is
    // measured before the gate asserts, because when the combined figure
    // misses its target this is the number that says whether the substrate
    // or the verifier is responsible
    let iterations = 500_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut sink = 0u16;
        for _ in 0..iterations {
            let scheme = registry
                .resolve_for_verification(ArtifactKind::Jwt, black_box("Ed25519"), 0)
                .expect("resolves");
            sink = sink.wrapping_add(scheme.scheme_id.0);
        }
        black_box(sink);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    record_metric("Signature Dispatch", "registry resolve only", "ns", runs);

    let after = take_util_snapshot();
    record_test_util("Signature Dispatch", before, after);

    assert!(v.passed, "signature dispatch exceeded its target");
}

// =============================================================================
// 7. Service principal key rotation
// =============================================================================

/// Rotating one principal's key: generate a keypair, install it, and mark
/// the outgoing one retiring
#[tokio::test]
async fn bench_sp_key_rotation() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Service Principal Key Rotation ===");

    let registry = scheme_registry();
    let iterations = 200usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let store = PrincipalKeyStore::new();
        store.issue("sp1", "Ed25519", 0).expect("issues");
        let start = Instant::now();
        for i in 0..iterations {
            store
                .rotate(&registry, "sp1", Some("Ed25519"), 3_600, i as u64)
                .expect("rotates");
        }
        runs.push(start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64);
    }

    let v = validate_metric_with_unit(
        "SP Key Rotation",
        "rotate one principal",
        "ms",
        runs,
        SP_ROTATION_TARGET_MS,
        false,
    );
    assert!(v.passed, "SP key rotation exceeded its target");

    let after = take_util_snapshot();
    record_test_util("SP Key Rotation", before, after);
}

// =============================================================================
// 8. Catalog schema migration, 100K rows
// =============================================================================

fn add_source(row: &mut Vec<u8>) -> Result<(), String> {
    row.extend_from_slice(b"|local");
    Ok(())
}

fn drop_source(row: &mut Vec<u8>) -> Result<(), String> {
    match row.len().checked_sub(6) {
        Some(cut) if &row[cut..] == b"|local" => {
            row.truncate(cut);
            Ok(())
        }
        _ => Err("no source column".to_string()),
    }
}

/// One catalog table's rows moved to the next schema version, in the one
/// transaction the runner uses
#[tokio::test]
async fn bench_catalog_migration() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Catalog Schema Migration, 100K rows ===");

    let registry = CatalogSchemaRegistry::from_parts(
        &[CatalogTableRegistration {
            catalog_table: "zyron_sys.auth.groups",
            current_schema_version: FormatVersion::new(1, 1),
            introduced_in_binary_version: "0.11.0",
            doc: "groups",
        }],
        &[CatalogSchemaEvolution {
            catalog_table: "zyron_sys.auth.groups",
            from_version: FormatVersion::V1,
            to_version: FormatVersion::new(1, 1),
            migration_function_ref: "bench::add_source",
            reversible: true,
            introduced_in_binary_version: "0.11.0",
            forward: add_source,
            backward: Some(drop_source),
            description: "adds the source column with a default",
        }],
    )
    .expect("loads");

    let rows: Vec<Vec<u8>> = (0..100_000)
        .map(|i| format!("group-{i}").into_bytes())
        .collect();

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let store = InMemoryCatalogStore::new();
        store.seed("zyron_sys.auth.groups", FormatVersion::V1, rows.clone());
        let start = Instant::now();
        let (migrated, failures) = migrations::migrate_catalog(&registry, store.as_ref());
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(migrated[0].rows_migrated, 100_000);
    }

    let v = validate_metric_with_unit(
        "Catalog Migration",
        "100K rows, one table",
        "ms",
        runs,
        CATALOG_MIGRATION_TARGET_MS,
        false,
    );
    assert!(v.passed, "catalog migration exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Catalog Migration", before, after);
}

// =============================================================================
// 9. User-object rewriter, 1000 objects
// =============================================================================

/// Building the objects a rewrite pass and a compat gate walk
fn user_objects(count: usize) -> Vec<UserObject> {
    (0..count)
        .map(|i| match i % 10 {
            0 => UserObject {
                name: format!("ambiguous_{i}"),
                kind: ObjectKind::View,
                sql: format!(
                    "CREATE VIEW ambiguous_{i} AS SELECT old_agg(amount) FROM sales \
                     WHERE region = 'north' GROUP BY region"
                ),
            },
            1 | 2 | 3 => UserObject {
                name: format!("untouched_{i}"),
                kind: ObjectKind::View,
                sql: format!(
                    "CREATE VIEW untouched_{i} AS SELECT a, b FROM other_table \
                     WHERE a > {i} ORDER BY b"
                ),
            },
            _ => UserObject {
                name: format!("safe_{i}"),
                kind: ObjectKind::View,
                sql: format!(
                    "CREATE VIEW safe_{i} AS SELECT w.id, w.name FROM warehouse_x w \
                     JOIN regions r ON w.region_id = r.id WHERE w.id > {i}"
                ),
            },
        })
        .collect()
}

/// Parsing, walking, and rewriting a thousand user-authored objects
#[tokio::test]
async fn bench_user_object_rewriter() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== User-Object Rewriter, 1000 objects ===");

    let objects = user_objects(1_000);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let board = UpgradeBoard::new();
        let start = Instant::now();
        let pass =
            migrations::rewrite_objects(&board, &objects, UserObjectRewritePolicy::AutoSafe, 0);
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(pass.applied, 600, "six in ten are the safe class");
        assert_eq!(pass.queued, 100, "one in ten is the ambiguous class");
    }

    let v = validate_metric_with_unit(
        "User-Object Rewriter",
        "1000 objects",
        "ms",
        runs,
        REWRITER_1K_TARGET_MS,
        false,
    );
    assert!(v.passed, "the rewriter exceeded its target");

    // The dry run on its own, which is what EXPLAIN REWRITE pays
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut proposals = 0usize;
        for object in &objects {
            proposals += rewriter::dry_run(&object.name, object.kind, &object.sql)
                .expect("classifies")
                .len();
        }
        black_box(proposals);
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    record_metric("User-Object Rewriter", "dry run, 1000 objects", "ms", runs);

    let after = take_util_snapshot();
    record_test_util("User-Object Rewriter", before, after);
}

// =============================================================================
// 10. Deprecation warning rate-limit check
// =============================================================================

/// The bucket lookup a deprecated use pays before anything is logged
#[tokio::test]
async fn bench_rate_limit_check() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Deprecation Warning Rate-Limit Check ===");

    let limiter = WarningRateLimiter::new(10);
    let items = ["CREATE PIPELINE", "warehouse", "date_trunc_zone_arg"];
    let tenants: Vec<String> = (0..64).map(|i| format!("tenant-{i}")).collect();

    let iterations = 500_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut allowed = 0usize;
        for i in 0..iterations {
            allowed += usize::from(limiter.allow(
                black_box(items[i % items.len()]),
                black_box(&tenants[i % tenants.len()]),
                (i / 1_000) as u64,
            ));
        }
        black_box(allowed);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }

    let v = validate_metric_with_unit(
        "Rate-Limit Check",
        "allow",
        "ns",
        runs,
        RATE_LIMIT_CHECK_TARGET_NS,
        false,
    );
    assert!(v.passed, "the rate-limit check exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Rate-Limit Check", before, after);
}

// =============================================================================
// 11. Compatibility gate, 10K user objects
// =============================================================================

/// The whole gate over ten thousand objects, which is the classification a
/// large cluster's upgrade pays once
#[tokio::test]
async fn bench_compat_gate() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Compatibility Gate, 10K user objects ===");

    let substrate = zyron_common::format::substrate().expect("loads");
    let target = zyron_server::upgrade::running_capabilities(substrate, "0.12.0");
    let persisted = compat_gate::persisted_versions(&substrate.formats);
    let objects = user_objects(10_000);

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let report = compat_gate::run(GateInput {
            from_version: "0.11.0",
            to_version: "0.12.0",
            manifest: None,
            registry: &substrate.formats,
            target: &target,
            persisted: &persisted,
            config_keys: &[],
            objects: &objects,
            peers: &[],
            apps: &[],
            policy: UserObjectRewritePolicy::AutoSafe,
        })
        .expect("runs");
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(report.classification.total(), 10_000);
        assert!(report.blockers.is_empty(), "{}", report.blocker_text());
    }

    let v = validate_metric_with_unit(
        "Compat Gate",
        "10K objects",
        "ms",
        runs,
        COMPAT_GATE_10K_TARGET_MS,
        false,
    );
    assert!(v.passed, "the compat gate exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Compat Gate", before, after);
}

// =============================================================================
// 12. Format documentation query
// =============================================================================

/// Reading the documentation view, which computes its rows from the registry
/// rather than from stored rows
#[tokio::test]
async fn bench_format_documentation_query() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Format Documentation Query ===");

    let iterations = 2_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut rows = 0usize;
        for _ in 0..iterations {
            let (_, built) =
                zyron_wire::system_format_views::build("storage", "format_documentation")
                    .expect("builds");
            rows += built.len();
        }
        black_box(rows);
        runs.push(start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64);
    }

    let v = validate_metric_with_unit(
        "Format Documentation",
        "one read",
        "ms",
        runs,
        FORMAT_DOC_QUERY_TARGET_MS,
        false,
    );
    assert!(v.passed, "the documentation query exceeded its target");

    // The registry view beside it, which is the one an operator reads most
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = zyron_wire::system_format_views::build("storage", "format_registry")
                .expect("builds");
        }
        runs.push(start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64);
    }
    record_metric("Format Documentation", "format_registry read", "ms", runs);

    let after = take_util_snapshot();
    record_test_util("Format Documentation", before, after);
}

// =============================================================================
// 13. Substrate load and catalog registration at startup
// =============================================================================

/// Collecting every registration and checking the whole set, which is what a
/// server pays before it opens a data directory
#[tokio::test]
async fn bench_substrate_startup() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Substrate Load at Startup ===");

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let substrate = FormatSubstrate::load().expect("loads");
        substrate.verify_complete().expect("complete");
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(substrate.formats.len(), ALL_FORMAT_KINDS.len());
    }

    let v = validate_metric_with_unit(
        "Substrate Load",
        "load and verify every registration",
        "ms",
        runs,
        SUBSTRATE_LOAD_TARGET_MS,
        false,
    );
    assert!(v.passed, "the substrate load exceeded its target");
    tprintln!(
        "  Formats registered: {} ({} reserved)",
        ALL_FORMAT_KINDS.len(),
        zyron_server::release_check::reserved_kinds().len()
    );

    // The system catalog's own registration, which is the map inserts a
    // startup pays for every view
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut seen = 0usize;
        for object in zyron_catalog::SYSTEM_OBJECTS {
            let name = object.canonical_name();
            seen += usize::from(zyron_catalog::system_catalog::find(&name).is_some());
        }
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
        assert_eq!(seen, zyron_catalog::SYSTEM_OBJECTS.len());
    }
    let v = validate_metric_with_unit(
        "Catalog Registration",
        "resolve every system object",
        "ms",
        runs,
        CATALOG_REGISTRATION_TARGET_MS,
        false,
    );
    assert!(v.passed, "catalog registration exceeded its target");
    tprintln!(
        "  System objects: {} across {} schemas",
        zyron_catalog::SYSTEM_OBJECTS.len(),
        zyron_catalog::SYSTEM_SCHEMAS.len()
    );

    let after = take_util_snapshot();
    record_test_util("Substrate Load", before, after);
}

// =============================================================================
// 14. Deprecation scan on the statement path
// =============================================================================

/// The scan a statement pays when the registry holds deprecated items. This
/// release deprecates nothing, so the measured path is the one a release
/// that does will take
#[tokio::test]
async fn bench_deprecation_scan() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Deprecation Scan on the Statement Path ===");

    let records: Vec<DeprecationRecord> = ["CREATE PIPELINE", "RUN PIPELINE", "DROP PIPELINE"]
        .into_iter()
        .map(|item_id| DeprecationRecord {
            item_kind: DeprecatedItemKind::DdlKeyword,
            item_id,
            deprecated_since_version: "0.0.1",
            warn_until_version: "99.0.0",
            error_since_version: "99.0.0",
            removed_since_version: "99.1.0",
            replacement_ref: Some("CREATE WORKFLOW"),
            migration_guide_url: Some("/docs/sql/workflows.md"),
            no_guide_required: false,
            summary: "pipelines became workflows",
            before_example: "CREATE PIPELINE p",
            after_example: "CREATE WORKFLOW w",
            migration_snippet: "SELECT 1",
            common_pitfall: "a stage is not a task",
        })
        .collect();
    let substrate = FormatSubstrate {
        formats: FormatRegistry::from_parts(&[], &[], &[]).expect("loads"),
        schemes: SchemeRegistry::from_parts(Vec::new(), Vec::new()),
        catalog_schemas: CatalogSchemaRegistry::from_parts(&[], &[]).expect("loads"),
        deprecations: DeprecationRegistry::from_records(records).expect("loads"),
        wire_versions: WireVersionRegistry::from_versions(Vec::new()),
        migrations: MigrationBoard::new(),
        warning_limiter: WarningRateLimiter::new(10),
        warning_log: WarningLog::default(),
    };

    // A statement that trips nothing, which is what almost every statement
    // is and so is the cost that matters
    let clean = "SELECT id, name FROM sales.orders WHERE region = 'north' ORDER BY id LIMIT 10";
    let iterations = 200_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut found = 0usize;
        for i in 0..iterations {
            found += substrate
                .scan_sql_for_deprecations(black_box(clean), "tenant", i as u64)
                .len();
        }
        assert_eq!(found, 0);
        runs.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    record_metric(
        "Deprecation Scan",
        "statement with no deprecated item",
        "ns",
        runs,
    );

    let after = take_util_snapshot();
    record_test_util("Deprecation Scan", before, after);
}

// =============================================================================
// 15. Discord contact channel delivery
// =============================================================================

/// The embed body a Discord channel encodes for one event, and the address
/// check every channel construction pays. Both are pure CPU on the path an
/// upgrade step takes to reach a person
#[tokio::test]
async fn bench_discord_embed_and_url_validation() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Discord Embed Encode and URL Validation ===");

    // One event of each kind, so the encode measurement spans the whole
    // field and color map
    let events = vec![
        UpgradeEvent::PendingDetected {
            from_version: "0.12.0".to_string(),
            to_version: "0.13.0".to_string(),
            gate_summary: "0 blocker(s), 4 safe rewrites".to_string(),
        },
        UpgradeEvent::Started {
            from_version: "0.12.0".to_string(),
            to_version: "0.13.0".to_string(),
        },
        UpgradeEvent::NodeCompleted {
            node_id: "node-2".to_string(),
            to_version: "0.13.0".to_string(),
            nodes_remaining: 1,
        },
        UpgradeEvent::RolledBack {
            node_id: "node-2".to_string(),
            reason: "p99 latency stayed above the baseline multiplier".to_string(),
        },
        UpgradeEvent::Paused {
            reason: "operator".to_string(),
        },
        UpgradeEvent::Blocked {
            reason: "an unsafe rewrite has no acknowledgment".to_string(),
        },
        UpgradeEvent::Completed {
            to_version: "0.13.0".to_string(),
            outcome: UpgradeOutcome::Completed,
            detail: "3 node(s) upgraded".to_string(),
        },
    ];

    let iterations = 100_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut bytes = 0usize;
        for i in 0..iterations {
            let payload =
                notification::discord_payload(black_box(&events[i % events.len()]), 1_757_251_845);
            bytes += payload.to_string().len();
        }
        black_box(bytes);
        runs.push(start.elapsed().as_micros() as f64 / iterations as f64);
    }
    let v = validate_metric_with_unit(
        "Discord Embed Encode",
        "one event",
        "us",
        runs,
        DISCORD_EMBED_ENCODE_TARGET_US,
        false,
    );
    assert!(v.passed, "the embed encode exceeded its target");

    // The address check, over the forms it accepts and the forms it refuses,
    // because a refusal that scans the whole address is the slower half
    let addresses = [
        "https://discord.com/api/webhooks/123456789012345678/aB9_-zZtoken",
        "https://canary.discord.com/api/webhooks/1/a",
        "https://ptb.discordapp.com/api/webhooks/98765432109876543/tok",
        "http://discord.com/api/webhooks/1/a",
        "https://discord.evil.com/api/webhooks/1/a",
        "https://discord.com/api/webhooks/1/a/extra",
    ];
    let iterations = 1_000_000usize;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut accepted = 0usize;
        for i in 0..iterations {
            accepted += usize::from(notification::discord_webhook_url_is_wellformed(black_box(
                addresses[i % addresses.len()],
            )));
        }
        black_box(accepted);
        runs.push(start.elapsed().as_micros() as f64 / iterations as f64);
    }
    let v = validate_metric_with_unit(
        "Discord URL Validation",
        "one address",
        "us",
        runs,
        DISCORD_URL_VALIDATION_TARGET_US,
        false,
    );
    assert!(v.passed, "the address check exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Discord Embed and URL", before, after);
}

/// A rate-limited delivery end to end: the channel answers 429 asking for a
/// second, the sink waits it out and posts again. What is measured is the
/// wall time an upgrade step pays for one throttled channel, which is the
/// interval the channel asked for plus two round trips
#[tokio::test]
async fn bench_discord_rate_limited_delivery() {
    zyron_bench_harness::init("format_substrate");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Discord Delivery Through a 1s Rate Limit ===");

    let event = UpgradeEvent::NodeCompleted {
        node_id: "node-2".to_string(),
        to_version: "0.13.0".to_string(),
        nodes_remaining: 1,
    };
    let sink = notification::HttpNotificationSink::new(30).expect("builds");

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        // A fresh server per run, so the first post of each run is the one
        // that gets throttled
        let server = httpmock::MockServer::start_async().await;
        let limited = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST).path("/hook");
                then.status(429).header("retry-after", "1");
            })
            .await;
        let channel = notification::ContactChannel::Discord {
            webhook_url: server.url("/hook"),
        };

        let start = Instant::now();
        let delivery = tokio::join!(sink.deliver(&channel, &event), async {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            limited.delete_async().await;
            server
                .mock_async(|when, then| {
                    when.method(httpmock::Method::POST).path("/hook");
                    then.status(200);
                })
                .await;
        })
        .0;
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);

        assert!(delivery.delivered, "{}", delivery.detail);
        assert!(
            delivery.detail.contains("second attempt answered 200"),
            "{}",
            delivery.detail
        );
    }

    let v = validate_metric_with_unit(
        "Discord Rate-Limited Delivery",
        "one event",
        "ms",
        runs,
        DISCORD_RATE_LIMITED_DELIVERY_TARGET_MS,
        false,
    );
    assert!(v.passed, "the throttled delivery exceeded its target");

    let after = take_util_snapshot();
    record_test_util("Discord Rate-Limited Delivery", before, after);
}
