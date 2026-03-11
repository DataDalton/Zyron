#![allow(non_snake_case, unused_assignments, dead_code)]

//! Phase 3: Catalog Validation Tests
//!
//! Comprehensive integration tests for ZyronDB Phase 3 components:
//! - DDL persistence (create/drop database, schema, table, index)
//! - Name resolution (search path, qualified/unqualified)
//! - Column type roundtrip (all supported data types)
//! - Constraint storage (PK, UNIQUE, NOT NULL, DEFAULT, CHECK)
//! - Statistics collection (ANALYZE, histogram, MCV)
//! - Cache performance (lookup latency, hit rate)
//!
//! Performance Targets:
//! | Test             | Metric     | Minimum Threshold | Hardware Ceiling                          |
//! |------------------|------------|-------------------|-------------------------------------------|
//! | Table lookup     | latency    | 50ns              | ~5ns (hash map lookup, L1 cache hit)      |
//! | Schema resolve   | latency    | 100ns             | ~10ns (search path walk + hash lookup)    |
//! | DDL create       | latency    | 200us             | ~10us (WAL append + catalog insert)       |
//! | DDL drop         | latency    | 80us              | ~5us (WAL append + catalog remove)        |
//! | ANALYZE          | throughput | 8M rows/sec       | ~375M rows/sec (DDR5 sequential scan)     |
//! | Histogram build  | latency    | 10ms/col          | ~500us/col (sort-based, 100K samples)     |
//! | Cache hit rate   | ratio      | 100%              | 100% (catalog fits in memory)             |
//! | Recovery         | time       | 1ms               | ~100us (small WAL replay for metadata)    |
//!
//! Validation Requirements:
//! - Each test runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target

use std::io::Write as _;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::*;
use zyron_catalog::stats::analyze_table;
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_common::TypeId;
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType, Expr, LiteralValue, TableConstraint};
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, Tuple};
use zyron_wal::{WalWriter, WalWriterConfig};

// =============================================================================
// Performance Target Constants
// =============================================================================

const TABLE_LOOKUP_TARGET_NS: f64 = 50.0;
const SCHEMA_RESOLVE_TARGET_NS: f64 = 100.0;
const DDL_CREATE_TARGET_US: f64 = 200.0;
const DDL_DROP_TARGET_US: f64 = 80.0;
const ANALYZE_TARGET_ROWS_SEC: f64 = 8_000_000.0;
const HISTOGRAM_TARGET_MS_PER_COL: f64 = 10.0;
const CACHE_HIT_RATE_TARGET: f64 = 1.0; // 100%
const RECOVERY_TARGET_MS: f64 = 1.0;

// Validation constants
const VALIDATION_RUNS: usize = 5;
const REGRESSION_THRESHOLD: f64 = 2.0; // Fail if any run >2x worse than target

// Serialize benchmarks to avoid CPU contention between tests.
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Validation Infrastructure
// =============================================================================

/// Writes to both stdout and the run's dated text log file.
macro_rules! tprintln {
    () => {{
        std::println!();
        write_raw_output("");
    }};
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        std::println!("{}", msg);
        write_raw_output(&msg);
    }};
}

struct ValidationResult {
    passed: bool,
    regression_detected: bool,
    average: f64,
}

/// Formats a number with comma separators for readability.
fn format_with_commas(n: f64) -> String {
    let s = format!("{:.0}", n);
    let bytes: Vec<char> = s.chars().collect();
    let mut result = String::new();
    let len = bytes.len();
    for (i, c) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    result
}

fn validate_metric(
    test: &str,
    name: &str,
    runs: Vec<f64>,
    target: f64,
    higher_is_better: bool,
) -> ValidationResult {
    let average = runs.iter().sum::<f64>() / runs.len() as f64;
    let min = runs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance = runs.iter().map(|x| (x - average).powi(2)).sum::<f64>() / runs.len() as f64;
    let std_dev = variance.sqrt();

    let passed = if higher_is_better {
        average >= target
    } else {
        average <= target
    };

    let regression_threshold = if higher_is_better {
        target / REGRESSION_THRESHOLD
    } else {
        target * REGRESSION_THRESHOLD
    };

    let regression_detected = runs.iter().any(|&r| {
        if higher_is_better {
            r < regression_threshold
        } else {
            r > regression_threshold
        }
    });

    let status = if passed { "PASS" } else { "FAIL" };
    let regr_status = if regression_detected { "REGR!" } else { "OK" };
    let comparison = if higher_is_better { ">=" } else { "<=" };

    tprintln!("  {} [{}/{}]:", name, status, regr_status);
    tprintln!(
        "    Runs: [{}]",
        runs.iter()
            .map(|x| format_with_commas(*x))
            .collect::<Vec<_>>()
            .join(", ")
    );
    tprintln!(
        "    Average: {} {} {} (target)",
        format_with_commas(average),
        comparison,
        format_with_commas(target)
    );
    tprintln!(
        "    Min/Max: {} / {}, StdDev: {}",
        format_with_commas(min),
        format_with_commas(max),
        format_with_commas(std_dev)
    );

    write_benchmark_record(test, name, average, runs, target, passed, higher_is_better);

    ValidationResult {
        passed,
        regression_detected,
        average,
    }
}

fn check_performance(
    test: &str,
    name: &str,
    actual: f64,
    target: f64,
    higher_is_better: bool,
) -> bool {
    let passed = if higher_is_better {
        actual >= target
    } else {
        actual <= target
    };
    let status = if passed { "PASS" } else { "FAIL" };
    let comparison = if higher_is_better { ">=" } else { "<=" };
    tprintln!(
        "  {} [{}]: {} {} {} (target)",
        name,
        status,
        format_with_commas(actual),
        comparison,
        format_with_commas(target)
    );
    write_benchmark_record(
        test,
        name,
        actual,
        vec![actual],
        target,
        passed,
        higher_is_better,
    );
    passed
}

// =============================================================================
// Benchmark Output
// =============================================================================
//
// One .json file per run with a fully nested structure:
//   {
//     hardware/commit context,
//     "tests": {
//       "DDL Persistence": {
//         "util_before": { cpu_pct, ram_used_gb },
//         "util_after":  { cpu_pct, ram_used_gb },
//         "Create table latency (us)": { average, runs: [...], target, passed, higher_is_better },
//         ...
//       },
//       ...
//     }
//   }

#[derive(Clone)]
struct MetricRecord {
    test: String,
    metric: String,
    average: f64,
    runs: Vec<f64>,
    target: f64,
    passed: bool,
    higher_is_better: bool,
}

#[derive(Clone)]
struct UtilSnapshot {
    cpu_pct: f64,
    ram_used_gb: f64,
}

#[derive(Clone)]
struct UtilRecord {
    test: String,
    before: UtilSnapshot,
    after: UtilSnapshot,
}

static GIT_COMMIT: OnceLock<String> = OnceLock::new();
static PLATFORM_HW: OnceLock<PlatformHardware> = OnceLock::new();
static RUN_ID: OnceLock<String> = OnceLock::new();
static RAW_LOG: OnceLock<Mutex<std::fs::File>> = OnceLock::new();
static COLLECTED_METRICS: OnceLock<Mutex<Vec<MetricRecord>>> = OnceLock::new();
static COLLECTED_UTILS: OnceLock<Mutex<Vec<UtilRecord>>> = OnceLock::new();

struct PlatformHardware {
    cpu: String,
    ram_gb: f64,
    gpus: Vec<String>,
}

// Appends "-local" to the short hash when the working tree has uncommitted changes.
fn git_commit() -> &'static str {
    GIT_COMMIT.get_or_init(|| {
        std::process::Command::new("git")
            .args(["describe", "--always", "--dirty=-local", "--abbrev=7"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    })
}

// Howard Hinnant's algorithm: converts Unix seconds to "YYYY-MM-DD HH:MM:SSZ".
fn unix_to_datetime(ts: u64) -> String {
    let secs = ts % 60;
    let mins = (ts / 60) % 60;
    let hours = (ts / 3600) % 24;
    let z = ts / 86400 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}Z",
        y, m, d, hours, mins, secs
    )
}

// CARGO_MANIFEST_DIR = crates/zyron-catalog, so go up 2 levels to workspace root.
fn benchmark_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks")
}

// Formats a slice of strings as a JSON array of quoted strings.
fn json_str_array(items: &[String]) -> String {
    let inner = items
        .iter()
        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", inner)
}

// Single PowerShell invocation fetches CPU, RAM, and all GPUs on Windows.
#[cfg(target_os = "windows")]
fn platform_hw_impl() -> PlatformHardware {
    let script = "$cpu = (Get-WmiObject Win32_Processor).Name.Trim(); \
                  $ram = (Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory; \
                  $gpus = (Get-WmiObject Win32_VideoController | ForEach-Object { $_.Name.Trim() }) -join ';;'; \
                  Write-Output \"$cpu||$ram||$gpus\"";
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let parts: Vec<&str> = out.trim().splitn(3, "||").collect();
    let cpu = parts.first().copied().unwrap_or("unknown").to_string();
    let ram_bytes: u64 = parts
        .get(1)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let gpus = parts
        .get(2)
        .map(|s| {
            s.split(";;")
                .map(|g| g.trim().to_string())
                .filter(|g| !g.is_empty())
                .collect()
        })
        .unwrap_or_default();
    PlatformHardware {
        cpu,
        ram_gb: ram_bytes as f64 / (1024.0_f64.powi(3)),
        gpus,
    }
}

#[cfg(target_os = "linux")]
fn platform_hw_impl() -> PlatformHardware {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let ram_kb: u64 = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(0);
    let gpus = std::process::Command::new("lspci")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            s.lines()
                .filter(|l| l.contains("VGA") || l.contains("3D controller"))
                .filter_map(|l| l.split(':').last())
                .map(|s| s.trim().to_string())
                .collect()
        })
        .unwrap_or_default();
    PlatformHardware {
        cpu,
        ram_gb: ram_kb as f64 / (1024.0 * 1024.0),
        gpus,
    }
}

#[cfg(target_os = "macos")]
fn platform_hw_impl() -> PlatformHardware {
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let ram_bytes: u64 = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let gpus = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            s.lines()
                .filter(|l| l.contains("Chipset Model:"))
                .filter_map(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
                .collect()
        })
        .unwrap_or_default();
    PlatformHardware {
        cpu,
        ram_gb: ram_bytes as f64 / (1024.0_f64.powi(3)),
        gpus,
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn platform_hw_impl() -> PlatformHardware {
    PlatformHardware {
        cpu: "unknown".to_string(),
        ram_gb: 0.0,
        gpus: vec![],
    }
}

#[cfg(target_os = "windows")]
fn take_util_snapshot() -> UtilSnapshot {
    let script = "$cpu = (Get-WmiObject Win32_Processor).LoadPercentage; \
                  $os = Get-WmiObject Win32_OperatingSystem; \
                  $usedKb = $os.TotalVisibleMemorySize - $os.FreePhysicalMemory; \
                  Write-Output \"$cpu||$usedKb\"";
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", script])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    let parts: Vec<&str> = out.trim().splitn(2, "||").collect();
    let cpu_pct: f64 = parts
        .first()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);
    let used_kb: f64 = parts
        .get(1)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);
    UtilSnapshot {
        cpu_pct,
        ram_used_gb: used_kb / (1024.0 * 1024.0),
    }
}

#[cfg(target_os = "linux")]
fn take_util_snapshot() -> UtilSnapshot {
    let cpu_pct = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    let mem_info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let total_kb: u64 = mem_info
        .lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let avail_kb: u64 = mem_info
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    UtilSnapshot {
        cpu_pct,
        ram_used_gb: (total_kb - avail_kb) as f64 / (1024.0 * 1024.0),
    }
}

#[cfg(target_os = "macos")]
fn take_util_snapshot() -> UtilSnapshot {
    let cpu_pct = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split_whitespace()
                .next()
                .and_then(|v| v.parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    let ram_bytes: u64 = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    let page_size: u64 = std::process::Command::new("pagesize")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(4096);
    let pages_free: u64 = std::process::Command::new("vm_stat")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Pages free:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().trim_end_matches('.').parse().ok())
        })
        .unwrap_or(0);
    let ram_used_gb =
        (ram_bytes.saturating_sub(pages_free * page_size)) as f64 / (1024.0_f64.powi(3));
    UtilSnapshot {
        cpu_pct,
        ram_used_gb,
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn take_util_snapshot() -> UtilSnapshot {
    UtilSnapshot {
        cpu_pct: 0.0,
        ram_used_gb: 0.0,
    }
}

fn platform_hw() -> &'static PlatformHardware {
    PLATFORM_HW.get_or_init(platform_hw_impl)
}

fn logical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

fn run_id() -> &'static str {
    RUN_ID.get_or_init(|| {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let date_tag = unix_to_datetime(ts)
            .replace(' ', "_")
            .replace(':', "")
            .replace('Z', "");
        format!("{}_{}", date_tag, git_commit())
    })
}

fn collected_metrics() -> &'static Mutex<Vec<MetricRecord>> {
    COLLECTED_METRICS.get_or_init(|| Mutex::new(Vec::new()))
}

fn collected_utils() -> &'static Mutex<Vec<UtilRecord>> {
    COLLECTED_UTILS.get_or_init(|| Mutex::new(Vec::new()))
}

fn record_test_util(test: &str, before: UtilSnapshot, after: UtilSnapshot) {
    let record = UtilRecord {
        test: test.to_string(),
        before,
        after,
    };
    let utils_snap = if let Ok(mut g) = collected_utils().lock() {
        if let Some(existing) = g.iter_mut().find(|u| u.test == test) {
            *existing = record;
        } else {
            g.push(record);
        }
        g.clone()
    } else {
        return;
    };
    let metrics_snap = collected_metrics()
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    write_run_json(&metrics_snap, &utils_snap);
}

fn build_run_json(metrics: &[MetricRecord], utils: &[UtilRecord]) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let date = unix_to_datetime(ts);
    let hw = platform_hw();
    let cores = logical_cores();
    let cpu = hw.cpu.replace('"', "\\\"");
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let id = run_id();
    let commit = git_commit();
    let gpus_json = json_str_array(&hw.gpus);

    let mut test_names: Vec<&str> = Vec::new();
    for m in metrics {
        if !test_names.contains(&m.test.as_str()) {
            test_names.push(&m.test);
        }
    }

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"id\": \"{id}\",\n"));
    out.push_str(&format!("  \"date\": \"{date}\",\n"));
    out.push_str(&format!("  \"ts\": {ts},\n"));
    out.push_str(&format!("  \"commit\": \"{commit}\",\n"));
    out.push_str(&format!("  \"cpu\": \"{cpu}\",\n"));
    out.push_str(&format!("  \"cores\": {cores},\n"));
    out.push_str(&format!("  \"ram_gb\": {:.1},\n", hw.ram_gb));
    out.push_str(&format!("  \"gpus\": {gpus_json},\n"));
    out.push_str(&format!("  \"os\": \"{os}\",\n"));
    out.push_str(&format!("  \"arch\": \"{arch}\",\n"));
    out.push_str("  \"tests\": {\n");

    for (ti, test_name) in test_names.iter().enumerate() {
        let escaped_test = test_name.replace('"', "\\\"");
        out.push_str(&format!("    \"{escaped_test}\": {{\n"));

        if let Some(u) = utils.iter().find(|u| u.test.as_str() == *test_name) {
            out.push_str(&format!(
                "      \"util_before\": {{ \"cpu_pct\": {:.1}, \"ram_used_gb\": {:.2} }},\n",
                u.before.cpu_pct, u.before.ram_used_gb
            ));
            out.push_str(&format!(
                "      \"util_after\": {{ \"cpu_pct\": {:.1}, \"ram_used_gb\": {:.2} }},\n",
                u.after.cpu_pct, u.after.ram_used_gb
            ));
        }

        let test_metrics: Vec<&MetricRecord> = metrics
            .iter()
            .filter(|m| m.test.as_str() == *test_name)
            .collect();
        for (mi, m) in test_metrics.iter().enumerate() {
            let escaped_metric = m.metric.replace('"', "\\\"");
            let comma = if mi + 1 < test_metrics.len() {
                ","
            } else {
                ""
            };
            let runs_json = m
                .runs
                .iter()
                .map(|v| format!("{:.2}", v))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("      \"{escaped_metric}\": {{\n"));
            out.push_str(&format!("        \"average\": {:.6},\n", m.average));
            out.push_str(&format!("        \"runs\": [{runs_json}],\n"));
            out.push_str(&format!("        \"target\": {:.6},\n", m.target));
            out.push_str(&format!("        \"passed\": {},\n", m.passed));
            out.push_str(&format!(
                "        \"higher_is_better\": {}\n",
                m.higher_is_better
            ));
            out.push_str(&format!("      }}{comma}\n"));
        }

        let test_comma = if ti + 1 < test_names.len() {
            ","
        } else {
            ""
        };
        out.push_str(&format!("    }}{test_comma}\n"));
    }

    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

fn write_run_json(metrics: &[MetricRecord], utils: &[UtilRecord]) {
    let dir = benchmark_dir();
    let _ = std::fs::create_dir_all(&dir);
    let fname = format!("phase3_{}.json", run_id());
    let json = build_run_json(metrics, utils);
    let _ = std::fs::write(dir.join(fname), json.as_bytes());
}

fn raw_log_file() -> &'static Mutex<std::fs::File> {
    RAW_LOG.get_or_init(|| {
        let dir = benchmark_dir();
        let _ = std::fs::create_dir_all(&dir);
        let fname = format!("phase3_{}.txt", run_id());
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dir.join(&fname))
            .unwrap_or_else(|_| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(dir.join("phase3_latest.txt"))
                    .expect("failed to open benchmark log")
            });
        Mutex::new(f)
    })
}

fn write_raw_output(line: &str) {
    if let Ok(mut guard) = raw_log_file().lock() {
        let _ = writeln!(guard, "{}", line);
    }
}

fn write_benchmark_record(
    test: &str,
    metric: &str,
    average: f64,
    runs: Vec<f64>,
    target: f64,
    passed: bool,
    higher_is_better: bool,
) {
    let record = MetricRecord {
        test: test.to_string(),
        metric: metric.to_string(),
        average,
        runs,
        target,
        passed,
        higher_is_better,
    };
    let metrics_snap = if let Ok(mut g) = collected_metrics().lock() {
        g.push(record);
        g.clone()
    } else {
        return;
    };
    let utils_snap = collected_utils()
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    write_run_json(&metrics_snap, &utils_snap);
}

// =============================================================================
// Helper functions
// =============================================================================

/// Creates a temporary catalog test environment with DiskManager, BufferPool, WAL, and Catalog.
async fn setup_catalog(
    dir: &std::path::Path,
) -> (Arc<DiskManager>, Arc<BufferPool>, Arc<WalWriter>, Catalog) {
    let data_dir = dir.join("data");
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );

    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Catalog::new(storage, cache, Arc::clone(&wal)).await.unwrap();

    (disk, pool, wal, catalog)
}

/// Column definitions for a 10-column test table.
fn make_10_column_defs() -> Vec<ColumnDef> {
    vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::BigInt,
            nullable: Some(false),
            default: None,
            constraints: vec![ColumnConstraint::PrimaryKey],
        },
        ColumnDef {
            name: "name".to_string(),
            data_type: DataType::Varchar(Some(255)),
            nullable: Some(false),
            default: None,
            constraints: vec![ColumnConstraint::NotNull],
        },
        ColumnDef {
            name: "email".to_string(),
            data_type: DataType::Varchar(Some(320)),
            nullable: Some(false),
            default: None,
            constraints: vec![ColumnConstraint::Unique],
        },
        ColumnDef {
            name: "age".to_string(),
            data_type: DataType::Int,
            nullable: Some(true),
            default: None,
            constraints: vec![],
        },
        ColumnDef {
            name: "balance".to_string(),
            data_type: DataType::Decimal(Some(10), Some(2)),
            nullable: Some(true),
            default: Some(Expr::Literal(LiteralValue::Float(0.00))),
            constraints: vec![],
        },
        ColumnDef {
            name: "is_active".to_string(),
            data_type: DataType::Boolean,
            nullable: Some(false),
            default: Some(Expr::Literal(LiteralValue::Boolean(true))),
            constraints: vec![],
        },
        ColumnDef {
            name: "created_at".to_string(),
            data_type: DataType::Timestamp,
            nullable: Some(false),
            default: None,
            constraints: vec![],
        },
        ColumnDef {
            name: "metadata".to_string(),
            data_type: DataType::Jsonb,
            nullable: Some(true),
            default: None,
            constraints: vec![],
        },
        ColumnDef {
            name: "avatar".to_string(),
            data_type: DataType::Bytea,
            nullable: Some(true),
            default: None,
            constraints: vec![],
        },
        ColumnDef {
            name: "tags".to_string(),
            data_type: DataType::Text,
            nullable: Some(true),
            default: None,
            constraints: vec![],
        },
    ]
}

/// Column definitions covering every supported data type.
fn make_all_types_column_defs() -> Vec<ColumnDef> {
    vec![
        ColumnDef { name: "c_bool".into(), data_type: DataType::Boolean, nullable: Some(false), default: None, constraints: vec![] },
        ColumnDef { name: "c_tinyint".into(), data_type: DataType::TinyInt, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_smallint".into(), data_type: DataType::SmallInt, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_int".into(), data_type: DataType::Int, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_bigint".into(), data_type: DataType::BigInt, nullable: Some(false), default: None, constraints: vec![] },
        ColumnDef { name: "c_int128".into(), data_type: DataType::Int128, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_uint8".into(), data_type: DataType::UInt8, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_uint16".into(), data_type: DataType::UInt16, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_uint32".into(), data_type: DataType::UInt32, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_uint64".into(), data_type: DataType::UInt64, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_uint128".into(), data_type: DataType::UInt128, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_real".into(), data_type: DataType::Real, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_double".into(), data_type: DataType::DoublePrecision, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_decimal".into(), data_type: DataType::Decimal(Some(18), Some(4)), nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_char".into(), data_type: DataType::Char(Some(10)), nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_varchar".into(), data_type: DataType::Varchar(Some(255)), nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_text".into(), data_type: DataType::Text, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_binary".into(), data_type: DataType::Binary(Some(64)), nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_varbinary".into(), data_type: DataType::Varbinary(Some(128)), nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_bytea".into(), data_type: DataType::Bytea, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_date".into(), data_type: DataType::Date, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_time".into(), data_type: DataType::Time, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_timestamp".into(), data_type: DataType::Timestamp, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_timestamptz".into(), data_type: DataType::TimestampTz, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_interval".into(), data_type: DataType::Interval, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_uuid".into(), data_type: DataType::Uuid, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_json".into(), data_type: DataType::Json, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_jsonb".into(), data_type: DataType::Jsonb, nullable: Some(true), default: None, constraints: vec![] },
        ColumnDef { name: "c_vector".into(), data_type: DataType::Vector(Some(128)), nullable: Some(true), default: None, constraints: vec![] },
    ]
}

// =============================================================================
// 1. DDL Persistence Test
// =============================================================================

#[tokio::test]
async fn phase3_01_ddl_persistence() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== DDL Persistence ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir: wal_dir.clone(),
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );

    // Phase 1: Create objects
    let db_id;
    let schema_id;
    let table_id;
    let idx1_id;
    let idx2_id;
    {
        let storage =
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
            .await
            .unwrap();

        db_id = catalog.create_database("testdb", "admin").await.unwrap();
        schema_id = catalog.create_schema(db_id, "app", "admin").await.unwrap();

        let col_defs = make_10_column_defs();
        table_id = catalog
            .create_table(schema_id, "users", &col_defs, &[])
            .await
            .unwrap();

        idx1_id = catalog
            .create_index(
                table_id,
                schema_id,
                "idx_users_email",
                &["email".to_string()],
                true,
                IndexType::BTree,
            )
            .await
            .unwrap();

        idx2_id = catalog
            .create_index(
                table_id,
                schema_id,
                "idx_users_name_age",
                &["name".to_string(), "age".to_string()],
                false,
                IndexType::BTree,
            )
            .await
            .unwrap();

        tprintln!("  Created: database={}, schema={}, table={}, idx1={}, idx2={}",
            db_id, schema_id, table_id, idx1_id, idx2_id);
    }

    // Phase 2: Reload catalog from storage and verify
    {
        let storage =
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .unwrap();

        // Verify database
        let db = catalog.get_database("testdb").unwrap();
        assert_eq!(db.id, db_id);
        assert_eq!(db.name, "testdb");
        assert_eq!(db.owner, "admin");
        tprintln!("  Database reload: PASS");

        // Verify schema
        let schema = catalog.get_schema(db_id, "app").unwrap();
        assert_eq!(schema.id, schema_id);
        assert_eq!(schema.name, "app");
        tprintln!("  Schema reload: PASS");

        // Verify table
        let table = catalog.get_table(schema_id, "users").unwrap();
        assert_eq!(table.id, table_id);
        assert_eq!(table.name, "users");
        assert_eq!(table.columns.len(), 10);
        tprintln!("  Table reload: PASS (10 columns)");

        // Verify column metadata
        assert_eq!(table.columns[0].name, "id");
        assert_eq!(table.columns[0].type_id, TypeId::Int64);
        assert!(!table.columns[0].nullable);
        assert_eq!(table.columns[1].name, "name");
        assert_eq!(table.columns[1].type_id, TypeId::Varchar);
        assert_eq!(table.columns[1].max_length, Some(255));
        assert_eq!(table.columns[2].name, "email");
        assert_eq!(table.columns[2].max_length, Some(320));
        tprintln!("  Column metadata: PASS");

        // Verify indexes
        let indexes = catalog.get_indexes_for_table(table_id);
        assert_eq!(indexes.len(), 2);

        let email_idx = indexes.iter().find(|i| i.name == "idx_users_email").unwrap();
        assert_eq!(email_idx.id, idx1_id);
        assert!(email_idx.unique);
        assert_eq!(email_idx.columns.len(), 1);

        let composite_idx = indexes
            .iter()
            .find(|i| i.name == "idx_users_name_age")
            .unwrap();
        assert_eq!(composite_idx.id, idx2_id);
        assert!(!composite_idx.unique);
        assert_eq!(composite_idx.columns.len(), 2);
        tprintln!("  Index reload: PASS (2 indexes, 1 unique, 1 composite)");
    }

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("DDL Persistence", before, after);
}

// =============================================================================
// 2. Name Resolution Test
// =============================================================================

#[tokio::test]
async fn phase3_02_name_resolution() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Name Resolution ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    // "zyron" database is bootstrapped with id=SYSTEM_DATABASE_ID
    let db_id = SYSTEM_DATABASE_ID;

    // "public" schema is bootstrapped as DEFAULT_SCHEMA_ID
    let app_schema_id = catalog
        .create_schema(db_id, "app", "system")
        .await
        .unwrap();

    // Create public.users
    let public_users_cols = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];
    let public_users_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "users", &public_users_cols, &[])
        .await
        .unwrap();

    // Create app.users
    let app_users_cols = vec![ColumnDef {
        name: "user_id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];
    let app_users_id = catalog
        .create_table(app_schema_id, "users", &app_users_cols, &[])
        .await
        .unwrap();

    // Create app.orders
    let orders_cols = vec![ColumnDef {
        name: "order_id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];
    let app_orders_id = catalog
        .create_table(app_schema_id, "orders", &orders_cols, &[])
        .await
        .unwrap();

    // Set search path to ['app', 'public']
    let resolver = catalog.resolver(
        db_id,
        vec!["app".to_string(), "public".to_string()],
    );

    // Resolve unqualified 'users' -> should find app.users (app is first in search path)
    let resolved = resolver.resolve_table(None, "users").await.unwrap();
    assert_eq!(resolved.id, app_users_id);
    assert_eq!(resolved.schema_id, app_schema_id);
    tprintln!("  Resolve 'users' -> app.users: PASS");

    // Resolve unqualified 'orders' -> should find app.orders
    let resolved = resolver.resolve_table(None, "orders").await.unwrap();
    assert_eq!(resolved.id, app_orders_id);
    tprintln!("  Resolve 'orders' -> app.orders: PASS");

    // Resolve qualified 'public.users' -> should find public.users
    let resolved = resolver
        .resolve_table(Some("public"), "users")
        .await
        .unwrap();
    assert_eq!(resolved.id, public_users_id);
    assert_eq!(resolved.schema_id, DEFAULT_SCHEMA_ID);
    tprintln!("  Resolve 'public.users' -> public.users: PASS");

    // Resolve 'nonexistent' -> should return error
    let result = resolver.resolve_table(None, "nonexistent").await;
    assert!(result.is_err());
    tprintln!("  Resolve 'nonexistent' -> error: PASS");

    // Resolve column within table
    let table = resolver.resolve_table(Some("app"), "users").await.unwrap();
    let col = resolver.resolve_column(&table, "user_id").unwrap();
    assert_eq!(col.type_id, TypeId::Int64);
    tprintln!("  Resolve column 'user_id': PASS");

    // Resolve nonexistent column -> error
    let result = resolver.resolve_column(&table, "nonexistent");
    assert!(result.is_err());
    tprintln!("  Resolve nonexistent column -> error: PASS");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Name Resolution", before, after);
}

// =============================================================================
// 3. Column Type Test
// =============================================================================

#[tokio::test]
async fn phase3_03_column_types() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Column Type Roundtrip ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    let col_defs = make_all_types_column_defs();
    let expected_types = vec![
        ("c_bool", TypeId::Boolean, false, None),
        ("c_tinyint", TypeId::Int8, true, None),
        ("c_smallint", TypeId::Int16, true, None),
        ("c_int", TypeId::Int32, true, None),
        ("c_bigint", TypeId::Int64, false, None),
        ("c_int128", TypeId::Int128, true, None),
        ("c_uint8", TypeId::UInt8, true, None),
        ("c_uint16", TypeId::UInt16, true, None),
        ("c_uint32", TypeId::UInt32, true, None),
        ("c_uint64", TypeId::UInt64, true, None),
        ("c_uint128", TypeId::UInt128, true, None),
        ("c_real", TypeId::Float32, true, None),
        ("c_double", TypeId::Float64, true, None),
        ("c_decimal", TypeId::Decimal, true, None),
        ("c_char", TypeId::Char, true, Some(10)),
        ("c_varchar", TypeId::Varchar, true, Some(255)),
        ("c_text", TypeId::Text, true, None),
        ("c_binary", TypeId::Binary, true, Some(64)),
        ("c_varbinary", TypeId::Varbinary, true, Some(128)),
        ("c_bytea", TypeId::Bytea, true, None),
        ("c_date", TypeId::Date, true, None),
        ("c_time", TypeId::Time, true, None),
        ("c_timestamp", TypeId::Timestamp, true, None),
        ("c_timestamptz", TypeId::TimestampTz, true, None),
        ("c_interval", TypeId::Interval, true, None),
        ("c_uuid", TypeId::Uuid, true, None),
        ("c_json", TypeId::Json, true, None),
        ("c_jsonb", TypeId::Jsonb, true, None),
        ("c_vector", TypeId::Vector, true, Some(128)),
    ];

    let table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "all_types", &col_defs, &[])
        .await
        .unwrap();

    let table = catalog.get_table_by_id(table_id).unwrap();
    assert_eq!(table.columns.len(), expected_types.len());

    for (i, (name, type_id, nullable, max_len)) in expected_types.iter().enumerate() {
        let col = &table.columns[i];
        assert_eq!(col.name, *name, "column {} name mismatch", i);
        assert_eq!(col.type_id, *type_id, "column {} ({}) type mismatch", i, name);
        assert_eq!(col.nullable, *nullable, "column {} ({}) nullable mismatch", i, name);
        assert_eq!(
            col.max_length, *max_len,
            "column {} ({}) max_length mismatch",
            i, name
        );
    }

    tprintln!("  All {} column types roundtripped: PASS", expected_types.len());

    // Verify default expression is preserved
    let col_defs_with_default = vec![
        ColumnDef {
            name: "amount".to_string(),
            data_type: DataType::Decimal(Some(10), Some(2)),
            nullable: Some(true),
            default: Some(Expr::Literal(LiteralValue::Float(42.0))),
            constraints: vec![],
        },
    ];
    let tid = catalog
        .create_table(DEFAULT_SCHEMA_ID, "defaults_test", &col_defs_with_default, &[])
        .await
        .unwrap();
    let t = catalog.get_table_by_id(tid).unwrap();
    assert!(t.columns[0].default_expr.is_some());
    tprintln!("  Default expression preserved: PASS");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Column Types", before, after);
}

// =============================================================================
// 4. Constraint Test
// =============================================================================

#[tokio::test]
async fn phase3_04_constraints() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Constraint Storage ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    // Table with PRIMARY KEY column constraint
    let pk_cols = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![ColumnConstraint::PrimaryKey],
    }];
    let pk_table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "pk_test", &pk_cols, &[])
        .await
        .unwrap();
    let pk_table = catalog.get_table_by_id(pk_table_id).unwrap();
    let pk_constraint = pk_table
        .constraints
        .iter()
        .find(|c| c.constraint_type == ConstraintType::PrimaryKey);
    assert!(pk_constraint.is_some());
    assert_eq!(pk_constraint.unwrap().columns.len(), 1);
    tprintln!("  PRIMARY KEY (column-level): PASS");

    // Table with composite PRIMARY KEY via table constraint
    let composite_cols = vec![
        ColumnDef {
            name: "a".to_string(),
            data_type: DataType::Int,
            nullable: Some(false),
            default: None,
            constraints: vec![],
        },
        ColumnDef {
            name: "b".to_string(),
            data_type: DataType::Int,
            nullable: Some(false),
            default: None,
            constraints: vec![],
        },
    ];
    let composite_constraints = vec![TableConstraint::PrimaryKey(vec![
        "a".to_string(),
        "b".to_string(),
    ])];
    let cpk_table_id = catalog
        .create_table(
            DEFAULT_SCHEMA_ID,
            "cpk_test",
            &composite_cols,
            &composite_constraints,
        )
        .await
        .unwrap();
    let cpk_table = catalog.get_table_by_id(cpk_table_id).unwrap();
    let cpk = cpk_table
        .constraints
        .iter()
        .find(|c| c.constraint_type == ConstraintType::PrimaryKey)
        .unwrap();
    assert_eq!(cpk.columns.len(), 2);
    tprintln!("  PRIMARY KEY (table-level, composite): PASS");

    // Table with UNIQUE column constraint
    let unique_cols = vec![
        ColumnDef {
            name: "id".to_string(),
            data_type: DataType::BigInt,
            nullable: Some(false),
            default: None,
            constraints: vec![],
        },
        ColumnDef {
            name: "email".to_string(),
            data_type: DataType::Varchar(Some(255)),
            nullable: Some(false),
            default: None,
            constraints: vec![ColumnConstraint::Unique],
        },
    ];
    let uq_table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "unique_test", &unique_cols, &[])
        .await
        .unwrap();
    let uq_table = catalog.get_table_by_id(uq_table_id).unwrap();
    let uq = uq_table
        .constraints
        .iter()
        .find(|c| c.constraint_type == ConstraintType::Unique);
    assert!(uq.is_some());
    tprintln!("  UNIQUE constraint: PASS");

    // Table with NOT NULL column constraints
    let nn_cols = vec![ColumnDef {
        name: "required_field".to_string(),
        data_type: DataType::Text,
        nullable: Some(false),
        default: None,
        constraints: vec![ColumnConstraint::NotNull],
    }];
    let nn_table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "notnull_test", &nn_cols, &[])
        .await
        .unwrap();
    let nn_table = catalog.get_table_by_id(nn_table_id).unwrap();
    let nn = nn_table
        .constraints
        .iter()
        .find(|c| c.constraint_type == ConstraintType::NotNull);
    assert!(nn.is_some());
    assert!(!nn_table.columns[0].nullable);
    tprintln!("  NOT NULL constraint: PASS");

    // Table with DEFAULT values
    let def_cols = vec![ColumnDef {
        name: "status".to_string(),
        data_type: DataType::Varchar(Some(50)),
        nullable: Some(true),
        default: Some(Expr::Literal(LiteralValue::String("active".to_string()))),
        constraints: vec![],
    }];
    let def_table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "default_test", &def_cols, &[])
        .await
        .unwrap();
    let def_table = catalog.get_table_by_id(def_table_id).unwrap();
    assert!(def_table.columns[0].default_expr.is_some());
    tprintln!("  DEFAULT value: PASS");

    // Verify constraints survive reload
    let tables = catalog.list_tables(DEFAULT_SCHEMA_ID);
    assert!(tables.len() >= 5, "expected at least 5 tables, got {}", tables.len());
    tprintln!("  All constraint tables listed: PASS ({} tables)", tables.len());

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Constraints", before, after);
}

// =============================================================================
// 5. DDL Operations Test (create + drop cycle)
// =============================================================================

#[tokio::test]
async fn phase3_05_ddl_create_drop() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== DDL Create/Drop Cycle ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    // Create and verify table exists
    let col_defs = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];
    let tid = catalog
        .create_table(DEFAULT_SCHEMA_ID, "temp_table", &col_defs, &[])
        .await
        .unwrap();
    assert!(catalog.get_table(DEFAULT_SCHEMA_ID, "temp_table").is_ok());
    tprintln!("  Create table: PASS");

    // Create index
    let _idx_id = catalog
        .create_index(
            tid,
            DEFAULT_SCHEMA_ID,
            "idx_temp",
            &["id".to_string()],
            false,
            IndexType::BTree,
        )
        .await
        .unwrap();
    assert_eq!(catalog.get_indexes_for_table(tid).len(), 1);
    tprintln!("  Create index: PASS");

    // Drop index
    catalog.drop_index(tid, "idx_temp").await.unwrap();
    assert_eq!(catalog.get_indexes_for_table(tid).len(), 0);
    tprintln!("  Drop index: PASS");

    // Drop table
    catalog.drop_table(DEFAULT_SCHEMA_ID, "temp_table").await.unwrap();
    assert!(catalog.get_table(DEFAULT_SCHEMA_ID, "temp_table").is_err());
    tprintln!("  Drop table: PASS");

    // Create and drop schema
    let db_id = SYSTEM_DATABASE_ID;
    catalog.create_schema(db_id, "temp_schema", "admin").await.unwrap();
    assert!(catalog.get_schema(db_id, "temp_schema").is_ok());
    catalog.drop_schema(db_id, "temp_schema").await.unwrap();
    assert!(catalog.get_schema(db_id, "temp_schema").is_err());
    tprintln!("  Create/drop schema: PASS");

    // Create and drop database
    catalog.create_database("temp_db", "admin").await.unwrap();
    assert!(catalog.get_database("temp_db").is_ok());
    catalog.drop_database("temp_db").await.unwrap();
    assert!(catalog.get_database("temp_db").is_err());
    tprintln!("  Create/drop database: PASS");

    // Duplicate detection
    let _ = catalog.create_database("dup_db", "admin").await.unwrap();
    let dup_result = catalog.create_database("dup_db", "admin").await;
    assert!(dup_result.is_err());
    tprintln!("  Duplicate database detection: PASS");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("DDL Create/Drop", before, after);
}

// =============================================================================
// 6. Statistics Test
// =============================================================================

#[tokio::test]
async fn phase3_06_statistics() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Statistics Collection ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 8192 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );

    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
        .await
        .unwrap();

    // Create a table for statistics
    let col_defs = vec![ColumnDef {
        name: "value".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(true),
        default: None,
        constraints: vec![],
    }];
    let table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "stats_test", &col_defs, &[])
        .await
        .unwrap();
    let table = catalog.get_table_by_id(table_id).unwrap();

    // Insert 100,000 rows into a separate heap file for analysis
    let stats_heap = HeapFile::new(
        Arc::clone(&disk),
        Arc::clone(&pool),
        HeapFileConfig {
            heap_file_id: table.heap_file_id,
            fsm_file_id: table.fsm_file_id,
        },
    )
    .unwrap();
    stats_heap.init_cache().await.unwrap();

    let row_count = 100_000u64;
    let batch_size = 10_000;

    tprintln!("  Inserting {} rows...", row_count);
    for batch_start in (0..row_count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size as u64).min(row_count);
        let tuples: Vec<Tuple> = (batch_start..batch_end)
            .map(|i| {
                let data = i.to_le_bytes().to_vec();
                Tuple::new(data, 0)
            })
            .collect();
        stats_heap.insert_batch(&tuples).await.unwrap();
    }

    // Run ANALYZE
    let analyze_start = Instant::now();
    let (table_stats, col_stats) = analyze_table(&table, &stats_heap).await.unwrap();
    let analyze_elapsed = analyze_start.elapsed();

    // Verify row count
    assert_eq!(
        table_stats.row_count, row_count,
        "row count mismatch: expected {}, got {}",
        row_count, table_stats.row_count
    );
    tprintln!("  Row count: {} (expected {}): PASS", table_stats.row_count, row_count);

    // Verify page count > 0
    assert!(table_stats.page_count > 0);
    tprintln!("  Page count: {}: PASS", table_stats.page_count);

    // Verify avg_row_size is reasonable (8 bytes per u64)
    assert!(table_stats.avg_row_size > 0);
    tprintln!("  Avg row size: {} bytes: PASS", table_stats.avg_row_size);

    // Verify column statistics populated
    assert!(!col_stats.is_empty());
    let cs = &col_stats[0];
    assert!(cs.distinct_count > 0);
    tprintln!("  Distinct count: {}: PASS", cs.distinct_count);

    // Verify histogram built
    assert!(cs.histogram.is_some());
    let hist = cs.histogram.as_ref().unwrap();
    assert!(hist.num_buckets > 0);
    let total_in_buckets: u64 = hist.counts.iter().sum();
    assert_eq!(total_in_buckets, row_count);
    tprintln!(
        "  Histogram: {} buckets, total={}: PASS",
        hist.num_buckets, total_in_buckets
    );

    // Verify MCV
    assert!(!cs.most_common_values.is_empty());
    tprintln!("  MCV: {} entries: PASS", cs.most_common_values.len());

    // Store stats in catalog
    catalog.put_stats(table_id, table_stats.clone(), col_stats.clone());
    let retrieved = catalog.get_stats(table_id);
    assert!(retrieved.is_some());
    let (ts, _cs_list) = retrieved.unwrap();
    assert_eq!(ts.row_count, row_count);
    tprintln!("  Stats store/retrieve: PASS");

    // Benchmark analyze throughput
    let analyze_us = analyze_elapsed.as_micros() as f64;
    let rows_per_sec = row_count as f64 / (analyze_us / 1_000_000.0);
    tprintln!("  ANALYZE: {:.0} us for {} rows ({:.0} rows/sec)",
        analyze_us, row_count, rows_per_sec);

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Statistics", before, after);
}

// =============================================================================
// 7. Recovery Test
// =============================================================================

#[tokio::test]
async fn phase3_07_recovery() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Crash Recovery ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));

    // Phase 1: Create catalog objects and "crash" (close WAL without checkpoint)
    let table_id;
    {
        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir: wal_dir.clone(),
                fsync_enabled: false,
                ..Default::default()
            })
            .unwrap(),
        );

        let storage =
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
            .await
            .unwrap();

        // Create several objects (all WAL-logged)
        let db_id = catalog.create_database("recovery_db", "admin").await.unwrap();
        let schema_id = catalog
            .create_schema(db_id, "recovery_schema", "admin")
            .await
            .unwrap();

        let col_defs = vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::BigInt,
                nullable: Some(false),
                default: None,
                constraints: vec![],
            },
            ColumnDef {
                name: "data".to_string(),
                data_type: DataType::Text,
                nullable: Some(true),
                default: None,
                constraints: vec![],
            },
        ];
        table_id = catalog
            .create_table(schema_id, "recovery_table", &col_defs, &[])
            .await
            .unwrap();

        // Ensure WAL is flushed
        wal.flush().unwrap();
        wal.close().unwrap();
        // "Crash" - drop catalog and cache, simulate restart
    }

    // Phase 2: "Recover" - create fresh catalog from same storage
    {
        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir: wal_dir.clone(),
                fsync_enabled: false,
                ..Default::default()
            })
            .unwrap(),
        );

        let storage =
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .unwrap();

        // Verify objects survived the "crash"
        let db = catalog.get_database("recovery_db").unwrap();
        assert_eq!(db.name, "recovery_db");
        tprintln!("  Database survived crash: PASS");

        let schema = catalog.get_schema(db.id, "recovery_schema").unwrap();
        assert_eq!(schema.name, "recovery_schema");
        tprintln!("  Schema survived crash: PASS");

        let table = catalog.get_table(schema.id, "recovery_table").unwrap();
        assert_eq!(table.id, table_id);
        assert_eq!(table.columns.len(), 2);
        assert_eq!(table.columns[0].name, "id");
        assert_eq!(table.columns[1].name, "data");
        tprintln!("  Table survived crash: PASS (2 columns)");

        wal.close().unwrap();
    }

    let after = take_util_snapshot();
    record_test_util("Crash Recovery", before, after);
}

// =============================================================================
// 8. Performance Benchmarks
// =============================================================================

#[tokio::test]
async fn phase3_08_bench_table_lookup() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Table Lookup Latency ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    // Create a table to look up
    let col_defs = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];
    catalog
        .create_table(DEFAULT_SCHEMA_ID, "bench_table", &col_defs, &[])
        .await
        .unwrap();

    // Warm the cache
    for _ in 0..100 {
        let _ = catalog.get_table(DEFAULT_SCHEMA_ID, "bench_table").unwrap();
    }

    let iterations = 1_000_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(
                catalog.get_table(DEFAULT_SCHEMA_ID, "bench_table"),
            );
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let ns_per_op = elapsed_ns / iterations as f64;
        runs.push(ns_per_op);
    }

    let v = validate_metric(
        "Table Lookup",
        "Lookup latency (ns/op)",
        runs,
        TABLE_LOOKUP_TARGET_NS,
        false,
    );
    assert!(v.passed, "Table lookup latency exceeded target");
    assert!(!v.regression_detected, "Table lookup regression detected");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Table Lookup", before, after);
}

#[tokio::test]
async fn phase3_09_bench_schema_resolve() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Schema Resolve Latency ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    let db_id = SYSTEM_DATABASE_ID;
    let app_id = catalog.create_schema(db_id, "app", "admin").await.unwrap();
    let col_defs = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];
    catalog
        .create_table(app_id, "target", &col_defs, &[])
        .await
        .unwrap();

    let resolver = catalog.resolver(db_id, vec!["app".to_string(), "public".to_string()]);

    // Warm
    for _ in 0..100 {
        let _ = resolver.resolve_table(None, "target").await.unwrap();
    }

    let iterations = 500_000u64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(
                resolver.resolve_table(None, "target").await,
            );
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let ns_per_op = elapsed_ns / iterations as f64;
        runs.push(ns_per_op);
    }

    let v = validate_metric(
        "Schema Resolve",
        "Resolve latency (ns/op)",
        runs,
        SCHEMA_RESOLVE_TARGET_NS,
        false,
    );
    assert!(v.passed, "Schema resolve latency exceeded target");
    assert!(!v.regression_detected, "Schema resolve regression detected");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Schema Resolve", before, after);
}

#[tokio::test]
async fn phase3_10_bench_ddl_create() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: DDL Create Latency ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    let col_defs = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];

    let iterations = 200;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        let mut total_us = 0.0;
        for i in 0..iterations {
            let table_name = format!("create_bench_{run}_{i}");
            let start = Instant::now();
            catalog
                .create_table(DEFAULT_SCHEMA_ID, &table_name, &col_defs, &[])
                .await
                .unwrap();
            total_us += start.elapsed().as_micros() as f64;
        }
        let us_per_op = total_us / iterations as f64;
        runs.push(us_per_op);
    }

    let v = validate_metric(
        "DDL Create",
        "Create table latency (us/op)",
        runs,
        DDL_CREATE_TARGET_US,
        false,
    );
    assert!(v.passed, "DDL create latency exceeded target");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("DDL Create", before, after);
}

#[tokio::test]
async fn phase3_11_bench_ddl_drop() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: DDL Drop Latency ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    let col_defs = vec![ColumnDef {
        name: "id".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(false),
        default: None,
        constraints: vec![],
    }];

    let iterations = 200;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        // Pre-create tables
        for i in 0..iterations {
            let table_name = format!("drop_bench_{run}_{i}");
            catalog
                .create_table(DEFAULT_SCHEMA_ID, &table_name, &col_defs, &[])
                .await
                .unwrap();
        }

        let mut total_us = 0.0;
        for i in 0..iterations {
            let table_name = format!("drop_bench_{run}_{i}");
            let start = Instant::now();
            catalog
                .drop_table(DEFAULT_SCHEMA_ID, &table_name)
                .await
                .unwrap();
            total_us += start.elapsed().as_micros() as f64;
        }
        let us_per_op = total_us / iterations as f64;
        runs.push(us_per_op);
    }

    let v = validate_metric(
        "DDL Drop",
        "Drop table latency (us/op)",
        runs,
        DDL_DROP_TARGET_US,
        false,
    );
    assert!(v.passed, "DDL drop latency exceeded target");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("DDL Drop", before, after);
}

#[tokio::test]
async fn phase3_12_bench_analyze() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: ANALYZE Throughput ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 16384 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            fsync_enabled: false,
            ..Default::default()
        })
        .unwrap(),
    );

    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
        .await
        .unwrap();

    let col_defs = vec![ColumnDef {
        name: "value".to_string(),
        data_type: DataType::BigInt,
        nullable: Some(true),
        default: None,
        constraints: vec![],
    }];
    let table_id = catalog
        .create_table(DEFAULT_SCHEMA_ID, "analyze_bench", &col_defs, &[])
        .await
        .unwrap();
    let table = catalog.get_table_by_id(table_id).unwrap();

    let stats_heap = HeapFile::new(
        Arc::clone(&disk),
        Arc::clone(&pool),
        HeapFileConfig {
            heap_file_id: table.heap_file_id,
            fsm_file_id: table.fsm_file_id,
        },
    )
    .unwrap();
    stats_heap.init_cache().await.unwrap();

    // Insert 100K rows
    let row_count = 100_000u64;
    let batch_size = 10_000;
    for batch_start in (0..row_count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size as u64).min(row_count);
        let tuples: Vec<Tuple> = (batch_start..batch_end)
            .map(|i| Tuple::new(i.to_le_bytes().to_vec(), 0))
            .collect();
        stats_heap.insert_batch(&tuples).await.unwrap();
    }

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let _ = analyze_table(&table, &stats_heap).await.unwrap();
        let elapsed_secs = start.elapsed().as_secs_f64();
        let rows_per_sec = row_count as f64 / elapsed_secs;
        runs.push(rows_per_sec);
    }

    let v = validate_metric(
        "ANALYZE",
        "ANALYZE throughput (rows/sec)",
        runs,
        ANALYZE_TARGET_ROWS_SEC,
        true,
    );
    assert!(v.passed, "ANALYZE throughput below target");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("ANALYZE", before, after);
}

#[tokio::test]
async fn phase3_13_bench_histogram() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Histogram Build Latency ===");
    let before = take_util_snapshot();

    use zyron_catalog::stats::Histogram;

    // Prepare 100K sorted samples (simulating what analyze_table produces)
    let num_samples = 100_000usize;
    let sorted_values: Vec<Vec<u8>> = (0..num_samples as u64)
        .map(|i| i.to_le_bytes().to_vec())
        .collect();

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let hist = std::hint::black_box(Histogram::build(&sorted_values, 100));
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(hist.num_buckets > 0);
        runs.push(elapsed_ms);
    }

    let v = validate_metric(
        "Histogram Build",
        "Histogram build latency (ms/col)",
        runs,
        HISTOGRAM_TARGET_MS_PER_COL,
        false,
    );
    assert!(v.passed, "Histogram build latency exceeded target");

    let after = take_util_snapshot();
    record_test_util("Histogram Build", before, after);
}

#[tokio::test]
async fn phase3_14_bench_cache_hit_rate() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Cache Hit Rate ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let (_disk, _pool, wal, catalog) = setup_catalog(dir.path()).await;

    // Create 50 tables
    let num_tables = 50;
    for i in 0..num_tables {
        let col_defs = vec![ColumnDef {
            name: "id".to_string(),
            data_type: DataType::BigInt,
            nullable: Some(false),
            default: None,
            constraints: vec![],
        }];
        catalog
            .create_table(DEFAULT_SCHEMA_ID, &format!("cache_t{}", i), &col_defs, &[])
            .await
            .unwrap();
    }

    // Warm cache
    for i in 0..num_tables {
        let _ = catalog.get_table(DEFAULT_SCHEMA_ID, &format!("cache_t{}", i));
    }

    // Measure hit rate: all lookups should succeed from cache
    let iterations = 10_000;
    let mut hits = 0u64;
    let mut total = 0u64;

    for _ in 0..iterations {
        for i in 0..num_tables {
            total += 1;
            if catalog
                .get_table(DEFAULT_SCHEMA_ID, &format!("cache_t{}", i))
                .is_ok()
            {
                hits += 1;
            }
        }
    }

    let hit_rate = hits as f64 / total as f64;
    let passed = check_performance(
        "Cache Hit Rate",
        "Cache hit rate",
        hit_rate,
        CACHE_HIT_RATE_TARGET,
        true,
    );
    assert!(passed, "Cache hit rate below target");

    wal.close().unwrap();
    let after = take_util_snapshot();
    record_test_util("Cache Hit Rate", before, after);
}

#[tokio::test]
async fn phase3_15_bench_recovery() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Benchmark: Recovery Time ===");
    let before = take_util_snapshot();

    let dir = tempdir().unwrap();
    let data_dir = dir.path().join("data");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: data_dir.clone(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 4096 }));

    // Create catalog objects that will be recovered
    {
        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir: wal_dir.clone(),
                fsync_enabled: false,
                ..Default::default()
            })
            .unwrap(),
        );

        let storage =
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
            .await
            .unwrap();

        let db_id = catalog.create_database("bench_db", "admin").await.unwrap();
        let sid = catalog.create_schema(db_id, "bench_schema", "admin").await.unwrap();
        let col_defs = vec![ColumnDef {
            name: "id".to_string(),
            data_type: DataType::BigInt,
            nullable: Some(false),
            default: None,
            constraints: vec![],
        }];
        for i in 0..20 {
            catalog
                .create_table(sid, &format!("recovery_t{}", i), &col_defs, &[])
                .await
                .unwrap();
        }

        wal.flush().unwrap();
        wal.close().unwrap();
    }

    // Measure recovery time across 5 runs
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();

        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir: wal_dir.clone(),
                fsync_enabled: false,
                ..Default::default()
            })
            .unwrap(),
        );

        let storage =
            HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
        storage.init_cache().await.unwrap();
        let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
        let cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog = Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .unwrap();

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        runs.push(elapsed_ms);

        // Verify correctness of recovered data
        assert!(catalog.get_database("bench_db").is_ok());

        wal.close().unwrap();
    }

    let v = validate_metric(
        "Recovery",
        "Recovery time (ms)",
        runs,
        RECOVERY_TARGET_MS,
        false,
    );
    assert!(v.passed, "Recovery time exceeded target");

    let after = take_util_snapshot();
    record_test_util("Recovery", before, after);
}

// =============================================================================
// Summary
// =============================================================================

#[tokio::test]
async fn phase3_99_summary() {
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n============================================");
    tprintln!("  Phase 3: Catalog - All Tests Complete");
    tprintln!("============================================");
    tprintln!("  DDL Persistence:    1 test (create + reload)");
    tprintln!("  Name Resolution:    1 test (6 cases)");
    tprintln!("  Column Types:       1 test (29 types)");
    tprintln!("  Constraints:        1 test (5 constraint types)");
    tprintln!("  DDL Create/Drop:    1 test (8 operations)");
    tprintln!("  Statistics:         1 test (100K rows)");
    tprintln!("  Crash Recovery:     1 test (3 object types)");
    tprintln!("  Benchmarks:         8 performance tests");
    tprintln!("============================================");
}
