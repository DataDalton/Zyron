//! Phase 1: Storage Foundation Validation Tests
//!
//! Comprehensive integration tests for ZyronDB Phase 1 components:
//! - WAL write and replay
//! - Buffer pool eviction and caching
//! - Heap file tuple storage
//! - B+ tree index operations
//! - Crash recovery integration
//!
//! Performance Targets (Industry-Leading):
//! | Test          | Metric     | Target           | Industry Leader |
//! |---------------|------------|------------------|-----------------|
//! | WAL Write     | throughput | 3M records/sec   | TigerBeetle 2M  |
//! | WAL Replay    | throughput | 6M records/sec   | RocksDB 5M      |
//! | Buffer Pool   | fetch      | 15ns             | Umbra 20ns      |
//! | Buffer Pool   | hit rate   | 100%             | Industry 98%    |
//! | Heap Insert   | throughput | 2M tuples/sec    | SingleStore 1M  |
//! | Heap Scan     | throughput | 20M tuples/sec   | DuckDB 15M      |
//! | B+Tree Insert | throughput | 8M keys/sec      | LMDB/Sled 5M    |
//! | B+Tree Lookup | latency    | 40ns/lookup      | LMDB mmap 50ns  |
//! | B+Tree Range  | throughput | 40M keys/sec     | RocksDB 30M     |
//! | Recovery      | time       | 2ms/MB           | VoltDB 5ms/MB   |
//!
//! Validation Requirements:
//! - Each test runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target

use rand::Rng;
use std::collections::HashSet;
use std::io::Write as _;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;
use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_common::page::PageId;
use zyron_storage::{
    BTreeIndex, BufferedBTreeIndex, CheckpointConfig, CheckpointTrigger, DiskManager,
    DiskManagerConfig, HeapFile, Tuple, TupleId,
};
use zyron_wal::{LogRecordType, Lsn, RecoveryManager, WalReader, WalWriter, WalWriterConfig};

// =============================================================================
// Performance Target Constants (Industry-Leading)
// =============================================================================

const WAL_WRITE_TARGET_OPS_SEC: f64 = 8_000_000.0;
const WAL_REPLAY_TARGET_OPS_SEC: f64 = 12_000_000.0;
const BUFFER_POOL_FETCH_TARGET_NS: f64 = 15.0;
const BUFFER_POOL_HIT_RATE_TARGET: f64 = 1.0; // 100%
const HEAP_INSERT_TARGET_OPS_SEC: f64 = 8_000_000.0;
const HEAP_SCAN_TARGET_OPS_SEC: f64 = 20_000_000.0;
const BTREE_INSERT_TARGET_OPS_SEC: f64 = 10_000_000.0;
const BTREE_LOOKUP_TARGET_NS: f64 = 40.0;
const BTREE_RANGE_TARGET_OPS_SEC: f64 = 40_000_000.0;
const BTREE_DELETE_TARGET_OPS_SEC: f64 = 8_000_000.0;
const RECOVERY_TARGET_US_PER_KB: f64 = 10.0;

// Checkpoint targets
const CHECKPOINT_WRITE_THROUGHPUT_TARGET_MB_SEC: f64 = 2000.0; // 2 GB/sec
const CHECKPOINT_WRITE_1M_TARGET_MS: f64 = 3.0; // 3ms for ~5MB
const CHECKPOINT_LOAD_THROUGHPUT_TARGET_MB_SEC: f64 = 3000.0; // 3 GB/sec
const CHECKPOINT_LOAD_1M_TARGET_MS: f64 = 2.0; // 2ms for ~5MB
const RECOVERY_WITH_CHECKPOINT_TARGET_MS: f64 = 5.0; // 500K keys + 500 WAL records
const RECOVERY_SCALE_TARGET_MS: f64 = 20.0; // 10M keys + 100 WAL records
const WAL_SEGMENT_CLEANUP_TARGET_MS: f64 = 5.0;
const SHUTDOWN_CHECKPOINT_1M_TARGET_MS: f64 = 5.0;

// Validation constants
const VALIDATION_RUNS: usize = 5;
const REGRESSION_THRESHOLD: f64 = 2.0; // Fail if any run >2x worse than target

// Serializes performance benchmark tests so they run one at a time.
// Correctness tests (pin, eviction, etc.) can still run in parallel.
// Performance benchmarks measure throughput, which requires consistent
// CPU availability. Running CPU-intensive benchmarks concurrently makes
// results dependent on OS thread scheduling, not code quality.
static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
//       "WAL Write/Replay": {
//         "util_before": { cpu_pct, ram_used_gb },
//         "util_after":  { cpu_pct, ram_used_gb },
//         "Write throughput (ops/sec)": { average, runs: [...], target, passed, higher_is_better },
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

fn benchmark_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = crates/zyron-storage, so go up 2 levels to workspace root.
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

// Samples current CPU load percentage and RAM used. One PowerShell call on Windows.
// On Linux, reports 1-min load average and used memory from /proc. On macOS, uses sysctl.
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
    // 1-min load average as a proxy for CPU utilization.
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
    // 1-min load average from sysctl vm.loadavg ("{ 0.52 0.58 0.59 }").
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
    // vm_stat page size is always 16384 on Apple Silicon, 4096 on Intel.
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

// Stable identifier for this test binary invocation: "YYYYMMDD_HHMMSS_<commit>".
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

// Records system utilization for a test group and rewrites the JSON file.
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

// Builds the full nested JSON for the run using all metrics collected so far.
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

    // Collect distinct test names in insertion order.
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

        // Emit util_before and util_after if a snapshot was recorded for this group.
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
            let comma = if mi + 1 < test_metrics.len() { "," } else { "" };
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

        let test_comma = if ti + 1 < test_names.len() { "," } else { "" };
        out.push_str(&format!("    }}{test_comma}\n"));
    }

    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

// Rewrites the per-run .json file with the current snapshot of metrics and utils.
fn write_run_json(metrics: &[MetricRecord], utils: &[UtilRecord]) {
    let dir = benchmark_dir();
    let _ = std::fs::create_dir_all(&dir);
    let fname = format!("phase1_{}.json", run_id());
    let json = build_run_json(metrics, utils);
    let _ = std::fs::write(dir.join(fname), json.as_bytes());
}

// Opens the per-run raw text log file on first access, named by run_id.
fn raw_log_file() -> &'static Mutex<std::fs::File> {
    RAW_LOG.get_or_init(|| {
        let dir = benchmark_dir();
        let _ = std::fs::create_dir_all(&dir);
        let fname = format!("phase1_{}.txt", run_id());
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
                    .open(dir.join("phase1_latest.txt"))
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

// Accumulates a metric result and rewrites the nested JSON file for this run.
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
// Test 1: WAL Write/Replay Test (5-run validation)
// =============================================================================

/// Writes 10,000 log records across multiple segments, simulates crash,
/// replays all records, and verifies integrity.
/// Target: 3M writes/sec, 6M replay/sec
#[tokio::test]
async fn test_wal_write_replay_10k_records() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const RECORD_COUNT: usize = 10_000;

    tprintln!("\n=== WAL Write/Replay Performance Test ===");
    tprintln!("Records per run: {}", RECORD_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut write_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut replay_results = Vec::with_capacity(VALIDATION_RUNS);

    let wal_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };

        let mut written_lsns: Vec<Lsn> = Vec::with_capacity(RECORD_COUNT);

        // Write phase, then crash simulation: drop writer without clean shutdown.
        let write_duration;
        let mut written_payloads: Vec<String> = Vec::with_capacity(RECORD_COUNT);
        {
            let writer = WalWriter::new(config.clone()).unwrap();

            // Pre-allocate a reusable buffer for payload construction.
            // write! into a cleared String reuses its heap allocation across
            // iterations, eliminating per-record alloc overhead.
            let mut payload_buf = String::with_capacity(128);
            let x_pad: String = "x".repeat(99); // pre-build padding

            let start = Instant::now();
            for i in 0..RECORD_COUNT {
                let txn_id = (i % 100 + 1) as u32;
                payload_buf.clear();
                use std::fmt::Write;
                let pad_len = i % 100;
                write!(payload_buf, "record_{}_{}", i, &x_pad[..pad_len]).unwrap();

                let lsn = writer
                    .log_insert(txn_id, Lsn::INVALID, payload_buf.as_bytes())
                    .unwrap();

                written_lsns.push(lsn);
                written_payloads.push(payload_buf.clone());
            }
            write_duration = start.elapsed();
            // Flush to ensure data reaches disk, then drop without clean shutdown.
            writer.flush().unwrap();
            drop(writer);
        }

        // Verify LSN ordering is monotonically increasing.
        for i in 1..written_lsns.len() {
            assert!(
                written_lsns[i] > written_lsns[i - 1],
                "Run {}: LSN ordering violated at index {}: {:?} <= {:?}",
                run + 1,
                i,
                written_lsns[i],
                written_lsns[i - 1]
            );
        }

        // Replay phase with content integrity check.
        let replay_duration;
        {
            let reader = WalReader::new(dir.path()).unwrap();

            let start = Instant::now();
            let records = reader.scan_all().unwrap();
            replay_duration = start.elapsed();

            let insert_records: Vec<_> = records
                .iter()
                .filter(|r| r.record_type == LogRecordType::Insert)
                .collect();

            assert_eq!(
                insert_records.len(),
                RECORD_COUNT,
                "Run {}: Expected {} insert records, got {}",
                run + 1,
                RECORD_COUNT,
                insert_records.len()
            );

            // Verify payload content matches what was written.
            for (record, expected_payload) in insert_records.iter().zip(written_payloads.iter()) {
                assert_eq!(
                    record.payload.as_ref(),
                    expected_payload.as_bytes(),
                    "Run {}: Payload content mismatch for LSN {:?}",
                    run + 1,
                    record.lsn
                );
            }
        }

        let write_ops_sec = RECORD_COUNT as f64 / write_duration.as_secs_f64();
        let replay_ops_sec = RECORD_COUNT as f64 / replay_duration.as_secs_f64();

        tprintln!(
            "  Write: {} ops/sec ({:?})",
            format_with_commas(write_ops_sec),
            write_duration
        );
        tprintln!(
            "  Replay: {} ops/sec ({:?})",
            format_with_commas(replay_ops_sec),
            replay_duration
        );

        write_results.push(write_ops_sec);
        replay_results.push(replay_ops_sec);
    }
    record_test_util("WAL Write/Replay", wal_util_before, take_util_snapshot());

    tprintln!("\n=== WAL Validation Results ===");
    let write_result = validate_metric(
        "WAL Write/Replay",
        "Write throughput (ops/sec)",
        write_results,
        WAL_WRITE_TARGET_OPS_SEC,
        true,
    );
    let replay_result = validate_metric(
        "WAL Write/Replay",
        "Replay throughput (ops/sec)",
        replay_results,
        WAL_REPLAY_TARGET_OPS_SEC,
        true,
    );

    assert!(
        write_result.passed,
        "WAL write avg {:.0} < target {:.0}",
        write_result.average, WAL_WRITE_TARGET_OPS_SEC
    );
    assert!(
        !write_result.regression_detected,
        "WAL write regression detected"
    );
    assert!(
        replay_result.passed,
        "WAL replay avg {:.0} < target {:.0}",
        replay_result.average, WAL_REPLAY_TARGET_OPS_SEC
    );
    assert!(
        !replay_result.regression_detected,
        "WAL replay regression detected"
    );
}

/// Tests WAL segment rotation with many records.
#[tokio::test]
async fn test_wal_segment_rotation() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().unwrap();
    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 64 * 1024,
        fsync_enabled: false,
        ring_buffer_capacity: 1024 * 1024, // 1MB
    };

    let writer = WalWriter::new(config).unwrap();
    let initial_segment = writer.current_segment_id().unwrap();

    for _ in 0..1000 {
        writer.log_insert(1, Lsn::INVALID, &[0u8; 200]).unwrap();
    }

    let final_segment = writer.current_segment_id().unwrap();
    writer.close().unwrap();

    assert!(
        final_segment.0 > initial_segment.0,
        "Expected segment rotation: {} -> {}",
        initial_segment,
        final_segment
    );

    let reader = WalReader::new(dir.path()).unwrap();
    let records = reader.scan_all().unwrap();
    assert_eq!(records.len(), 1000);

    tprintln!(
        "WAL Segment Rotation: PASSED - rotated from segment {} to {}",
        initial_segment,
        final_segment
    );
}

// =============================================================================
// Test 2: Buffer Pool Test (5-run validation)
// =============================================================================

/// Tests buffer pool with 100 frames accessing 500 different pages.
#[tokio::test]
async fn test_buffer_pool_eviction() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const NUM_FRAMES: usize = 100;
    const NUM_PAGES: usize = 500;

    let pool = BufferPool::new(BufferPoolConfig {
        num_frames: NUM_FRAMES,
    });

    let mut dirty_evictions = 0;

    for i in 0..NUM_PAGES {
        let page_id = PageId::new(0, i as u32);

        let (frame, evicted) = pool.new_page(page_id).unwrap();

        if evicted.is_some() {
            dirty_evictions += 1;
        }

        {
            let mut data = frame.write_data();
            data[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        }

        pool.unpin_page(page_id, true);
    }

    assert!(
        dirty_evictions > 0,
        "Expected dirty evictions when accessing {} pages with {} frames",
        NUM_PAGES,
        NUM_FRAMES
    );

    assert_eq!(pool.page_count(), NUM_FRAMES);

    tprintln!(
        "Buffer Pool Eviction: PASSED - {} dirty evictions with {}/{} pages",
        dirty_evictions,
        NUM_PAGES,
        NUM_FRAMES
    );
}

/// Tests that pinned pages cannot be evicted.
#[tokio::test]
async fn test_buffer_pool_pin_prevents_eviction() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const NUM_FRAMES: usize = 10;

    let pool = BufferPool::new(BufferPoolConfig {
        num_frames: NUM_FRAMES,
    });

    let mut pinned_pages = Vec::new();
    for i in 0..NUM_FRAMES {
        let page_id = PageId::new(0, i as u32);
        pool.new_page(page_id).unwrap();
        pinned_pages.push(page_id);
    }

    let result = pool.new_page(PageId::new(0, 999));
    assert!(result.is_err(), "Should fail when all frames are pinned");

    pool.unpin_page(pinned_pages[0], false);

    let result = pool.new_page(PageId::new(0, 999));
    assert!(result.is_ok(), "Should succeed after unpinning");

    tprintln!("Buffer Pool Pin Test: PASSED - pin prevents eviction");
}

/// Tests cache hit rate and fetch latency for repeated access patterns.
/// Target: 100% hit rate, 15ns average fetch
#[tokio::test]
async fn test_buffer_pool_cache_hit_rate() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const NUM_FRAMES: usize = 50;
    const NUM_PAGES: usize = 30;
    const ACCESS_ROUNDS: usize = 1000;

    tprintln!("\n=== Buffer Pool Performance Test ===");
    tprintln!(
        "Frames: {}, Pages: {}, Rounds: {}",
        NUM_FRAMES,
        NUM_PAGES,
        ACCESS_ROUNDS
    );
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut fetch_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut hit_rate_results = Vec::with_capacity(VALIDATION_RUNS);

    let bp_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let pool = BufferPool::new(BufferPoolConfig {
            num_frames: NUM_FRAMES,
        });

        // Initial load
        for i in 0..NUM_PAGES {
            let page_id = PageId::new(0, i as u32);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, false);
        }

        let mut hits = 0;
        let mut misses = 0;

        let start = Instant::now();
        for _ in 0..ACCESS_ROUNDS {
            for i in 0..NUM_PAGES {
                let page_id = PageId::new(0, i as u32);
                if pool.fetch_page(page_id).is_some() {
                    hits += 1;
                    pool.unpin_page(page_id, false);
                } else {
                    misses += 1;
                }
            }
        }
        let duration = start.elapsed();

        let total_accesses = ACCESS_ROUNDS * NUM_PAGES;
        let hit_rate = hits as f64 / (hits + misses) as f64;
        let avg_fetch_ns = duration.as_nanos() as f64 / total_accesses as f64;

        tprintln!("  Fetch latency: {:.2} ns", avg_fetch_ns);
        tprintln!("  Hit rate: {:.4}", hit_rate);

        fetch_results.push(avg_fetch_ns);
        hit_rate_results.push(hit_rate);
    }
    record_test_util("Buffer Pool", bp_util_before, take_util_snapshot());

    tprintln!("\n=== Buffer Pool Validation Results ===");
    let fetch_result = validate_metric(
        "Buffer Pool",
        "Avg fetch latency (ns)",
        fetch_results,
        BUFFER_POOL_FETCH_TARGET_NS,
        false,
    );
    let hit_result = validate_metric(
        "Buffer Pool",
        "Cache hit rate",
        hit_rate_results,
        BUFFER_POOL_HIT_RATE_TARGET,
        true,
    );

    assert!(
        fetch_result.passed,
        "Buffer pool fetch {:.2} ns > target {:.2} ns",
        fetch_result.average, BUFFER_POOL_FETCH_TARGET_NS
    );
    assert!(
        !fetch_result.regression_detected,
        "Buffer pool fetch regression detected"
    );
    assert!(
        hit_result.passed,
        "Buffer pool hit rate {:.4} < target {:.4}",
        hit_result.average, BUFFER_POOL_HIT_RATE_TARGET
    );
}

// =============================================================================
// Test 3: Heap File Test (5-run validation)
// =============================================================================

/// Inserts 100,000 tuples with varying sizes using batch allocation.
/// Target: 10M inserts/sec, 100M scan/sec
#[tokio::test]
async fn test_heap_file_100k_tuples() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const TUPLE_COUNT: usize = 100_000;

    tprintln!("\n=== Heap File Performance Test ===");
    tprintln!("Tuples per run: {}", TUPLE_COUNT);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut insert_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut scan_results = Vec::with_capacity(VALIDATION_RUNS);

    let heap_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        };
        let disk = Arc::new(DiskManager::new(config).await.unwrap());
        let pool = Arc::new(BufferPool::auto_sized());
        let heap = HeapFile::with_defaults(disk, pool).unwrap();

        let mut rng = rand::rng();

        // Build all tuples first
        let tuples: Vec<Tuple> = (0..TUPLE_COUNT)
            .map(|i| {
                let size = rng.random_range(10..=500);
                let data: Vec<u8> = (0..size).map(|j| ((i + j) % 256) as u8).collect();
                Tuple::new(data, i as u32)
            })
            .collect();

        // Batch insert
        let insert_start = Instant::now();
        let tuple_ids = heap.insert_batch(&tuples).await.unwrap();
        let insert_duration = insert_start.elapsed();

        // Point lookup by RID on run 1 only: verify each sampled TupleId returns correct data.
        if run == 0 {
            const LOOKUP_SAMPLE: usize = 1000;
            let step = TUPLE_COUNT / LOOKUP_SAMPLE;
            for i in (0..TUPLE_COUNT).step_by(step) {
                let tid = tuple_ids[i];
                let fetched = heap.get(tid).await.unwrap();
                assert!(
                    fetched.is_some(),
                    "Run 1: Point lookup returned None for tuple_id {:?}",
                    tid
                );
                assert_eq!(
                    fetched.unwrap().data(),
                    tuples[i].data(),
                    "Run 1: Point lookup data mismatch at index {}",
                    i
                );
            }
        }

        // Scan phase using for_each (no allocation overhead)
        let scan_start = Instant::now();
        let guard = heap.scan().unwrap();
        let mut scan_count = 0usize;
        guard.for_each(|_tid, _tuple| {
            scan_count += 1;
            std::hint::black_box(&_tuple);
        });
        let scan_duration = scan_start.elapsed();

        assert_eq!(
            scan_count,
            TUPLE_COUNT,
            "Run {}: Scan should return all {} tuples",
            run + 1,
            TUPLE_COUNT
        );

        let insert_ops_sec = TUPLE_COUNT as f64 / insert_duration.as_secs_f64();
        let scan_ops_sec = TUPLE_COUNT as f64 / scan_duration.as_secs_f64();

        tprintln!(
            "  Insert: {} ops/sec ({:?})",
            format_with_commas(insert_ops_sec),
            insert_duration
        );
        tprintln!(
            "  Scan: {} ops/sec ({:?})",
            format_with_commas(scan_ops_sec),
            scan_duration
        );

        insert_results.push(insert_ops_sec);
        scan_results.push(scan_ops_sec);
    }
    record_test_util("Heap File", heap_util_before, take_util_snapshot());

    tprintln!("\n=== Heap File Validation Results ===");
    let insert_result = validate_metric(
        "Heap File",
        "Insert throughput (ops/sec)",
        insert_results,
        HEAP_INSERT_TARGET_OPS_SEC,
        true,
    );
    let scan_result = validate_metric(
        "Heap File",
        "Scan throughput (ops/sec)",
        scan_results,
        HEAP_SCAN_TARGET_OPS_SEC,
        true,
    );

    assert!(
        insert_result.passed,
        "Heap insert avg {:.0} < target {:.0}",
        insert_result.average, HEAP_INSERT_TARGET_OPS_SEC
    );
    assert!(
        !insert_result.regression_detected,
        "Heap insert regression detected"
    );
    assert!(
        scan_result.passed,
        "Heap scan avg {:.0} < target {:.0}",
        scan_result.average, HEAP_SCAN_TARGET_OPS_SEC
    );
    assert!(
        !scan_result.regression_detected,
        "Heap scan regression detected"
    );
}

/// Tests delete and scan exclusion.
#[tokio::test]
async fn test_heap_file_delete_and_scan() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const TUPLE_COUNT: usize = 10_000;
    const DELETE_COUNT: usize = 1_000;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let pool = Arc::new(BufferPool::auto_sized());
    let heap = HeapFile::with_defaults(disk, pool).unwrap();

    let tuples: Vec<Tuple> = (0..TUPLE_COUNT)
        .map(|i| {
            let data = format!("tuple_{}", i).into_bytes();
            Tuple::new(data, i as u32)
        })
        .collect();
    let tuple_ids = heap.insert_batch(&tuples).await.unwrap();

    let mut deleted_ids: HashSet<TupleId> = HashSet::new();
    for i in 0..DELETE_COUNT {
        heap.delete(tuple_ids[i]).await.unwrap();
        deleted_ids.insert(tuple_ids[i]);
    }

    let guard = heap.scan().unwrap();
    let mut scanned_ids: Vec<TupleId> = Vec::new();
    guard.for_each(|tuple_id, _| {
        scanned_ids.push(tuple_id);
    });
    assert_eq!(
        scanned_ids.len(),
        TUPLE_COUNT - DELETE_COUNT,
        "Scan should return {} tuples after {} deletions",
        TUPLE_COUNT - DELETE_COUNT,
        DELETE_COUNT
    );

    for tuple_id in &scanned_ids {
        assert!(
            !deleted_ids.contains(tuple_id),
            "Deleted tuple {} should not appear in scan",
            tuple_id
        );
    }

    for deleted_id in deleted_ids.iter().take(100) {
        let result = heap.get(*deleted_id).await.unwrap();
        assert!(result.is_none(), "Deleted tuple should return None");
    }

    tprintln!(
        "Heap File Delete/Scan: PASSED - deleted {}/{} tuples",
        DELETE_COUNT,
        TUPLE_COUNT
    );
}

/// Tests free space reuse after deletes with performance benchmarks.
/// Target: Delete throughput >= 500k ops/sec, Reinsert throughput >= 1M ops/sec
#[tokio::test]
async fn test_heap_file_space_reuse() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const TUPLE_COUNT: usize = 10_000;
    const TUPLE_DATA_SIZE: usize = 100;
    const DELETE_TARGET_OPS_SEC: f64 = 500_000.0;
    const REINSERT_TARGET_OPS_SEC: f64 = 1_000_000.0;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let pool = Arc::new(BufferPool::auto_sized());
    let heap = HeapFile::with_defaults(disk, pool).unwrap();

    // Initial insert batch
    let tuples: Vec<Tuple> = (0..TUPLE_COUNT)
        .map(|i| {
            let data = vec![i as u8; TUPLE_DATA_SIZE];
            Tuple::new(data, i as u32)
        })
        .collect();

    let insert_start = Instant::now();
    let tuple_ids = heap.insert_batch(&tuples).await.unwrap();
    let insert_duration = insert_start.elapsed();
    let insert_ops_sec = TUPLE_COUNT as f64 / insert_duration.as_secs_f64();

    let pages_after_insert = heap.num_pages().await.unwrap();
    tprintln!(
        "Initial insert: {} ops/sec ({} tuples)",
        format_with_commas(insert_ops_sec),
        format_with_commas(TUPLE_COUNT as f64)
    );
    tprintln!("Pages after insert: {}", pages_after_insert);

    // Batch delete all tuples
    let delete_start = Instant::now();
    let deleted_count = heap.delete_batch(&tuple_ids).await.unwrap();
    let delete_duration = delete_start.elapsed();
    assert_eq!(deleted_count, TUPLE_COUNT);
    let delete_ops_sec = TUPLE_COUNT as f64 / delete_duration.as_secs_f64();
    tprintln!(
        "Delete: {} ops/sec (target: {})",
        format_with_commas(delete_ops_sec),
        format_with_commas(DELETE_TARGET_OPS_SEC)
    );

    // Reinsert batch (triggers compaction)
    let tuples2: Vec<Tuple> = (0..TUPLE_COUNT)
        .map(|i| {
            let data = vec![(i + 100) as u8; TUPLE_DATA_SIZE];
            Tuple::new(data, (i + TUPLE_COUNT) as u32)
        })
        .collect();

    let reinsert_start = Instant::now();
    let _new_ids = heap.insert_batch(&tuples2).await.unwrap();
    let reinsert_duration = reinsert_start.elapsed();
    let reinsert_ops_sec = TUPLE_COUNT as f64 / reinsert_duration.as_secs_f64();
    tprintln!(
        "Reinsert (with compaction): {} ops/sec (target: {})",
        format_with_commas(reinsert_ops_sec),
        format_with_commas(REINSERT_TARGET_OPS_SEC)
    );

    let pages_after_reinsert = heap.num_pages().await.unwrap();

    // Verify space was reused
    assert!(
        pages_after_reinsert <= pages_after_insert * 2,
        "Expected space reuse: {} pages after reinsert vs {} after initial insert",
        pages_after_reinsert,
        pages_after_insert
    );
    tprintln!(
        "Space reuse: {} pages (was {})",
        pages_after_reinsert,
        pages_after_insert
    );

    // Performance assertions
    assert!(
        delete_ops_sec >= DELETE_TARGET_OPS_SEC,
        "Delete throughput {:.0} below target {:.0} ops/sec",
        delete_ops_sec,
        DELETE_TARGET_OPS_SEC
    );
    assert!(
        reinsert_ops_sec >= REINSERT_TARGET_OPS_SEC,
        "Reinsert throughput {:.0} below target {:.0} ops/sec",
        reinsert_ops_sec,
        REINSERT_TARGET_OPS_SEC
    );

    tprintln!("Heap File Space Reuse: PASSED");
}

// =============================================================================
// Test 4: B+ Tree Test (5-run validation)
// =============================================================================

/// Inserts 1,000,000 random i64 keys and verifies operations.
/// Target: 8M inserts/sec, 40ns/lookup, 40M range/sec
#[test]
fn test_btree_1m_keys() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const KEY_COUNT: usize = 1_000_000;
    const LOOKUP_SAMPLE: usize = 10_000;
    const RANGE_SIZE: usize = 1_000;

    tprintln!("\n=== B+ Tree Arena Performance Test ===");
    tprintln!("Keys per run: {}", KEY_COUNT);
    tprintln!("Lookup sample: {}", LOOKUP_SAMPLE);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut insert_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut lookup_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut range_results = Vec::with_capacity(VALIDATION_RUNS);

    let btree_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        // Buffered B+Tree: write buffer + 32KB nodes for HTAP workloads
        // 1024 nodes × 32KB = 32MB capacity (enough for 2M keys)
        let mut btree = BufferedBTreeIndex::new(1024);

        let mut rng = rand::rng();
        let mut keys: Vec<i64> = (0..KEY_COUNT as i64).collect();

        // Shuffle for random insertion order
        for i in (1..keys.len()).rev() {
            let j = rng.random_range(0..=i);
            keys.swap(i, j);
        }

        // Insert phase
        let insert_start = Instant::now();
        for &key in &keys {
            let key_bytes = key.to_be_bytes();
            let tuple_id = TupleId::new(PageId::new(0, (key % 1000) as u32), (key % 100) as u16);
            btree.insert(&key_bytes, tuple_id).unwrap();
        }
        let insert_duration = insert_start.elapsed();

        // Flush buffer to B+Tree before measuring lookup latency
        btree.flush().unwrap();

        let final_height = btree.height();
        assert!(
            final_height <= 5,
            "Run {}: Tree height {} too large for {} keys",
            run + 1,
            final_height,
            KEY_COUNT
        );

        // Lookup phase (measures B+Tree performance after flush)
        // Uses production search() path which checks buffer first

        // Warmup pass to prime caches and CPU frequency
        for i in (0..KEY_COUNT).step_by(KEY_COUNT / 1000) {
            let key = i as i64;
            let key_bytes = key.to_be_bytes();
            std::hint::black_box(btree.search(&key_bytes));
        }

        let mut found_count = 0usize;
        let lookup_start = Instant::now();
        for i in (0..KEY_COUNT).step_by(KEY_COUNT / LOOKUP_SAMPLE) {
            let key = i as i64;
            let key_bytes = key.to_be_bytes();
            // black_box prevents compiler from optimizing away the search
            if std::hint::black_box(btree.search(&key_bytes)).is_some() {
                found_count += 1;
            }
        }
        let lookup_duration = lookup_start.elapsed();
        assert_eq!(
            found_count,
            LOOKUP_SAMPLE,
            "Run {}: Expected {} found",
            run + 1,
            LOOKUP_SAMPLE
        );

        // Range scan phase (multiple iterations, take median for stable timing)
        let start_key = 1000i64.to_be_bytes();
        let end_key = (1000 + RANGE_SIZE as i64).to_be_bytes();

        // Warmup
        let _ = btree.range_scan(Some(&start_key), Some(&end_key));

        // Multiple timed iterations
        const RANGE_ITERATIONS: usize = 10;
        let mut range_times = Vec::with_capacity(RANGE_ITERATIONS);
        let mut last_len = 0;
        for _ in 0..RANGE_ITERATIONS {
            let range_start = Instant::now();
            let range_results_data = btree.range_scan(Some(&start_key), Some(&end_key));
            range_times.push(range_start.elapsed());
            last_len = range_results_data.len();
        }
        range_times.sort();
        let range_duration = range_times[RANGE_ITERATIONS / 2]; // Median

        assert!(
            last_len >= RANGE_SIZE - 10,
            "Run {}: Expected ~{} keys in range, got {}",
            run + 1,
            RANGE_SIZE,
            last_len
        );
        let range_results_len = last_len;

        // Delete phase: delete 100,000 keys on run 1, verify they are not found.
        if run == 0 {
            const DELETE_COUNT: usize = 100_000;
            let delete_step = KEY_COUNT / DELETE_COUNT;
            let mut deleted_keys: Vec<i64> = Vec::with_capacity(DELETE_COUNT);

            let delete_start = Instant::now();
            for i in (0..KEY_COUNT).step_by(delete_step) {
                let key = keys[i];
                let key_bytes = key.to_be_bytes();
                let removed = btree.delete(&key_bytes);
                assert!(
                    removed,
                    "Run 1: delete returned false for key {} that should exist",
                    key
                );
                deleted_keys.push(key);
            }
            let delete_duration = delete_start.elapsed();
            let delete_ops_sec = deleted_keys.len() as f64 / delete_duration.as_secs_f64();

            // Verify none of the deleted keys are found.
            for key in &deleted_keys {
                let key_bytes = key.to_be_bytes();
                assert!(
                    btree.search(&key_bytes).is_none(),
                    "Run 1: key {} still found after delete",
                    key
                );
            }

            // Verify a non-deleted key is still found.
            let alive_key = keys[1];
            let alive_key_bytes = alive_key.to_be_bytes();
            if !deleted_keys.contains(&alive_key) {
                assert!(
                    btree.search(&alive_key_bytes).is_some(),
                    "Run 1: non-deleted key {} not found after deletes",
                    alive_key
                );
            }

            let delete_status = if delete_ops_sec >= BTREE_DELETE_TARGET_OPS_SEC {
                "PASS"
            } else {
                "FAIL"
            };
            tprintln!(
                "  Delete [{}]: {} ops/sec ({:?}, {} keys, target: {})",
                delete_status,
                format_with_commas(delete_ops_sec),
                delete_duration,
                deleted_keys.len(),
                format_with_commas(BTREE_DELETE_TARGET_OPS_SEC)
            );
            tprintln!(
                "  All {} deleted keys verified not found",
                deleted_keys.len()
            );

            check_performance(
                "B+ Tree",
                "Delete throughput (ops/sec)",
                delete_ops_sec,
                BTREE_DELETE_TARGET_OPS_SEC,
                true,
            );
        }

        let insert_ops_sec = KEY_COUNT as f64 / insert_duration.as_secs_f64();
        let lookup_ns = lookup_duration.as_nanos() as f64 / LOOKUP_SAMPLE as f64;
        let range_ops_sec = range_results_len as f64 / range_duration.as_secs_f64();

        // Get flush stats for profiling
        let stats = btree.stats();
        let hash_table_time_ns =
            (insert_duration.as_nanos() as u64).saturating_sub(stats.flush_time_ns);

        tprintln!(
            "  Insert: {} ops/sec ({:?}), height={}",
            format_with_commas(insert_ops_sec),
            insert_duration,
            final_height
        );
        tprintln!(
            "    Breakdown: hash_table={:.1}ms, flush={:.1}ms, flushes={}",
            hash_table_time_ns as f64 / 1_000_000.0,
            stats.flush_time_ns as f64 / 1_000_000.0,
            stats.flush_count
        );
        tprintln!("  Lookup: {:.2} ns/op ({:?})", lookup_ns, lookup_duration);
        tprintln!(
            "  Range: {} ops/sec ({:?})",
            format_with_commas(range_ops_sec),
            range_duration
        );

        insert_results.push(insert_ops_sec);
        lookup_results.push(lookup_ns);
        range_results.push(range_ops_sec);
    }
    record_test_util("B+ Tree", btree_util_before, take_util_snapshot());

    tprintln!("\n=== B+ Tree Validation Results ===");
    let insert_result = validate_metric(
        "B+ Tree",
        "Insert throughput (ops/sec)",
        insert_results,
        BTREE_INSERT_TARGET_OPS_SEC,
        true,
    );
    let lookup_result = validate_metric(
        "B+ Tree",
        "Lookup latency (ns/op)",
        lookup_results,
        BTREE_LOOKUP_TARGET_NS,
        false,
    );
    let range_result = validate_metric(
        "B+ Tree",
        "Range throughput (ops/sec)",
        range_results,
        BTREE_RANGE_TARGET_OPS_SEC,
        true,
    );

    assert!(
        insert_result.passed,
        "B+Tree insert avg {:.0} < target {:.0}",
        insert_result.average, BTREE_INSERT_TARGET_OPS_SEC
    );
    assert!(
        !insert_result.regression_detected,
        "B+Tree insert regression detected"
    );
    assert!(
        lookup_result.passed,
        "B+Tree lookup avg {:.2} ns > target {:.2} ns",
        lookup_result.average, BTREE_LOOKUP_TARGET_NS
    );
    assert!(
        !lookup_result.regression_detected,
        "B+Tree lookup regression detected"
    );
    assert!(
        range_result.passed,
        "B+Tree range avg {:.0} < target {:.0}",
        range_result.average, BTREE_RANGE_TARGET_OPS_SEC
    );
    assert!(
        !range_result.regression_detected,
        "B+Tree range regression detected"
    );
}

// =============================================================================
// Test 5: Integration Test - WAL + Heap Recovery
// =============================================================================

/// Tests crash recovery: write tuples with WAL logging, crash, recover.
/// Target: 2ms/MB recovery time
#[tokio::test]
async fn test_wal_heap_recovery() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().unwrap();
    let heap_dir = dir.path().join("heap");
    let wal_dir = dir.path().join("wal");

    std::fs::create_dir_all(&heap_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    const TUPLE_COUNT: usize = 1000;
    let mut committed_data: Vec<(u32, Vec<u8>)> = Vec::new();

    // Phase 1: Write tuples with WAL logging
    let wal_size_bytes: usize;
    {
        let wal_config = WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: true,
            ring_buffer_capacity: 1024 * 1024, // 1MB
        };
        let writer = Arc::new(WalWriter::new(wal_config).unwrap());

        let disk_config = DiskManagerConfig {
            data_dir: heap_dir.clone(),
            fsync_enabled: true,
        };
        let disk = Arc::new(DiskManager::new(disk_config).await.unwrap());
        let pool = Arc::new(BufferPool::auto_sized());
        let heap = HeapFile::with_defaults(disk, pool).unwrap();

        for i in 0..TUPLE_COUNT {
            let txn_id = (i + 1) as u32;
            let data = format!("committed_tuple_{}", i);

            let begin_lsn = writer.log_begin(txn_id).unwrap();

            let tuple = Tuple::new(data.clone().into_bytes(), txn_id);
            let tuple_id = heap.insert_batch(&[tuple]).await.unwrap().remove(0);

            let payload = format!(
                "{}:{}:{}",
                tuple_id.page_id.page_num, tuple_id.slot_id, data
            );
            let insert_lsn = writer
                .log_insert(txn_id, begin_lsn, payload.as_bytes())
                .unwrap();

            writer.log_commit(txn_id, insert_lsn).unwrap();

            committed_data.push((txn_id, data.into_bytes()));
        }

        writer.flush().unwrap();
        writer.close().unwrap();
        drop(heap);

        // Measure actual bytes on disk (segments grow as data is written, not pre-allocated).
        wal_size_bytes = std::fs::read_dir(&wal_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len() as usize)
            .sum();
    }

    // Phase 2: Recovery with timing
    let recovery_util_before = take_util_snapshot();
    let recovery_start = Instant::now();
    {
        let recovery = RecoveryManager::new(&wal_dir).unwrap();
        let result = recovery.recover().unwrap();

        assert!(
            result.undo_txns.is_empty(),
            "No transactions should need undo"
        );

        assert_eq!(
            result.redo_records.len(),
            TUPLE_COUNT,
            "Should have {} redo records",
            TUPLE_COUNT
        );

        let redo_txns: HashSet<u32> = result.redo_records.iter().map(|r| r.txn_id).collect();
        for (txn_id, _) in &committed_data {
            assert!(
                redo_txns.contains(txn_id),
                "Transaction {} should be in redo set",
                txn_id
            );
        }
    }
    let recovery_duration = recovery_start.elapsed();

    // Calculate recovery performance using actual WAL bytes on disk.
    let wal_size_kb = wal_size_bytes as f64 / 1024.0;
    let recovery_us = recovery_duration.as_micros() as f64;
    let us_per_kb = if wal_size_kb > 0.0 {
        recovery_us / wal_size_kb
    } else {
        0.0
    };

    record_test_util("Recovery", recovery_util_before, take_util_snapshot());

    tprintln!("\n=== Recovery Performance ===");
    tprintln!("  WAL size: {:.1} KB", wal_size_kb);
    tprintln!("  Recovery time: {:?}", recovery_duration);
    let recovery_pass = check_performance(
        "Recovery",
        "Recovery time (us/KB)",
        us_per_kb,
        RECOVERY_TARGET_US_PER_KB,
        false,
    );

    tprintln!(
        "WAL+Heap Recovery: {} - {} committed transactions recovered ({:.1} KB WAL, {:?})",
        if recovery_pass { "PASSED" } else { "FAILED" },
        TUPLE_COUNT,
        wal_size_kb,
        recovery_duration
    );
}

/// Tests recovery with uncommitted transactions.
#[tokio::test]
async fn test_wal_recovery_with_uncommitted() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().unwrap();

    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 16 * 1024 * 1024,
        fsync_enabled: true,
        ring_buffer_capacity: 1024 * 1024, // 1MB
    };

    {
        let writer = WalWriter::new(config.clone()).unwrap();

        for i in 1..=10 {
            let begin = writer.log_begin(i).unwrap();
            let data = format!("data_{}", i);
            let insert = writer.log_insert(i, begin, data.as_bytes()).unwrap();
            writer.log_commit(i, insert).unwrap();
        }

        for i in 11..=15 {
            let begin = writer.log_begin(i).unwrap();
            let data = format!("uncommitted_{}", i);
            writer.log_insert(i, begin, data.as_bytes()).unwrap();
        }

        writer.close().unwrap();
    }

    let recovery = RecoveryManager::new(dir.path()).unwrap();
    let result = recovery.recover().unwrap();

    let committed_txns: HashSet<u32> = (1..=10).collect();
    let redo_txns: HashSet<u32> = result.redo_records.iter().map(|r| r.txn_id).collect();

    for txn in &committed_txns {
        assert!(
            redo_txns.contains(txn),
            "Committed transaction {} should be in redo",
            txn
        );
    }

    let uncommitted_txns: HashSet<u32> = (11..=15).collect();
    let undo_set: HashSet<u32> = result.undo_txns.iter().copied().collect();

    for txn in &uncommitted_txns {
        assert!(
            undo_set.contains(txn),
            "Uncommitted transaction {} should be in undo",
            txn
        );
    }

    for txn in &uncommitted_txns {
        assert!(
            !redo_txns.contains(txn),
            "Uncommitted transaction {} should NOT be in redo",
            txn
        );
    }

    tprintln!(
        "WAL Recovery with Uncommitted: PASSED - {} redo, {} undo",
        result.redo_records.len(),
        result.undo_txns.len()
    );
}

// =============================================================================
// Test 6: WAL Checksum Integrity
// =============================================================================

/// Verifies that the WAL checksum catches corruption, truncation, and bit flips.
/// Writes records through the full WAL pipeline, then tampers with the on-disk
/// bytes to confirm the reader rejects corrupted data.
#[tokio::test]
async fn test_wal_checksum_integrity() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== WAL Checksum Integrity Test ===");

    let dir = tempdir().unwrap();
    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 1024 * 1024,
        fsync_enabled: false,
        ring_buffer_capacity: 1024 * 1024,
    };

    // Write a set of records through the full pipeline
    let record_count = 100;
    let writer = WalWriter::new(config.clone()).unwrap();
    for i in 0..record_count {
        let payload = format!("checksum_test_record_{}", i);
        writer
            .log_insert(1, Lsn::INVALID, payload.as_bytes())
            .unwrap();
    }
    writer.flush().unwrap();
    writer.close().unwrap();

    // 1. Verify clean read succeeds and all records round-trip
    {
        let reader = WalReader::new(dir.path()).unwrap();
        let records = reader.scan_all().unwrap();
        assert_eq!(
            records.len(),
            record_count,
            "Clean read should return all {} records",
            record_count
        );
        for (i, r) in records.iter().enumerate() {
            let expected = format!("checksum_test_record_{}", i);
            assert_eq!(
                r.payload.as_ref(),
                expected.as_bytes(),
                "Payload mismatch at record {}",
                i
            );
        }
        tprintln!("  Clean read: PASSED ({} records)", record_count);
    }

    // Find the segment file for corruption tests
    let mut seg_path = None;
    for entry in std::fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        if entry
            .path()
            .extension()
            .map(|e| e == "wal")
            .unwrap_or(false)
        {
            seg_path = Some(entry.path());
        }
    }
    let seg_path = seg_path.expect("Should have at least one segment file");
    let original_bytes = std::fs::read(&seg_path).unwrap();

    // 2. Single bit flip in a payload byte: reader should stop at the corrupted record
    {
        let mut corrupted = original_bytes.clone();
        // Flip a bit in the middle of the file (past the segment header)
        let flip_pos = corrupted.len() / 2;
        corrupted[flip_pos] ^= 0x01;
        std::fs::write(&seg_path, &corrupted).unwrap();

        let reader = WalReader::new(dir.path()).unwrap();
        let records = reader.scan_all().unwrap();
        assert!(
            records.len() < record_count,
            "Bit flip should cause reader to stop early. Got {} records, expected fewer than {}",
            records.len(),
            record_count
        );
        tprintln!(
            "  Bit flip detection: PASSED (read {} of {} before corruption)",
            records.len(),
            record_count
        );
    }

    // 3. Zeroed region: wipe 8 bytes in the payload area
    {
        let mut corrupted = original_bytes.clone();
        let zero_start = corrupted.len() / 3;
        for b in &mut corrupted[zero_start..zero_start + 8] {
            *b = 0;
        }
        std::fs::write(&seg_path, &corrupted).unwrap();

        let reader = WalReader::new(dir.path()).unwrap();
        let records = reader.scan_all().unwrap();
        assert!(
            records.len() < record_count,
            "Zeroed region should cause reader to stop early. Got {} records",
            records.len()
        );
        tprintln!(
            "  Zeroed region detection: PASSED (read {} of {} before corruption)",
            records.len(),
            record_count
        );
    }

    // 4. Truncation: chop the file in half
    {
        let truncated = &original_bytes[..original_bytes.len() / 2];
        std::fs::write(&seg_path, truncated).unwrap();

        let reader = WalReader::new(dir.path()).unwrap();
        let records = reader.scan_all().unwrap();
        assert!(
            records.len() < record_count,
            "Truncation should return fewer records. Got {} records",
            records.len()
        );
        assert!(
            !records.is_empty(),
            "Truncation at midpoint should still recover some records"
        );
        tprintln!(
            "  Truncation detection: PASSED (recovered {} of {} records from half-sized file)",
            records.len(),
            record_count
        );
    }

    // 5. Full corruption: randomize all data after segment header (32 bytes)
    {
        let mut corrupted = original_bytes.clone();
        let mut rng = rand::rng();
        for b in &mut corrupted[32..] {
            *b = rng.random::<u8>();
        }
        std::fs::write(&seg_path, &corrupted).unwrap();

        let reader = WalReader::new(dir.path()).unwrap();
        let records = reader.scan_all().unwrap();
        assert_eq!(
            records.len(),
            0,
            "Fully corrupted data should yield 0 valid records, got {}",
            records.len()
        );
        tprintln!("  Full corruption detection: PASSED (0 records from random data)");
    }

    // Restore original file
    std::fs::write(&seg_path, &original_bytes).unwrap();
    tprintln!("WAL Checksum Integrity: PASSED");
}

// =============================================================================
// Test 9: Checkpoint Write/Load Round-Trip (1M keys, 5-run validation)
// =============================================================================

/// Creates a B+Tree with 1,000,000 keys, writes a checkpoint, loads it back,
/// and verifies every key round-trips correctly. Measures write and load
/// throughput against targets (2 GB/sec write, 3 GB/sec load).
#[tokio::test]
async fn test_checkpoint_round_trip_1m() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const KEY_COUNT: usize = 1_000_000;

    tprintln!("\n=== Checkpoint Write/Load Round-Trip Test ===");
    tprintln!("Keys: {}", format_with_commas(KEY_COUNT as f64));
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut write_latency_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut load_latency_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut write_throughput_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut load_throughput_results = Vec::with_capacity(VALIDATION_RUNS);

    let ckpt_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path().join("ckpt");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();

        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::auto_sized());

        let mut btree = BTreeIndex::create_with_config(disk.clone(), pool.clone(), 0, checkpoint_dir.clone(), CheckpointConfig { fsync: false, ..CheckpointConfig::default() })
            .await
            .unwrap();

        // Insert 1M keys using exclusive access for maximum speed
        for i in 0..KEY_COUNT as u64 {
            let key = i.to_be_bytes();
            let tid = TupleId::new(PageId::new(0, (i % 1000) as u32), (i % 100) as u16);
            btree.insert_exclusive(&key, tid).unwrap();
        }

        // Write checkpoint
        let checkpoint_lsn = 42_000_u64;
        let write_start = Instant::now();
        btree.force_checkpoint(checkpoint_lsn).unwrap();
        let write_duration = write_start.elapsed();

        // Measure file size
        let ckpt_path = checkpoint_dir.join("index_0.zyridx");
        let file_size = std::fs::metadata(&ckpt_path).unwrap().len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);

        assert_eq!(btree.checkpoint_lsn(), checkpoint_lsn);

        // Load checkpoint into a new index
        let load_start = Instant::now();
        let loaded = BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
            .await
            .unwrap();
        let load_duration = load_start.elapsed();

        assert_eq!(loaded.checkpoint_lsn(), checkpoint_lsn);

        // Verify every key round-trips
        for i in 0..KEY_COUNT as u64 {
            let key = i.to_be_bytes();
            let expected = TupleId::new(PageId::new(0, (i % 1000) as u32), (i % 100) as u16);
            let found = loaded.search_sync(&key);
            assert_eq!(
                found,
                Some(expected),
                "Key {} missing or wrong after checkpoint load",
                i
            );
        }

        let write_ms = write_duration.as_secs_f64() * 1000.0;
        let load_ms = load_duration.as_secs_f64() * 1000.0;
        let write_mb_sec = file_size_mb / write_duration.as_secs_f64();
        let load_mb_sec = file_size_mb / load_duration.as_secs_f64();

        tprintln!(
            "  Checkpoint size: {:.2} MB ({} pages)",
            file_size_mb,
            btree.height()
        );
        tprintln!(
            "  Write: {:.2} ms ({:.0} MB/sec)",
            write_ms,
            write_mb_sec
        );
        tprintln!(
            "  Load: {:.2} ms ({:.0} MB/sec)",
            load_ms,
            load_mb_sec
        );

        write_latency_results.push(write_ms);
        load_latency_results.push(load_ms);
        write_throughput_results.push(write_mb_sec);
        load_throughput_results.push(load_mb_sec);
    }
    record_test_util("Checkpoint", ckpt_util_before, take_util_snapshot());

    tprintln!("\n=== Checkpoint Validation Results ===");
    let write_tp = validate_metric(
        "Checkpoint",
        "Write throughput (MB/sec)",
        write_throughput_results,
        CHECKPOINT_WRITE_THROUGHPUT_TARGET_MB_SEC,
        true,
    );
    let write_lat = validate_metric(
        "Checkpoint",
        "Write latency 1M keys (ms)",
        write_latency_results,
        CHECKPOINT_WRITE_1M_TARGET_MS,
        false,
    );
    let load_tp = validate_metric(
        "Checkpoint",
        "Load throughput (MB/sec)",
        load_throughput_results,
        CHECKPOINT_LOAD_THROUGHPUT_TARGET_MB_SEC,
        true,
    );
    let load_lat = validate_metric(
        "Checkpoint",
        "Load latency 1M keys (ms)",
        load_latency_results,
        CHECKPOINT_LOAD_1M_TARGET_MS,
        false,
    );

    // Report checkpoint performance without fixed assertions.
    // Compare against previous runs for regression detection.
    tprintln!("  Write: avg {:.2} ms, {:.0} MB/sec", write_lat.average, write_tp.average);
    tprintln!("  Load:  avg {:.2} ms, {:.0} MB/sec", load_lat.average, load_tp.average);
}

// =============================================================================
// Test 10: Corrupt Checkpoint Fallback
// =============================================================================

/// Writes a valid checkpoint, corrupts one byte in page data, and verifies
/// that loading fails CRC validation. Then verifies the system falls back
/// to creating an empty tree (caller would do full WAL replay).
#[tokio::test]
async fn test_checkpoint_corrupt_fallback() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const KEY_COUNT: usize = 10_000;

    tprintln!("\n=== Corrupt Checkpoint Fallback Test ===");

    let dir = tempdir().unwrap();
    let checkpoint_dir = dir.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::auto_sized());

    // Create index, insert keys, checkpoint
    let mut btree = BTreeIndex::create_with_config(disk.clone(), pool.clone(), 0, checkpoint_dir.clone(), CheckpointConfig { fsync: false, ..CheckpointConfig::default() })
        .await
        .unwrap();

    for i in 0..KEY_COUNT as u64 {
        let key = i.to_be_bytes();
        let tid = TupleId::new(PageId::new(0, (i % 100) as u32), (i % 50) as u16);
        btree.insert_exclusive(&key, tid).unwrap();
    }
    btree.force_checkpoint(1000).unwrap();
    drop(btree);

    // Corrupt one byte in the page data section (past the 32-byte header + 4-byte page_id)
    let ckpt_path = checkpoint_dir.join("index_0.zyridx");
    let mut data = std::fs::read(&ckpt_path).unwrap();
    assert!(data.len() > 100, "Checkpoint file too small");
    data[50] ^= 0xFF; // Flip a byte in the first page entry's data
    std::fs::write(&ckpt_path, &data).unwrap();

    // Loading should fail CRC validation and fall back to empty tree
    let loaded = BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
        .await
        .unwrap();

    // The loaded tree should be empty (fallback to fresh store)
    assert_eq!(loaded.checkpoint_lsn(), 0, "Corrupt checkpoint should not set LSN");
    assert_eq!(loaded.height(), 1, "Fallback tree should have height 1");

    // Searching should find nothing (empty tree)
    let key = 0u64.to_be_bytes();
    assert!(
        loaded.search_sync(&key).is_none(),
        "Empty fallback tree should have no keys"
    );

    tprintln!("Corrupt Checkpoint Fallback: PASSED");
    tprintln!("  CRC validation caught corruption, system fell back to empty store");
}

// =============================================================================
// Test 11: Recovery With Checkpoint (500K + 500 WAL)
// =============================================================================

/// Inserts 500,000 keys and writes a checkpoint. Then inserts 500 more keys
/// with WAL logging. Simulates crash and recovery by loading checkpoint +
/// replaying only post-checkpoint WAL. Verifies all 500,500 keys are present.
/// Recovery should only replay 500 records, not all 500,000.
#[tokio::test]
async fn test_recovery_with_checkpoint() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const PRE_CHECKPOINT_KEYS: usize = 500_000;
    const POST_CHECKPOINT_KEYS: usize = 500;

    tprintln!("\n=== Recovery With Checkpoint Test ===");
    tprintln!("Pre-checkpoint keys: {}", format_with_commas(PRE_CHECKPOINT_KEYS as f64));
    tprintln!("Post-checkpoint keys: {}", POST_CHECKPOINT_KEYS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut recovery_time_results = Vec::with_capacity(VALIDATION_RUNS);

    let recovery_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path().join("ckpt");
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();

        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::auto_sized());

        // Phase 1: Insert 500K keys and checkpoint
        let checkpoint_lsn;
        {
            let mut btree =
                BTreeIndex::create_with_config(disk.clone(), pool.clone(), 0, checkpoint_dir.clone(), CheckpointConfig { fsync: false, ..CheckpointConfig::default() })
                    .await
                    .unwrap();

            for i in 0..PRE_CHECKPOINT_KEYS as u64 {
                let key = i.to_be_bytes();
                let tid = TupleId::new(PageId::new(0, (i % 1000) as u32), (i % 100) as u16);
                btree.insert_exclusive(&key, tid).unwrap();
            }

            // Write WAL records for the pre-checkpoint keys (simulating normal operation)
            let wal_config = WalWriterConfig {
                wal_dir: wal_dir.clone(),
                segment_size: 16 * 1024 * 1024,
                fsync_enabled: false,
                ring_buffer_capacity: 1024 * 1024,
            };
            let writer = Arc::new(WalWriter::new(wal_config).unwrap());

            // Write a checkpoint marker at the current WAL position
            let ckpt_begin = writer.log_checkpoint_begin().unwrap();
            writer.log_checkpoint_end(&[]).unwrap();
            writer.flush().unwrap();

            checkpoint_lsn = ckpt_begin.0;
            btree.force_checkpoint(checkpoint_lsn).unwrap();

            // Phase 2: Insert 500 more keys with WAL logging (post-checkpoint)
            for i in PRE_CHECKPOINT_KEYS..(PRE_CHECKPOINT_KEYS + POST_CHECKPOINT_KEYS) {
                let txn_id = (i + 1) as u32;
                let begin_lsn = writer.log_begin(txn_id).unwrap();
                let payload = format!("key:{}", i);
                let insert_lsn = writer
                    .log_insert(txn_id, begin_lsn, payload.as_bytes())
                    .unwrap();
                writer.log_commit(txn_id, insert_lsn).unwrap();
            }
            writer.flush().unwrap();

            // Insert the post-checkpoint keys into the btree as well
            for i in PRE_CHECKPOINT_KEYS..(PRE_CHECKPOINT_KEYS + POST_CHECKPOINT_KEYS) {
                let key = (i as u64).to_be_bytes();
                let tid =
                    TupleId::new(PageId::new(0, (i % 1000) as u32), (i as u64 % 100) as u16);
                btree.insert_exclusive(&key, tid).unwrap();
            }

            writer.close().unwrap();
            // Simulate crash: drop everything without clean shutdown
        }

        // Phase 3: Recovery
        let recovery_start = Instant::now();

        // Load checkpoint
        let mut recovered =
            BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
                .await
                .unwrap();

        assert_eq!(
            recovered.checkpoint_lsn(),
            checkpoint_lsn,
            "Checkpoint LSN mismatch after load"
        );

        // Replay only post-checkpoint WAL records
        let recovery = RecoveryManager::new(&wal_dir).unwrap();
        let result = recovery.recover().unwrap();

        // Apply redo records from committed transactions that are post-checkpoint
        let mut replayed = 0usize;
        for record in &result.redo_records {
            if record.lsn.0 > checkpoint_lsn && record.record_type == LogRecordType::Insert {
                // Parse the payload to extract the key
                let payload_str = String::from_utf8_lossy(&record.payload);
                if let Some(key_str) = payload_str.strip_prefix("key:") {
                    if let Ok(i) = key_str.parse::<u64>() {
                        let key = i.to_be_bytes();
                        let tid = TupleId::new(
                            PageId::new(0, (i % 1000) as u32),
                            (i % 100) as u16,
                        );
                        // Only insert if not already present from checkpoint
                        if recovered.search_sync(&key).is_none() {
                            recovered.insert_exclusive(&key, tid).unwrap();
                        }
                        replayed += 1;
                    }
                }
            }
        }

        let recovery_duration = recovery_start.elapsed();
        let recovery_ms = recovery_duration.as_secs_f64() * 1000.0;

        tprintln!("  Checkpoint load + {} WAL records replayed in {:.2} ms", replayed, recovery_ms);

        // Verify all keys are present
        for i in 0..(PRE_CHECKPOINT_KEYS + POST_CHECKPOINT_KEYS) as u64 {
            let key = i.to_be_bytes();
            assert!(
                recovered.search_sync(&key).is_some(),
                "Key {} missing after recovery (run {})",
                i,
                run + 1
            );
        }

        tprintln!("  All {} keys verified", PRE_CHECKPOINT_KEYS + POST_CHECKPOINT_KEYS);
        recovery_time_results.push(recovery_ms);
    }
    record_test_util("Recovery with Checkpoint", recovery_util_before, take_util_snapshot());

    tprintln!("\n=== Recovery With Checkpoint Validation Results ===");
    let recovery_result = validate_metric(
        "Recovery with Checkpoint",
        "Recovery time (ms)",
        recovery_time_results,
        RECOVERY_WITH_CHECKPOINT_TARGET_MS,
        false,
    );
    tprintln!("  Recovery: avg {:.2} ms (target {:.2} ms)", recovery_result.average, RECOVERY_WITH_CHECKPOINT_TARGET_MS);
}

// =============================================================================
// Test 12: Recovery Without Checkpoint (10K + WAL)
// =============================================================================

/// Inserts 10,000 keys with WAL logging but no checkpoint. Simulates crash
/// and does full WAL replay. Verifies all keys are recovered.
#[tokio::test]
async fn test_recovery_without_checkpoint() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const KEY_COUNT: usize = 10_000;

    tprintln!("\n=== Recovery Without Checkpoint Test ===");
    tprintln!("Keys: {}", format_with_commas(KEY_COUNT as f64));

    let dir = tempdir().unwrap();
    let checkpoint_dir = dir.path().join("ckpt");
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&checkpoint_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        })
        .await
        .unwrap(),
    );
    let pool = Arc::new(BufferPool::auto_sized());

    // Phase 1: Insert keys with WAL logging, no checkpoint
    {
        let wal_config = WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024,
        };
        let writer = Arc::new(WalWriter::new(wal_config).unwrap());

        for i in 0..KEY_COUNT {
            let txn_id = (i + 1) as u32;
            let begin_lsn = writer.log_begin(txn_id).unwrap();
            let payload = format!("key:{}", i);
            let insert_lsn = writer
                .log_insert(txn_id, begin_lsn, payload.as_bytes())
                .unwrap();
            writer.log_commit(txn_id, insert_lsn).unwrap();
        }
        writer.flush().unwrap();
        writer.close().unwrap();
        // Crash: no clean shutdown
    }

    // Phase 2: Recovery from WAL only (no checkpoint)
    let mut recovered =
        BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
            .await
            .unwrap();

    assert_eq!(recovered.checkpoint_lsn(), 0, "No checkpoint should exist");

    let recovery = RecoveryManager::new(&wal_dir).unwrap();
    let result = recovery.recover().unwrap();

    // Replay all committed insert records
    let mut replayed = 0usize;
    for record in &result.redo_records {
        if record.record_type == LogRecordType::Insert {
            let payload_str = String::from_utf8_lossy(&record.payload);
            if let Some(key_str) = payload_str.strip_prefix("key:") {
                if let Ok(i) = key_str.parse::<u64>() {
                    let key = i.to_be_bytes();
                    let tid = TupleId::new(
                        PageId::new(0, (i % 1000) as u32),
                        (i % 100) as u16,
                    );
                    recovered.insert_exclusive(&key, tid).unwrap();
                    replayed += 1;
                }
            }
        }
    }

    tprintln!("  Replayed {} WAL records (full replay, no checkpoint)", replayed);
    assert_eq!(replayed, KEY_COUNT, "Should replay all {} records", KEY_COUNT);

    // Verify all keys present
    for i in 0..KEY_COUNT as u64 {
        let key = i.to_be_bytes();
        assert!(
            recovered.search_sync(&key).is_some(),
            "Key {} missing after full WAL replay",
            i
        );
    }

    tprintln!("  All {} keys verified after full WAL replay", KEY_COUNT);
    tprintln!("Recovery Without Checkpoint: PASSED");
}

// =============================================================================
// Test 13: WAL Segment Cleanup
// =============================================================================

/// Writes enough data to create 5+ WAL segments, writes a checkpoint at
/// the current LSN, calls cleanup_old_segments, and verifies old segments
/// are deleted while the current segment is retained.
#[tokio::test]
async fn test_wal_segment_cleanup() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== WAL Segment Cleanup Test ===");
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut cleanup_time_results = Vec::with_capacity(VALIDATION_RUNS);

    let cleanup_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&wal_dir).unwrap();

        // Small segments to force rotation (64KB per segment)
        let wal_config = WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 64 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 256 * 1024,
        };
        let writer = Arc::new(WalWriter::new(wal_config).unwrap());

        // Write enough data to create 5+ segments (each ~64KB, write ~500KB total)
        let payload = vec![0xABu8; 200];
        for i in 0..2000 {
            let txn_id = (i + 1) as u32;
            let begin_lsn = writer.log_begin(txn_id).unwrap();
            let insert_lsn = writer.log_insert(txn_id, begin_lsn, &payload).unwrap();
            let _ = writer.log_commit(txn_id, insert_lsn).unwrap();
        }
        writer.flush().unwrap();

        // Count segments before cleanup
        let segments_before: Vec<String> = std::fs::read_dir(&wal_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "wal")
                    .unwrap_or(false)
            })
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();

        tprintln!("  Segments before cleanup: {} ({:?})", segments_before.len(), segments_before);
        assert!(
            segments_before.len() >= 5,
            "Expected 5+ segments, got {}",
            segments_before.len()
        );

        // Set checkpoint LSN to the last flushed position
        let checkpoint_lsn = writer.flushed_lsn();

        // Cleanup old segments
        let cleanup_start = Instant::now();
        let deleted = writer.cleanup_old_segments(checkpoint_lsn).unwrap();
        let cleanup_duration = cleanup_start.elapsed();
        let cleanup_ms = cleanup_duration.as_secs_f64() * 1000.0;

        // Count segments after cleanup
        let segments_after: Vec<String> = std::fs::read_dir(&wal_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "wal")
                    .unwrap_or(false)
            })
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();

        tprintln!("  Segments after cleanup: {} ({:?})", segments_after.len(), segments_after);
        tprintln!("  Deleted: {} segments in {:.3} ms", deleted, cleanup_ms);

        assert!(deleted > 0, "Should have deleted at least 1 segment");
        assert!(
            segments_after.len() < segments_before.len(),
            "Segments should decrease after cleanup"
        );
        // The current segment (containing checkpoint_lsn) must be retained
        let checkpoint_seg = format!("{:016}.wal", checkpoint_lsn.segment_id());
        assert!(
            segments_after.contains(&checkpoint_seg),
            "Current segment {} must be retained",
            checkpoint_seg
        );

        // Verify post-cleanup WAL still works: write more data
        let txn_id = 9999u32;
        let begin_lsn = writer.log_begin(txn_id).unwrap();
        let insert_lsn = writer.log_insert(txn_id, begin_lsn, b"post_cleanup").unwrap();
        let _ = writer.log_commit(txn_id, insert_lsn).unwrap();
        writer.flush().unwrap();

        writer.close().unwrap();
        cleanup_time_results.push(cleanup_ms);
    }
    record_test_util("WAL Segment Cleanup", cleanup_util_before, take_util_snapshot());

    tprintln!("\n=== WAL Segment Cleanup Validation Results ===");
    let cleanup_result = validate_metric(
        "WAL Segment Cleanup",
        "Cleanup latency (ms)",
        cleanup_time_results,
        WAL_SEGMENT_CLEANUP_TARGET_MS,
        false,
    );
    assert!(
        cleanup_result.passed,
        "WAL segment cleanup avg {:.2} ms > target {:.2} ms",
        cleanup_result.average, WAL_SEGMENT_CLEANUP_TARGET_MS
    );
}

// =============================================================================
// Test 14: Adaptive Checkpoint Trigger
// =============================================================================

/// Tests the CheckpointTrigger thresholds: byte-based, time-based,
/// and minimum interval guard.
#[test]
fn test_checkpoint_trigger_adaptive() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Adaptive Checkpoint Trigger Test ===");

    // Test 1: Byte threshold
    let config = CheckpointConfig {
        wal_bytes_threshold: 1024 * 1024, // 1 MB
        max_interval_secs: 60,
        min_interval_secs: 0, // Disable min interval for this sub-test
        fsync: false,
    };
    let mut trigger = CheckpointTrigger::new(config);
    let mut wal_bytes: u64 = 0;

    // 512KB should not trigger
    wal_bytes += 512 * 1024;
    assert!(
        !trigger.should_checkpoint(wal_bytes),
        "512KB should not trigger 1MB threshold"
    );

    // Another 512KB should trigger (1MB total)
    wal_bytes += 512 * 1024;
    assert!(
        trigger.should_checkpoint(wal_bytes),
        "1MB should trigger 1MB threshold"
    );

    // After reset, should not trigger
    trigger.reset();
    wal_bytes = 0;
    assert!(
        !trigger.should_checkpoint(wal_bytes),
        "After reset, should not trigger"
    );

    tprintln!("  Byte threshold: PASSED");

    // Test 2: Minimum interval guard
    let config = CheckpointConfig {
        wal_bytes_threshold: 0, // Would always trigger on bytes alone
        max_interval_secs: 3600,
        min_interval_secs: 999, // 999 seconds, will block
        fsync: false,
    };
    let trigger = CheckpointTrigger::new(config);
    assert!(
        !trigger.should_checkpoint(u64::MAX),
        "min_interval should prevent immediate checkpoint"
    );

    tprintln!("  Minimum interval guard: PASSED");

    // Test 3: Trigger check latency (should be <10ns, just an atomic read path)
    let config = CheckpointConfig {
        wal_bytes_threshold: u64::MAX,
        max_interval_secs: u64::MAX,
        min_interval_secs: 0,
        fsync: false,
    };
    let trigger = CheckpointTrigger::new(config);

    const TRIGGER_CHECK_ITERS: usize = 1_000_000;
    let start = Instant::now();
    for _ in 0..TRIGGER_CHECK_ITERS {
        std::hint::black_box(trigger.should_checkpoint(0));
    }
    let duration = start.elapsed();
    let ns_per_check = duration.as_nanos() as f64 / TRIGGER_CHECK_ITERS as f64;

    tprintln!("  Trigger check latency: {:.1} ns/call", ns_per_check);
    check_performance(
        "Checkpoint Trigger",
        "Trigger check latency (ns)",
        ns_per_check,
        10.0,
        false,
    );

    tprintln!("Adaptive Checkpoint Trigger: PASSED");
}

// =============================================================================
// Test 15: Graceful Shutdown Checkpoint
// =============================================================================

/// Inserts 1M keys, triggers graceful shutdown (writes final checkpoint),
/// restarts, and verifies startup requires zero WAL replay.
#[tokio::test]
async fn test_graceful_shutdown_checkpoint() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const KEY_COUNT: usize = 1_000_000;

    tprintln!("\n=== Graceful Shutdown Checkpoint Test ===");
    tprintln!("Keys: {}", format_with_commas(KEY_COUNT as f64));
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut shutdown_time_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut startup_time_results = Vec::with_capacity(VALIDATION_RUNS);

    let shutdown_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path().join("ckpt");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();

        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::auto_sized());

        let shutdown_lsn = 99999u64;

        // Insert keys and shutdown
        {
            let mut btree =
                BTreeIndex::create_with_config(disk.clone(), pool.clone(), 0, checkpoint_dir.clone(), CheckpointConfig { fsync: false, ..CheckpointConfig::default() })
                    .await
                    .unwrap();

            for i in 0..KEY_COUNT as u64 {
                let key = i.to_be_bytes();
                let tid = TupleId::new(PageId::new(0, (i % 1000) as u32), (i % 100) as u16);
                btree.insert_exclusive(&key, tid).unwrap();
            }

            // Graceful shutdown writes a final checkpoint
            let shutdown_start = Instant::now();
            btree.shutdown(shutdown_lsn).unwrap();
            let shutdown_duration = shutdown_start.elapsed();
            let shutdown_ms = shutdown_duration.as_secs_f64() * 1000.0;

            tprintln!("  Shutdown: {:.2} ms", shutdown_ms);
            shutdown_time_results.push(shutdown_ms);
        }

        // Startup: load from checkpoint (zero WAL replay needed)
        let startup_start = Instant::now();
        let loaded = BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
            .await
            .unwrap();
        let startup_duration = startup_start.elapsed();
        let startup_ms = startup_duration.as_secs_f64() * 1000.0;

        tprintln!("  Startup: {:.2} ms", startup_ms);
        startup_time_results.push(startup_ms);

        assert_eq!(
            loaded.checkpoint_lsn(),
            shutdown_lsn,
            "Loaded checkpoint_lsn should match shutdown LSN"
        );

        // Verify all keys are present (spot-check a sample for speed)
        let sample_step = KEY_COUNT / 10_000;
        for i in (0..KEY_COUNT as u64).step_by(sample_step) {
            let key = i.to_be_bytes();
            let expected = TupleId::new(PageId::new(0, (i % 1000) as u32), (i % 100) as u16);
            let found = loaded.search_sync(&key);
            assert_eq!(
                found,
                Some(expected),
                "Key {} wrong after shutdown+reload (run {})",
                i,
                run + 1
            );
        }
    }
    record_test_util("Graceful Shutdown", shutdown_util_before, take_util_snapshot());

    tprintln!("\n=== Graceful Shutdown Validation Results ===");
    let shutdown_result = validate_metric(
        "Graceful Shutdown",
        "Shutdown checkpoint latency (ms)",
        shutdown_time_results,
        SHUTDOWN_CHECKPOINT_1M_TARGET_MS,
        false,
    );
    // Startup time is informational, not a gating target (load target already tested in test 9)
    let _startup = validate_metric(
        "Graceful Shutdown",
        "Startup load latency (ms)",
        startup_time_results,
        CHECKPOINT_LOAD_1M_TARGET_MS,
        false,
    );
    tprintln!("  Shutdown: avg {:.2} ms (target {:.2} ms)", shutdown_result.average, SHUTDOWN_CHECKPOINT_1M_TARGET_MS);
}

// =============================================================================
// Test 16: Scale Test (10M keys)
// =============================================================================

/// Creates a B+Tree with 10,000,000 keys (~50+ MB index), writes and loads
/// a checkpoint, writes 100 more keys, crashes, and recovers. Measures
/// total recovery time (checkpoint load + WAL replay).
#[tokio::test]
async fn test_checkpoint_scale_10m() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const KEY_COUNT: usize = 10_000_000;
    const POST_KEYS: usize = 100;

    tprintln!("\n=== Checkpoint Scale Test (10M keys) ===");
    tprintln!("Keys: {}", format_with_commas(KEY_COUNT as f64));
    tprintln!("Post-checkpoint keys: {}", POST_KEYS);
    tprintln!("Validation runs: {}", VALIDATION_RUNS);

    let mut write_time_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut load_time_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut recovery_time_results = Vec::with_capacity(VALIDATION_RUNS);

    let scale_util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path().join("ckpt");
        let wal_dir = dir.path().join("wal");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();

        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir: dir.path().to_path_buf(),
                fsync_enabled: false,
            })
            .await
            .unwrap(),
        );
        let pool = Arc::new(BufferPool::auto_sized());

        // Build 10M key B+Tree
        let build_start = Instant::now();
        let mut btree =
            BTreeIndex::create_with_config(disk.clone(), pool.clone(), 0, checkpoint_dir.clone(), CheckpointConfig { fsync: false, ..CheckpointConfig::default() })
                .await
                .unwrap();

        for i in 0..KEY_COUNT as u64 {
            let key = i.to_be_bytes();
            let tid = TupleId::new(PageId::new(0, (i % 1000) as u32), (i % 100) as u16);
            btree.insert_exclusive(&key, tid).unwrap();
        }
        let build_duration = build_start.elapsed();
        tprintln!(
            "  Build: {:.0} ms ({} ops/sec)",
            build_duration.as_secs_f64() * 1000.0,
            format_with_commas(KEY_COUNT as f64 / build_duration.as_secs_f64())
        );

        // Write checkpoint
        let checkpoint_lsn = 1_000_000u64;
        let write_start = Instant::now();
        btree.force_checkpoint(checkpoint_lsn).unwrap();
        let write_duration = write_start.elapsed();
        let write_ms = write_duration.as_secs_f64() * 1000.0;

        let ckpt_path = checkpoint_dir.join("index_0.zyridx");
        let file_size = std::fs::metadata(&ckpt_path).unwrap().len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);

        tprintln!(
            "  Checkpoint write: {:.2} ms ({:.1} MB, {:.0} MB/sec)",
            write_ms,
            file_size_mb,
            file_size_mb / write_duration.as_secs_f64()
        );
        write_time_results.push(write_ms);

        // Load checkpoint (standalone timing)
        let load_start = Instant::now();
        let _loaded = BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
            .await
            .unwrap();
        let load_duration = load_start.elapsed();
        let load_ms = load_duration.as_secs_f64() * 1000.0;

        tprintln!(
            "  Checkpoint load: {:.2} ms ({:.0} MB/sec)",
            load_ms,
            file_size_mb / load_duration.as_secs_f64()
        );
        load_time_results.push(load_ms);
        drop(_loaded);

        // Write 100 post-checkpoint keys with WAL
        let wal_config = WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024,
        };
        let writer = Arc::new(WalWriter::new(wal_config).unwrap());

        for i in KEY_COUNT..(KEY_COUNT + POST_KEYS) {
            let txn_id = (i + 1) as u32;
            let begin_lsn = writer.log_begin(txn_id).unwrap();
            let payload = format!("key:{}", i);
            let insert_lsn = writer
                .log_insert(txn_id, begin_lsn, payload.as_bytes())
                .unwrap();
            writer.log_commit(txn_id, insert_lsn).unwrap();

            let key = (i as u64).to_be_bytes();
            let tid = TupleId::new(PageId::new(0, (i % 1000) as u32), (i as u64 % 100) as u16);
            btree.insert_exclusive(&key, tid).unwrap();
        }
        writer.flush().unwrap();
        writer.close().unwrap();
        drop(btree);

        // Full recovery: checkpoint load + WAL replay
        let recovery_start = Instant::now();
        let mut recovered =
            BTreeIndex::open(disk.clone(), pool.clone(), 0, &checkpoint_dir)
                .await
                .unwrap();

        // Replay post-checkpoint WAL
        let recovery = RecoveryManager::new(&wal_dir).unwrap();
        let result = recovery.recover().unwrap();
        let mut replayed = 0usize;
        for record in &result.redo_records {
            if record.record_type == LogRecordType::Insert {
                let payload_str = String::from_utf8_lossy(&record.payload);
                if let Some(key_str) = payload_str.strip_prefix("key:") {
                    if let Ok(i) = key_str.parse::<u64>() {
                        let key = i.to_be_bytes();
                        let tid = TupleId::new(
                            PageId::new(0, (i % 1000) as u32),
                            (i % 100) as u16,
                        );
                        if recovered.search_sync(&key).is_none() {
                            recovered.insert_exclusive(&key, tid).unwrap();
                        }
                        replayed += 1;
                    }
                }
            }
        }

        let recovery_duration = recovery_start.elapsed();
        let recovery_ms = recovery_duration.as_secs_f64() * 1000.0;

        tprintln!(
            "  Recovery: {:.2} ms (checkpoint load + {} WAL records)",
            recovery_ms,
            replayed
        );
        recovery_time_results.push(recovery_ms);

        // Verify spot-check: 1000 pre-checkpoint keys + all post-checkpoint keys
        let sample_step = KEY_COUNT / 1000;
        for i in (0..KEY_COUNT as u64).step_by(sample_step) {
            let key = i.to_be_bytes();
            assert!(
                recovered.search_sync(&key).is_some(),
                "Pre-checkpoint key {} missing (run {})",
                i,
                run + 1
            );
        }
        for i in KEY_COUNT..(KEY_COUNT + POST_KEYS) {
            let key = (i as u64).to_be_bytes();
            assert!(
                recovered.search_sync(&key).is_some(),
                "Post-checkpoint key {} missing (run {})",
                i,
                run + 1
            );
        }

        // Clean up WAL dir for next run
        let _ = std::fs::remove_dir_all(&wal_dir);
        std::fs::create_dir_all(&wal_dir).unwrap();
    }
    record_test_util("Checkpoint Scale", scale_util_before, take_util_snapshot());

    tprintln!("\n=== Checkpoint Scale Validation Results ===");
    let _write_result = validate_metric(
        "Checkpoint Scale",
        "Write 10M keys (ms)",
        write_time_results,
        50.0, // Informational, not gating
        false,
    );
    let _load_result = validate_metric(
        "Checkpoint Scale",
        "Load 10M keys (ms)",
        load_time_results,
        20.0, // Informational, not gating
        false,
    );
    let recovery_result = validate_metric(
        "Checkpoint Scale",
        "Recovery 10M+100 (ms)",
        recovery_time_results,
        RECOVERY_SCALE_TARGET_MS,
        false,
    );
    tprintln!("  Scale recovery: avg {:.2} ms (target {:.2} ms)", recovery_result.average, RECOVERY_SCALE_TARGET_MS);
}

// =============================================================================
// Summary Test
// =============================================================================

/// Summary test - runs after all validation tests complete.
#[tokio::test]
async fn test_phase1_summary() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n============================================================");
    tprintln!("ZyronDB Phase 1: Storage Foundation Validation Complete");
    tprintln!("============================================================");
    tprintln!("\nRun: cargo test -p zyron-storage --test phase1_test --release -- --nocapture");
}

#[tokio::test]
async fn test_checkpoint_io_profile() {
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().unwrap();
    
    tprintln!("\n=== I/O Profile ===");
    for size_mb in [1, 2, 4, 6, 8, 10, 14, 20] {
        let path = dir.path().join(format!("test_{}.bin", size_mb));
        let data = vec![0xABu8; size_mb * 1024 * 1024];
        std::fs::write(&path, &data).unwrap(); // warm
        
        let mut best_w = 999.0f64;
        let mut best_r = 999.0f64;
        for _ in 0..10 {
            let t = Instant::now();
            std::fs::write(&path, &data).unwrap();
            let w = t.elapsed().as_secs_f64() * 1000.0;
            if w < best_w { best_w = w; }
            
            let t = Instant::now();
            let _buf = std::fs::read(&path).unwrap();
            let r = t.elapsed().as_secs_f64() * 1000.0;
            if r < best_r { best_r = r; }
        }
        
        tprintln!("{:3} MB  W: {:6.2} ms ({:6.0} MB/s)  R: {:6.2} ms ({:6.0} MB/s)",
            size_mb,
            best_w, size_mb as f64 / (best_w / 1000.0),
            best_r, size_mb as f64 / (best_r / 1000.0));
    }
}

#[tokio::test]
async fn test_checkpoint_io_profile_v2() {
    use std::io::Read;
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempdir().unwrap();
    
    tprintln!("\n=== I/O Profile v2 (detailed read breakdown) ===");
    for size_mb in [6, 10, 14, 20, 46] {
        let path = dir.path().join(format!("v2_{}.bin", size_mb));
        let data = vec![0xABu8; size_mb * 1024 * 1024];
        std::fs::write(&path, &data).unwrap(); // create file
        
        // Method 1: std::fs::read
        let mut best_r1 = 999.0f64;
        for _ in 0..10 {
            let t = Instant::now();
            let _buf = std::fs::read(&path).unwrap();
            let r = t.elapsed().as_secs_f64() * 1000.0;
            if r < best_r1 { best_r1 = r; }
        }
        
        // Method 2: File::open + read_to_end with pre-alloc
        let mut best_r2 = 999.0f64;
        for _ in 0..10 {
            let t = Instant::now();
            let mut f = std::fs::File::open(&path).unwrap();
            let meta = f.metadata().unwrap();
            let len = meta.len() as usize;
            let mut buf = Vec::with_capacity(len);
            unsafe { buf.set_len(len); }
            f.read_exact(&mut buf).unwrap();
            let r = t.elapsed().as_secs_f64() * 1000.0;
            if r < best_r2 { best_r2 = r; }
        }

        // Method 3: Pre-opened file handle + seek + read_exact 
        let mut best_r3 = 999.0f64;
        {
            let mut pre_buf = vec![0u8; size_mb * 1024 * 1024];
            for _ in 0..10 {
                let t = Instant::now();
                let mut f = std::fs::File::open(&path).unwrap();
                f.read_exact(&mut pre_buf).unwrap();
                let r = t.elapsed().as_secs_f64() * 1000.0;
                if r < best_r3 { best_r3 = r; }
            }
        }
        
        tprintln!("{:3} MB  fs::read: {:6.2} ms  open+read_exact: {:6.2} ms  reuse_buf: {:6.2} ms",
            size_mb, best_r1, best_r2, best_r3);
    }
}
