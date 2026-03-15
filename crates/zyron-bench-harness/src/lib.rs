//! Shared benchmark harness for ZyronDB integration test suites.
//!
//! Each test file calls `init("suite_name")` once, then uses `validate_metric`,
//! `check_performance`, and the `tprintln!` macro for output. Results are written
//! to `benchmarks/<suite_name>/<suite_name>_<run_id>.{json,txt}`.

use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

// Re-export so test files can just `use zyron_bench_harness::*;`
pub use std::time::Instant;

// =============================================================================
// Suite name (set once per test binary via `init`)
// =============================================================================

static SUITE_NAME: OnceLock<String> = OnceLock::new();

/// Registers the suite name that determines the output subdirectory and file prefix.
/// Call this once at the top of each test file (typically inside a test or helper).
/// Subsequent calls with the same name are harmless; calls with a different name panic.
pub fn init(name: &str) {
    SUITE_NAME.get_or_init(|| name.to_string());
}

fn suite_name() -> &'static str {
    SUITE_NAME.get().map(|s| s.as_str()).unwrap_or("unknown")
}

// =============================================================================
// Validation constants
// =============================================================================

pub const VALIDATION_RUNS: usize = 5;
pub const REGRESSION_THRESHOLD: f64 = 2.0;

// =============================================================================
// tprintln! macro -- writes to both stdout and the run's text log file
// =============================================================================

#[macro_export]
macro_rules! tprintln {
    () => {{
        std::println!();
        $crate::write_raw_output("");
    }};
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        std::println!("{}", msg);
        $crate::write_raw_output(&msg);
    }};
}

// =============================================================================
// Core data types
// =============================================================================

pub struct ValidationResult {
    pub passed: bool,
    pub regression_detected: bool,
    pub average: f64,
}

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
pub struct UtilSnapshot {
    pub cpu_pct: f64,
    pub ram_used_gb: f64,
}

#[derive(Clone)]
struct UtilRecord {
    test: String,
    before: UtilSnapshot,
    after: UtilSnapshot,
}

struct PlatformHardware {
    cpu: String,
    ram_gb: f64,
    gpus: Vec<String>,
}

// =============================================================================
// Global state (per test binary)
// =============================================================================

static GIT_COMMIT: OnceLock<String> = OnceLock::new();
static PLATFORM_HW: OnceLock<PlatformHardware> = OnceLock::new();
static RUN_ID: OnceLock<String> = OnceLock::new();
static RAW_LOG: OnceLock<Mutex<std::fs::File>> = OnceLock::new();
static COLLECTED_METRICS: OnceLock<Mutex<Vec<MetricRecord>>> = OnceLock::new();
static COLLECTED_UTILS: OnceLock<Mutex<Vec<UtilRecord>>> = OnceLock::new();

// =============================================================================
// Formatting
// =============================================================================

pub fn format_with_commas(n: f64) -> String {
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

// =============================================================================
// Metric validation
// =============================================================================

pub fn validate_metric(
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

pub fn check_performance(
    test: &str,
    metric_name: &str,
    value: f64,
    target: f64,
    higher_is_better: bool,
) -> bool {
    let passed = if higher_is_better {
        value >= target
    } else {
        value <= target
    };
    let status = if passed { "PASS" } else { "FAIL" };
    let comparison = if higher_is_better { ">=" } else { "<=" };
    tprintln!(
        "  {} [{}]: {} {} {} (target)",
        metric_name,
        status,
        format_with_commas(value),
        comparison,
        format_with_commas(target),
    );
    write_benchmark_record(
        test,
        metric_name,
        value,
        vec![value],
        target,
        passed,
        higher_is_better,
    );
    passed
}

// =============================================================================
// Utilization snapshots
// =============================================================================

#[cfg(target_os = "windows")]
pub fn take_util_snapshot() -> UtilSnapshot {
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
pub fn take_util_snapshot() -> UtilSnapshot {
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
pub fn take_util_snapshot() -> UtilSnapshot {
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
pub fn take_util_snapshot() -> UtilSnapshot {
    UtilSnapshot {
        cpu_pct: 0.0,
        ram_used_gb: 0.0,
    }
}

/// Records system utilization for a test group and rewrites the JSON file.
pub fn record_test_util(test: &str, before: UtilSnapshot, after: UtilSnapshot) {
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

// =============================================================================
// Platform hardware detection
// =============================================================================

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

fn platform_hw() -> &'static PlatformHardware {
    PLATFORM_HW.get_or_init(platform_hw_impl)
}

pub fn logical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

// =============================================================================
// Run identity and git info
// =============================================================================

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

pub fn run_id() -> &'static str {
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

// =============================================================================
// Metric and util collection
// =============================================================================

fn collected_metrics() -> &'static Mutex<Vec<MetricRecord>> {
    COLLECTED_METRICS.get_or_init(|| Mutex::new(Vec::new()))
}

fn collected_utils() -> &'static Mutex<Vec<UtilRecord>> {
    COLLECTED_UTILS.get_or_init(|| Mutex::new(Vec::new()))
}

// =============================================================================
// JSON output
// =============================================================================

fn json_str_array(items: &[String]) -> String {
    let inner = items
        .iter()
        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", inner)
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

// =============================================================================
// File output -- writes to benchmarks/<suite_name>/ subdirectory
// =============================================================================

fn benchmark_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR points to the crate directory. Walk up to workspace root.
    // For zyron-bench-harness itself this is crates/zyron-bench-harness, but the
    // callers compile this crate as a dependency so the env var resolves to the
    // caller's crate directory. We go up 2 levels (crates/<name> -> workspace root).
    //
    // To make this work regardless of which crate calls us, we search upward for
    // the workspace Cargo.toml that contains [workspace].
    let start = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dir = start.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(contents) = std::fs::read_to_string(&cargo_toml) {
                if contents.contains("[workspace]") {
                    return dir.join("benchmarks").join(suite_name());
                }
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }
    // Fallback: two levels up from the harness crate itself
    start
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks")
        .join(suite_name())
}

fn write_run_json(metrics: &[MetricRecord], utils: &[UtilRecord]) {
    let dir = benchmark_dir();
    let _ = std::fs::create_dir_all(&dir);
    let fname = format!("{}_{}.json", suite_name(), run_id());
    let json = build_run_json(metrics, utils);
    let _ = std::fs::write(dir.join(fname), json.as_bytes());
}

fn raw_log_file() -> &'static Mutex<std::fs::File> {
    RAW_LOG.get_or_init(|| {
        let dir = benchmark_dir();
        let _ = std::fs::create_dir_all(&dir);
        let name = suite_name();
        let fname = format!("{}_{}.txt", name, run_id());
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
                    .open(dir.join(format!("{}_latest.txt", name)))
                    .expect("failed to open benchmark log")
            });
        Mutex::new(f)
    })
}

pub fn write_raw_output(line: &str) {
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
