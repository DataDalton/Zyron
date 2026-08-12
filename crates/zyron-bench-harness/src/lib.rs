//! Shared benchmark harness for ZyronDB integration test suites.
//!
//! Each test file calls `init("suite_name")` once, then uses `validate_metric`,
//! `check_performance`, and the `tprintln!` macro for output. Results are written
//! to `benchmarks/<suite_name>/<suite_name>_<run_id>.{json,txt}`.

use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

// Re-export so test files can just `use zyron_bench_harness::*;`
pub use std::time::Instant;

pub mod production;
pub use production::{
    buffer_pool_config, compaction_config, create_dirs, data_and_wal_dirs, disk_config,
    wal_config,
};

// =============================================================================
// Suite name (set once per test binary via `init`)
// =============================================================================

static SUITE_NAME: OnceLock<String> = OnceLock::new();

/// Registers the suite name that determines the output subdirectory and file prefix.
/// Call this once at the top of each test file (typically inside a test or helper).
/// Subsequent calls with the same name are harmless. Calls with a different name panic.
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

/// Whether this build can produce a number worth comparing to a target.
///
/// An unoptimized build is not slower by a constant factor. It is slower by an
/// amount that depends on how much inlining, bounds-check elision and
/// vectorization each path would have had, so one routine loses 5x and another
/// 100x. Every target in this repo was set against an optimized build, which
/// makes a debug comparison meaningless in both directions: it fails code that
/// is fine, and it cannot fail code that is not.
///
/// Benchmarks are therefore run with `--release`. In a debug build the suites
/// still execute and still print, so they keep working as correctness tests,
/// but nothing is judged against a target.
#[inline]
pub fn measuring() -> bool {
    !cfg!(debug_assertions)
}

/// Whether an expensive benchmark should do its work at all.
///
/// Some suites build structures large enough that the run is measured in hours
/// without optimization, for a number that would be discarded anyway. Those
/// call this and return early, printing why, rather than spending the time.
pub fn skip_expensive(test: &str, what: &str) -> bool {
    if measuring() {
        return false;
    }
    tprintln!(
        "  {} [SKIPPED]: {} is only measured in an optimized build.\n             Run: cargo test --release -p <crate> --test <suite>",
        test,
        what
    );
    true
}

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

/// Which storage format produced a number.
///
/// Most metrics have none, because most of what a suite measures is not
/// specific to a format. One that carries a format is comparable against
/// the same metric from the other format, and the run file pairs the two
/// and reports their ratio rather than leaving it to be worked out from
/// two files
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Heap,
    Lake,
}

impl Format {
    pub fn as_str(self) -> &'static str {
        match self {
            Format::Heap => "heap",
            Format::Lake => "lake",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which side of a ratio a bound constrains.
///
/// `AtMost(1.0)` on lake over heap says lake must be at least as fast;
/// `AtLeast(50.0)` on heap over lake says heap must be fifty times ahead.
/// Which way round is the claim, so it is stated rather than inferred
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RatioBound {
    AtMost(f64),
    AtLeast(f64),
}

impl RatioBound {
    fn admits(self, value: f64) -> bool {
        match self {
            RatioBound::AtMost(limit) => value <= limit,
            RatioBound::AtLeast(limit) => value >= limit,
        }
    }

    fn comparison(self) -> &'static str {
        match self {
            RatioBound::AtMost(_) => "<=",
            RatioBound::AtLeast(_) => ">=",
        }
    }

    fn limit(self) -> f64 {
        match self {
            RatioBound::AtMost(v) | RatioBound::AtLeast(v) => v,
        }
    }
}

/// Whether a quantity means the same thing in an unoptimized build.
///
/// This decides whether a bound on it can be applied outside a measuring
/// build, and it is a property of what is being counted rather than of how
/// the number was collected, so the caller states it
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exactness {
    /// Counts. Bytes read, rows scanned, files pruned, index entries
    /// probed. Optimization changes how fast the work happens, not how much
    /// of it there is, so these are identical in every build profile and a
    /// bound on one holds everywhere
    Exact,
    /// Timings and anything derived from one. An unoptimized build is not
    /// slower by a constant factor, so a bound on one means nothing outside
    /// a measuring build and is not applied there
    ProfileDependent,
}

#[derive(Clone)]
struct MetricRecord {
    test: String,
    metric: String,
    /// None when the metric is not specific to a storage format
    format: Option<Format>,
    average: f64,
    runs: Vec<f64>,
    /// None when the number was recorded without being judged, which is
    /// what a cross-format measurement is: its claim is its ratio
    target: Option<f64>,
    passed: Option<bool>,
    higher_is_better: bool,
}

#[derive(Clone)]
struct RatioRecord {
    test: String,
    metric: String,
    numerator: Format,
    denominator: Format,
    numerator_value: f64,
    denominator_value: f64,
    value: f64,
    /// None until a release measurement sets one
    bound: Option<RatioBound>,
    passed: Option<bool>,
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
static COLLECTED_RATIOS: OnceLock<Mutex<Vec<RatioRecord>>> = OnceLock::new();

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

/// Renders a measurement without throwing away what it is.
///
/// `format_with_commas` rounds to whole units, which is right for a count
/// of rows or a microsecond total and destroys a rate: any skip rate above
/// a half prints as one and reads as perfect when it is not, and a
/// fractional bound prints as zero and reads as no bound at all. Values
/// below a thousand keep four decimals, larger ones get the grouped
/// integer form, so a fraction and a file count each print as themselves
pub fn format_measurement(n: f64) -> String {
    if n.abs() < 1_000.0 && n.fract() != 0.0 {
        format!("{:.4}", n)
    } else {
        format_with_commas(n)
    }
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
    validate_metric_inner(test, name, None, runs, target, higher_is_better)
}

/// Validates a metric that belongs to one storage format.
///
/// The run file groups the two formats' numbers under one metric name and
/// reports their ratio, so a comparison is read from one file rather than
/// assembled from two. Use this only where the metric genuinely means the
/// same thing on both formats. A metric with no counterpart belongs in
/// `validate_metric`, where an empty cell is the honest answer
pub fn validate_metric_for(
    format: Format,
    test: &str,
    name: &str,
    runs: Vec<f64>,
    target: f64,
    higher_is_better: bool,
) -> ValidationResult {
    validate_metric_inner(test, name, Some(format), runs, target, higher_is_better)
}

fn validate_metric_inner(
    test: &str,
    name: &str,
    format: Option<Format>,
    runs: Vec<f64>,
    target: f64,
    higher_is_better: bool,
) -> ValidationResult {
    // The format is shown beside the metric but is not folded into its
    // name, so the run file can still group both formats under one metric
    let display = match format {
        Some(f) => format!("{} [{}]", name, f),
        None => name.to_string(),
    };
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

    // A debug build did not measure this, so it does not get to judge it.
    // Reporting FAIL here is what turned every suite run without --release
    // into a page of regressions that said nothing about the code
    if !measuring() {
        tprintln!("  {} [NOT MEASURED, unoptimized build]:", display);
        tprintln!(
            "    Runs: [{}]",
            runs.iter()
                .map(|x| format_with_commas(*x))
                .collect::<Vec<_>>()
                .join(", ")
        );
        tprintln!(
            "    Average: {} (target {} {}, not applied)",
            format_with_commas(average),
            if higher_is_better { ">=" } else { "<=" },
            format_with_commas(target)
        );
        tprintln!("    Re-run with --release for a number worth comparing");
        // Recorded like any other run. Every run writes, and the run file's
        // profile field is what tells a reader this one was not measured.
        // The target is carried but no verdict is, because none was reached
        write_benchmark_record(test, name, format, average, runs, Some(target), None, higher_is_better);
        return ValidationResult {
            passed: true,
            regression_detected: false,
            average,
        };
    }

    let status = if passed { "PASS" } else { "FAIL" };
    let regr_status = if regression_detected { "REGR!" } else { "OK" };
    let comparison = if higher_is_better { ">=" } else { "<=" };

    tprintln!("  {} [{}/{}]:", display, status, regr_status);
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

    write_benchmark_record(
        test,
        name,
        format,
        average,
        runs,
        Some(target),
        Some(passed),
        higher_is_better,
    );

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
    check_performance_inner(test, metric_name, None, value, target, higher_is_better)
}

/// Checks a single measurement that belongs to one storage format.
///
/// The format-free `check_performance` is the right call for a metric that
/// has no counterpart in the other format, which most do not
pub fn check_performance_for(
    format: Format,
    test: &str,
    metric_name: &str,
    value: f64,
    target: f64,
    higher_is_better: bool,
) -> bool {
    check_performance_inner(
        test,
        metric_name,
        Some(format),
        value,
        target,
        higher_is_better,
    )
}

fn check_performance_inner(
    test: &str,
    metric_name: &str,
    format: Option<Format>,
    value: f64,
    target: f64,
    higher_is_better: bool,
) -> bool {
    let display = match format {
        Some(f) => format!("{} [{}]", metric_name, f),
        None => metric_name.to_string(),
    };
    if !measuring() {
        tprintln!(
            "  {} [NOT MEASURED, unoptimized build]: {} (target {} {}, not applied)",
            display,
            format_with_commas(value),
            if higher_is_better { ">=" } else { "<=" },
            format_with_commas(target)
        );
        let _ = test;
        return true;
    }
    let passed = if higher_is_better {
        value >= target
    } else {
        value <= target
    };
    let status = if passed { "PASS" } else { "FAIL" };
    let comparison = if higher_is_better { ">=" } else { "<=" };
    tprintln!(
        "  {} [{}]: {} {} {} (target)",
        display,
        status,
        format_with_commas(value),
        comparison,
        format_with_commas(target),
    );
    write_benchmark_record(
        test,
        metric_name,
        format,
        value,
        vec![value],
        Some(target),
        Some(passed),
        higher_is_better,
    );
    passed
}

// =============================================================================
// Cross-format ratios
// =============================================================================

/// Asserts one format's measurement against the other's, as a ratio.
///
/// A ratio is what a cross-format claim should be made of. Both numbers
/// come from one run on one machine, so the hardware divides out: the
/// answer means the same thing on the machine that set the target and on
/// the one running it today. Every absolute target in this repo went stale
/// when the CPU changed, and a ratio does not have that failure mode.
///
/// Both values must measure the same thing in the same units, and the
/// caller is responsible for having verified that the two formats returned
/// the same rows. A timing whose answers differ is a measurement of a
/// cheaper wrong thing.
///
/// In an unoptimized build this prints and records nothing to judge, the
/// same rule every other assertion here follows
/// Records one format's measurement without judging it, returning its
/// average so a ratio can be taken.
///
/// A cross-format metric's claim is its ratio, not either absolute value,
/// so both sides are recorded and only the ratio carries a bound. Putting
/// an absolute target on one side would reintroduce exactly the staleness
/// this suite exists to avoid: it would have to be re-set on every machine
pub fn record_metric_for(
    format: Format,
    test: &str,
    name: &str,
    unit: &str,
    runs: Vec<f64>,
) -> f64 {
    let average = if runs.is_empty() {
        0.0
    } else {
        runs.iter().sum::<f64>() / runs.len() as f64
    };
    let u = |v: f64| format!("{}{}", format_measurement(v), unit);
    tprintln!("  {} [{}]:", name, format);
    tprintln!(
        "    Runs: [{}]",
        runs.iter().map(|x| u(*x)).collect::<Vec<_>>().join(", ")
    );
    tprintln!("    Average: {} (no target set)", u(average));
    if runs.len() > 1 {
        let min = runs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance =
            runs.iter().map(|r| (r - average).powi(2)).sum::<f64>() / runs.len() as f64;
        tprintln!(
            "    Min/Max: {} / {}, StdDev: {}",
            u(min),
            u(max),
            u(variance.sqrt())
        );
    }
    write_benchmark_record(test, name, Some(format), average, runs, None, None, false);
    average
}

/// Records an absolute measurement that belongs to no storage format and
/// carries no target, returning its average.
///
/// This is what a measurement of something only one format has looks like
/// before any baseline exists: file pruning effectiveness, manifest
/// planning cost, transaction log commit rate. `validate_metric` would
/// demand a target, and a target invented without an optimized run on
/// known hardware is a number somebody made up that goes stale the first
/// time the machine changes. Recording keeps it in the run file so a
/// baseline can be read straight off it later
pub fn record_metric(test: &str, name: &str, unit: &str, runs: Vec<f64>) -> f64 {
    let average = if runs.is_empty() {
        0.0
    } else {
        runs.iter().sum::<f64>() / runs.len() as f64
    };
    // The unit rides on every number rather than on the metric name, so a
    // line can be read on its own without scrolling back to find out what
    // the figures are in
    let u = |v: f64| format!("{}{}", format_measurement(v), unit);
    tprintln!("  {}:", name);
    tprintln!(
        "    Runs: [{}]",
        runs.iter().map(|x| u(*x)).collect::<Vec<_>>().join(", ")
    );
    tprintln!("    Average: {} (no target set)", u(average));
    if runs.len() > 1 {
        let min = runs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = runs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance =
            runs.iter().map(|r| (r - average).powi(2)).sum::<f64>() / runs.len() as f64;
        tprintln!(
            "    Min/Max: {} / {}, StdDev: {}",
            u(min),
            u(max),
            u(variance.sqrt())
        );
    }
    write_benchmark_record(test, name, None, average, runs, None, None, false);
    average
}

/// Asserts a bound on a counted quantity that belongs to no storage format.
///
/// The count analogue of `record_metric`: a skip rate, a file count, a
/// retry count. Optimization does not change any of them, so the bound
/// applies in every build profile and the assertion is worth running on
/// every test pass rather than only on a baselined one
pub fn assert_exact_metric(
    test: &str,
    name: &str,
    value: f64,
    bound: RatioBound,
) -> bool {
    let admits = bound.admits(value);
    tprintln!(
        "  {} [{}]: {} (bound {} {})",
        name,
        if admits { "PASS" } else { "FAIL" },
        format_measurement(value),
        bound.comparison(),
        format_measurement(bound.limit())
    );
    write_benchmark_record(
        test,
        name,
        None,
        value,
        vec![value],
        Some(bound.limit()),
        Some(admits),
        matches!(bound, RatioBound::AtLeast(_)),
    );
    admits
}

/// The ratio, or None when the pair cannot produce one.
///
/// A zero or non-finite denominator is a broken measurement rather than
/// an infinite ratio, and calling it a pass would hide that the baseline
/// measured nothing
fn ratio_value(numerator: f64, denominator: f64) -> Option<f64> {
    if !denominator.is_finite() || denominator <= 0.0 || !numerator.is_finite() {
        return None;
    }
    Some(numerator / denominator)
}

pub fn assert_ratio(
    test: &str,
    metric: &str,
    numerator: (Format, f64),
    denominator: (Format, f64),
    bound: RatioBound,
) -> bool {
    assert_ratio_with(
        test,
        metric,
        numerator,
        denominator,
        bound,
        Exactness::ProfileDependent,
    )
}

/// Asserts a bound on a ratio between two counts rather than two timings.
///
/// A byte count or a row count is the same number in an unoptimized build,
/// so its bound applies there too. That makes it the one kind of
/// cross-format claim worth checking on every run rather than only on a
/// baselined one, and it is usually the claim that explains a timing:
/// a format is not faster by accident, it is faster because it read less
pub fn assert_exact_ratio(
    test: &str,
    metric: &str,
    numerator: (Format, f64),
    denominator: (Format, f64),
    bound: RatioBound,
) -> bool {
    assert_ratio_with(test, metric, numerator, denominator, bound, Exactness::Exact)
}

fn assert_ratio_with(
    test: &str,
    metric: &str,
    numerator: (Format, f64),
    denominator: (Format, f64),
    bound: RatioBound,
    exactness: Exactness,
) -> bool {
    match record_ratio_inner(test, metric, numerator, denominator, Some(bound), exactness) {
        // A profile-dependent bound is not applied outside a measuring
        // build, and reporting a bound that was never checked as failed
        // would be a lie in the other direction
        Some(value) => {
            if measuring() || exactness == Exactness::Exact {
                bound.admits(value)
            } else {
                true
            }
        }
        None => false,
    }
}

/// Records a ratio without judging it, returning it, or None when the
/// pair is not comparable.
///
/// This is what a cross-format claim looks like before its bound has been
/// set from an optimized run. A bound invented from an unoptimized one is
/// not a target, it is a number somebody made up, and it would have to be
/// moved the first time real hardware disagreed. Recording keeps the
/// comparison in the run file so a baseline can be read straight off it
pub fn record_ratio(
    test: &str,
    metric: &str,
    numerator: (Format, f64),
    denominator: (Format, f64),
) -> Option<f64> {
    record_ratio_inner(
        test,
        metric,
        numerator,
        denominator,
        None,
        Exactness::ProfileDependent,
    )
}

fn record_ratio_inner(
    test: &str,
    metric: &str,
    numerator: (Format, f64),
    denominator: (Format, f64),
    bound: Option<RatioBound>,
    exactness: Exactness,
) -> Option<f64> {
    let (num_format, num_value) = numerator;
    let (den_format, den_value) = denominator;
    let label = format!("{} [{} / {}]", metric, num_format, den_format);
    let bound_text = match bound {
        Some(b) => format!("bound {} {}", b.comparison(), format_measurement(b.limit())),
        None => "no bound set".to_string(),
    };

    let Some(value) = ratio_value(num_value, den_value) else {
        tprintln!(
            "  {} [FAIL]: not comparable, {} over {}",
            label,
            format_with_commas(num_value),
            format_with_commas(den_value)
        );
        return None;
    };

    if !measuring() {
        // A count is the same number here as it is in an optimized build, so
        // its bound is applied and its verdict reported. A timing is not, so
        // it is shown and left unjudged. Either way no record is written:
        // a run file that carried counts but no timings would read as a
        // benchmark result while being nothing of the kind
        match (exactness, bound) {
            (Exactness::Exact, Some(b)) => tprintln!(
                "  {} [{}]: {:.3} ({})",
                label,
                if b.admits(value) { "PASS" } else { "FAIL" },
                value,
                bound_text
            ),
            _ => tprintln!(
                "  {}: {:.3} ({}, not applied in an unoptimized build)",
                label,
                value,
                bound_text
            ),
        }
        let record = RatioRecord {
            test: test.to_string(),
            metric: metric.to_string(),
            numerator: num_format,
            denominator: den_format,
            numerator_value: num_value,
            denominator_value: den_value,
            value,
            bound,
            // A verdict only where one was actually reached: an exact bound
            // is applied in any profile, a profile-dependent one is not
            passed: match (exactness, bound) {
                (Exactness::Exact, Some(b)) => Some(b.admits(value)),
                _ => None,
            },
        };
        if let Ok(mut g) = collected_ratios().lock() {
            g.push(record);
        }
        write_current_run();
        return Some(value);
    }

    let passed = bound.map(|b| b.admits(value));
    // A ratio with no bound carries no verdict, so it carries no tag. The
    // bound text already says whether one was set
    let verdict = match passed {
        Some(true) => " [PASS]",
        Some(false) => " [FAIL]",
        None => "",
    };
    tprintln!(
        "  {}{}: {:.3} ({}), from {} and {}",
        label,
        verdict,
        value,
        bound_text,
        format_measurement(num_value),
        format_measurement(den_value)
    );

    let record = RatioRecord {
        test: test.to_string(),
        metric: metric.to_string(),
        numerator: num_format,
        denominator: den_format,
        numerator_value: num_value,
        denominator_value: den_value,
        value,
        bound,
        passed,
    };
    if let Ok(mut g) = collected_ratios().lock() {
        g.push(record);
    }
    write_current_run();
    Some(value)
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
    if let Ok(mut g) = collected_utils().lock() {
        if let Some(existing) = g.iter_mut().find(|u| u.test == test) {
            *existing = record;
        } else {
            g.push(record);
        }
    } else {
        return;
    }
    write_current_run();
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

fn collected_ratios() -> &'static Mutex<Vec<RatioRecord>> {
    COLLECTED_RATIOS.get_or_init(|| Mutex::new(Vec::new()))
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

/// One metric's numbers, at the given indent. `trailing` adds a comma
/// after the last field for a caller that appends more
fn push_metric_body(out: &mut String, m: &MetricRecord, indent: &str, trailing: bool) {
    let runs_json = m
        .runs
        .iter()
        .map(|v| format!("{:.2}", v))
        .collect::<Vec<_>>()
        .join(", ");
    // A number recorded without a target is not a pass and not a failure,
    // so both fields read null rather than a value that would be taken
    // for a judgement nobody made
    let target = match m.target {
        Some(v) => format!("{:.6}", v),
        None => "null".to_string(),
    };
    let passed = match m.passed {
        Some(v) => v.to_string(),
        None => "null".to_string(),
    };
    out.push_str(&format!("{indent}\"average\": {:.6},\n", m.average));
    out.push_str(&format!("{indent}\"runs\": [{runs_json}],\n"));
    out.push_str(&format!("{indent}\"target\": {target},\n"));
    out.push_str(&format!("{indent}\"passed\": {passed},\n"));
    out.push_str(&format!(
        "{indent}\"higher_is_better\": {}{}\n",
        m.higher_is_better,
        if trailing { "," } else { "" }
    ));
}

fn build_run_json(
    metrics: &[MetricRecord],
    utils: &[UtilRecord],
    ratios: &[RatioRecord],
) -> String {
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
    // Which build produced these numbers. A record that does not say cannot be
    // told apart later from one that was measured, and an unoptimized timing
    // read as a real one is worse than having no record at all
    out.push_str(&format!(
        "  \"profile\": \"{}\",\n",
        if measuring() { "release" } else { "debug" }
    ));
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

        // Grouped by metric name so the same metric measured on both
        // formats lands in one object with its ratio, instead of two keys
        // that a reader has to pair up, or worse the same key twice
        let mut metric_names: Vec<&str> = Vec::new();
        for m in metrics.iter().filter(|m| m.test.as_str() == *test_name) {
            if !metric_names.contains(&m.metric.as_str()) {
                metric_names.push(&m.metric);
            }
        }
        for (mi, metric_name) in metric_names.iter().enumerate() {
            let escaped_metric = metric_name.replace('"', "\\\"");
            let comma = if mi + 1 < metric_names.len() { "," } else { "" };
            let group: Vec<&MetricRecord> = metrics
                .iter()
                .filter(|m| m.test.as_str() == *test_name && m.metric.as_str() == *metric_name)
                .collect();
            let ratio = ratios
                .iter()
                .find(|r| r.test.as_str() == *test_name && r.metric.as_str() == *metric_name);

            out.push_str(&format!("      \"{escaped_metric}\": {{\n"));
            if group.len() == 1 && group[0].format.is_none() && ratio.is_none() {
                push_metric_body(&mut out, group[0], "        ", false);
            } else {
                for (gi, m) in group.iter().enumerate() {
                    // A format-free record inside a grouped metric is keyed
                    // by its position, which keeps every key distinct
                    let key = match m.format {
                        Some(f) => f.as_str().to_string(),
                        None => format!("run_{}", gi + 1),
                    };
                    let trailing = gi + 1 < group.len() || ratio.is_some();
                    out.push_str(&format!("        \"{key}\": {{\n"));
                    push_metric_body(&mut out, m, "          ", false);
                    out.push_str(&format!("        }}{}\n", if trailing { "," } else { "" }));
                }
                if let Some(r) = ratio {
                    out.push_str("        \"ratio\": {\n");
                    out.push_str(&format!("          \"of\": \"{}\",\n", r.numerator));
                    out.push_str(&format!("          \"to\": \"{}\",\n", r.denominator));
                    out.push_str(&format!("          \"value\": {:.6},\n", r.value));
                    out.push_str(&format!(
                        "          \"of_value\": {:.6},\n",
                        r.numerator_value
                    ));
                    out.push_str(&format!(
                        "          \"to_value\": {:.6},\n",
                        r.denominator_value
                    ));
                    // A ratio whose bound has not been set from an
                    // optimized run reads null rather than a number that
                    // would be taken for a target somebody chose
                    let bound = match r.bound {
                        Some(b) => format!("\"{} {:.6}\"", b.comparison(), b.limit()),
                        None => "null".to_string(),
                    };
                    let passed = match r.passed {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    out.push_str(&format!("          \"bound\": {bound},\n"));
                    out.push_str(&format!("          \"passed\": {passed}\n"));
                    out.push_str("        }\n");
                }
            }
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

fn write_run_json(metrics: &[MetricRecord], utils: &[UtilRecord], ratios: &[RatioRecord]) {
    let dir = benchmark_dir();
    let _ = std::fs::create_dir_all(&dir);
    let fname = format!("{}_{}.json", suite_name(), run_id());
    let json = build_run_json(metrics, utils, ratios);
    let _ = std::fs::write(dir.join(fname), json.as_bytes());
}

/// Rewrites the run file from everything collected so far.
///
/// The file is complete after every assertion rather than only at the end,
/// so a suite that fails partway still leaves the numbers it did take
fn write_current_run() {
    let metrics = collected_metrics()
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    let utils = collected_utils()
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    let ratios = collected_ratios()
        .lock()
        .ok()
        .map(|g| g.clone())
        .unwrap_or_default();
    write_run_json(&metrics, &utils, &ratios);
}

fn raw_log_file() -> &'static Mutex<std::fs::File> {
    RAW_LOG.get_or_init(|| {
        let dir = benchmark_dir();
        let _ = std::fs::create_dir_all(&dir);
        let name = suite_name();
        let fname = format!("{}_{}.txt", name, run_id());
        let mut f = std::fs::OpenOptions::new()
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

        // Every run writes a log, so every log has to say what produced it.
        // The JSON has carried a profile field all along and the text log
        // carried nothing, which left an unoptimized run's file looking
        // exactly like a measured one: same directory, same name shape, same
        // columns of numbers. The header is written before any line of output
        // so it cannot be missed by reading from the top
        let hw = platform_hw();
        let _ = writeln!(f, "# suite:   {}", name);
        let _ = writeln!(f, "# run:     {}", run_id());
        let _ = writeln!(f, "# profile: {}", if measuring() { "release" } else { "debug" });
        let _ = writeln!(f, "# cpu:     {}", hw.cpu);
        let _ = writeln!(f, "# cores:   {}", logical_cores());
        if !measuring() {
            let _ = writeln!(f, "#");
            let _ = writeln!(
                f,
                "# NOT A BENCHMARK RESULT. This is an unoptimized build, so every"
            );
            let _ = writeln!(
                f,
                "# timing below is meaningless: an unoptimized build is not slower"
            );
            let _ = writeln!(
                f,
                "# by a constant factor, so these numbers do not even rank correctly"
            );
            let _ = writeln!(
                f,
                "# against each other. Counted quantities (rates, file and row"
            );
            let _ = writeln!(
                f,
                "# counts, PASS/FAIL on an exact bound) are valid and are the only"
            );
            let _ = writeln!(
                f,
                "# thing worth reading here. For real numbers: cargo test --release"
            );
        }
        let _ = writeln!(f, "#");

        Mutex::new(f)
    })
}

pub fn write_raw_output(line: &str) {
    if let Ok(mut guard) = raw_log_file().lock() {
        let _ = writeln!(guard, "{}", line);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_benchmark_record(
    test: &str,
    metric: &str,
    format: Option<Format>,
    average: f64,
    runs: Vec<f64>,
    target: Option<f64>,
    passed: Option<bool>,
    higher_is_better: bool,
) {
    let record = MetricRecord {
        test: test.to_string(),
        metric: metric.to_string(),
        format,
        average,
        runs,
        target,
        passed,
        higher_is_better,
    };
    if let Ok(mut g) = collected_metrics().lock() {
        g.push(record);
    } else {
        return;
    }
    write_current_run();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(test: &str, name: &str, format: Option<Format>, average: f64) -> MetricRecord {
        MetricRecord {
            test: test.to_string(),
            metric: name.to_string(),
            format,
            average,
            runs: vec![average],
            target: Some(average),
            passed: Some(true),
            higher_is_better: false,
        }
    }

    /// Every key inside one object, at the nesting level it sits at.
    ///
    /// The run file is written by hand rather than by a serializer, so a
    /// repeated metric name used to emit the same key twice, which is not
    /// valid JSON and silently loses one of the two numbers
    fn duplicate_keys(json: &str) -> Vec<String> {
        let mut scopes: Vec<Vec<String>> = vec![Vec::new()];
        let mut duplicates = Vec::new();
        let mut chars = json.chars().peekable();
        let mut current = String::new();
        let mut in_string = false;
        let mut escaped = false;
        let mut pending_key: Option<String> = None;

        while let Some(c) = chars.next() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                    // A string followed by a colon is a key
                    let mut lookahead = chars.clone();
                    while matches!(lookahead.peek(), Some(' ') | Some('\n')) {
                        lookahead.next();
                    }
                    if lookahead.peek() == Some(&':') {
                        pending_key = Some(std::mem::take(&mut current));
                    } else {
                        current.clear();
                    }
                    continue;
                }
                if in_string {
                    current.push(c);
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                ':' => {
                    if let Some(key) = pending_key.take() {
                        let scope = scopes.last_mut().expect("a scope is always open");
                        if scope.contains(&key) {
                            duplicates.push(key.clone());
                        }
                        scope.push(key);
                    }
                }
                '{' => scopes.push(Vec::new()),
                '}' => {
                    scopes.pop();
                    if scopes.is_empty() {
                        scopes.push(Vec::new());
                    }
                }
                _ => {}
            }
        }
        duplicates
    }

    #[test]
    fn test_a_metric_measured_on_both_formats_lands_under_one_name_with_its_ratio() {
        let metrics = vec![
            metric("point_lookup", "latency us", Some(Format::Heap), 40.0),
            metric("point_lookup", "latency us", Some(Format::Lake), 32.0),
        ];
        let ratios = vec![RatioRecord {
            test: "point_lookup".into(),
            metric: "latency us".into(),
            numerator: Format::Lake,
            denominator: Format::Heap,
            numerator_value: 32.0,
            denominator_value: 40.0,
            value: 0.8,
            bound: Some(RatioBound::AtMost(1.0)),
            passed: Some(true),
        }];
        let json = build_run_json(&metrics, &[], &ratios);

        assert!(json.contains("\"latency us\""), "{json}");
        assert!(json.contains("\"heap\""), "{json}");
        assert!(json.contains("\"lake\""), "{json}");
        assert!(json.contains("\"ratio\""), "{json}");
        assert!(json.contains("\"of\": \"lake\""), "{json}");
        assert!(json.contains("\"to\": \"heap\""), "{json}");
        assert!(json.contains("\"value\": 0.800000"), "{json}");
        assert!(json.contains("\"bound\": \"<= 1.000000\""), "{json}");
        assert_eq!(
            json.matches("\"latency us\"").count(),
            1,
            "one metric name, not one per format"
        );
        assert!(
            duplicate_keys(&json).is_empty(),
            "duplicate keys in {json}"
        );
    }

    #[test]
    fn test_a_metric_with_no_format_keeps_the_shape_every_existing_suite_writes() {
        let metrics = vec![metric("insert", "rows per second", None, 120_000.0)];
        let json = build_run_json(&metrics, &[], &[]);
        assert!(json.contains("\"rows per second\": {"), "{json}");
        assert!(json.contains("\"average\": 120000.000000"), "{json}");
        assert!(!json.contains("\"heap\""), "no format dimension was asked for");
        assert!(!json.contains("\"ratio\""), "{json}");
        assert!(duplicate_keys(&json).is_empty(), "duplicate keys in {json}");
    }

    #[test]
    fn test_two_records_of_one_metric_name_stay_distinct_keys() {
        // Without grouping these emitted the same key twice, which is not
        // valid JSON and drops one of the two numbers on read
        let metrics = vec![
            metric("scan", "throughput", None, 10.0),
            metric("scan", "throughput", None, 20.0),
        ];
        let json = build_run_json(&metrics, &[], &[]);
        assert!(json.contains("\"run_1\""), "{json}");
        assert!(json.contains("\"run_2\""), "{json}");
        assert!(duplicate_keys(&json).is_empty(), "duplicate keys in {json}");
    }

    #[test]
    fn test_several_tests_and_metrics_stay_well_formed() {
        let metrics = vec![
            metric("load", "seconds", Some(Format::Heap), 9.0),
            metric("load", "seconds", Some(Format::Lake), 3.0),
            metric("load", "bytes read", None, 4096.0),
            metric("aggregate", "seconds", Some(Format::Lake), 1.0),
        ];
        let ratios = vec![RatioRecord {
            test: "load".into(),
            metric: "seconds".into(),
            numerator: Format::Heap,
            denominator: Format::Lake,
            numerator_value: 9.0,
            denominator_value: 3.0,
            value: 3.0,
            bound: Some(RatioBound::AtLeast(2.0)),
            passed: Some(true),
        }];
        let json = build_run_json(&metrics, &[], &ratios);
        assert!(duplicate_keys(&json).is_empty(), "duplicate keys in {json}");
        assert_eq!(
            json.matches('{').count(),
            json.matches('}').count(),
            "braces balance in {json}"
        );
        assert!(json.contains("\"bound\": \">= 2.000000\""), "{json}");
    }

    #[test]
    fn test_a_bound_states_which_way_the_claim_runs() {
        assert!(RatioBound::AtMost(1.0).admits(0.8));
        assert!(RatioBound::AtMost(1.0).admits(1.0));
        assert!(!RatioBound::AtMost(1.0).admits(1.01));
        assert!(RatioBound::AtLeast(50.0).admits(51.0));
        assert!(RatioBound::AtLeast(50.0).admits(50.0));
        assert!(!RatioBound::AtLeast(50.0).admits(49.9));
        assert_eq!(RatioBound::AtMost(1.0).comparison(), "<=");
        assert_eq!(RatioBound::AtLeast(1.0).comparison(), ">=");
    }

    #[test]
    fn test_a_broken_denominator_is_reported_rather_than_divided_by() {
        assert_eq!(ratio_value(32.0, 40.0), Some(0.8));
        // A baseline that measured nothing is a broken benchmark, not an
        // infinite speedup
        assert_eq!(ratio_value(1.0, 0.0), None);
        assert_eq!(ratio_value(1.0, -1.0), None);
        assert_eq!(ratio_value(1.0, f64::NAN), None);
        assert_eq!(ratio_value(1.0, f64::INFINITY), None);
        assert_eq!(ratio_value(f64::NAN, 1.0), None);
        assert_eq!(ratio_value(f64::INFINITY, 1.0), None);
        // A zero numerator against a real denominator is a real ratio
        assert_eq!(ratio_value(0.0, 4.0), Some(0.0));
    }
}
