// Analytics function registry. Tracks every analytical function the
// planner can resolve at bind time. Populated once via the builder and
// then read-only for the lifetime of the process.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalyticsFunctionKind {
    // Returns a relation; usable in FROM clause
    TableReturning,
    // Returns a scalar; usable in SELECT projections
    Scalar,
    // Window function; needs OVER clause
    Window,
}

#[derive(Debug, Clone)]
pub struct AnalyticsFunction {
    pub name: &'static str,
    pub kind: AnalyticsFunctionKind,
    pub min_args: usize,
    pub max_args: Option<usize>,
    pub description: &'static str,
    // Output schema for table-returning functions, as (col_name, type_label)
    pub output_schema: &'static [(&'static str, &'static str)],
}

// Maximum supported function-name length in bytes. Lookup folds the input
// to lowercase into a stack buffer of this size to avoid any heap
// allocation on the hot SQL-bind path. All registered names today are
// well under 32 bytes; the buffer leaves comfortable headroom.
const MAX_NAME_LEN: usize = 64;

/// Read-mostly registry of analytics functions. Construction goes through
/// the builder; once built, the inner map is immutable and accessed
/// without any synchronisation overhead beyond an Arc clone.
///
/// Why: registered once at server startup, then queried on every SQL
/// parse/bind that mentions an analytics function. The previous
/// RwLock-backed form took a shared lock on every query and the lookup
/// allocated a String to upper-case the input; both go away here. The
/// public API is preserved.
pub struct AnalyticsRegistry {
    // The map keys are lowercased function names so lookup can compare
    // against an in-place lowercased input without case re-folding the
    // stored side. Lookups never mutate this; the parking_lot Mutex on
    // `extras` is for callers that genuinely want post-startup register()
    // (deprecated path, kept so existing code compiles).
    primary: HashMap<NameKey, AnalyticsFunction>,
    extras: Mutex<HashMap<NameKey, AnalyticsFunction>>,
}

// Owned lowercased name used as the HashMap key. The Borrow<str> impl
// matches String's, which lets `HashMap::get(&str)` find a NameKey entry
// without allocating an owned key for the lookup.
//
// Hash and PartialEq must be byte-for-byte identical to str's, otherwise
// the Borrow contract is violated and lookups would silently miss. We
// derive both, and since the inner String hashes the same as the str it
// borrows from, the contract holds.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct NameKey(String);

impl std::borrow::Borrow<str> for NameKey {
    #[inline]
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl AnalyticsRegistry {
    /// Used by the builder. Most callers should go through `default_registry()`
    /// or call `builder()`.
    pub fn from_entries(entries: Vec<AnalyticsFunction>) -> Arc<Self> {
        let mut primary = HashMap::with_capacity(entries.len());
        for f in entries {
            primary.insert(NameKey(ascii_lowercase_owned(f.name)), f);
        }
        Arc::new(Self {
            primary,
            extras: Mutex::new(HashMap::new()),
        })
    }

    /// Empty registry. Prefer the builder for setting things up.
    pub fn new() -> Arc<Self> {
        Self::from_entries(Vec::new())
    }

    /// Post-startup registration. Goes into a Mutex-protected side map so
    /// the primary stays lock-free; both are consulted on lookup.
    pub fn register(&self, f: AnalyticsFunction) {
        let mut guard = self.extras.lock();
        guard.insert(NameKey(ascii_lowercase_owned(f.name)), f);
    }

    /// Case-insensitive lookup with no heap allocation on the hot path.
    /// The input name is folded to lowercase into a stack buffer and
    /// matched against the lowercased keys in the primary map. Names
    /// longer than `MAX_NAME_LEN` fall back to the heap-allocated path
    /// (no analytics function we ship is anywhere near that long).
    pub fn lookup(&self, name: &str) -> Option<AnalyticsFunction> {
        if name.len() <= MAX_NAME_LEN {
            let mut buf = [0u8; MAX_NAME_LEN];
            let bytes = name.as_bytes();
            for (i, &b) in bytes.iter().enumerate() {
                buf[i] = b.to_ascii_lowercase();
            }
            // SAFETY: only ASCII bytes were copied. ASCII bytes are
            // single-byte UTF-8 codepoints, so lowercasing each input
            // byte (which only touches ASCII letters; non-ASCII bytes are
            // copied verbatim) preserves UTF-8 validity. The total byte
            // length is unchanged. Non-ASCII characters in identifiers
            // hash unchanged, matching SQL's case-insensitivity rules
            // for ASCII letters.
            let lower: &str = unsafe { std::str::from_utf8_unchecked(&buf[..bytes.len()]) };
            if let Some(f) = self.primary.get(lower) {
                return Some(f.clone());
            }
            // Extras path: rare, take the lock briefly
            let extras = self.extras.lock();
            extras.get(lower).cloned()
        } else {
            // Long-name slow path: heap-allocate the lowercased input.
            // No analytics function we ship is anywhere near MAX_NAME_LEN
            // bytes, so this branch is essentially unreachable today.
            let lower = ascii_lowercase_owned(name);
            if let Some(f) = self.primary.get(lower.as_str()) {
                return Some(f.clone());
            }
            self.extras.lock().get(lower.as_str()).cloned()
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .primary
            .keys()
            .map(|k| k.0.to_ascii_uppercase())
            .collect();
        let extras = self.extras.lock();
        for k in extras.keys() {
            v.push(k.0.to_ascii_uppercase());
        }
        v.sort();
        v.dedup();
        v
    }

    pub fn len(&self) -> usize {
        self.primary.len() + self.extras.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[inline]
fn ascii_lowercase_owned(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        out.push(b.to_ascii_lowercase() as char);
    }
    out
}

/// Default analytics catalog. The first call builds the entry list and
/// freezes it into a process-wide `Arc<AnalyticsRegistry>`; every
/// subsequent call is an `Arc::clone` of the cached value (a pointer
/// copy, no map rebuild). The previous form rebuilt the whole HashMap
/// each call, which was hot for the SQL binder and the analytics
/// executor operators that consult it per query.
pub fn default_registry() -> Arc<AnalyticsRegistry> {
    static CACHED: OnceLock<Arc<AnalyticsRegistry>> = OnceLock::new();
    CACHED.get_or_init(build_default_registry).clone()
}

fn build_default_registry() -> Arc<AnalyticsRegistry> {
    let mut entries: Vec<AnalyticsFunction> = Vec::new();

    // Table-returning analytical functions
    entries.push(AnalyticsFunction {
        name: "COHORT_RETENTION",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: None,
        description: "Compute a cohort retention matrix from an event stream",
        output_schema: &[
            ("cohort", "TEXT"),
            ("period", "INT32"),
            ("value", "FLOAT64"),
        ],
    });
    entries.push(AnalyticsFunction {
        name: "FUNNEL_ANALYSIS",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: None,
        description: "Compute funnel step conversion and drop-off rates",
        output_schema: &[
            ("step", "TEXT"),
            ("users_count", "INT64"),
            ("conversion_rate", "FLOAT64"),
            ("drop_off_rate", "FLOAT64"),
            ("avg_time_to_next_ms", "FLOAT64"),
        ],
    });
    entries.push(AnalyticsFunction {
        name: "DATA_PROFILE",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(1),
        description: "Single pass per-column profile of a table",
        output_schema: &[
            ("column_name", "TEXT"),
            ("data_type", "TEXT"),
            ("null_count", "INT64"),
            ("distinct_count", "INT64"),
            ("mean", "FLOAT64"),
            ("median", "FLOAT64"),
            ("stddev", "FLOAT64"),
        ],
    });
    entries.push(AnalyticsFunction {
        name: "COLUMN_PROFILE",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: Some(2),
        description: "Single pass profile for a specific column",
        output_schema: &[("statistic", "TEXT"), ("value", "TEXT")],
    });
    entries.push(AnalyticsFunction {
        name: "CORRELATION_MATRIX",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: None,
        description: "Pairwise Pearson correlation matrix across selected columns",
        output_schema: &[
            ("col_a", "TEXT"),
            ("col_b", "TEXT"),
            ("correlation", "FLOAT64"),
        ],
    });

    // Feature store and lineage
    entries.push(AnalyticsFunction {
        name: "GET_FEATURES",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 3,
        max_args: Some(4),
        description: "Point-in-time correct retrieval of feature values",
        output_schema: &[("entity_key", "TEXT"), ("feature_name", "TEXT"), ("value", "TEXT")],
    });
    entries.push(AnalyticsFunction {
        name: "FEATURE_LINEAGE",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(1),
        description: "Source tables, columns, and dependencies for a qualified feature",
        output_schema: &[
            ("source_table", "TEXT"),
            ("source_column", "TEXT"),
            ("transform", "TEXT"),
            ("dependency", "TEXT"),
            ("last_computed_ms", "INT64"),
        ],
    });
    entries.push(AnalyticsFunction {
        name: "FEATURE_PARITY_CHECK",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: Some(2),
        description: "Compares offline and online feature retrievals for divergence",
        output_schema: &[
            ("entity_key", "TEXT"),
            ("feature_name", "TEXT"),
            ("offline", "TEXT"),
            ("online", "TEXT"),
        ],
    });

    // ML inference
    entries.push(AnalyticsFunction {
        name: "PREDICT",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 2,
        max_args: None,
        description: "Apply a trained model to a row, returns prediction",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "PREDICT_BATCH",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: None,
        description: "Apply a trained model to a query result, returns predictions",
        output_schema: &[("row_idx", "INT64"), ("prediction", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "EXPLAIN_PREDICTION",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: None,
        description: "Per-feature contribution explanation for a single prediction",
        output_schema: &[("feature", "TEXT"), ("contribution", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "MODEL_LINEAGE",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(1),
        description: "Training metadata and feature dependencies for a model",
        output_schema: &[
            ("attribute", "TEXT"),
            ("value", "TEXT"),
        ],
    });

    // Causal inference
    entries.push(AnalyticsFunction {
        name: "PROPENSITY_SCORE",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 2,
        max_args: None,
        description: "Logistic-regression propensity score for treatment given covariates",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "ATE",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 3,
        max_args: None,
        description: "Average Treatment Effect via inverse-propensity weighting",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "ATT",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 3,
        max_args: None,
        description: "Average Treatment Effect on the Treated",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "DIFF_IN_DIFF",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 4,
        max_args: Some(4),
        description: "Difference-in-differences estimator on (outcome, treatment, time, post)",
        output_schema: &[],
    });

    // Predictive analytics
    entries.push(AnalyticsFunction {
        name: "FORECAST",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: None,
        description: "Time series forecast (ES, ARIMA, Holt-Winters, linear trend, decomposition)",
        output_schema: &[("step", "INT64"), ("value", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "ANOMALY_DETECT",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(3),
        description: "Anomaly detection on a series, returns per-row score and flag",
        output_schema: &[("idx", "INT64"), ("is_anomaly", "BOOL"), ("score", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "TREND",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 1,
        max_args: Some(2),
        description: "Linear trend slope and intercept over a series",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "SEASONALITY_DETECT",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(2),
        description: "Detects periodic patterns in a series via autocorrelation peaks",
        output_schema: &[("period", "INT64"), ("strength", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "ACF",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(2),
        description: "Auto-correlation function up to a maximum lag",
        output_schema: &[("lag", "INT64"), ("value", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "PACF",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(2),
        description: "Partial auto-correlation up to a maximum lag",
        output_schema: &[("lag", "INT64"), ("value", "FLOAT64")],
    });
    entries.push(AnalyticsFunction {
        name: "CHANGE_POINTS",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(2),
        description: "CUSUM change point indices",
        output_schema: &[("idx", "INT64")],
    });

    // Drift and quality
    entries.push(AnalyticsFunction {
        name: "PSI",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 2,
        max_args: Some(2),
        description: "Population Stability Index between two histograms",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "KS_TEST",
        kind: AnalyticsFunctionKind::Scalar,
        min_args: 2,
        max_args: Some(2),
        description: "Kolmogorov-Smirnov D statistic between two samples",
        output_schema: &[],
    });
    entries.push(AnalyticsFunction {
        name: "DATE_FEATURES",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 1,
        max_args: Some(1),
        description: "Decomposes a timestamp into year, month, day, dow, hour, weekend, doy, woy, quarter",
        output_schema: &[
            ("year", "INT32"),
            ("month", "INT32"),
            ("day", "INT32"),
            ("dow", "INT32"),
            ("hour", "INT32"),
            ("is_weekend", "BOOL"),
            ("doy", "INT32"),
            ("woy", "INT32"),
            ("quarter", "INT32"),
        ],
    });
    entries.push(AnalyticsFunction {
        name: "POLYNOMIAL_FEATURES",
        kind: AnalyticsFunctionKind::TableReturning,
        min_args: 2,
        max_args: None,
        description: "Polynomial expansion of features up to a given degree",
        output_schema: &[("term", "TEXT"), ("value", "FLOAT64")],
    });

    // Scalar / window functions
    for (name, kind, args, desc) in [
        (
            "YOY",
            AnalyticsFunctionKind::Window,
            2,
            "Year-over-year value",
        ),
        (
            "YOY_GROWTH",
            AnalyticsFunctionKind::Window,
            2,
            "YoY percentage growth",
        ),
        (
            "MOM",
            AnalyticsFunctionKind::Window,
            2,
            "Month-over-month value",
        ),
        (
            "MOM_GROWTH",
            AnalyticsFunctionKind::Window,
            2,
            "MoM percentage growth",
        ),
        (
            "WOW",
            AnalyticsFunctionKind::Window,
            2,
            "Week-over-week value",
        ),
        (
            "QOQ",
            AnalyticsFunctionKind::Window,
            2,
            "Quarter-over-quarter value",
        ),
        (
            "SAME_PERIOD_LAST_YEAR",
            AnalyticsFunctionKind::Window,
            2,
            "Value from the same period one year prior",
        ),
        (
            "PERIOD_COMPARE",
            AnalyticsFunctionKind::Window,
            4,
            "Generic period-over-period comparison",
        ),
        (
            "YTD_SUM",
            AnalyticsFunctionKind::Window,
            2,
            "Year-to-date running sum",
        ),
        (
            "QTD_SUM",
            AnalyticsFunctionKind::Window,
            2,
            "Quarter-to-date running sum",
        ),
        (
            "MTD_SUM",
            AnalyticsFunctionKind::Window,
            2,
            "Month-to-date running sum",
        ),
        (
            "ZSCORE",
            AnalyticsFunctionKind::Window,
            1,
            "Z-score within partition",
        ),
        (
            "IQR_OUTLIER",
            AnalyticsFunctionKind::Window,
            1,
            "Outside 1.5*IQR",
        ),
        (
            "MAD_OUTLIER",
            AnalyticsFunctionKind::Window,
            1,
            "Modified Z-score outlier flag",
        ),
        (
            "CORR",
            AnalyticsFunctionKind::Scalar,
            2,
            "Pearson correlation",
        ),
        (
            "SPEARMAN_CORR",
            AnalyticsFunctionKind::Scalar,
            2,
            "Spearman rank correlation",
        ),
        (
            "KENDALL_TAU",
            AnalyticsFunctionKind::Scalar,
            2,
            "Kendall tau",
        ),
        (
            "MUTUAL_INFORMATION",
            AnalyticsFunctionKind::Scalar,
            2,
            "Mutual information (binned)",
        ),
        (
            "GROUPING",
            AnalyticsFunctionKind::Scalar,
            1,
            "Grouping bit indicator",
        ),
        (
            "GROUPING_ID",
            AnalyticsFunctionKind::Scalar,
            1,
            "Grouping bitmask",
        ),
    ] {
        entries.push(AnalyticsFunction {
            name,
            kind,
            min_args: args,
            max_args: None,
            description: desc,
            output_schema: &[],
        });
    }

    AnalyticsRegistry::from_entries(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_contains_required_functions() {
        let r = default_registry();
        for name in [
            "COHORT_RETENTION",
            "FUNNEL_ANALYSIS",
            "DATA_PROFILE",
            "CORRELATION_MATRIX",
            "YOY",
            "MOM_GROWTH",
            "ZSCORE",
            "CORR",
            "GROUPING",
        ] {
            assert!(r.lookup(name).is_some(), "missing function {}", name);
        }
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let r = default_registry();
        assert!(r.lookup("yoy").is_some());
        assert!(r.lookup("YoY").is_some());
    }
}
