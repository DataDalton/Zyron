//! Data quality check evaluation for pipeline stages.

/// Severity level for a quality check failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualitySeverity {
    Warn,
    Error,
    Fatal,
}

/// Action to take when a quality check fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnFailure {
    Continue,
    Abort,
    Quarantine,
}

/// Built-in and custom quality check types.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityCheckType {
    NotNull {
        column: String,
    },
    Unique {
        columns: Vec<String>,
    },
    Range {
        column: String,
        min: Option<f64>,
        max: Option<f64>,
    },
    Pattern {
        column: String,
        regex: String,
    },
    Referential {
        column: String,
        ref_table: String,
        ref_column: String,
    },
    Freshness {
        column: String,
        max_age_secs: u64,
    },
    RowCount {
        min: Option<u64>,
        max: Option<u64>,
    },
    DistributionStable {
        column: String,
        window_secs: u64,
    },
    Custom {
        expression: String,
    },
}

/// A named quality check with severity and failure action.
#[derive(Debug, Clone)]
pub struct QualityCheck {
    pub name: String,
    pub check_type: QualityCheckType,
    pub severity: QualitySeverity,
    pub on_failure: OnFailure,
}

/// Result of evaluating a single quality check.
#[derive(Debug, Clone)]
pub struct QualityResult {
    pub check_name: String,
    pub passed: bool,
    pub failing_rows: u64,
    pub total_rows: u64,
    pub failure_percentage: f64,
    pub details: Option<String>,
}

impl QualityResult {
    /// Create a passing result.
    pub fn pass(name: &str, total_rows: u64) -> Self {
        Self {
            check_name: name.to_string(),
            passed: true,
            failing_rows: 0,
            total_rows,
            failure_percentage: 0.0,
            details: None,
        }
    }

    /// Create a failing result.
    pub fn fail(name: &str, failing_rows: u64, total_rows: u64, details: Option<String>) -> Self {
        let pct = if total_rows > 0 {
            (failing_rows as f64 / total_rows as f64) * 100.0
        } else {
            0.0
        };
        Self {
            check_name: name.to_string(),
            passed: false,
            failing_rows,
            total_rows,
            failure_percentage: pct,
            details,
        }
    }
}

/// Check if any Fatal quality check with Abort action has failed.
pub fn should_abort(results: &[QualityResult], checks: &[QualityCheck]) -> bool {
    for (result, check) in results.iter().zip(checks.iter()) {
        if !result.passed
            && check.severity == QualitySeverity::Fatal
            && check.on_failure == OnFailure::Abort
        {
            return true;
        }
    }
    false
}

/// Count how many checks failed at or above the given severity.
pub fn failure_count(results: &[QualityResult], min_severity: QualitySeverity) -> usize {
    results
        .iter()
        .zip(std::iter::repeat(min_severity))
        .filter(|(r, _sev)| !r.passed)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_result_pass() {
        let r = QualityResult::pass("not_null_check", 1000);
        assert!(r.passed);
        assert_eq!(r.failing_rows, 0);
        assert_eq!(r.failure_percentage, 0.0);
    }

    #[test]
    fn test_quality_result_fail() {
        let r = QualityResult::fail(
            "unique_check",
            50,
            1000,
            Some("duplicates found".to_string()),
        );
        assert!(!r.passed);
        assert_eq!(r.failing_rows, 50);
        assert!((r.failure_percentage - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_should_abort_fatal_abort() {
        let checks = vec![QualityCheck {
            name: "fatal_check".to_string(),
            check_type: QualityCheckType::NotNull {
                column: "id".to_string(),
            },
            severity: QualitySeverity::Fatal,
            on_failure: OnFailure::Abort,
        }];
        let results = vec![QualityResult::fail("fatal_check", 10, 100, None)];
        assert!(should_abort(&results, &checks));
    }

    #[test]
    fn test_should_not_abort_warn() {
        let checks = vec![QualityCheck {
            name: "warn_check".to_string(),
            check_type: QualityCheckType::NotNull {
                column: "name".to_string(),
            },
            severity: QualitySeverity::Warn,
            on_failure: OnFailure::Continue,
        }];
        let results = vec![QualityResult::fail("warn_check", 5, 100, None)];
        assert!(!should_abort(&results, &checks));
    }

    #[test]
    fn test_should_not_abort_fatal_continue() {
        let checks = vec![QualityCheck {
            name: "fatal_continue".to_string(),
            check_type: QualityCheckType::RowCount {
                min: Some(100),
                max: None,
            },
            severity: QualitySeverity::Fatal,
            on_failure: OnFailure::Continue,
        }];
        let results = vec![QualityResult::fail("fatal_continue", 0, 50, None)];
        assert!(!should_abort(&results, &checks));
    }

    #[test]
    fn test_should_not_abort_all_pass() {
        let checks = vec![QualityCheck {
            name: "ok_check".to_string(),
            check_type: QualityCheckType::NotNull {
                column: "id".to_string(),
            },
            severity: QualitySeverity::Fatal,
            on_failure: OnFailure::Abort,
        }];
        let results = vec![QualityResult::pass("ok_check", 1000)];
        assert!(!should_abort(&results, &checks));
    }

    #[test]
    fn test_quality_check_types() {
        let _range = QualityCheckType::Range {
            column: "price".to_string(),
            min: Some(0.0),
            max: Some(10000.0),
        };
        let _pattern = QualityCheckType::Pattern {
            column: "email".to_string(),
            regex: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
        };
        let _custom = QualityCheckType::Custom {
            expression: "amount > 0 AND status IN ('active', 'pending')".to_string(),
        };
        let _drift = QualityCheckType::DistributionStable {
            column: "revenue".to_string(),
            window_secs: 7 * 86400,
        };
    }
}
