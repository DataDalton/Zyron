// ZyronDB analytics engine
// Phase 16: ROLLUP/CUBE/GROUPING SETS, period-over-period comparisons,
// cohort and funnel analysis, single-pass data profiling, outlier detection,
// and correlation statistics

pub mod cohort;
pub mod correlation;
pub mod funnel;
pub mod grouping;
pub mod outlier;
pub mod period_compare;
pub mod profiling;
pub mod registry;
pub mod value;

pub use cohort::{
    CohortAnalyser, CohortAnalysis, CohortDefinition, CohortEvent, CohortMetric, CohortPeriod,
    CohortResult, CohortRow, CohortType, retention_analysis,
};
pub use correlation::{
    CorrelationMatrix, CorrelationMatrixBuilder, KendallAggregator, MutualInformationEstimator,
    PearsonAggregator, SpearmanAggregator, correlation_matrix, kendall_tau, mutual_information,
    pearson_corr, spearman_corr,
};
pub use funnel::{
    FunnelAnalyser, FunnelConfig, FunnelEvent, FunnelResult, FunnelStep, StepResult,
    funnel_analysis,
};
pub use grouping::{
    GroupingSetExpander, GroupingSetType, GroupingSetsRunner, RowKey, expand_grouping_sets,
    grouping_bit, grouping_id_bits,
};
pub use outlier::{
    IsolationForest, MadDetector, OutlierDecision, ZScoreEvaluator, iqr_outlier, mad_outlier,
    zscore,
};
pub use period_compare::{
    PeriodUnit, mom, mom_growth, mtd_sum, period_compare_value, qoq, qtd_sum,
    same_period_last_year, wow, yoy, yoy_growth, ytd_sum,
};
pub use profiling::{
    ColumnProfile, HistogramBin, NumericRange, PatternFrequency, PercentileSet, TableProfile,
    column_profile, profile_table,
};
pub use registry::{
    AnalyticsFunction, AnalyticsFunctionKind, AnalyticsRegistry, default_registry,
};
pub use value::{AnalyticsRow, AnalyticsValue, MS_PER_DAY, MS_PER_HOUR};
