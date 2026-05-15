// ZyronDB analytics engine
// ROLLUP/CUBE/GROUPING SETS, period-over-period comparisons, cohort and
// funnel analysis, single-pass data profiling, outlier detection,
// correlation statistics, feature store, in-database ML, causal
// inference, time series forecasting, anomaly detection

pub mod causal;
pub mod cohort;
pub mod correlation;
pub mod featureLineage;
pub mod featureStore;
pub mod fft;
pub mod funnel;
pub mod grouping;
pub mod ml;
pub mod mlInference;
pub mod numeric;
pub mod outlier;
pub mod period_compare;
pub mod predictive;
pub mod profiling;
pub mod registry;
pub mod value;

pub use causal::{CausalEstimate, ate, ateWithCi, att, diffInDiff, propensityScore};
pub use cohort::{
    CohortAnalyser, CohortAnalysis, CohortDefinition, CohortEvent, CohortMetric, CohortPeriod,
    CohortResult, CohortRow, CohortType, retention_analysis,
};
pub use cohort::retention_analysis as retentionAnalysis;
pub use correlation::{
    CorrelationMatrix, CorrelationMatrixBuilder, KendallAggregator, MutualInformationEstimator,
    PearsonAggregator, SpearmanAggregator, correlation_matrix, kendall_tau, mutual_information,
    pearson_corr, spearman_corr,
};
pub use correlation::correlation_matrix as correlationMatrix;
pub use correlation::kendall_tau as kendallTau;
pub use correlation::mutual_information as mutualInformation;
pub use correlation::pearson_corr as pearsonCorr;
pub use correlation::spearman_corr as spearmanCorr;
pub use featureLineage::{FeatureLineageRegistry, LineageEntry, extractTablesAndColumns};
pub use featureStore::{
    FeatureDefinition, FeatureFrame, FeatureGroup, FeatureRow, FeatureStore, FeatureValue,
    ParityCheckResult, featureLineageRegistry, featureParityCheck, featureStore,
};
pub use funnel::{
    FunnelAnalyser, FunnelConfig, FunnelEvent, FunnelResult, FunnelStep, StepResult,
    funnel_analysis,
};
pub use funnel::funnel_analysis as funnelAnalysis;
pub use grouping::{
    GroupingSetExpander, GroupingSetType, GroupingSetsRunner, RowKey, expand_grouping_sets,
    grouping_bit, grouping_id_bits,
};
pub use grouping::expand_grouping_sets as expandGroupingSets;
pub use grouping::grouping_bit as groupingBit;
pub use grouping::grouping_id_bits as groupingIdBits;
pub use ml::{
    Hyperparameters, ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TrainingData,
    TreeNode,
};
pub use mlInference::{InferenceHandle, ModelCache, modelCache, predictBatch, predictOne};
pub use numeric::{
    BloomFilter, KahanSum, OnlineCovariance, OnlineMoments, OnlineQuantile, bootstrapCi,
    choleskySolve, columnStandardize,
};
pub use outlier::{
    IsolationForest, MadDetector, OutlierDecision, ZScoreEvaluator, iqr_outlier, mad_outlier,
    zscore,
};
pub use outlier::iqr_outlier as iqrOutlier;
pub use outlier::mad_outlier as madOutlier;
pub use outlier::zscore as zScore;
pub use period_compare::{
    PeriodUnit, mom, mom_growth, mtd_sum, period_compare_value, qoq, qtd_sum,
    same_period_last_year, wow, yoy, yoy_growth, ytd_sum,
};
pub use period_compare::mom_growth as momGrowth;
pub use period_compare::mtd_sum as mtdSum;
pub use period_compare::period_compare_value as periodCompareValue;
pub use period_compare::qtd_sum as qtdSum;
pub use period_compare::same_period_last_year as samePeriodLastYear;
pub use period_compare::yoy_growth as yoyGrowth;
pub use period_compare::ytd_sum as ytdSum;
pub use predictive::{
    AnomalyMethod, ForecastMethod, ForecastPoint, SeasonalComponent, acf, anomalyDetect, arima,
    changePoints, exponentialSmoothing, forecast, holtWinters, linearTrendForecast, pacf,
    seasonalDecompose, seasonalityDetect, trend,
};
pub use profiling::{
    ColumnProfile, HistogramBin, NumericRange, PatternFrequency, PercentileSet, TableProfile,
    column_profile, profile_table,
};
pub use profiling::column_profile as columnProfile;
pub use profiling::profile_table as profileTable;
pub use registry::{
    AnalyticsFunction, AnalyticsFunctionKind, AnalyticsRegistry, default_registry,
};
pub use registry::default_registry as defaultRegistry;
pub use value::{AnalyticsRow, AnalyticsValue, MS_PER_DAY, MS_PER_HOUR};
