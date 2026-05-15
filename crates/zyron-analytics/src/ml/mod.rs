#![allow(non_snake_case)]
// In-database ML, types and shared structures
// Each algorithm has its own module under ml/
// Models serialize through TrainedModel for catalog persistence

pub mod annKnn;
pub mod decisionTree;
pub mod evaluation;
pub mod f64Kernels;
pub mod gradientBoosting;
pub mod kmeans;
pub mod knn;
pub mod linearRegression;
pub mod logisticRegression;
pub mod randomForest;
pub mod transforms;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Algorithm tag stored alongside the trained artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    LinearRegression,
    LogisticRegression,
    DecisionTreeRegression,
    DecisionTreeClassification,
    RandomForestRegression,
    RandomForestClassification,
    GradientBoostingRegression,
    GradientBoostingClassification,
    KMeans,
    KnnRegression,
    KnnClassification,
}

impl ModelType {
    pub fn fromStr(s: &str) -> Option<Self> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "linear_regression" | "linear" => Some(Self::LinearRegression),
            "logistic_regression" | "logistic" => Some(Self::LogisticRegression),
            "decision_tree_regression" | "tree_regression" => Some(Self::DecisionTreeRegression),
            "decision_tree" | "decision_tree_classification" | "tree" => {
                Some(Self::DecisionTreeClassification)
            }
            "random_forest_regression" => Some(Self::RandomForestRegression),
            "random_forest" | "random_forest_classification" => {
                Some(Self::RandomForestClassification)
            }
            "gradient_boosting_regression" | "gbm_regression" => {
                Some(Self::GradientBoostingRegression)
            }
            "gradient_boosting" | "gbm" => Some(Self::GradientBoostingClassification),
            "kmeans" | "k_means" => Some(Self::KMeans),
            "knn_regression" => Some(Self::KnnRegression),
            "knn" | "knn_classification" => Some(Self::KnnClassification),
            _ => None,
        }
    }

    pub fn isClassification(self) -> bool {
        matches!(
            self,
            ModelType::LogisticRegression
                | ModelType::DecisionTreeClassification
                | ModelType::RandomForestClassification
                | ModelType::GradientBoostingClassification
                | ModelType::KnnClassification
        )
    }

    pub fn isRegression(self) -> bool {
        matches!(
            self,
            ModelType::LinearRegression
                | ModelType::DecisionTreeRegression
                | ModelType::RandomForestRegression
                | ModelType::GradientBoostingRegression
                | ModelType::KnnRegression
        )
    }
}

/// One node of a binary decision tree
/// 32 bytes, cache-line friendly
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct TreeNode {
    pub featureIdx: i32,
    pub threshold: f64,
    pub left: i32,
    pub right: i32,
    pub leafValue: f64,
}

impl TreeNode {
    pub fn leaf(value: f64) -> Self {
        Self {
            featureIdx: -1,
            threshold: 0.0,
            left: -1,
            right: -1,
            leafValue: value,
        }
    }

    pub fn isLeaf(&self) -> bool {
        self.featureIdx < 0
    }
}

/// Auxiliary structure for non-linear models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelData {
    None,
    Tree {
        nodes: Vec<TreeNode>,
    },
    Forest {
        trees: Vec<Vec<TreeNode>>,
    },
    BoostedTrees {
        baseScore: f64,
        learningRate: f64,
        trees: Vec<Vec<TreeNode>>,
    },
    KMeans {
        k: usize,
        nFeatures: usize,
    },
    Knn {
        x: Vec<f64>,
        y: Vec<f64>,
        k: usize,
        nFeatures: usize,
    },
}

/// Hyperparameter container, all values stored as f64 for uniform handling
/// Boolean flags are encoded as 0.0 / 1.0
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub values: HashMap<String, f64>,
    pub stringValues: HashMap<String, String>,
}

impl Hyperparameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn setF64(&mut self, key: &str, value: f64) {
        self.values.insert(key.to_string(), value);
    }

    pub fn setStr(&mut self, key: &str, value: &str) {
        self.stringValues.insert(key.to_string(), value.to_string());
    }

    pub fn getF64(&self, key: &str) -> Option<f64> {
        self.values.get(key).copied()
    }

    pub fn getF64Or(&self, key: &str, default: f64) -> f64 {
        self.values.get(key).copied().unwrap_or(default)
    }

    pub fn getUsizeOr(&self, key: &str, default: usize) -> usize {
        self.values
            .get(key)
            .map(|v| v.max(0.0) as usize)
            .unwrap_or(default)
    }

    pub fn getU64Or(&self, key: &str, default: u64) -> u64 {
        self.values
            .get(key)
            .map(|v| v.max(0.0) as u64)
            .unwrap_or(default)
    }

    pub fn getBoolOr(&self, key: &str, default: bool) -> bool {
        self.values.get(key).map(|v| *v != 0.0).unwrap_or(default)
    }

    pub fn getStr(&self, key: &str) -> Option<&str> {
        self.stringValues.get(key).map(|s| s.as_str())
    }
}

/// Configuration handed to a trainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub modelType: ModelType,
    pub featureColumns: Vec<String>,
    pub targetColumn: Option<String>,
    pub hyperparameters: Hyperparameters,
}

impl ModelConfig {
    pub fn new(modelType: ModelType, featureColumns: Vec<String>) -> Self {
        Self {
            modelType,
            featureColumns,
            targetColumn: None,
            hyperparameters: Hyperparameters::new(),
        }
    }
}

/// Final trained model, persisted to the catalog and loaded into the
/// inference cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedModel {
    pub modelId: String,
    pub modelType: ModelType,
    pub featureColumns: Vec<String>,
    pub targetColumn: Option<String>,
    pub featureMean: Vec<f64>,
    pub featureStd: Vec<f64>,
    pub weights: Vec<f64>,
    pub data: ModelData,
    pub metrics: HashMap<String, f64>,
    pub hyperparameters: Hyperparameters,
    pub createdAtMs: i64,
    pub trainingRows: u64,
    pub quantized: bool,
}

impl TrainedModel {
    pub fn new(modelId: String, modelType: ModelType) -> Self {
        Self {
            modelId,
            modelType,
            featureColumns: Vec::new(),
            targetColumn: None,
            featureMean: Vec::new(),
            featureStd: Vec::new(),
            weights: Vec::new(),
            data: ModelData::None,
            metrics: HashMap::new(),
            hyperparameters: Hyperparameters::new(),
            createdAtMs: 0,
            trainingRows: 0,
            quantized: false,
        }
    }
}

/// Standard evaluation outputs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    // Regression metrics
    pub rmse: Option<f64>,
    pub mae: Option<f64>,
    pub mape: Option<f64>,
    pub rSquared: Option<f64>,
    // Classification metrics
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1Score: Option<f64>,
    pub logLoss: Option<f64>,
    // Clustering metrics
    pub inertia: Option<f64>,
    pub silhouette: Option<f64>,
}

impl ModelMetrics {
    pub fn intoMap(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        if let Some(v) = self.rmse {
            m.insert("rmse".to_string(), v);
        }
        if let Some(v) = self.mae {
            m.insert("mae".to_string(), v);
        }
        if let Some(v) = self.mape {
            m.insert("mape".to_string(), v);
        }
        if let Some(v) = self.rSquared {
            m.insert("r_squared".to_string(), v);
        }
        if let Some(v) = self.accuracy {
            m.insert("accuracy".to_string(), v);
        }
        if let Some(v) = self.precision {
            m.insert("precision".to_string(), v);
        }
        if let Some(v) = self.recall {
            m.insert("recall".to_string(), v);
        }
        if let Some(v) = self.f1Score {
            m.insert("f1_score".to_string(), v);
        }
        if let Some(v) = self.logLoss {
            m.insert("log_loss".to_string(), v);
        }
        if let Some(v) = self.inertia {
            m.insert("inertia".to_string(), v);
        }
        if let Some(v) = self.silhouette {
            m.insert("silhouette".to_string(), v);
        }
        m
    }
}

/// Layout of training matrix passed to learners
/// Row-major (n, p), labels separate
pub struct TrainingData<'a> {
    pub xs: &'a [f64],
    pub ys: &'a [f64],
    pub n: usize,
    pub p: usize,
}

impl<'a> TrainingData<'a> {
    pub fn new(xs: &'a [f64], ys: &'a [f64], n: usize, p: usize) -> Self {
        debug_assert_eq!(xs.len(), n * p);
        debug_assert_eq!(ys.len(), n);
        Self { xs, ys, n, p }
    }

    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.p;
        &self.xs[start..start + self.p]
    }
}
