//! Pipeline engine with DAG validation, execution ordering, and preview mode.

use crate::ids::PipelineId;
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

/// How a pipeline stage refreshes its target table from its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshMode {
    Full,
    Incremental,
    AppendOnly,
    Merge,
}

/// Current execution status of a pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStatus {
    Idle,
    Running,
    Failed(String),
    Completed,
}

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineStageConfig {
    pub name: String,
    pub source: String,
    pub target: String,
    pub refresh_mode: RefreshMode,
    pub transform_sql: Option<String>,
    pub quality_checks: Vec<String>,
}

/// SLA constraints for pipeline execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineSla {
    pub max_duration_ms: Option<u64>,
    pub max_staleness_ms: Option<u64>,
}

/// A declarative data pipeline with ordered stages.
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub id: PipelineId,
    pub name: String,
    pub stages: Vec<PipelineStageConfig>,
    pub enabled: bool,
    pub created_at: i64,
    pub sla: Option<PipelineSla>,
}

/// Runtime state for a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineState {
    pub last_run_at: Option<i64>,
    pub last_success_at: Option<i64>,
    pub rows_processed: u64,
    pub run_duration_ms: u64,
    pub status: PipelineStatus,
}

impl PipelineState {
    pub fn new() -> Self {
        Self {
            last_run_at: None,
            last_success_at: None,
            rows_processed: 0,
            run_duration_ms: 0,
            status: PipelineStatus::Idle,
        }
    }
}

/// Result of a pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineRunResult {
    pub pipeline_name: String,
    pub stages_executed: u32,
    pub total_rows: u64,
    pub duration_ms: u64,
    pub quality_results: Vec<String>,
}

/// Manages pipeline lifecycle: creation, validation, and execution ordering.
pub struct PipelineManager {
    pipelines: scc::HashMap<String, Arc<Pipeline>>,
    states: scc::HashMap<String, PipelineState>,
}

impl PipelineManager {
    pub fn new() -> Self {
        Self {
            pipelines: scc::HashMap::new(),
            states: scc::HashMap::new(),
        }
    }

    /// Register a new pipeline after validating its DAG.
    pub fn create_pipeline(&self, pipeline: Pipeline) -> Result<()> {
        // Validate DAG before inserting
        validate_dag(&pipeline.stages)?;

        let name = pipeline.name.clone();
        let arc = Arc::new(pipeline);
        if self.pipelines.insert_sync(name.clone(), arc).is_err() {
            return Err(ZyronError::PipelineAlreadyExists(name));
        }
        let _ = self.states.insert_sync(name, PipelineState::new());
        Ok(())
    }

    /// Remove a pipeline by name.
    pub fn drop_pipeline(&self, name: &str) -> Result<()> {
        if self.pipelines.remove_sync(name).is_none() {
            return Err(ZyronError::PipelineNotFound(name.to_string()));
        }
        let _ = self.states.remove_sync(name);
        Ok(())
    }

    /// Enable a pipeline for scheduled execution.
    pub fn enable_pipeline(&self, name: &str) -> Result<()> {
        let updated = self.pipelines.update_sync(name, |_k, v| {
            let mut p = Pipeline::clone(v);
            p.enabled = true;
            *v = Arc::new(p);
        });
        if updated.is_none() {
            return Err(ZyronError::PipelineNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Disable a pipeline to prevent scheduled execution.
    pub fn disable_pipeline(&self, name: &str) -> Result<()> {
        let updated = self.pipelines.update_sync(name, |_k, v| {
            let mut p = Pipeline::clone(v);
            p.enabled = false;
            *v = Arc::new(p);
        });
        if updated.is_none() {
            return Err(ZyronError::PipelineNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Get a pipeline by name.
    pub fn get_pipeline(&self, name: &str) -> Option<Arc<Pipeline>> {
        self.pipelines.read_sync(name, |_k, v| Arc::clone(v))
    }

    /// List all registered pipelines.
    pub fn list_pipelines(&self) -> Vec<Arc<Pipeline>> {
        let mut result = Vec::new();
        self.pipelines.iter_sync(|_k, v| {
            result.push(Arc::clone(v));
            true
        });
        result
    }

    /// Get runtime state for a pipeline.
    pub fn get_state(&self, name: &str) -> Option<PipelineState> {
        self.states.read_sync(name, |_k, v| v.clone())
    }

    /// Get the topological execution order for a pipeline's stages.
    pub fn execution_order(&self, name: &str) -> Result<Vec<usize>> {
        let pipeline = self
            .get_pipeline(name)
            .ok_or_else(|| ZyronError::PipelineNotFound(name.to_string()))?;
        validate_dag(&pipeline.stages)
    }

    /// Return the number of registered pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }
}

/// Validate that pipeline stages form a DAG (no circular dependencies).
/// Returns topological execution order on success.
pub fn validate_dag(stages: &[PipelineStageConfig]) -> Result<Vec<usize>> {
    if stages.is_empty() {
        return Ok(Vec::new());
    }

    // Build adjacency list using stage indices.
    // A stage's target may be another stage's source, creating an edge.
    let n = stages.len();

    // Map target names to stage indices that produce them
    let mut target_to_stage: hashbrown::HashMap<&str, usize> = hashbrown::HashMap::new();
    for (i, stage) in stages.iter().enumerate() {
        target_to_stage.insert(stage.target.as_str(), i);
    }

    // Build in-degree and adjacency
    let mut in_degree = vec![0u32; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, stage) in stages.iter().enumerate() {
        // If this stage's source is another stage's target, add dependency edge
        if let Some(&producer) = target_to_stage.get(stage.source.as_str()) {
            if producer != i {
                adj[producer].push(i);
                in_degree[i] += 1;
            }
        }
    }

    // Kahn's algorithm for topological sort
    let mut queue: Vec<usize> = Vec::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push(i);
        }
    }

    let mut order = Vec::with_capacity(n);
    let mut head = 0;

    while head < queue.len() {
        let node = queue[head];
        head += 1;
        order.push(node);

        for &next in &adj[node] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push(next);
            }
        }
    }

    if order.len() != n {
        // Find stages involved in cycle for error message
        let cycle_stages: Vec<String> = stages
            .iter()
            .enumerate()
            .filter(|(i, _)| in_degree[*i] > 0)
            .map(|(_, s)| s.name.clone())
            .collect();
        return Err(ZyronError::CircularPipelineDependency(
            cycle_stages.join(", "),
        ));
    }

    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::PipelineId;

    fn stage(name: &str, source: &str, target: &str) -> PipelineStageConfig {
        PipelineStageConfig {
            name: name.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            refresh_mode: RefreshMode::Full,
            transform_sql: None,
            quality_checks: Vec::new(),
        }
    }

    #[test]
    fn test_dag_linear() {
        let stages = vec![
            stage("bronze", "raw_data", "bronze_table"),
            stage("silver", "bronze_table", "silver_table"),
            stage("gold", "silver_table", "gold_table"),
        ];
        let order = validate_dag(&stages).expect("linear DAG should be valid");
        assert_eq!(order.len(), 3);
        // bronze must come before silver, silver before gold
        let pos_bronze = order.iter().position(|&x| x == 0).expect("bronze");
        let pos_silver = order.iter().position(|&x| x == 1).expect("silver");
        let pos_gold = order.iter().position(|&x| x == 2).expect("gold");
        assert!(pos_bronze < pos_silver);
        assert!(pos_silver < pos_gold);
    }

    #[test]
    fn test_dag_diamond() {
        // source -> A -> C, source -> B -> C
        let stages = vec![
            stage("a", "source", "a_out"),
            stage("b", "source", "b_out"),
            stage("c_from_a", "a_out", "c_out"),
            stage("c_from_b", "b_out", "c_out"),
        ];
        let order = validate_dag(&stages).expect("diamond DAG should be valid");
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_dag_cycle_detected() {
        let stages = vec![
            stage("a", "c_out", "a_out"),
            stage("b", "a_out", "b_out"),
            stage("c", "b_out", "c_out"),
        ];
        let result = validate_dag(&stages);
        assert!(result.is_err());
        match result {
            Err(ZyronError::CircularPipelineDependency(_)) => {}
            other => panic!("Expected CircularPipelineDependency, got {:?}", other),
        }
    }

    #[test]
    fn test_dag_empty() {
        let order = validate_dag(&[]).expect("empty DAG should be valid");
        assert!(order.is_empty());
    }

    #[test]
    fn test_dag_independent_stages() {
        let stages = vec![
            stage("a", "src_a", "out_a"),
            stage("b", "src_b", "out_b"),
            stage("c", "src_c", "out_c"),
        ];
        let order = validate_dag(&stages).expect("independent stages should be valid");
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_pipeline_manager_crud() {
        let mgr = PipelineManager::new();
        let pipeline = Pipeline {
            id: PipelineId(1),
            name: "test_pipeline".to_string(),
            stages: vec![stage("s1", "src", "dst")],
            enabled: true,
            created_at: 1000,
            sla: None,
        };
        mgr.create_pipeline(pipeline).expect("create");
        assert_eq!(mgr.pipeline_count(), 1);

        let p = mgr.get_pipeline("test_pipeline").expect("should exist");
        assert!(p.enabled);

        mgr.disable_pipeline("test_pipeline").expect("disable");
        let p = mgr.get_pipeline("test_pipeline").expect("should exist");
        assert!(!p.enabled);

        mgr.drop_pipeline("test_pipeline").expect("drop");
        assert_eq!(mgr.pipeline_count(), 0);
    }

    #[test]
    fn test_pipeline_duplicate_rejected() {
        let mgr = PipelineManager::new();
        let pipeline = Pipeline {
            id: PipelineId(1),
            name: "dup".to_string(),
            stages: vec![stage("s1", "src", "dst")],
            enabled: true,
            created_at: 1000,
            sla: None,
        };
        mgr.create_pipeline(pipeline.clone()).expect("first create");
        let result = mgr.create_pipeline(pipeline);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_not_found() {
        let mgr = PipelineManager::new();
        let result = mgr.drop_pipeline("nonexistent");
        assert!(matches!(result, Err(ZyronError::PipelineNotFound(_))));
    }

    #[test]
    fn test_pipeline_state() {
        let mgr = PipelineManager::new();
        let pipeline = Pipeline {
            id: PipelineId(1),
            name: "stateful".to_string(),
            stages: vec![stage("s1", "src", "dst")],
            enabled: true,
            created_at: 1000,
            sla: None,
        };
        mgr.create_pipeline(pipeline).expect("create");
        let state = mgr.get_state("stateful").expect("state should exist");
        assert_eq!(state.status, PipelineStatus::Idle);
        assert_eq!(state.rows_processed, 0);
    }
}
