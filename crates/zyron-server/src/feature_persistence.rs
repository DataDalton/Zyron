#![allow(non_snake_case)]
// Feature store and model on-disk persistence
// Snapshot files survive restarts so registered feature groups and
// trained models are recovered into the in-memory FeatureStore and ModelCache

use std::path::PathBuf;
use std::sync::Arc;
use zyron_analytics::featureLineage::{LineageEntry, extractTablesAndColumns};
use zyron_analytics::{FeatureGroup, TrainedModel};
use zyron_common::error::{Result, ZyronError};
use zyron_wire::connection::ServerState;

fn featureSnapshotPath(state: &Arc<ServerState>) -> PathBuf {
    state.data_dir.join("feature_store.json")
}

fn modelsDirPath(state: &Arc<ServerState>) -> PathBuf {
    state.data_dir.join("models")
}

/// Restores feature groups and trained models from the data directory
/// Called once at server startup, after ServerState is built
pub fn restore_feature_state(state: &Arc<ServerState>) -> Result<()> {
    restoreFeatureGroups(state)?;
    restoreModels(state)?;
    Ok(())
}

fn restoreFeatureGroups(state: &Arc<ServerState>) -> Result<()> {
    let path = featureSnapshotPath(state);
    if !path.exists() {
        return Ok(());
    }
    let bytes = std::fs::read(&path).map_err(ZyronError::from)?;
    let groups: Vec<FeatureGroup> = serde_json::from_slice(&bytes).map_err(|e| {
        ZyronError::ExecutionError(format!("feature snapshot decode: {}", e))
    })?;
    let mut lineageGuard = state.feature_lineage.write();
    for g in groups {
        let groupName = g.name.clone();
        for fd in &g.features {
            let qualifiedName = format!("{}.{}", groupName, fd.name);
            let (tables, cols) = extractTablesAndColumns(&fd.transformExpr);
            let mut entry = LineageEntry::new(qualifiedName.clone());
            entry.sourceTables = tables;
            entry.sourceColumns = cols;
            entry.transformChain = vec![fd.transformExpr.clone()];
            entry.lastComputedMs = g.lastRefreshMs;
            lineageGuard.register(qualifiedName, entry);
        }
        if let Err(e) = state.feature_store.registerFeatureGroup(g) {
            tracing::warn!(target: "zyron::server", "feature group restore: {}", e);
        }
    }
    Ok(())
}

fn restoreModels(state: &Arc<ServerState>) -> Result<()> {
    let dir = modelsDirPath(state);
    if !dir.exists() {
        return Ok(());
    }
    let entries = std::fs::read_dir(&dir).map_err(ZyronError::from)?;
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(target: "zyron::server", "model dir read: {}", e);
                continue;
            }
        };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<TrainedModel>(&bytes) {
                Ok(model) => {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    state.model_cache.install(name, model);
                }
                Err(e) => {
                    tracing::warn!(
                        target: "zyron::server",
                        "model decode failed for {}: {}",
                        path.display(),
                        e
                    );
                }
            },
            Err(e) => {
                tracing::warn!(
                    target: "zyron::server",
                    "model read failed for {}: {}",
                    path.display(),
                    e
                );
            }
        }
    }
    Ok(())
}
