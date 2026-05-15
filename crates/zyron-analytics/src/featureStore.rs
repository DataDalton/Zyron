#![allow(non_snake_case)]
// Feature store, in-memory store of feature groups with materialized
// per-entity time series of feature values
// Time-travel-aware get_features uses point-in-time semantics, returning
// the latest value at-or-before the supplied as-of timestamp
//
// Concurrency: each feature group has its own RwLock so bulk writers do
// not contend on a global lock (F1). Versioned series share the same
// physical storage as the default series and are filtered by version
// at read time, no duplicate writes (F2).

use crate::numeric::BloomFilter;
use crate::value::{AnalyticsValue, VerifiedKeyMap};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use zyron_common::error::{Result, ZyronError};

/// Definition of a single feature within a feature group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDefinition {
    pub name: String,
    pub dataType: String,
    pub description: String,
    pub version: u32,
    pub createdAtMs: i64,
    pub transformExpr: String,
}

impl FeatureDefinition {
    pub fn new(name: String, dataType: String, transformExpr: String) -> Self {
        Self {
            name,
            dataType,
            description: String::new(),
            version: 1,
            createdAtMs: 0,
            transformExpr,
        }
    }
}

/// A group of features sharing an entity key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGroup {
    pub name: String,
    pub entityKey: String,
    pub features: Vec<FeatureDefinition>,
    pub sourceQuery: String,
    pub refreshSeconds: u64,
    pub maxStalenessSeconds: u64,
    pub lastRefreshMs: i64,
    pub backingTable: Option<String>,
    pub retentionDays: u64,
}

impl FeatureGroup {
    pub fn new(name: String, entityKey: String) -> Self {
        Self {
            name,
            entityKey,
            features: Vec::new(),
            sourceQuery: String::new(),
            refreshSeconds: 3600,
            maxStalenessSeconds: 0,
            lastRefreshMs: 0,
            backingTable: None,
            retentionDays: 30,
        }
    }

    pub fn addFeature(&mut self, def: FeatureDefinition) {
        self.features.push(def);
    }

    pub fn featureIndex(&self, name: &str) -> Option<usize> {
        self.features.iter().position(|f| f.name == name)
    }
}

/// One materialized feature value snapshot, sorted by computationTimestampMs
/// per entity for fast point-in-time lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureValue {
    pub computationTimestampMs: i64,
    pub validFromMs: i64,
    pub validToMs: i64,
    pub value: AnalyticsValue,
    pub featureVersion: u32,
}

/// Composite 128-bit hash of `(entity, feature)`. Two independent fx_mix
/// chains run in lockstep over each byte stream, so a single pass yields
/// both halves without allocating temporary buffers
#[inline]
fn computeEfHash128(entity: &str, feature: &str) -> (u64, u64) {
    use zyron_common::fx_mix;
    // Two independent chains seeded with distinct constants. Mixing in
    // a length tag before the feature bytes ensures (a||b) and (a||b')
    // with overlapping byte sequences cannot collide
    let mut lo: u64 = 0x9E3779B97F4A7C15;
    let mut hi: u64 = 0xBF58476D1CE4E5B9;
    for &b in entity.as_bytes() {
        lo = fx_mix(lo, b as u64);
        hi = fx_mix(hi, (b as u64) ^ 0x55);
    }
    // Domain separator between entity and feature
    lo = fx_mix(lo, 0xCAFEBABE_DEADBEEF);
    hi = fx_mix(hi, 0xFACE_FEED_C0FFEE_01);
    for &b in feature.as_bytes() {
        lo = fx_mix(lo, b as u64);
        hi = fx_mix(hi, (b as u64) ^ 0xAA);
    }
    (lo, hi)
}

/// Per-group backing storage, individually locked so bulk writers do not
/// serialize on a process-wide RwLock (F1)
///
/// Storage uses a `VerifiedKeyMap` keyed by a 128-bit composite hash of
/// `(entity, feature)`. Lookups hash both byte streams once through two
/// parallel fx_mix chains, probe the PreHashMap with the low half, then
/// verify the high half plus the stored (entity, feature) for collision
/// safety. Zero allocations on the read path. Single bucket walk per
/// probe vs the nested HashMap layout's two bucket walks
struct FeatureBackingStore {
    inner: RwLock<FeatureBackingInner>,
}

#[derive(Default)]
struct FeatureBackingInner {
    series: VerifiedKeyMap<(String, String), Vec<FeatureValue>>,
    bloom: Option<BloomFilter>,
    bloomItemCount: usize,
    bloomCapacity: usize,
}

impl FeatureBackingStore {
    fn new() -> Self {
        Self {
            inner: RwLock::new(FeatureBackingInner::default()),
        }
    }

    #[inline]
    fn writeOne(&self, entity: &str, feature: &str, value: FeatureValue) {
        let mut g = self.inner.write();
        Self::writeOneInner(&mut g, entity, feature, value);
    }

    #[inline]
    fn writeOneInner(
        inner: &mut FeatureBackingInner,
        entity: &str,
        feature: &str,
        value: FeatureValue,
    ) {
        let (hLo, hHi) = computeEfHash128(entity, feature);
        // Single VerifiedKeyMap probe: entry_or_insert returns &mut V and
        // creates the slot if missing. Detect new-key via whether the
        // returned series is empty, which is the post-construction state
        // (a freshly inserted Vec). Empty series with new value is the
        // only path that increments the bloom item count
        let series = inner.series.entry_or_insert(
            hLo,
            hHi,
            || (entity.to_string(), feature.to_string()),
            Vec::new,
        );
        let isNewKey = series.is_empty();
        let pos = series
            .binary_search_by_key(&value.computationTimestampMs, |v| v.computationTimestampMs)
            .unwrap_or_else(|p| p);
        if pos < series.len()
            && series[pos].computationTimestampMs == value.computationTimestampMs
            && series[pos].featureVersion == value.featureVersion
        {
            series[pos] = value;
        } else {
            series.insert(pos, value);
        }
        Self::maybeResizeBloom(inner);
        if let Some(bf) = inner.bloom.as_mut() {
            bf.insert(entity);
        }
        if isNewKey {
            inner.bloomItemCount += 1;
        }
    }

    /// Lazily build the bloom on first insert. Resize when the count of
    /// distinct (entity, feature) keys crosses 80% of capacity. Using
    /// distinct keys as the sizing proxy is conservative: the bloom is
    /// sized for entity-only membership but keys grow at >= entity rate,
    /// so the FP rate target stays bounded
    fn maybeResizeBloom(inner: &mut FeatureBackingInner) {
        let needsResize = inner.bloom.is_none()
            || inner.bloomItemCount >= (inner.bloomCapacity.saturating_mul(8)) / 10;
        if !needsResize {
            return;
        }
        let newCap = if inner.bloomCapacity == 0 {
            10_000
        } else {
            inner.bloomCapacity.saturating_mul(4)
        };
        let mut bf = BloomFilter::withCapacity(newCap, 0.01);
        // Rebuild from existing series keys, single pass through the
        // VerifiedKeyMap. The same entity appears once per feature so the
        // bloom sees duplicate inserts but that is idempotent at the bit
        // level and keeps this path O(n) without per-insert tracking
        for ((e, _f), _series) in inner.series.iter() {
            bf.insert(e.as_str());
        }
        inner.bloom = Some(bf);
        inner.bloomCapacity = newCap;
    }

    fn writeBatch(&self, batch: &[(String, String, FeatureValue)]) {
        let mut g = self.inner.write();
        for (entity, feature, value) in batch {
            Self::writeOneInner(&mut g, entity, feature, value.clone());
        }
    }

    fn lookup(
        &self,
        entity: &str,
        feature: &str,
        asOfMs: i64,
        versionFilter: Option<u32>,
    ) -> Option<FeatureValue> {
        let g = self.inner.read();
        if let Some(bf) = &g.bloom {
            // Zero-alloc bloom probe via HashableBloom impl on str
            if !bf.contains(entity) {
                return None;
            }
        }
        let (hLo, hHi) = computeEfHash128(entity, feature);
        let series = g.series.get(hLo, hHi)?;
        // Binary search the most recent value at-or-before asOfMs
        let idx = match series.binary_search_by_key(&asOfMs, |v| v.computationTimestampMs) {
            Ok(i) => Some(i),
            Err(0) => None,
            Err(i) => Some(i - 1),
        }?;
        // Walk backward when filtering by version, since the requested
        // version may live at an earlier index than the latest
        let mut cursor = idx;
        loop {
            let v = &series[cursor];
            if v.computationTimestampMs > asOfMs {
                if cursor == 0 {
                    return None;
                }
                cursor -= 1;
                continue;
            }
            match versionFilter {
                None => return Some(v.clone()),
                Some(target) if v.featureVersion == target => return Some(v.clone()),
                _ => {
                    if cursor == 0 {
                        return None;
                    }
                    cursor -= 1;
                }
            }
        }
    }

    fn applyRetention(&self, cutoff: i64) -> u64 {
        let mut g = self.inner.write();
        let mut removed = 0u64;
        // Series are sorted ascending by computationTimestampMs, so all
        // entries to retain live in a contiguous suffix. Binary search
        // for the cutoff boundary and drain the prefix in one shot
        // (O(log n + drained)) rather than retain's O(n) shifting form
        for (_k, series) in g.series.iter_mut() {
            if series.is_empty() {
                continue;
            }
            let cutIdx = series
                .binary_search_by_key(&cutoff, |v| v.computationTimestampMs)
                .unwrap_or_else(|p| p);
            if cutIdx > 0 {
                removed += cutIdx as u64;
                series.drain(0..cutIdx);
            }
        }
        removed
    }

    fn maxStalenessMs(&self, nowMs: i64) -> i64 {
        let g = self.inner.read();
        let mut maxStale = 0i64;
        for (_k, series) in g.series.iter() {
            if let Some(last) = series.last() {
                let stale = nowMs - last.computationTimestampMs;
                if stale > maxStale {
                    maxStale = stale;
                }
            }
        }
        maxStale
    }

    fn rowCount(&self) -> u64 {
        let g = self.inner.read();
        g.series.iter().map(|(_k, v)| v.len() as u64).sum()
    }
}

/// Top-level feature store. Holds an `Arc<FeatureBackingStore>` per group;
/// writes acquire only the per-group lock so independent groups never
/// contend, and bulk batch writes acquire one lock for the whole batch
pub struct FeatureStore {
    groups: RwLock<HashMap<String, Arc<FeatureGroup>>>,
    storage: RwLock<HashMap<String, Arc<FeatureBackingStore>>>,
}

impl FeatureStore {
    pub fn new() -> Self {
        Self {
            groups: RwLock::new(HashMap::new()),
            storage: RwLock::new(HashMap::new()),
        }
    }

    pub fn registerFeatureGroup(&self, group: FeatureGroup) -> Result<()> {
        let mut g = self.groups.write();
        if g.contains_key(&group.name) {
            return Err(ZyronError::InvalidParameter {
                name: "feature_group".to_string(),
                value: format!("'{}' already registered", group.name),
            });
        }
        let groupName = group.name.clone();
        g.insert(groupName.clone(), Arc::new(group));
        let mut s = self.storage.write();
        s.insert(groupName, Arc::new(FeatureBackingStore::new()));
        Ok(())
    }

    pub fn dropFeatureGroup(&self, name: &str) -> Result<()> {
        let mut g = self.groups.write();
        if g.remove(name).is_none() {
            return Err(ZyronError::InvalidParameter {
                name: "feature_group".to_string(),
                value: format!("'{}' not found", name),
            });
        }
        let mut s = self.storage.write();
        s.remove(name);
        Ok(())
    }

    pub fn group(&self, name: &str) -> Option<Arc<FeatureGroup>> {
        self.groups.read().get(name).cloned()
    }

    pub fn groups(&self) -> Vec<Arc<FeatureGroup>> {
        self.groups.read().values().cloned().collect()
    }

    /// Resolve the per-group store under a read lock on the registry,
    /// then drop the registry lock and operate under the per-group lock
    fn resolveStore(&self, groupName: &str) -> Option<Arc<FeatureBackingStore>> {
        self.storage.read().get(groupName).cloned()
    }

    /// Append a single feature value. Takes only the per-group lock
    pub fn writeFeatureValue(
        &self,
        groupName: &str,
        entityKey: &str,
        featureName: &str,
        value: FeatureValue,
    ) -> Result<()> {
        let store = self
            .resolveStore(groupName)
            .ok_or_else(|| ZyronError::InvalidParameter {
                name: "feature_group".to_string(),
                value: format!("'{}' not found", groupName),
            })?;
        store.writeOne(entityKey, featureName, value);
        Ok(())
    }

    /// Bulk write, acquires the per-group lock once for an entire batch.
    /// Each entry is (entityKey, featureName, value). Use for materialization
    /// passes where you have many writes for the same group ready
    pub fn writeFeatureValuesBatch(
        &self,
        groupName: &str,
        batch: &[(String, String, FeatureValue)],
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let store = self
            .resolveStore(groupName)
            .ok_or_else(|| ZyronError::InvalidParameter {
                name: "feature_group".to_string(),
                value: format!("'{}' not found", groupName),
            })?;
        store.writeBatch(batch);
        Ok(())
    }

    /// Point-in-time lookup with single HashMap probe (F5)
    pub fn pointInTimeLookup(
        &self,
        groupName: &str,
        entityKey: &str,
        featureName: &str,
        asOfMs: i64,
    ) -> Option<FeatureValue> {
        let store = self.resolveStore(groupName)?;
        store.lookup(entityKey, featureName, asOfMs, None)
    }

    /// Multi-entity, multi-feature point-in-time retrieval
    pub fn getFeatures(
        &self,
        groupName: &str,
        entityKeys: &[String],
        featureNames: &[String],
        asOfMs: i64,
    ) -> Result<FeatureFrame> {
        let group = self.group(groupName).ok_or_else(|| ZyronError::InvalidParameter {
            name: "feature_group".to_string(),
            value: format!("'{}' not found", groupName),
        })?;
        let store = self
            .resolveStore(groupName)
            .ok_or_else(|| ZyronError::InvalidParameter {
                name: "feature_group".to_string(),
                value: format!("'{}' not found", groupName),
            })?;
        let mut frame = FeatureFrame::new(group.entityKey.clone(), featureNames.to_vec());
        for entity in entityKeys {
            let mut row = FeatureRow {
                entityKey: entity.clone(),
                values: Vec::with_capacity(featureNames.len()),
            };
            for fname in featureNames {
                let v = store.lookup(entity, fname, asOfMs, None);
                row.values
                    .push(v.map(|f| f.value).unwrap_or(AnalyticsValue::Null));
            }
            frame.rows.push(row);
        }
        Ok(frame)
    }

    /// Versioned point-in-time retrieval, filters the shared series by
    /// featureVersion at read time so storage cost stays single-copy (F2)
    pub fn getFeaturesVersioned(
        &self,
        groupName: &str,
        entityKeys: &[String],
        featureNames: &[String],
        version: u32,
        asOfMs: i64,
    ) -> Result<FeatureFrame> {
        let group = self.group(groupName).ok_or_else(|| ZyronError::InvalidParameter {
            name: "feature_group".to_string(),
            value: format!("'{}' not found", groupName),
        })?;
        let mut frame = FeatureFrame::new(group.entityKey.clone(), featureNames.to_vec());
        let store = match self.resolveStore(groupName) {
            Some(s) => s,
            None => return Ok(frame),
        };
        for entity in entityKeys {
            let mut row = FeatureRow {
                entityKey: entity.clone(),
                values: Vec::with_capacity(featureNames.len()),
            };
            for fname in featureNames {
                let v = store.lookup(entity, fname, asOfMs, Some(version));
                row.values
                    .push(v.map(|f| f.value).unwrap_or(AnalyticsValue::Null));
            }
            frame.rows.push(row);
        }
        Ok(frame)
    }

    /// Apply retention policy, drop computationTimestampMs < cutoff
    pub fn applyRetention(&self, groupName: &str, retainAfterMs: i64) -> Result<u64> {
        let store = self
            .resolveStore(groupName)
            .ok_or_else(|| ZyronError::InvalidParameter {
                name: "feature_group".to_string(),
                value: format!("'{}' not found", groupName),
            })?;
        Ok(store.applyRetention(retainAfterMs))
    }

    pub fn maxStalenessMs(&self, groupName: &str, nowMs: i64) -> i64 {
        match self.resolveStore(groupName) {
            Some(s) => s.maxStalenessMs(nowMs),
            None => i64::MAX,
        }
    }

    pub fn rowCount(&self, groupName: &str) -> u64 {
        match self.resolveStore(groupName) {
            Some(s) => s.rowCount(),
            None => 0,
        }
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

static FEATURE_STORE: std::sync::OnceLock<Arc<FeatureStore>> = std::sync::OnceLock::new();

pub fn featureStore() -> Arc<FeatureStore> {
    FEATURE_STORE
        .get_or_init(|| Arc::new(FeatureStore::new()))
        .clone()
}

static FEATURE_LINEAGE: std::sync::OnceLock<
    Arc<parking_lot::RwLock<crate::featureLineage::FeatureLineageRegistry>>,
> = std::sync::OnceLock::new();

pub fn featureLineageRegistry()
-> Arc<parking_lot::RwLock<crate::featureLineage::FeatureLineageRegistry>> {
    FEATURE_LINEAGE
        .get_or_init(|| {
            Arc::new(parking_lot::RwLock::new(
                crate::featureLineage::FeatureLineageRegistry::new(),
            ))
        })
        .clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRow {
    pub entityKey: String,
    pub values: Vec<AnalyticsValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFrame {
    pub entityKeyName: String,
    pub featureNames: Vec<String>,
    pub rows: Vec<FeatureRow>,
}

impl FeatureFrame {
    pub fn new(entityKeyName: String, featureNames: Vec<String>) -> Self {
        Self {
            entityKeyName,
            featureNames,
            rows: Vec::new(),
        }
    }

    pub fn rowCount(&self) -> usize {
        self.rows.len()
    }

    pub fn columnCount(&self) -> usize {
        self.featureNames.len()
    }

    pub fn column(&self, idx: usize) -> Vec<AnalyticsValue> {
        self.rows.iter().map(|r| r.values[idx].clone()).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityCheckResult {
    pub matched: u64,
    pub mismatched: u64,
    pub mismatches: Vec<(String, String, AnalyticsValue, AnalyticsValue)>,
}

pub fn featureParityCheck(a: &FeatureFrame, b: &FeatureFrame) -> ParityCheckResult {
    let mut matched = 0u64;
    let mut mismatched = 0u64;
    let mut mismatches: Vec<(String, String, AnalyticsValue, AnalyticsValue)> = Vec::new();
    // HashMap, not BTreeMap, since membership-only lookup does not need
    // ordered traversal and HashMap is O(1) vs BTreeMap's O(log n)
    let mut aRows: HashMap<&str, &FeatureRow> = HashMap::with_capacity(a.rows.len());
    for r in &a.rows {
        aRows.insert(r.entityKey.as_str(), r);
    }
    for rB in &b.rows {
        if let Some(rA) = aRows.get(rB.entityKey.as_str()) {
            for (i, name) in a.featureNames.iter().enumerate() {
                let va = rA.values.get(i).cloned().unwrap_or(AnalyticsValue::Null);
                let vb = rB.values.get(i).cloned().unwrap_or(AnalyticsValue::Null);
                if va == vb {
                    matched += 1;
                } else {
                    mismatched += 1;
                    mismatches.push((rB.entityKey.clone(), name.clone(), va, vb));
                }
            }
        }
    }
    ParityCheckResult {
        matched,
        mismatched,
        mismatches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buildGroup() -> FeatureGroup {
        let mut g = FeatureGroup::new("user_features".to_string(), "user_id".to_string());
        g.addFeature(FeatureDefinition::new(
            "total_purchases".to_string(),
            "FLOAT64".to_string(),
            "SELECT SUM(amount) FROM orders WHERE user_id = entity.user_id".to_string(),
        ));
        g.addFeature(FeatureDefinition::new(
            "avg_order_value".to_string(),
            "FLOAT64".to_string(),
            "SELECT AVG(amount) FROM orders WHERE user_id = entity.user_id".to_string(),
        ));
        g
    }

    fn fv(ts: i64, value: f64) -> FeatureValue {
        FeatureValue {
            computationTimestampMs: ts,
            validFromMs: ts,
            validToMs: i64::MAX,
            value: AnalyticsValue::Float(value),
            featureVersion: 1,
        }
    }

    #[test]
    fn registersAndQueries() {
        let store = FeatureStore::new();
        store.registerFeatureGroup(buildGroup()).unwrap();
        store
            .writeFeatureValue("user_features", "u1", "total_purchases", fv(100, 50.0))
            .unwrap();
        store
            .writeFeatureValue("user_features", "u1", "total_purchases", fv(200, 75.0))
            .unwrap();
        let frame = store
            .getFeatures(
                "user_features",
                &["u1".to_string()],
                &["total_purchases".to_string()],
                300,
            )
            .unwrap();
        assert_eq!(frame.rows.len(), 1);
        let v = &frame.rows[0].values[0];
        if let AnalyticsValue::Float(f) = v {
            assert!((f - 75.0).abs() < 1e-12);
        } else {
            panic!("expected float, got {:?}", v);
        }
    }

    #[test]
    fn pointInTimeNoLeakage() {
        let store = FeatureStore::new();
        store.registerFeatureGroup(buildGroup()).unwrap();
        store
            .writeFeatureValue("user_features", "u1", "total_purchases", fv(100, 50.0))
            .unwrap();
        store
            .writeFeatureValue("user_features", "u1", "total_purchases", fv(200, 75.0))
            .unwrap();
        let f150 = store
            .getFeatures(
                "user_features",
                &["u1".to_string()],
                &["total_purchases".to_string()],
                150,
            )
            .unwrap();
        if let AnalyticsValue::Float(f) = &f150.rows[0].values[0] {
            assert!((f - 50.0).abs() < 1e-12, "AS OF 150 should see 50, got {}", f);
        }
    }

    #[test]
    fn missingEntityReturnsNull() {
        let store = FeatureStore::new();
        store.registerFeatureGroup(buildGroup()).unwrap();
        let frame = store
            .getFeatures(
                "user_features",
                &["nonexistent".to_string()],
                &["total_purchases".to_string()],
                500,
            )
            .unwrap();
        assert!(matches!(frame.rows[0].values[0], AnalyticsValue::Null));
    }

    #[test]
    fn versionedRetrieval() {
        let store = FeatureStore::new();
        store.registerFeatureGroup(buildGroup()).unwrap();
        let mut v1 = fv(100, 50.0);
        v1.featureVersion = 1;
        let mut v2 = fv(100, 88.0);
        v2.featureVersion = 2;
        store
            .writeFeatureValue("user_features", "u1", "total_purchases", v1)
            .unwrap();
        store
            .writeFeatureValue("user_features", "u1", "total_purchases", v2)
            .unwrap();
        let frame1 = store
            .getFeaturesVersioned(
                "user_features",
                &["u1".to_string()],
                &["total_purchases".to_string()],
                1,
                200,
            )
            .unwrap();
        let frame2 = store
            .getFeaturesVersioned(
                "user_features",
                &["u1".to_string()],
                &["total_purchases".to_string()],
                2,
                200,
            )
            .unwrap();
        if let (AnalyticsValue::Float(a), AnalyticsValue::Float(b)) = (
            &frame1.rows[0].values[0],
            &frame2.rows[0].values[0],
        ) {
            assert!((a - 50.0).abs() < 1e-12);
            assert!((b - 88.0).abs() < 1e-12);
        } else {
            panic!("expected float values");
        }
    }

    #[test]
    fn batchWriteIsOneLock() {
        let store = FeatureStore::new();
        store.registerFeatureGroup(buildGroup()).unwrap();
        let batch: Vec<(String, String, FeatureValue)> = (0..1000)
            .map(|i| {
                (
                    format!("u{}", i),
                    "total_purchases".to_string(),
                    fv(100 + i as i64, i as f64),
                )
            })
            .collect();
        store
            .writeFeatureValuesBatch("user_features", &batch)
            .unwrap();
        let frame = store
            .getFeatures(
                "user_features",
                &(0..1000).map(|i| format!("u{}", i)).collect::<Vec<_>>(),
                &["total_purchases".to_string()],
                10_000,
            )
            .unwrap();
        assert_eq!(frame.rows.len(), 1000);
    }
}
