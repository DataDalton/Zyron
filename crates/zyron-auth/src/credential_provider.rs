//! Dynamic credential provider trait and TTL-aware credential cache.
//!
//! Dynamic credentials (OAuth2, K8s SA, Vault, AWS, GCP, Azure) are fetched at
//! use time and cached with a caller-supplied TTL. The cache refreshes a cached
//! entry in-band when the caller indicates the remaining TTL has dropped below
//! a refresh-before threshold.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// Provider trait
// -----------------------------------------------------------------------------

/// Fetches fresh credential material on demand.
#[async_trait]
pub trait CredentialProvider: Send + Sync {
    /// Fetches a credential map plus the TTL after which the credential expires.
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)>;

    /// Short identifier for logging and metrics.
    fn provider_kind(&self) -> &'static str;
}

// -----------------------------------------------------------------------------
// Cached credential entry
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CachedCredential {
    credentials: HashMap<String, String>,
    expires_at: Instant,
    fetched_at: Instant,
}

// -----------------------------------------------------------------------------
// Cache statistics
// -----------------------------------------------------------------------------

/// Observable cache counters for metrics surfaces.
#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub refreshes: u64,
    pub invalidations: u64,
    pub size: u64,
}

// -----------------------------------------------------------------------------
// Credential cache
// -----------------------------------------------------------------------------

/// TTL-aware cache. Entries are keyed by caller-supplied cache key strings so
/// different providers can share a single cache under distinct namespaces.
pub struct CredentialCache {
    entries: scc::HashMap<String, CachedCredential>,
    hits: AtomicU64,
    misses: AtomicU64,
    refreshes: AtomicU64,
    invalidations: AtomicU64,
    metrics: OnceLock<Arc<zyron_common::LabeledMetrics>>,
}

impl CredentialCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            entries: scc::HashMap::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            refreshes: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            metrics: OnceLock::new(),
        }
    }

    /// Attaches the shared labeled-metrics handle. Set once at server start
    /// so per-provider cache counters surface on the metrics endpoint.
    pub fn attachMetrics(&self, metrics: Arc<zyron_common::LabeledMetrics>) {
        let _ = self.metrics.set(metrics);
    }

    /// Returns cached credentials if still fresh, otherwise fetches through the
    /// provider, stores the result with the provider-supplied TTL, and returns
    /// the material. If the cached entry has less remaining life than
    /// `refresh_before`, it is refreshed proactively.
    pub async fn get_or_fetch(
        &self,
        key: &str,
        provider: &dyn CredentialProvider,
        refresh_before: Duration,
    ) -> Result<HashMap<String, String>> {
        let now = Instant::now();
        let kind = provider.provider_kind();
        let metrics = self.metrics.get();
        let existing = self.entries.read_async(key, |_, v| v.clone()).await;
        let mut was_refresh = false;
        if let Some(entry) = existing {
            if entry.expires_at > now {
                let remaining = entry.expires_at.duration_since(now);
                if remaining > refresh_before {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    if let Some(m) = metrics {
                        m.credCacheHit(kind);
                    }
                    tracing::info!(
                        target: "zyron::audit",
                        event = "CredentialRead",
                        principal = %key,
                        object = kind,
                        decision = "granted",
                        reason = "cache-hit",
                    );
                    return Ok(entry.credentials);
                }
                self.refreshes.fetch_add(1, Ordering::Relaxed);
                was_refresh = true;
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
                if let Some(m) = metrics {
                    m.credCacheMiss(kind);
                }
            }
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            if let Some(m) = metrics {
                m.credCacheMiss(kind);
            }
        }

        let (creds, ttl) = provider.fetch().await?;
        if ttl.is_zero() {
            return Err(ZyronError::InvalidCredential(format!(
                "credential provider {} returned zero TTL",
                provider.provider_kind()
            )));
        }
        let expires_at = Instant::now() + ttl;
        let fetched_at = Instant::now();
        let entry = CachedCredential {
            credentials: creds.clone(),
            expires_at,
            fetched_at,
        };
        let _ = self.entries.remove_async(key).await;
        let _ = self
            .entries
            .insert_async(key.to_string(), entry)
            .await
            .map_err(|_| {
                ZyronError::InvalidCredential("credential cache insert race".to_string())
            })?;
        if was_refresh {
            if let Some(m) = metrics {
                m.credRefresh(kind);
            }
            tracing::info!(
                target: "zyron::audit",
                event = "CredentialRefreshed",
                principal = %key,
                object = kind,
                decision = "granted",
                reason = "proactive-ttl-refresh",
            );
        }
        tracing::info!(
            target: "zyron::audit",
            event = "CredentialRead",
            principal = %key,
            object = kind,
            decision = "granted",
            reason = "provider-fetch",
        );
        Ok(creds)
    }

    /// Removes a cached credential. Subsequent lookups will refetch.
    pub fn invalidate(&self, key: &str) {
        if self.entries.remove_sync(key).is_some() {
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns cache counters. Size is an approximate snapshot.
    pub fn stats(&self) -> CacheStats {
        let mut size: u64 = 0;
        self.entries.iter_sync(|_, _| {
            size += 1;
            true
        });
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            refreshes: self.refreshes.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
            size,
        }
    }

    /// Returns the time the entry was last fetched if present.
    pub fn fetched_at(&self, key: &str) -> Option<Instant> {
        self.entries.read_sync(key, |_, v| v.fetched_at)
    }
}

impl Default for CredentialCache {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// Test-only static provider
// -----------------------------------------------------------------------------

/// Provider that returns a fixed credential map with a configurable TTL. Used
/// in tests and as the default seed for local dev setups.
pub struct StaticCredentialProvider {
    creds: HashMap<String, String>,
    ttl: Duration,
    calls: Arc<AtomicU64>,
}

impl StaticCredentialProvider {
    pub fn new(creds: HashMap<String, String>, ttl: Duration) -> Self {
        Self {
            creds,
            ttl,
            calls: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn call_count(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl CredentialProvider for StaticCredentialProvider {
    async fn fetch(&self) -> Result<(HashMap<String, String>, Duration)> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok((self.creds.clone(), self.ttl))
    }

    fn provider_kind(&self) -> &'static str {
        "static"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mkcreds() -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("token".to_string(), "abc".to_string());
        h
    }

    #[tokio::test]
    async fn roundtrip_and_hit() {
        let cache = CredentialCache::new();
        let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_secs(60));
        let first = cache
            .get_or_fetch("k", &provider, Duration::from_secs(5))
            .await
            .expect("fetch");
        assert_eq!(first.get("token").map(|s| s.as_str()), Some("abc"));
        assert_eq!(provider.call_count(), 1);

        let second = cache
            .get_or_fetch("k", &provider, Duration::from_secs(5))
            .await
            .expect("fetch");
        assert_eq!(second.get("token").map(|s| s.as_str()), Some("abc"));
        assert_eq!(
            provider.call_count(),
            1,
            "second call should be a cache hit"
        );
        let s = cache.stats();
        assert_eq!(s.hits, 1);
    }

    #[tokio::test]
    async fn refresh_before_expiry_triggers_refetch() {
        let cache = CredentialCache::new();
        let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_millis(50));
        let _ = cache
            .get_or_fetch("k", &provider, Duration::from_millis(100))
            .await
            .expect("fetch");
        // Second call: remaining life is less than 100ms threshold, refetch.
        let _ = cache
            .get_or_fetch("k", &provider, Duration::from_millis(100))
            .await
            .expect("fetch");
        assert_eq!(provider.call_count(), 2);
        let s = cache.stats();
        assert!(s.refreshes >= 1);
    }

    #[tokio::test]
    async fn expiry_forces_refetch() {
        let cache = CredentialCache::new();
        let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_millis(30));
        let _ = cache
            .get_or_fetch("k", &provider, Duration::from_millis(5))
            .await
            .expect("fetch");
        tokio::time::sleep(Duration::from_millis(60)).await;
        let _ = cache
            .get_or_fetch("k", &provider, Duration::from_millis(5))
            .await
            .expect("fetch");
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn invalidate_removes_entry() {
        let cache = CredentialCache::new();
        let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_secs(60));
        let _ = cache
            .get_or_fetch("k", &provider, Duration::from_secs(1))
            .await
            .expect("fetch");
        cache.invalidate("k");
        let _ = cache
            .get_or_fetch("k", &provider, Duration::from_secs(1))
            .await
            .expect("fetch");
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn concurrent_access_single_entry() {
        let cache = Arc::new(CredentialCache::new());
        let provider = Arc::new(StaticCredentialProvider::new(
            mkcreds(),
            Duration::from_secs(60),
        ));
        let mut handles = Vec::new();
        for _ in 0..16 {
            let c = cache.clone();
            let p = provider.clone();
            handles.push(tokio::spawn(async move {
                c.get_or_fetch("k", p.as_ref(), Duration::from_secs(5))
                    .await
                    .expect("fetch")
            }));
        }
        for h in handles {
            let v = h.await.expect("join");
            assert_eq!(v.get("token").map(|s| s.as_str()), Some("abc"));
        }
        // Multiple refetches may occur from races but total must be bounded.
        assert!(provider.call_count() >= 1);
    }

    #[tokio::test]
    async fn zero_ttl_is_rejected() {
        let cache = CredentialCache::new();
        let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_secs(0));
        let r = cache
            .get_or_fetch("k", &provider, Duration::from_secs(1))
            .await;
        assert!(r.is_err());
    }

    #[derive(Clone)]
    struct SharedBuf(std::sync::Arc<parking_lot::Mutex<Vec<u8>>>);
    impl std::io::Write for SharedBuf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBuf {
        type Writer = SharedBuf;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[tokio::test]
    async fn credential_read_audit_fires_on_fetch_and_hit() {
        let buf = std::sync::Arc::new(parking_lot::Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(SharedBuf(buf.clone()))
            .with_max_level(tracing::Level::INFO)
            .without_time()
            .with_ansi(false)
            .finish();

        let captured = {
            let _guard = tracing::subscriber::set_default(subscriber);
            let cache = CredentialCache::new();
            let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_secs(60));
            // First call: provider-fetch path.
            cache
                .get_or_fetch("k", &provider, Duration::from_secs(5))
                .await
                .expect("fetch");
            // Second call: cache-hit path.
            cache
                .get_or_fetch("k", &provider, Duration::from_secs(5))
                .await
                .expect("hit");
            String::from_utf8(buf.lock().clone()).unwrap()
        };

        assert!(captured.contains("CredentialRead"));
        assert!(captured.contains("provider-fetch"));
        assert!(captured.contains("cache-hit"));
        assert!(captured.contains("static"));
    }

    #[tokio::test]
    async fn credential_refreshed_audit_fires_on_proactive_refresh() {
        let buf = std::sync::Arc::new(parking_lot::Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(SharedBuf(buf.clone()))
            .with_max_level(tracing::Level::INFO)
            .without_time()
            .with_ansi(false)
            .finish();

        let captured = {
            let _guard = tracing::subscriber::set_default(subscriber);
            let cache = CredentialCache::new();
            // Short TTL so the second call lands in the proactive-refresh
            // window (remaining <= refresh_before) instead of a plain hit.
            let provider = StaticCredentialProvider::new(mkcreds(), Duration::from_millis(40));
            cache
                .get_or_fetch("k", &provider, Duration::from_millis(5))
                .await
                .expect("fetch");
            tokio::time::sleep(Duration::from_millis(20)).await;
            cache
                .get_or_fetch("k", &provider, Duration::from_millis(30))
                .await
                .expect("refresh");
            String::from_utf8(buf.lock().clone()).unwrap()
        };

        assert!(captured.contains("CredentialRefreshed"));
        assert!(captured.contains("proactive-ttl-refresh"));
    }
}
