//! Connection pool with host-list failover and per-host health tracking.
//!
//! Maintains a bounded set of PgClient connections keyed across a list of
//! candidate hosts. Failed hosts are marked unavailable and rechecked on a
//! cooldown timer. A semaphore caps the total active connections.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use crate::pg_client::{ClientConfig, PgClient};
use crate::tls::ClientTlsConnector;
use crate::uri::UriHost;

// ----------------------------------------------------------------------------
// Credential provider
// ----------------------------------------------------------------------------

/// Abstraction over password sourcing so credentials can be rotated without
/// rebuilding the pool.
#[async_trait]
pub trait CredentialProvider: Send + Sync + 'static {
    async fn password(&self, user: &str) -> Option<String>;
}

/// Credential provider that returns a fixed static password.
pub struct StaticPassword(pub Option<String>);

#[async_trait]
impl CredentialProvider for StaticPassword {
    async fn password(&self, _user: &str) -> Option<String> {
        self.0.clone()
    }
}

// ----------------------------------------------------------------------------
// Host state
// ----------------------------------------------------------------------------

/// Role hint used to steer writes to a primary and reads to replicas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostRole {
    Primary,
    Replica,
    Unknown,
}

/// Lock-free health state for one host.
pub struct AtomicHealth {
    /// Non-zero when the host is considered unhealthy. Stores unix-epoch
    /// seconds at which the host should be reconsidered.
    cooldown_until: AtomicU64,
    /// Consecutive failures since the last success.
    consecutive_failures: AtomicU32,
}

impl AtomicHealth {
    pub fn new() -> Self {
        Self {
            cooldown_until: AtomicU64::new(0),
            consecutive_failures: AtomicU32::new(0),
        }
    }

    pub fn is_available_now(&self) -> bool {
        let now = now_epoch_secs();
        self.cooldown_until.load(Ordering::Relaxed) <= now
    }

    pub fn record_success(&self) {
        self.cooldown_until.store(0, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    pub fn record_failure(&self, breaker_threshold: u32, cooldown: Duration) {
        let fails = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if fails >= breaker_threshold {
            let until = now_epoch_secs() + cooldown.as_secs();
            self.cooldown_until.store(until, Ordering::Relaxed);
        }
    }
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Resolved host entry with live health state.
pub struct HostEntry {
    pub host: String,
    pub port: u16,
    pub role: HostRole,
    pub health: AtomicHealth,
}

impl HostEntry {
    pub fn from_uri(uri: &UriHost, role: HostRole) -> Self {
        Self {
            host: uri.host.clone(),
            port: uri.port,
            role,
            health: AtomicHealth::new(),
        }
    }

    pub fn addr(&self) -> Option<SocketAddr> {
        format!("{}:{}", self.host, self.port).parse().ok()
    }
}

// ----------------------------------------------------------------------------
// Config and stats
// ----------------------------------------------------------------------------

/// Runtime configuration for the pool.
pub struct PoolConfig {
    pub hosts: Vec<HostEntry>,
    pub min_idle: usize,
    pub max_size: usize,
    pub max_idle_secs: u64,
    pub max_lifetime_secs: u64,
    pub connect_timeout: Duration,
    pub statement_timeout: Duration,
    pub idle_timeout: Duration,
    pub keepalive: Duration,
    pub tls: Option<Arc<ClientTlsConnector>>,
    pub prefer_quic: bool,
    pub database: String,
    pub user: String,
    pub credential_provider: Arc<dyn CredentialProvider>,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_cooldown: Duration,
}

impl PoolConfig {
    /// Builds a plain config with one host, no TLS, and a static password.
    pub fn simple(
        host: &str,
        port: u16,
        user: &str,
        password: Option<&str>,
        database: &str,
    ) -> Self {
        Self {
            hosts: vec![HostEntry {
                host: host.into(),
                port,
                role: HostRole::Unknown,
                health: AtomicHealth::new(),
            }],
            min_idle: 0,
            max_size: 8,
            max_idle_secs: 300,
            max_lifetime_secs: 1800,
            connect_timeout: Duration::from_secs(5),
            statement_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(60),
            keepalive: Duration::from_secs(30),
            tls: None,
            prefer_quic: false,
            database: database.into(),
            user: user.into(),
            credential_provider: Arc::new(StaticPassword(password.map(|s| s.into()))),
            circuit_breaker_threshold: 3,
            circuit_breaker_cooldown: Duration::from_secs(5),
        }
    }
}

/// Snapshot of pool counters.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub idle: usize,
    pub in_use: usize,
    pub max_size: usize,
    pub total_acquired: u64,
    pub total_failed: u64,
    pub hosts_healthy: usize,
    pub hosts_total: usize,
}

/// Errors returned by the pool.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("Pool shutdown")]
    Shutdown,
    #[error("All candidate hosts are unavailable")]
    NoHealthyHosts,
    #[error("Connect timed out")]
    Timeout,
    #[error("Connect failed: {0}")]
    Connect(String),
}

pub type PoolResult<T> = std::result::Result<T, PoolError>;

// ----------------------------------------------------------------------------
// Pool internals
// ----------------------------------------------------------------------------

struct IdleEntry {
    client: PgClient,
    host_idx: usize,
    created_at: Instant,
    last_used: Instant,
}

struct PoolInner {
    config: PoolConfig,
    idle: Mutex<VecDeque<IdleEntry>>,
    in_use: AtomicU32,
    total_acquired: AtomicU64,
    total_failed: AtomicU64,
    semaphore: Arc<Semaphore>,
    shutdown: std::sync::atomic::AtomicBool,
}

/// Connection pool handle.
#[derive(Clone)]
pub struct ConnectionPool {
    inner: Arc<PoolInner>,
}

impl ConnectionPool {
    /// Builds a new pool from the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        let sem = Arc::new(Semaphore::new(config.max_size));
        let inner = Arc::new(PoolInner {
            semaphore: sem,
            idle: Mutex::new(VecDeque::new()),
            in_use: AtomicU32::new(0),
            total_acquired: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            shutdown: std::sync::atomic::AtomicBool::new(false),
            config,
        });
        Self { inner }
    }

    /// Acquires a connection, opening a new one if the idle queue is empty.
    /// Drops are returned to the pool by the guard.
    pub async fn acquire(&self) -> PoolResult<PooledConnection> {
        self.acquire_role(HostRole::Unknown).await
    }

    /// Acquires a connection with a preferred host role.
    pub async fn acquire_role(&self, role: HostRole) -> PoolResult<PooledConnection> {
        if self.inner.shutdown.load(Ordering::Relaxed) {
            return Err(PoolError::Shutdown);
        }

        // Enforce max_size.
        let permit = self
            .inner
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| PoolError::Shutdown)?;

        // Try to reuse an idle entry.
        let reuse = {
            let mut idle = self.inner.idle.lock();
            loop {
                match idle.pop_front() {
                    Some(entry) => {
                        let lifetime_expired = entry.created_at.elapsed().as_secs()
                            >= self.inner.config.max_lifetime_secs;
                        let idle_expired =
                            entry.last_used.elapsed().as_secs() >= self.inner.config.max_idle_secs;
                        if lifetime_expired || idle_expired {
                            // Drop and continue scanning.
                            drop(entry);
                            continue;
                        }
                        if role != HostRole::Unknown {
                            let host = &self.inner.config.hosts[entry.host_idx];
                            if host.role != role && host.role != HostRole::Unknown {
                                idle.push_back(entry);
                                break None;
                            }
                        }
                        break Some(entry);
                    }
                    None => break None,
                }
            }
        };

        if let Some(entry) = reuse {
            self.inner.in_use.fetch_add(1, Ordering::Relaxed);
            self.inner.total_acquired.fetch_add(1, Ordering::Relaxed);
            return Ok(PooledConnection {
                client: Some(entry.client),
                created_at: entry.created_at,
                host_idx: entry.host_idx,
                _permit: permit,
                pool: self.inner.clone(),
            });
        }

        // Open a new connection. Iterate the host list.
        let (client, host_idx) = match self.open_new(role).await {
            Ok(r) => r,
            Err(e) => {
                self.inner.total_failed.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        };
        self.inner.in_use.fetch_add(1, Ordering::Relaxed);
        self.inner.total_acquired.fetch_add(1, Ordering::Relaxed);
        Ok(PooledConnection {
            client: Some(client),
            created_at: Instant::now(),
            host_idx,
            _permit: permit,
            pool: self.inner.clone(),
        })
    }

    async fn open_new(&self, role: HostRole) -> PoolResult<(PgClient, usize)> {
        let config = ClientConfig {
            user: self.inner.config.user.clone(),
            database: self.inner.config.database.clone(),
            application_name: "zyron-pool".into(),
            password: self
                .inner
                .config
                .credential_provider
                .password(&self.inner.config.user)
                .await,
            connect_timeout: self.inner.config.connect_timeout,
            statement_timeout: self.inner.config.statement_timeout,
        };

        let mut last_err: Option<PoolError> = None;
        for (idx, host) in self.inner.config.hosts.iter().enumerate() {
            if !host.health.is_available_now() {
                continue;
            }
            if role != HostRole::Unknown && host.role != role && host.role != HostRole::Unknown {
                continue;
            }
            let addr = match host.addr() {
                Some(a) => a,
                None => {
                    host.health.record_failure(
                        self.inner.config.circuit_breaker_threshold,
                        self.inner.config.circuit_breaker_cooldown,
                    );
                    continue;
                }
            };
            let connect_res = if let Some(tls) = self.inner.config.tls.as_ref() {
                PgClient::connect_tls(addr, tls, &config).await
            } else {
                PgClient::connect(addr, &config).await
            };
            match connect_res {
                Ok(client) => {
                    host.health.record_success();
                    return Ok((client, idx));
                }
                Err(e) => {
                    host.health.record_failure(
                        self.inner.config.circuit_breaker_threshold,
                        self.inner.config.circuit_breaker_cooldown,
                    );
                    last_err = Some(PoolError::Connect(e.to_string()));
                }
            }
        }
        Err(last_err.unwrap_or(PoolError::NoHealthyHosts))
    }

    /// Marks the pool as shutting down and closes all idle connections.
    pub async fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        let mut idle = self.inner.idle.lock();
        while let Some(entry) = idle.pop_front() {
            let _ = entry.client.close().await;
        }
    }

    /// Returns a snapshot of pool counters.
    pub fn stats(&self) -> PoolStats {
        let healthy = self
            .inner
            .config
            .hosts
            .iter()
            .filter(|h| h.health.is_available_now())
            .count();
        PoolStats {
            idle: self.inner.idle.lock().len(),
            in_use: self.inner.in_use.load(Ordering::Relaxed) as usize,
            max_size: self.inner.config.max_size,
            total_acquired: self.inner.total_acquired.load(Ordering::Relaxed),
            total_failed: self.inner.total_failed.load(Ordering::Relaxed),
            hosts_healthy: healthy,
            hosts_total: self.inner.config.hosts.len(),
        }
    }
}

// ----------------------------------------------------------------------------
// Pooled connection guard
// ----------------------------------------------------------------------------

/// RAII guard that returns its underlying client to the pool on drop, unless
/// the caller explicitly discards it via `discard`.
pub struct PooledConnection {
    client: Option<PgClient>,
    created_at: Instant,
    host_idx: usize,
    _permit: tokio::sync::OwnedSemaphorePermit,
    pool: Arc<PoolInner>,
}

impl PooledConnection {
    /// Borrows the underlying PgClient for queries.
    pub fn client_mut(&mut self) -> &mut PgClient {
        self.client.as_mut().expect("client already taken")
    }

    /// Drops the connection without returning it to the pool, for instance
    /// after an unrecoverable protocol error.
    pub async fn discard(mut self) {
        if let Some(c) = self.client.take() {
            let _ = c.close().await;
        }
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        self.pool.in_use.fetch_sub(1, Ordering::Relaxed);
        if let Some(client) = self.client.take() {
            // If the connection is too old, drop instead of returning.
            if self.created_at.elapsed().as_secs() >= self.pool.config.max_lifetime_secs {
                tokio::spawn(async move {
                    let _ = client.close().await;
                });
                return;
            }
            let mut idle = self.pool.idle.lock();
            idle.push_back(IdleEntry {
                client,
                host_idx: self.host_idx,
                created_at: self.created_at,
                last_used: Instant::now(),
            });
        }
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_credential_provider() {
        let p = StaticPassword(Some("s3cr3t".into()));
        assert_eq!(p.password("alice").await.as_deref(), Some("s3cr3t"));
    }

    #[test]
    fn test_atomic_health_initial_available() {
        let h = AtomicHealth::new();
        assert!(h.is_available_now());
    }

    #[test]
    fn test_atomic_health_trips_after_threshold() {
        let h = AtomicHealth::new();
        h.record_failure(3, Duration::from_secs(60));
        assert!(h.is_available_now());
        h.record_failure(3, Duration::from_secs(60));
        assert!(h.is_available_now());
        h.record_failure(3, Duration::from_secs(60));
        assert!(!h.is_available_now());
    }

    #[test]
    fn test_atomic_health_success_resets() {
        let h = AtomicHealth::new();
        h.record_failure(1, Duration::from_secs(60));
        assert!(!h.is_available_now());
        h.record_success();
        assert!(h.is_available_now());
    }

    #[test]
    fn test_host_entry_from_uri() {
        let u = UriHost {
            host: "db.example.com".into(),
            port: 5432,
        };
        let e = HostEntry::from_uri(&u, HostRole::Primary);
        assert_eq!(e.host, "db.example.com");
        assert_eq!(e.role, HostRole::Primary);
    }

    #[test]
    fn test_host_entry_addr() {
        let e = HostEntry {
            host: "127.0.0.1".into(),
            port: 9999,
            role: HostRole::Unknown,
            health: AtomicHealth::new(),
        };
        assert_eq!(e.addr().unwrap().port(), 9999);
    }

    #[test]
    fn test_pool_config_simple_defaults() {
        let cfg = PoolConfig::simple("h", 5432, "u", Some("p"), "db");
        assert_eq!(cfg.user, "u");
        assert_eq!(cfg.database, "db");
        assert_eq!(cfg.hosts.len(), 1);
    }

    #[tokio::test]
    async fn test_pool_stats_empty() {
        let cfg = PoolConfig::simple("127.0.0.1", 1, "u", None, "db");
        let p = ConnectionPool::new(cfg);
        let s = p.stats();
        assert_eq!(s.idle, 0);
        assert_eq!(s.in_use, 0);
        assert_eq!(s.hosts_total, 1);
    }

    #[tokio::test]
    async fn test_pool_acquire_fails_bad_host() {
        let cfg = PoolConfig {
            hosts: vec![HostEntry {
                host: "127.0.0.1".into(),
                port: 1, // nothing listens on port 1
                role: HostRole::Unknown,
                health: AtomicHealth::new(),
            }],
            min_idle: 0,
            max_size: 1,
            max_idle_secs: 60,
            max_lifetime_secs: 60,
            connect_timeout: Duration::from_millis(200),
            statement_timeout: Duration::from_secs(1),
            idle_timeout: Duration::from_secs(60),
            keepalive: Duration::from_secs(30),
            tls: None,
            prefer_quic: false,
            database: "db".into(),
            user: "u".into(),
            credential_provider: Arc::new(StaticPassword(None)),
            circuit_breaker_threshold: 1,
            circuit_breaker_cooldown: Duration::from_secs(5),
        };
        let p = ConnectionPool::new(cfg);
        let r = p.acquire().await;
        assert!(r.is_err());
        let s = p.stats();
        assert_eq!(s.total_failed, 1);
        assert_eq!(s.hosts_healthy, 0);
    }

    #[tokio::test]
    async fn test_pool_shutdown_rejects_acquire() {
        let cfg = PoolConfig::simple("127.0.0.1", 1, "u", None, "db");
        let p = ConnectionPool::new(cfg);
        p.shutdown().await;
        let r = p.acquire().await;
        assert!(matches!(r, Err(PoolError::Shutdown)));
    }
}
