//! First contact with a declared peer.
//!
//! A peer is declared by an operator, who says where it is and may say
//! what it stores. Neither of those is authoritative about the peer: an
//! operator can mistype an address or be wrong about a node's mode, and a
//! mesh that acts on a belief it never checked will route work to a node
//! that cannot do it.
//!
//! So the peer is asked. It answers out of its own `zyron_nodes` view,
//! which reports the local node for certain, and what comes back replaces
//! what was assumed. Until then the mesh view says the peer's id and mode
//! are unknown rather than guessing, because an unknown that admits it is
//! unknown is safe and a wrong guess is not.
//!
//! A probe never blocks the declaration. `CREATE PEER` against a node that
//! is down still succeeds, records why the contact failed, and leaves the
//! peer to be reached later: a mesh that refuses to remember a peer until
//! it is up cannot be configured before it is running.

use std::net::ToSocketAddrs;
use std::time::Duration;

use zyron_common::{DeploymentMode, ZyronError};

use crate::pg_client::{ClientConfig, PgClient};

/// What a peer said about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerFacts {
    pub node_id: u64,
    pub name: String,
    pub mode: DeploymentMode,
}

/// How long a probe waits before deciding a peer is unreachable.
///
/// Short, because a probe runs on the DDL path and an operator declaring a
/// peer should not wait on a node that is down. The declaration succeeds
/// either way, so the only cost of giving up early is that the facts fill
/// in on a later attempt.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Asks a peer who it is.
///
/// One query against the peer's own mesh view, which is the smallest thing
/// that answers both questions a router needs: which node this is, so two
/// names for one node are recognizable, and what it stores, so work is not
/// sent to a node that cannot do it.
pub async fn probe_peer(
    address: &str,
    user: &str,
    database: &str,
) -> Result<PeerFacts, ZyronError> {
    let socket = resolve(address)?;
    let config = ClientConfig {
        user: user.to_string(),
        database: database.to_string(),
        application_name: "zyron-peer-probe".to_string(),
        password: None,
        connect_timeout: PROBE_TIMEOUT,
        statement_timeout: PROBE_TIMEOUT,
    };
    let mut client = PgClient::connect(socket, &config)
        .await
        .map_err(|e| ZyronError::ConfigError(format!("connecting to {}: {}", address, e)))?;
    let results = client
        .simple_query("SELECT node_name, node_id, mode FROM zyron_nodes WHERE is_local = 't'")
        .await
        .map_err(|e| ZyronError::ConfigError(format!("querying {}: {}", address, e)))?;

    let row = results.iter().find_map(|q| q.rows.first()).ok_or_else(|| {
        ZyronError::ConfigError(format!(
            "{} answered with no local node. It is reachable but is not a Zyron node, \
                 or it is one that never established an identity",
            address
        ))
    })?;
    let text = |index: usize| -> Option<String> {
        row.get(index)
            .and_then(|c| c.as_ref())
            .and_then(|b| std::str::from_utf8(b).ok())
            .map(|s| s.to_string())
    };
    let name = text(0).unwrap_or_default();
    let node_id = text(1)
        .and_then(|s| u64::from_str_radix(&s, 16).ok())
        .filter(|id| *id != 0)
        .ok_or_else(|| {
            ZyronError::ConfigError(format!(
                "{} reported no node id, so it has no identity to be known by",
                address
            ))
        })?;
    let mode = text(2)
        .and_then(|s| DeploymentMode::parse(&s))
        .ok_or_else(|| {
            ZyronError::ConfigError(format!("{} reported an unreadable mode", address))
        })?;
    Ok(PeerFacts {
        node_id,
        name,
        mode,
    })
}

/// Reads a leader's published log over the wire.
///
/// The leader exposes its version files through `zyron_lake_log`, so a
/// follower fetches metadata and nothing else: the entries name immutable
/// data files, and where storage is shared the follower already has them.
///
/// `limit` bounds one fetch so a follower far behind catches up in steps
/// rather than one long stall holding a connection open.
pub async fn fetch_remote_versions(
    address: &str,
    user: &str,
    database: &str,
    table: &str,
    from: u64,
    limit: usize,
) -> Result<Vec<zyron_lake::FollowedVersion>, ZyronError> {
    // The leader is asked for a table by name, and a name is not a place to
    // accept arbitrary text: it reaches the peer inside a query
    if !table
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    {
        return Err(ZyronError::ConfigError(format!(
            "\"{}\" is not a table name this can ask a peer for",
            table
        )));
    }
    let socket = resolve(address)?;
    let config = ClientConfig {
        user: user.to_string(),
        database: database.to_string(),
        application_name: "zyron-follower".to_string(),
        password: None,
        connect_timeout: PROBE_TIMEOUT,
        statement_timeout: PROBE_TIMEOUT,
    };
    let mut client = PgClient::connect(socket, &config)
        .await
        .map_err(|e| ZyronError::ConfigError(format!("connecting to {}: {}", address, e)))?;
    let sql = format!(
        "SELECT version, payload FROM zyron_lake_log \
         WHERE table_name = '{}' AND from_version = {} LIMIT {}",
        table, from, limit
    );
    let results = client
        .simple_query(&sql)
        .await
        .map_err(|e| ZyronError::ConfigError(format!("reading log from {}: {}", address, e)))?;

    let mut rows: Vec<(u64, String)> = Vec::new();
    for result in &results {
        for row in &result.rows {
            let text = |index: usize| -> Option<&str> {
                row.get(index)
                    .and_then(|c| c.as_ref())
                    .and_then(|b| std::str::from_utf8(b).ok())
            };
            let (Some(version), Some(payload)) = (text(0), text(1)) else {
                continue;
            };
            let Ok(version) = version.parse::<u64>() else {
                continue;
            };
            rows.push((version, payload.to_string()));
        }
    }
    rows.sort_by_key(|(version, _)| *version);
    zyron_lake::decode_log_rows(from, &rows)
}

/// Resolves a `host:port` address, reporting a name that does not resolve
/// as a configuration error rather than a connection failure, because they
/// call for different fixes
fn resolve(address: &str) -> Result<std::net::SocketAddr, ZyronError> {
    address
        .to_socket_addrs()
        .map_err(|e| ZyronError::ConfigError(format!("resolving \"{}\": {}", address, e)))?
        .next()
        .ok_or_else(|| ZyronError::ConfigError(format!("\"{}\" resolves to no address", address)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An address that cannot be resolved is a configuration mistake, and
    /// saying so beats reporting a connection failure the operator would
    /// then go looking for on the wire
    #[tokio::test]
    async fn test_an_unresolvable_address_is_a_configuration_error() {
        let err = probe_peer("no-such-host.invalid:5433", "zyron", "zyron")
            .await
            .expect_err("must not resolve");
        let message = err.to_string();
        assert!(message.contains("no-such-host.invalid"), "{message}");
    }

    /// A port nothing listens on is unreachable, and the peer stays
    /// declared so a later probe can find it
    #[tokio::test]
    async fn test_an_unreachable_peer_reports_why() {
        // Port 1 on loopback refuses immediately on every platform this
        // runs on, so the test measures the error path rather than a timeout
        let err = probe_peer("127.0.0.1:1", "zyron", "zyron")
            .await
            .expect_err("nothing listens there");
        let message = err.to_string();
        assert!(message.contains("127.0.0.1:1"), "{message}");
    }

    #[test]
    fn test_facts_carry_what_a_router_needs() {
        let facts = PeerFacts {
            node_id: 0x1234,
            name: "west".into(),
            mode: DeploymentMode::Lake,
        };
        // Which node, so two names for one node are recognizable, and what
        // it stores, so work is not sent where it cannot be done
        assert_ne!(facts.node_id, 0);
        assert!(facts.mode.runs_lake_tier());
        assert!(!facts.name.is_empty());
    }
}
