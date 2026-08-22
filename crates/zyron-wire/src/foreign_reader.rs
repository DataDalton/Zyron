//! Reaching a peer for a foreign table scan.
//!
//! The executor decides what to ask for and this decides how to ask it.
//! Splitting them that way is what lets a foreign scan be an ordinary
//! operator in a plan, so a foreign table can join a local one, while the
//! connection, the pool and the peer registry stay above the executor
//! where they belong.
//!
//! The request is rendered as SQL the peer would accept from any client,
//! because it is one. No private protocol, no shared build: a Zyron node
//! federates with another the same way a client reads from it.

use std::sync::Arc;

use zyron_common::{ForeignRequest, Result, ZyronError};
use zyron_executor::column::ScalarValue;
use zyron_executor::operator::foreign_scan::ForeignReader;

use crate::pool::{ConnectionPool, PoolConfig};
use crate::types::{text_to_scalar, type_id_to_pg_oid};

/// Connections held open per peer. A foreign scan is a read, so several can
/// be in flight at once against the same node, and the cap is what keeps a
/// query fanning out over a mesh from opening an unbounded number of them.
const MAX_CONNECTIONS_PER_PEER: usize = 8;

/// How long a peer has to answer, both to connect and to run the statement.
const PEER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Reads foreign tables over the wire, resolving peers through the node's
/// own registry so a scan names a peer and never an address.
///
/// Connections are pooled per peer and reused. Connecting per scan would pay
/// a TCP handshake and an authentication round trip before every read, which
/// on a query that scans a foreign table once per outer row is most of the
/// query, so the pool is what makes federation usable rather than a
/// convenience.
pub struct PeerForeignReader {
    peers: Arc<parking_lot::RwLock<Arc<zyron_common::PeerRegistry>>>,
    /// One pool per peer, keyed by the name and the address it was built
    /// for. A peer redeclared at a new address gets a new pool rather than
    /// connections to where it used to be
    pools: scc::HashMap<String, (String, Arc<ConnectionPool>)>,
    user: String,
    database: String,
    runtime: tokio::runtime::Handle,
}

impl PeerForeignReader {
    pub fn new(
        peers: Arc<parking_lot::RwLock<Arc<zyron_common::PeerRegistry>>>,
        user: String,
        database: String,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            peers,
            pools: scc::HashMap::new(),
            user,
            database,
            runtime,
        }
    }

    /// The pool for a peer, built on first use and reused after.
    ///
    /// A peer whose address changed is rebuilt rather than reused, because
    /// the old pool's idle connections point at the previous node and
    /// handing one of those to a scan would read the wrong database.
    fn pool_for(&self, peer: &str, address: &str) -> Result<Arc<ConnectionPool>> {
        if let Some(entry) = self.pools.read_sync(peer, |_, v| v.clone()) {
            if entry.0 == address {
                return Ok(entry.1);
            }
            self.pools.remove_sync(peer);
        }

        let (host, port) = split_address(address)?;
        let mut config = PoolConfig::simple(&host, port, &self.user, None, &self.database);
        config.max_size = MAX_CONNECTIONS_PER_PEER;
        config.connect_timeout = PEER_TIMEOUT;
        config.statement_timeout = PEER_TIMEOUT;
        let pool = Arc::new(ConnectionPool::new(config));

        // A concurrent builder may have won, and its pool is as good as this
        // one, so the loser uses the winner's rather than keeping a second
        // set of connections to the same node
        match self
            .pools
            .insert_sync(peer.to_string(), (address.to_string(), Arc::clone(&pool)))
        {
            Ok(()) => Ok(pool),
            Err((_, existing)) => Ok(existing.1),
        }
    }

    /// Where a peer is, and what it stores.
    ///
    /// A scan against an undeclared peer is a plan built on a peering that
    /// no longer exists, and saying so beats a connection error that sends
    /// an operator looking at the network
    fn resolve(&self, peer: &str) -> Result<String> {
        self.peers
            .read()
            .get(peer)
            .map(|p| p.address.clone())
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "no peer named \"{}\". It was dropped, or this plan was built \
                     against a peering that no longer exists",
                    peer
                ))
            })
    }
}

/// The SQL a peer is asked.
///
/// Identifiers are checked rather than quoted-and-hoped: they come from
/// the local catalog, but they end up inside a statement on another node,
/// and a boundary that trusts its input because of where the input came
/// from is the one that eventually does not.
pub fn render_request(request: &ForeignRequest) -> Result<String> {
    check_identifier(&request.table)?;
    for column in &request.columns {
        check_identifier(column)?;
    }
    if request.columns.is_empty() {
        return Err(ZyronError::ExecutionError(
            "a foreign scan must project at least one column".into(),
        ));
    }
    let mut sql = format!(
        "SELECT {} FROM {}",
        request.columns.join(", "),
        request.table
    );
    if let Some(predicate) = &request.predicate {
        sql.push_str(" WHERE ");
        sql.push_str(predicate);
    }
    if let Some(limit) = request.limit {
        sql.push_str(&format!(" LIMIT {}", limit));
    }
    Ok(sql)
}

/// An unquoted SQL identifier, which is all a table or column name may be
/// here. Anything else is refused rather than escaped
fn check_identifier(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$');
    if ok {
        Ok(())
    } else {
        Err(ZyronError::ExecutionError(format!(
            "\"{}\" is not a name this can ask a peer for",
            name
        )))
    }
}

/// Splits `host:port`, the form a peer address takes.
///
/// The host is left as written rather than resolved here, so the pool
/// re-resolves it per connection. A peer behind a name whose address changes
/// is then reached at its current one instead of wherever it was when the
/// pool was built.
fn split_address(address: &str) -> Result<(String, u16)> {
    let (host, port) = address.rsplit_once(':').ok_or_else(|| {
        ZyronError::ExecutionError(format!(
            "peer address \"{}\" has no port, expected host:port",
            address
        ))
    })?;
    let port: u16 = port.parse().map_err(|_| {
        ZyronError::ExecutionError(format!("peer address \"{}\" has no valid port", address))
    })?;
    if host.is_empty() {
        return Err(ZyronError::ExecutionError(format!(
            "peer address \"{}\" names no host",
            address
        )));
    }
    Ok((host.to_string(), port))
}

impl ForeignReader for PeerForeignReader {
    fn scan(&self, request: &ForeignRequest) -> Result<Vec<Vec<ScalarValue>>> {
        let address = self.resolve(&request.peer)?;
        let sql = render_request(request)?;
        let pool = self.pool_for(&request.peer, &address)?;
        let types = &request.column_types;

        // The operator is synchronous at this point, and blocking a runtime
        // worker on a network read would starve every other query sharing
        // it, so the round trip runs on the blocking pool
        let results = tokio::task::block_in_place(|| {
            self.runtime.block_on(async {
                let mut connection = pool.acquire().await.map_err(|e| {
                    ZyronError::ExecutionError(format!("connecting to {}: {}", address, e))
                })?;
                match connection.client_mut().simple_query(&sql).await {
                    Ok(results) => Ok(results),
                    Err(e) => {
                        // A failed statement may have left the connection
                        // mid-protocol, and returning it to the pool would
                        // hand the next scan a stream out of step
                        connection.discard().await;
                        Err(ZyronError::ExecutionError(format!(
                            "reading from {}: {}",
                            address, e
                        )))
                    }
                }
            })
        })?;

        // Sized from the answer so the row vector allocates once
        let total: usize = results.iter().map(|r| r.rows.len()).sum();
        let mut rows = Vec::with_capacity(total);
        for result in results {
            for mut raw in result.rows {
                if raw.len() < types.len() {
                    return Err(ZyronError::ExecutionError(format!(
                        "peer \"{}\" returned {} columns for \"{}\", the scan asked for {}",
                        request.peer,
                        raw.len(),
                        request.table,
                        types.len()
                    )));
                }
                let mut row = Vec::with_capacity(types.len());
                for (index, type_id) in types.iter().enumerate() {
                    // Taken by value: the cell's bytes are decoded once and
                    // dropped, rather than the whole answer being held in
                    // its wire form beside the decoded copy
                    let value = match raw[index].take() {
                        None => ScalarValue::Null,
                        Some(bytes) => text_to_scalar(&bytes, type_id_to_pg_oid(*type_id))
                            .map_err(|e| {
                                ZyronError::ExecutionError(format!(
                                    "peer returned a value for column {} that is not a {:?}: {}",
                                    index, type_id, e
                                ))
                            })?,
                    };
                    row.push(value);
                }
                rows.push(row);
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::TypeId;

    fn request() -> ForeignRequest {
        ForeignRequest {
            peer: "west".into(),
            table: "orders".into(),
            columns: vec!["id".into(), "total".into()],
            column_types: vec![TypeId::Int64, TypeId::Float64],
            predicate: None,
            limit: None,
        }
    }

    /// The peer is asked for exactly what the plan needs and nothing more
    #[test]
    fn test_a_request_renders_projection_predicate_and_limit() {
        let mut r = request();
        assert_eq!(
            render_request(&r).expect("render"),
            "SELECT id, total FROM orders"
        );

        r.predicate = Some("(id > 100)".into());
        assert_eq!(
            render_request(&r).expect("render"),
            "SELECT id, total FROM orders WHERE (id > 100)"
        );

        r.limit = Some(25);
        assert_eq!(
            render_request(&r).expect("render"),
            "SELECT id, total FROM orders WHERE (id > 100) LIMIT 25"
        );
    }

    /// A name that is not an identifier is refused rather than escaped,
    /// because it ends up inside a statement running on another node
    #[test]
    fn test_names_are_checked_before_they_reach_a_peer() {
        let mut r = request();
        r.table = "orders; DROP TABLE users".into();
        assert!(render_request(&r).is_err());

        let mut r = request();
        r.columns = vec!["id".into(), "total) FROM users --".into()];
        assert!(render_request(&r).is_err());

        let mut r = request();
        r.columns.clear();
        assert!(render_request(&r).is_err(), "a scan asks for something");
    }
}
