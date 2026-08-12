//! Node identity.
//!
//! A node in a mesh has to be recognizable across restarts, or every
//! restart looks like a new peer joining and the peers it was talking to
//! have no way to tell that its data is the same data. So the identity is
//! minted once, on first start, and persisted beside the data it belongs
//! to: move the data directory and the identity moves with it, which is
//! the behaviour an operator expects because the identity is a property of
//! the dataset, not of the process or the host.
//!
//! The file is small and fixed width, with a CRC, because it is read once
//! at startup and a corrupt one has to be reported rather than guessed at.
//! Regenerating an id that peers already know would silently fork the
//! mesh's view of who owns what.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::checksum::{hash32, hash64};
use crate::config::DeploymentMode;
use crate::error::{Result, ZyronError};

const IDENTITY_MAGIC: [u8; 8] = *b"ZYNODEID";
const IDENTITY_VERSION: u32 = 1;
/// magic 8, version 4, crc 4, node_id 8, created_us 8, name length 2,
/// then the name
const IDENTITY_HEADER_LEN: usize = 34;
/// A name longer than this is a configuration mistake rather than a name
const MAX_NODE_NAME: usize = 255;

/// File name under the data directory.
pub const IDENTITY_FILE: &str = "node_identity";

/// Who this node is, and what it is prepared to store.
///
/// The default is a node with no identity: id zero, which is reserved for
/// "no node", and an empty name. It exists for a harness that assembles a
/// server without a data directory to mint one in. A running server always
/// establishes a real identity at startup, so an id of zero anywhere in a
/// mesh means the node never did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeIdentity {
    /// Stable across restarts, minted once from entropy. Peers key their
    /// view of this node's datasets by it
    pub node_id: u64,
    /// Operator-facing name, changeable. Two nodes may share a name by
    /// mistake, which is why it is not the identity
    pub name: String,
    /// Microseconds since the epoch when the identity was minted
    pub created_us: i64,
    /// What this node stores, which decides what a peer can ask of it
    pub mode: DeploymentMode,
}

impl NodeIdentity {
    /// Reads the node's identity, minting one on first start.
    ///
    /// The mode comes from the running configuration rather than the file:
    /// an operator who changes a node from `db` to `unified` has changed
    /// what it stores, not who it is, and the id has to survive that.
    pub fn load_or_create(data_dir: &Path, name: &str, mode: DeploymentMode) -> Result<Self> {
        let path = identity_path(data_dir);
        match fs::read(&path) {
            Ok(bytes) => {
                let mut identity = Self::decode(&bytes, &path)?;
                identity.mode = mode;
                // A rename is an operator decision, so it is honored and
                // persisted rather than silently ignored on every start
                if identity.name != name {
                    identity.name = name.to_string();
                    identity.persist(data_dir)?;
                }
                Ok(identity)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let identity = Self {
                    node_id: mint_node_id(name),
                    name: name.to_string(),
                    created_us: now_us(),
                    mode,
                };
                identity.persist(data_dir)?;
                Ok(identity)
            }
            Err(e) => Err(ZyronError::IoError(format!(
                "reading {}: {}",
                path.display(),
                e
            ))),
        }
    }

    /// Writes the identity durably. Staged and renamed, so a crash leaves
    /// either the old identity or the new one and never a torn file that
    /// would read as corruption on the next start
    pub fn persist(&self, data_dir: &Path) -> Result<()> {
        if self.name.len() > MAX_NODE_NAME {
            return Err(ZyronError::ConfigError(format!(
                "node name is {} bytes, the limit is {}",
                self.name.len(),
                MAX_NODE_NAME
            )));
        }
        fs::create_dir_all(data_dir)?;
        let path = identity_path(data_dir);
        let temp = path.with_extension("tmp");
        let bytes = self.encode();
        {
            let mut file = fs::File::create(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &path)?;
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let name = self.name.as_bytes();
        let mut buf = Vec::with_capacity(IDENTITY_HEADER_LEN + name.len());
        buf.extend_from_slice(&IDENTITY_MAGIC);
        buf.extend_from_slice(&IDENTITY_VERSION.to_le_bytes());
        // CRC placeholder, filled once the rest is in place
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&self.node_id.to_le_bytes());
        buf.extend_from_slice(&self.created_us.to_le_bytes());
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name);
        let crc = identity_crc(&buf);
        buf[12..16].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn decode(bytes: &[u8], path: &Path) -> Result<Self> {
        let corrupt = |reason: &str| {
            ZyronError::ConfigError(format!(
                "node identity at {} is unusable: {}. Refusing to mint a new one, \
                 because peers already know this node by its current id",
                path.display(),
                reason
            ))
        };
        if bytes.len() < IDENTITY_HEADER_LEN {
            return Err(corrupt("file is shorter than its header"));
        }
        if bytes[0..8] != IDENTITY_MAGIC {
            return Err(corrupt("magic mismatch"));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != IDENTITY_VERSION {
            return Err(corrupt(&format!("unsupported version {}", version)));
        }
        let stored = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if stored != identity_crc(bytes) {
            return Err(corrupt("checksum mismatch"));
        }
        let node_id = u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| corrupt("truncated node id"))?,
        );
        let created_us = i64::from_le_bytes(
            bytes[24..32]
                .try_into()
                .map_err(|_| corrupt("truncated timestamp"))?,
        );
        let name_len = u16::from_le_bytes([bytes[32], bytes[33]]) as usize;
        let end = IDENTITY_HEADER_LEN + name_len;
        if bytes.len() < end {
            return Err(corrupt("name is truncated"));
        }
        let name = std::str::from_utf8(&bytes[IDENTITY_HEADER_LEN..end])
            .map_err(|_| corrupt("name is not valid UTF-8"))?
            .to_string();
        Ok(Self {
            node_id,
            name,
            created_us,
            // Overwritten by the caller from the running configuration
            mode: DeploymentMode::Unified,
        })
    }
}

/// Checksum over everything but the checksum field itself, using the
/// crate's own hash family so the identity file needs no extra dependency
fn identity_crc(buf: &[u8]) -> u32 {
    let mut covered = Vec::with_capacity(buf.len() - 4);
    covered.extend_from_slice(&buf[0..12]);
    covered.extend_from_slice(&buf[16..]);
    hash32(&covered)
}

pub fn identity_path(data_dir: &Path) -> PathBuf {
    data_dir.join(IDENTITY_FILE)
}

/// A node id from entropy, never zero.
///
/// Zero is reserved for "no node", so a field holding it is unset rather
/// than pointing at a real node that happened to draw it.
///
/// The seed comes from the operating system rather than the clock and the
/// name. Two nodes brought up together by the same orchestrator share both
/// of those, down to the microsecond, and would draw the same id, which is
/// the one thing a node id may never do.
fn mint_node_id(name: &str) -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    loop {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(now_us() as u64);
        hasher.write_u64(hash64(name.as_bytes()));
        let mut seed = hasher.finish();
        let candidate = crate::prng::splitMix64(&mut seed);
        if candidate != 0 {
            return candidate;
        }
    }
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// One peer this node has been told to talk to.
///
/// Two kinds of field here, and the distinction matters. `address` and the
/// declared `mode` are what the operator SAID. `node_id`, `observed_mode`
/// and `last_seen_us` are what the peer itself said when it was reached,
/// and those are the authority: an operator can be wrong about what a peer
/// stores, the peer cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerEntry {
    /// Operator-chosen name, unique on this node
    pub name: String,
    /// Where to reach it
    pub address: String,
    /// What the operator says it stores. `None` when they did not say
    pub mode: Option<DeploymentMode>,
    /// Microseconds since the epoch when the peer was declared
    pub added_us: i64,
    /// The peer's own id, learned on contact. `None` until then, which is
    /// why the mesh view reports it as unknown rather than guessing
    pub node_id: Option<u64>,
    /// What the peer said it stores, which overrides what was declared
    pub observed_mode: Option<DeploymentMode>,
    /// Microseconds since the epoch of the last successful contact, zero
    /// when never reached
    pub last_seen_us: i64,
    /// Why the last contact attempt failed, cleared by a success. Kept so
    /// an operator sees why a peer is unreachable rather than only that it
    /// is
    pub last_error: Option<String>,
}

impl PeerEntry {
    /// A newly declared peer, not yet contacted.
    pub fn declared(
        name: String,
        address: String,
        mode: Option<DeploymentMode>,
        added_us: i64,
    ) -> Self {
        Self {
            name,
            address,
            mode,
            added_us,
            node_id: None,
            observed_mode: None,
            last_seen_us: 0,
            last_error: None,
        }
    }

    /// What the peer stores, preferring what it said over what was declared
    pub fn effective_mode(&self) -> Option<DeploymentMode> {
        self.observed_mode.or(self.mode)
    }

    /// True when this peer has ever been reached
    pub fn is_known(&self) -> bool {
        self.node_id.is_some()
    }

    /// Records a successful contact, which is the only thing that may set
    /// the observed facts
    pub fn observed(&mut self, node_id: u64, mode: DeploymentMode, at_us: i64) {
        self.node_id = Some(node_id);
        self.observed_mode = Some(mode);
        self.last_seen_us = at_us;
        self.last_error = None;
    }

    /// Records a failed contact. What was learned before is kept: a peer
    /// that is unreachable now is still the peer it was, and forgetting its
    /// id would make a transient outage look like a different node
    pub fn unreachable(&mut self, reason: String) {
        self.last_error = Some(reason);
    }
}

/// File name under the data directory.
pub const PEERS_FILE: &str = "peers";

const PEERS_MAGIC: [u8; 8] = *b"ZYPEERS\0";
const PEERS_VERSION: u32 = 1;

/// The peers this node has been told about.
///
/// Node-local on purpose. Peer membership is not schema: it is this node's
/// view of who it talks to, and putting it in the catalog would replicate
/// one node's view to every other node that shares the database. It also
/// has to be readable before the catalog is up, because a node needs to
/// know its peers as part of starting rather than after.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerRegistry {
    peers: Vec<PeerEntry>,
}

impl PeerRegistry {
    /// Loads the registry, returning an empty one when the node has never
    /// been peered.
    pub fn load(data_dir: &Path) -> Result<Self> {
        let path = peers_path(data_dir);
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => {
                return Err(ZyronError::IoError(format!(
                    "reading {}: {}",
                    path.display(),
                    e
                )));
            }
        };
        Self::decode(&bytes, &path)
    }

    pub fn peers(&self) -> &[PeerEntry] {
        &self.peers
    }

    pub fn get(&self, name: &str) -> Option<&PeerEntry> {
        self.peers.iter().find(|p| p.name == name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut PeerEntry> {
        self.peers.iter_mut().find(|p| p.name == name)
    }

    /// Declares a peer. Returns false when one of that name already exists,
    /// so the caller can honor `IF NOT EXISTS` rather than silently
    /// replacing an address an operator set deliberately
    pub fn add(&mut self, entry: PeerEntry) -> bool {
        if self.get(&entry.name).is_some() {
            return false;
        }
        self.peers.push(entry);
        self.peers.sort_by(|a, b| a.name.cmp(&b.name));
        true
    }

    /// Removes a peer, returning whether it was there.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.peers.len();
        self.peers.retain(|p| p.name != name);
        self.peers.len() != before
    }

    /// Writes the registry durably, staged and renamed so a crash leaves
    /// the previous membership rather than a partial one
    pub fn persist(&self, data_dir: &Path) -> Result<()> {
        fs::create_dir_all(data_dir)?;
        let path = peers_path(data_dir);
        let temp = path.with_extension("tmp");
        let bytes = self.encode();
        {
            let mut file = fs::File::create(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp, &path)?;
        Ok(())
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64 + self.peers.len() * 64);
        buf.extend_from_slice(&PEERS_MAGIC);
        buf.extend_from_slice(&PEERS_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(self.peers.len() as u32).to_le_bytes());
        for peer in &self.peers {
            write_short_string(&mut buf, &peer.name);
            write_short_string(&mut buf, &peer.address);
            buf.push(mode_code(peer.mode));
            buf.extend_from_slice(&peer.added_us.to_le_bytes());
            buf.extend_from_slice(&peer.node_id.unwrap_or(0).to_le_bytes());
            buf.push(mode_code(peer.observed_mode));
            buf.extend_from_slice(&peer.last_seen_us.to_le_bytes());
            write_short_string(&mut buf, peer.last_error.as_deref().unwrap_or(""));
        }
        let crc = peers_crc(&buf);
        buf[12..16].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn decode(bytes: &[u8], path: &Path) -> Result<Self> {
        let corrupt = |reason: &str| {
            ZyronError::ConfigError(format!(
                "peer registry at {} is unusable: {}. Refusing to start with an \
                 empty membership, because a node that forgets its peers looks \
                 to them like a node that left",
                path.display(),
                reason
            ))
        };
        if bytes.len() < 20 {
            return Err(corrupt("file is shorter than its header"));
        }
        if bytes[0..8] != PEERS_MAGIC {
            return Err(corrupt("magic mismatch"));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != PEERS_VERSION {
            return Err(corrupt(&format!("unsupported version {}", version)));
        }
        let stored = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if stored != peers_crc(bytes) {
            return Err(corrupt("checksum mismatch"));
        }
        let count = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let mut offset = 20usize;
        let mut peers = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let name = read_short_string(bytes, &mut offset).ok_or_else(|| corrupt("peer name"))?;
            let address =
                read_short_string(bytes, &mut offset).ok_or_else(|| corrupt("peer address"))?;
            let mode = read_mode(bytes, &mut offset).ok_or_else(|| corrupt("peer mode"))?;
            let added_us = read_i64(bytes, &mut offset).ok_or_else(|| corrupt("peer timestamp"))?;
            let node_id =
                read_i64(bytes, &mut offset).ok_or_else(|| corrupt("peer node id"))? as u64;
            let observed_mode =
                read_mode(bytes, &mut offset).ok_or_else(|| corrupt("peer observed mode"))?;
            let last_seen_us =
                read_i64(bytes, &mut offset).ok_or_else(|| corrupt("peer last seen"))?;
            let last_error =
                read_short_string(bytes, &mut offset).ok_or_else(|| corrupt("peer error"))?;
            peers.push(PeerEntry {
                name,
                address,
                mode,
                added_us,
                node_id: (node_id != 0).then_some(node_id),
                observed_mode,
                last_seen_us,
                last_error: (!last_error.is_empty()).then_some(last_error),
            });
        }
        Ok(Self { peers })
    }
}

fn peers_crc(buf: &[u8]) -> u32 {
    let mut covered = Vec::with_capacity(buf.len() - 4);
    covered.extend_from_slice(&buf[0..12]);
    covered.extend_from_slice(&buf[16..]);
    hash32(&covered)
}

pub fn peers_path(data_dir: &Path) -> PathBuf {
    data_dir.join(PEERS_FILE)
}

fn mode_code(mode: Option<DeploymentMode>) -> u8 {
    match mode {
        None => 0,
        Some(DeploymentMode::Db) => 1,
        Some(DeploymentMode::Lake) => 2,
        Some(DeploymentMode::Unified) => 3,
    }
}

fn read_mode(bytes: &[u8], offset: &mut usize) -> Option<Option<DeploymentMode>> {
    let code = *bytes.get(*offset)?;
    *offset += 1;
    Some(match code {
        0 => None,
        1 => Some(DeploymentMode::Db),
        2 => Some(DeploymentMode::Lake),
        3 => Some(DeploymentMode::Unified),
        _ => return None,
    })
}

fn read_i64(bytes: &[u8], offset: &mut usize) -> Option<i64> {
    let end = offset.checked_add(8)?;
    if end > bytes.len() {
        return None;
    }
    let v = i64::from_le_bytes(bytes[*offset..end].try_into().ok()?);
    *offset = end;
    Some(v)
}

fn write_short_string(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(u16::MAX as usize);
    buf.extend_from_slice(&(len as u16).to_le_bytes());
    buf.extend_from_slice(&bytes[..len]);
}

fn read_short_string(bytes: &[u8], offset: &mut usize) -> Option<String> {
    if *offset + 2 > bytes.len() {
        return None;
    }
    let len = u16::from_le_bytes([bytes[*offset], bytes[*offset + 1]]) as usize;
    *offset += 2;
    let end = offset.checked_add(len)?;
    if end > bytes.len() {
        return None;
    }
    let s = std::str::from_utf8(&bytes[*offset..end]).ok()?.to_string();
    *offset = end;
    Some(s)
}

/// Microseconds since the epoch, for a caller stamping a peer entry.
pub fn peer_timestamp_us() -> i64 {
    now_us()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_survives_restart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first =
            NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Lake).expect("mint");
        assert_ne!(first.node_id, 0, "zero means no node");
        assert_eq!(first.name, "alpha");
        assert_eq!(first.mode, DeploymentMode::Lake);

        let second =
            NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Lake).expect("load");
        assert_eq!(
            second.node_id, first.node_id,
            "a restart is not a new node joining the mesh"
        );
        assert_eq!(second.created_us, first.created_us);
    }

    /// Changing what a node stores does not change who it is
    #[test]
    fn test_a_mode_change_keeps_the_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first =
            NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Db).expect("mint");
        let after = NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Unified)
            .expect("load");
        assert_eq!(after.node_id, first.node_id);
        assert_eq!(after.mode, DeploymentMode::Unified);
    }

    /// A rename is an operator decision, so it sticks
    #[test]
    fn test_a_rename_is_honored_and_persisted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let first =
            NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Lake).expect("mint");
        let renamed =
            NodeIdentity::load_or_create(dir.path(), "beta", DeploymentMode::Lake).expect("rename");
        assert_eq!(renamed.node_id, first.node_id);
        assert_eq!(renamed.name, "beta");
        let reread =
            NodeIdentity::load_or_create(dir.path(), "beta", DeploymentMode::Lake).expect("reread");
        assert_eq!(reread.name, "beta");
    }

    /// Minting a fresh id over a corrupt one would silently fork the mesh's
    /// view of who owns this node's data, so it is refused loudly instead
    #[test]
    fn test_a_corrupt_identity_is_refused_rather_than_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let original =
            NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Lake).expect("mint");

        let path = identity_path(dir.path());
        let mut bytes = fs::read(&path).expect("read");
        bytes[20] ^= 0xFF;
        fs::write(&path, &bytes).expect("corrupt");
        let err = NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Lake)
            .expect_err("a corrupt identity must not be silently replaced");
        let message = err.to_string();
        assert!(message.contains("checksum"), "{message}");

        // The file is left as it was, so an operator can restore it
        assert_eq!(fs::read(&path).expect("read"), bytes);

        // And a truncated file is equally refused
        fs::write(&path, &bytes[..10]).expect("truncate");
        assert!(NodeIdentity::load_or_create(dir.path(), "alpha", DeploymentMode::Lake).is_err());
        let _ = original;
    }

    /// Two nodes started at once with the same name must not collide
    #[test]
    fn test_ids_are_distinct_across_nodes() {
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..1000 {
            assert!(seen.insert(mint_node_id("same-name")), "id collision");
        }
    }

    #[test]
    fn test_a_long_name_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let long = "n".repeat(MAX_NODE_NAME + 1);
        assert!(NodeIdentity::load_or_create(dir.path(), &long, DeploymentMode::Lake).is_err());
    }

    #[test]
    fn test_peers_round_trip_and_stay_sorted() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(
            PeerRegistry::load(dir.path())
                .expect("empty")
                .peers()
                .is_empty()
        );

        let mut registry = PeerRegistry::default();
        assert!(registry.add(PeerEntry::declared(
            "west".into(),
            "10.0.0.2:5433".into(),
            Some(DeploymentMode::Lake),
            100,
        )));
        assert!(registry.add(PeerEntry::declared(
            "east".into(),
            "10.0.0.3:5433".into(),
            None,
            200,
        )));
        // A second declaration of the same name does not silently replace
        // an address the operator set on purpose
        assert!(!registry.add(PeerEntry::declared(
            "west".into(),
            "somewhere-else:5433".into(),
            None,
            300,
        )));
        registry.persist(dir.path()).expect("persist");

        let loaded = PeerRegistry::load(dir.path()).expect("load");
        assert_eq!(loaded, registry);
        let names: Vec<&str> = loaded.peers().iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["east", "west"], "peers list in a stable order");
        assert_eq!(
            loaded.get("west").and_then(|p| p.mode),
            Some(DeploymentMode::Lake)
        );
        assert_eq!(loaded.get("east").expect("east").mode, None);
        assert_eq!(loaded.get("east").expect("east").address, "10.0.0.3:5433");
    }

    #[test]
    fn test_removing_a_peer_persists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut registry = PeerRegistry::default();
        registry.add(PeerEntry::declared("west".into(), "a:1".into(), None, 1));
        registry.persist(dir.path()).expect("persist");
        assert!(registry.remove("west"));
        assert!(!registry.remove("west"), "removing twice reports honestly");
        registry.persist(dir.path()).expect("persist");
        assert!(
            PeerRegistry::load(dir.path())
                .expect("load")
                .peers()
                .is_empty()
        );
    }

    /// A node that silently forgot its peers would look to them like a node
    /// that left the mesh, so a damaged registry is reported instead
    #[test]
    fn test_a_corrupt_peer_registry_is_refused_rather_than_emptied() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut registry = PeerRegistry::default();
        registry.add(PeerEntry::declared(
            "west".into(),
            "a:1".into(),
            Some(DeploymentMode::Db),
            1,
        ));
        registry.persist(dir.path()).expect("persist");

        let path = peers_path(dir.path());
        let mut bytes = fs::read(&path).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&path, &bytes).expect("corrupt");
        let err = PeerRegistry::load(dir.path()).expect_err("must be refused");
        assert!(err.to_string().contains("checksum"), "{err}");
    }
}
