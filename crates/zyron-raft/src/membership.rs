//! Who is in the group, and what counts as a majority while that is changing.
//!
//! ## Two ways to change a group, and why both exist
//!
//! Adding a learner is safe on its own: a learner receives the log and never
//! votes, so no quorum anywhere moves and a single log entry can carry the
//! change. That is how a new node joins, and it is why a join never risks the
//! group even when the new node is hours behind.
//!
//! Changing the voters is not safe on its own. Between the moment one node
//! adopts a new voter set and the moment another does, the two disagree about
//! what a majority is, and two disjoint majorities can each elect a leader.
//! So a voter change goes through a joint configuration: an entry that names
//! both the old voters and the new ones, during which every decision needs a
//! majority of each. No pair of disjoint majorities exists across that entry,
//! which is what makes the transition safe rather than merely brief. Once the
//! joint entry commits, a second entry names the new voters alone.
//!
//! ## Learners do not count, in either direction
//!
//! A learner is absent from every quorum computation here. It cannot help a
//! commit and it cannot hold one up, which means promoting it is the only
//! moment its state matters, and the leader waits for it to catch up before
//! proposing that promotion.

use crate::NodeId;
use crate::codec::{Cursor, put_bool, put_str, put_u32, put_u64};
use serde::{Deserialize, Serialize};
use zyron_common::error::{Result, ZyronError};

/// Most nodes one configuration may name.
///
/// A consensus group past this size is a design mistake rather than a large
/// deployment: every commit waits on a majority of it. Sharding across groups
/// is how a deployment grows past this
pub const MAX_CLUSTER_NODES: usize = 64;

/// One member of the group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Stable across restarts, minted with the data directory
    pub node_id: NodeId,
    /// Where the node answers consensus RPCs, as `host:port`
    pub address: String,
    /// Counts toward quorum and may grant votes
    pub is_voter: bool,
    /// Receives replication, never votes, never counts toward a quorum
    pub is_learner: bool,
}

impl NodeConfig {
    /// A full member
    pub fn voter(node_id: NodeId, address: impl Into<String>) -> Self {
        Self {
            node_id,
            address: address.into(),
            is_voter: true,
            is_learner: false,
        }
    }

    /// A member that receives the log and is not counted by anyone
    pub fn learner(node_id: NodeId, address: impl Into<String>) -> Self {
        Self {
            node_id,
            address: address.into(),
            is_voter: false,
            is_learner: true,
        }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.node_id);
        put_str(buf, &self.address);
        put_bool(buf, self.is_voter);
        put_bool(buf, self.is_learner);
    }

    fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        let node_id = c.u64()?;
        let address = c.string()?;
        let is_voter = c.bool()?;
        let is_learner = c.bool()?;
        // A node is one or the other. A payload claiming both, or neither,
        // came from a peer that does not agree with this build about what a
        // member is, and guessing which it meant would let the two disagree
        // about a quorum
        if is_voter == is_learner {
            return Err(ZyronError::EncodingFailed(format!(
                "node {node_id} is described as both voter and learner"
            )));
        }
        Ok(Self {
            node_id,
            address,
            is_voter,
            is_learner,
        })
    }
}

/// The membership of one consensus group.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub nodes: Vec<NodeConfig>,
}

impl ClusterConfig {
    pub fn new(nodes: Vec<NodeConfig>) -> Self {
        Self { nodes }
    }

    /// Builds a configuration where every listed node votes
    pub fn of_voters(nodes: impl IntoIterator<Item = (NodeId, String)>) -> Self {
        Self {
            nodes: nodes
                .into_iter()
                .map(|(id, addr)| NodeConfig::voter(id, addr))
                .collect(),
        }
    }

    pub fn get(&self, node_id: NodeId) -> Option<&NodeConfig> {
        self.nodes.iter().find(|n| n.node_id == node_id)
    }

    pub fn contains(&self, node_id: NodeId) -> bool {
        self.get(node_id).is_some()
    }

    pub fn is_voter(&self, node_id: NodeId) -> bool {
        self.get(node_id).map(|n| n.is_voter).unwrap_or(false)
    }

    pub fn voter_ids(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.is_voter)
            .map(|n| n.node_id)
            .collect()
    }

    pub fn voter_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_voter).count()
    }

    /// How many voters have to agree for a decision to hold
    pub fn quorum(&self) -> usize {
        self.voter_count() / 2 + 1
    }

    pub fn address_of(&self, node_id: NodeId) -> Option<&str> {
        self.get(node_id).map(|n| n.address.as_str())
    }

    /// Adds a node, replacing any entry with the same id
    pub fn with_node(&self, node: NodeConfig) -> Self {
        let mut nodes: Vec<NodeConfig> = self
            .nodes
            .iter()
            .filter(|n| n.node_id != node.node_id)
            .cloned()
            .collect();
        nodes.push(node);
        nodes.sort_by_key(|n| n.node_id);
        Self { nodes }
    }

    pub fn without_node(&self, node_id: NodeId) -> Self {
        Self {
            nodes: self
                .nodes
                .iter()
                .filter(|n| n.node_id != node_id)
                .cloned()
                .collect(),
        }
    }

    /// Turns a learner into a voter, leaving everything else alone
    pub fn promoted(&self, node_id: NodeId) -> Self {
        Self {
            nodes: self
                .nodes
                .iter()
                .map(|n| {
                    if n.node_id == node_id {
                        NodeConfig::voter(n.node_id, n.address.clone())
                    } else {
                        n.clone()
                    }
                })
                .collect(),
        }
    }

    /// Refuses a configuration that cannot make progress or cannot be framed.
    ///
    /// Checked where a configuration enters the node rather than where it is
    /// used, so a group never reaches a state where no quorum is reachable
    pub fn validate(&self) -> Result<()> {
        if self.nodes.len() > MAX_CLUSTER_NODES {
            return Err(ZyronError::Internal(format!(
                "cluster configuration names {} nodes, the limit is {MAX_CLUSTER_NODES}",
                self.nodes.len()
            )));
        }
        if self.voter_count() == 0 {
            return Err(ZyronError::Internal(
                "cluster configuration has no voters, so no decision could ever commit".into(),
            ));
        }
        for (i, node) in self.nodes.iter().enumerate() {
            if node.is_voter == node.is_learner {
                return Err(ZyronError::Internal(format!(
                    "node {} is described as both voter and learner",
                    node.node_id
                )));
            }
            if node.address.is_empty() {
                return Err(ZyronError::Internal(format!(
                    "node {} has no address",
                    node.node_id
                )));
            }
            if self.nodes[..i].iter().any(|p| p.node_id == node.node_id) {
                return Err(ZyronError::Internal(format!(
                    "node {} appears twice in the configuration",
                    node.node_id
                )));
            }
        }
        Ok(())
    }

    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u32(buf, self.nodes.len() as u32);
        for node in &self.nodes {
            node.encode(buf);
        }
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        let count = c.u32()? as usize;
        if count > MAX_CLUSTER_NODES {
            return Err(ZyronError::EncodingFailed(format!(
                "cluster configuration claims {count} nodes, the limit is {MAX_CLUSTER_NODES}"
            )));
        }
        let mut nodes = Vec::with_capacity(count);
        for _ in 0..count {
            nodes.push(NodeConfig::decode(c)?);
        }
        Ok(Self { nodes })
    }

    pub fn encoded(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.nodes.len() * 32);
        self.encode(&mut buf);
        buf
    }
}

/// The configuration a node is deciding under.
///
/// `Joint` is the transitional state of a voter change: both configurations
/// are in force, and every quorum has to be satisfied in both
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Membership {
    Simple(ClusterConfig),
    Joint {
        old: ClusterConfig,
        new: ClusterConfig,
    },
}

impl Default for Membership {
    fn default() -> Self {
        Membership::Simple(ClusterConfig::default())
    }
}

impl Membership {
    pub fn simple(config: ClusterConfig) -> Self {
        Membership::Simple(config)
    }

    pub fn is_joint(&self) -> bool {
        matches!(self, Membership::Joint { .. })
    }

    /// The configuration to report and to reach nodes through.
    ///
    /// During a joint change this is the incoming one, because it is the
    /// superset that names every node the leader has to talk to
    pub fn effective(&self) -> &ClusterConfig {
        match self {
            Membership::Simple(c) => c,
            Membership::Joint { new, .. } => new,
        }
    }

    /// Whether this node is a voter under any configuration in force
    pub fn is_voter(&self, node_id: NodeId) -> bool {
        match self {
            Membership::Simple(c) => c.is_voter(node_id),
            Membership::Joint { old, new } => old.is_voter(node_id) || new.is_voter(node_id),
        }
    }

    pub fn contains(&self, node_id: NodeId) -> bool {
        match self {
            Membership::Simple(c) => c.contains(node_id),
            Membership::Joint { old, new } => old.contains(node_id) || new.contains(node_id),
        }
    }

    /// Every node this one replicates to, learners included, without itself.
    ///
    /// Sorted so the replication loop walks peers in a stable order and a
    /// test comparing two nodes' views compares equal lists
    pub fn peers(&self, self_id: NodeId) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = match self {
            Membership::Simple(c) => c.nodes.iter().map(|n| n.node_id).collect(),
            Membership::Joint { old, new } => {
                let mut v: Vec<NodeId> = old.nodes.iter().map(|n| n.node_id).collect();
                for n in &new.nodes {
                    if !v.contains(&n.node_id) {
                        v.push(n.node_id);
                    }
                }
                v
            }
        };
        ids.retain(|id| *id != self_id);
        ids.sort_unstable();
        ids
    }

    /// Where a node answers, taking the incoming configuration first because
    /// a joint change is what moves an address
    pub fn address_of(&self, node_id: NodeId) -> Option<&str> {
        match self {
            Membership::Simple(c) => c.address_of(node_id),
            Membership::Joint { old, new } => {
                new.address_of(node_id).or_else(|| old.address_of(node_id))
            }
        }
    }

    /// Whether the nodes `granted` names satisfy every configuration in force.
    ///
    /// Used for elections and for the leader's own quorum check. Under a joint
    /// configuration both majorities are required, which is the property that
    /// stops two leaders existing across a voter change
    pub fn has_quorum(&self, granted: impl Fn(NodeId) -> bool) -> bool {
        match self {
            Membership::Simple(c) => config_quorum(c, &granted),
            Membership::Joint { old, new } => {
                config_quorum(old, &granted) && config_quorum(new, &granted)
            }
        }
    }

    /// The highest log index replicated on a majority of every configuration
    /// in force.
    ///
    /// `match_of` reports what each voter is known to hold, the leader
    /// included. Sorting the voters' match indexes and taking the one at the
    /// midpoint gives the largest index a majority has reached
    pub fn quorum_match_index(&self, match_of: impl Fn(NodeId) -> u64) -> u64 {
        match self {
            Membership::Simple(c) => config_quorum_index(c, &match_of),
            Membership::Joint { old, new } => {
                config_quorum_index(old, &match_of).min(config_quorum_index(new, &match_of))
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Membership::Simple(c) => c.validate(),
            Membership::Joint { old, new } => {
                old.validate()?;
                new.validate()
            }
        }
    }
}

fn config_quorum(config: &ClusterConfig, granted: &impl Fn(NodeId) -> bool) -> bool {
    let voters = config.voter_count();
    if voters == 0 {
        return false;
    }
    let yes = config
        .nodes
        .iter()
        .filter(|n| n.is_voter && granted(n.node_id))
        .count();
    yes >= voters / 2 + 1
}

fn config_quorum_index(config: &ClusterConfig, match_of: &impl Fn(NodeId) -> u64) -> u64 {
    // Runs on every append reply and every commit recomputation, so the
    // voters' match indexes go on the stack. The array always fits because
    // every path a configuration enters through refuses more than
    // MAX_CLUSTER_NODES members
    let mut indexes = [0u64; MAX_CLUSTER_NODES];
    let mut count = 0usize;
    for node in config.nodes.iter().filter(|n| n.is_voter) {
        if count == indexes.len() {
            break;
        }
        indexes[count] = match_of(node.node_id);
        count += 1;
    }
    if count == 0 {
        return 0;
    }
    // Descending, then the element at position quorum minus one is the
    // highest index that a majority, half the voters plus one, has reached.
    // For an even count the midpoint would be one position too high, an
    // index only half the voters hold, which a group of two or four would
    // commit on a minority
    let indexes = &mut indexes[..count];
    indexes.sort_unstable_by(|a, b| b.cmp(a));
    indexes[count / 2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three() -> ClusterConfig {
        ClusterConfig::of_voters([
            (1, "a:1".to_string()),
            (2, "b:1".to_string()),
            (3, "c:1".to_string()),
        ])
    }

    #[test]
    fn quorum_of_three_is_two() {
        let m = Membership::simple(three());
        assert!(m.has_quorum(|id| id == 1 || id == 2));
        assert!(!m.has_quorum(|id| id == 1));
    }

    #[test]
    fn quorum_of_four_is_three() {
        let mut c = three();
        c.nodes.push(NodeConfig::voter(4, "d:1"));
        let m = Membership::simple(c);
        assert!(!m.has_quorum(|id| id == 1 || id == 2));
        assert!(m.has_quorum(|id| id <= 3));
    }

    #[test]
    fn learners_are_absent_from_every_quorum() {
        let mut c = three();
        c.nodes.push(NodeConfig::learner(4, "d:1"));
        let m = Membership::simple(c);
        // The learner agreeing adds nothing
        assert!(!m.has_quorum(|id| id == 1 || id == 4));
        // And a learner far ahead does not raise the commit index
        assert_eq!(m.quorum_match_index(|id| if id == 4 { 900 } else { 5 }), 5);
    }

    #[test]
    fn joint_requires_both_majorities() {
        let old = three();
        let new = ClusterConfig::of_voters([
            (3, "c:1".to_string()),
            (4, "d:1".to_string()),
            (5, "e:1".to_string()),
        ]);
        let m = Membership::Joint { old, new };
        // A majority of the old set alone is not enough
        assert!(!m.has_quorum(|id| id == 1 || id == 2));
        // Nor a majority of the new set alone
        assert!(!m.has_quorum(|id| id == 4 || id == 5));
        // Both together commit
        assert!(m.has_quorum(|id| matches!(id, 1 | 2 | 3 | 4)));
    }

    #[test]
    fn joint_commit_index_is_the_lower_of_the_two() {
        let old = three();
        let new = ClusterConfig::of_voters([
            (3, "c:1".to_string()),
            (4, "d:1".to_string()),
            (5, "e:1".to_string()),
        ]);
        let m = Membership::Joint { old, new };
        let idx = m.quorum_match_index(|id| match id {
            1 => 10,
            2 => 10,
            3 => 10,
            4 => 4,
            5 => 4,
            _ => 0,
        });
        // Old set has 10 on a majority, new set only 4
        assert_eq!(idx, 4);
    }

    #[test]
    fn quorum_index_picks_the_majority_watermark() {
        let m = Membership::simple(three());
        assert_eq!(
            m.quorum_match_index(|id| match id {
                1 => 9,
                2 => 7,
                3 => 2,
                _ => 0,
            }),
            7
        );
    }

    /// An even number of voters needs more than half of them. Two voters
    /// need both, four need three, and the watermark is what the last of
    /// those holds, never what the half ahead of it holds
    #[test]
    fn quorum_index_of_an_even_group_needs_more_than_half() {
        let two = Membership::simple(ClusterConfig::of_voters([
            (1, "a:1".to_string()),
            (2, "b:1".to_string()),
        ]));
        assert_eq!(two.quorum_match_index(|id| if id == 1 { 50 } else { 3 }), 3);

        let mut c = three();
        c.nodes.push(NodeConfig::voter(4, "d:1"));
        let four = Membership::simple(c);
        assert_eq!(
            four.quorum_match_index(|id| match id {
                1 => 10,
                2 => 10,
                3 => 4,
                4 => 1,
                _ => 0,
            }),
            4
        );
        assert_eq!(
            four.quorum_match_index(|id| match id {
                1 => 10,
                2 => 10,
                3 => 10,
                4 => 1,
                _ => 0,
            }),
            10
        );

        for count in 1..=MAX_CLUSTER_NODES as u64 {
            let m = Membership::simple(ClusterConfig::of_voters(
                (1..=count).map(|id| (id, format!("n{id}:1"))),
            ));
            // Voter `id` holds index `id`, so the highest index a majority
            // holds is the one at the quorum's lowest member
            let quorum = count / 2 + 1;
            assert_eq!(
                m.quorum_match_index(|id| id),
                count + 1 - quorum,
                "{count} voters"
            );
        }
    }

    #[test]
    fn config_round_trips_through_the_codec() {
        let mut c = three();
        c.nodes.push(NodeConfig::learner(9, "z:99"));
        let bytes = c.encoded();
        let mut cur = Cursor::new(&bytes);
        assert_eq!(ClusterConfig::decode(&mut cur).expect("decode"), c);
    }

    #[test]
    fn a_config_with_no_voters_is_refused() {
        let c = ClusterConfig::new(vec![NodeConfig::learner(1, "a:1")]);
        assert!(c.validate().is_err());
    }

    #[test]
    fn peers_omit_self_and_are_stable() {
        let m = Membership::simple(three());
        assert_eq!(m.peers(2), vec![1, 3]);
    }
}
