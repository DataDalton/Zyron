//! Getting entries onto a majority, and knowing when they got there.
//!
//! ## The consistency check, and why a rejection carries a hint
//!
//! Every AppendEntries names the entry before the ones it carries. A follower
//! that does not hold exactly that entry, at exactly that term, refuses. That
//! single check is what makes the logs converge: accepting the batch would
//! mean accepting a log whose prefix differs from the leader's, and no later
//! message could detect it.
//!
//! Raft's own description has the leader step `next_index` back by one and try
//! again. That is correct and it is O(divergence) round trips, which on a
//! follower that has been away for a while is a lot of round trips. So a
//! rejection carries a hint instead: the highest index the follower could
//! still plausibly match on. The leader takes it directly, and the same
//! consistency check runs again against the new position, so the hint is a
//! shortcut rather than a second source of truth. Each of the three shapes a
//! rejection can take moves the leader somewhere it has not already been
//! refused, which is what makes the search terminate.
//!
//! ## Pipelining
//!
//! The leader does not wait for a reply before sending the next batch. It
//! advances `next_index` when it sends and only trusts `match_index` when a
//! reply confirms it, so a lost or refused batch costs a reset of the guess
//! rather than a wrong commit. Replies may arrive out of order, so
//! `match_index` only ever moves forward.
//!
//! ## The read round
//!
//! Every AppendEntries carries a round number and every reply echoes it. When
//! a majority has echoed round R, the leader knows it still held the group at
//! the moment it sent R, which is what a linearizable read on a follower needs
//! and is why a read costs no extra message on the leader's side.

use std::sync::Arc;

use zyron_common::error::{Result, ZyronError};

use crate::NodeId;
use crate::codec::{Cursor, put_bool, put_u32, put_u64};
use crate::log::RaftLogEntry;

/// Most entries one message may claim, so a corrupt count cannot make a
/// receiver preallocate without bound
pub const MAX_ENTRIES_PER_MESSAGE: usize = 64 * 1024;

/// Entries for a follower, with the check that says where they attach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesRequest {
    pub term: u64,
    pub leader_id: NodeId,
    /// Index of the entry immediately before `entries`
    pub prev_log_index: u64,
    /// Term of that entry. A mismatch is what a follower refuses on
    pub prev_log_term: u64,
    pub entries: Vec<Arc<RaftLogEntry>>,
    /// How far the leader has committed, so the follower can apply
    pub leader_commit: u64,
    /// Stamped so a reply proves the leader held the group when it sent this
    pub read_round: u64,
}

impl AppendEntriesRequest {
    /// Whether this message carries no entries, which is the heartbeat shape.
    ///
    /// A heartbeat is the same message with an empty entry list rather than a
    /// message of its own, so the consistency check and the commit index ride
    /// along with it for free
    pub fn is_heartbeat(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_u64(buf, self.leader_id);
        put_u64(buf, self.prev_log_index);
        put_u64(buf, self.prev_log_term);
        put_u64(buf, self.leader_commit);
        put_u64(buf, self.read_round);
        put_u32(buf, self.entries.len() as u32);
        for entry in &self.entries {
            entry.encode_record(buf);
        }
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        let term = c.u64()?;
        let leader_id = c.u64()?;
        let prev_log_index = c.u64()?;
        let prev_log_term = c.u64()?;
        let leader_commit = c.u64()?;
        let read_round = c.u64()?;
        let count = c.u32()? as usize;
        if count > MAX_ENTRIES_PER_MESSAGE {
            return Err(ZyronError::EncodingFailed(format!(
                "AppendEntries claims {count} entries, the limit is {MAX_ENTRIES_PER_MESSAGE}"
            )));
        }
        let mut entries = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let (entry, used) = RaftLogEntry::decode_record(c.rest())?;
            c.advance(used)?;
            entries.push(Arc::new(entry));
        }
        Ok(Self {
            term,
            leader_id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
            read_round,
        })
    }

    /// Bytes this message occupies once encoded, used by the transport's
    /// inflight accounting
    pub fn encoded_len(&self) -> usize {
        52 + self.entries.iter().map(|e| e.encoded_len()).sum::<usize>()
    }
}

/// What a follower says back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesReply {
    /// The follower's term, which deposes a stale leader
    pub term: u64,
    pub success: bool,
    /// On success, the highest index the follower now holds and has made
    /// durable. Meaningless on failure
    pub match_index: u64,
    /// On failure, the index the follower suggests the leader check next.
    ///
    /// It is the highest index the follower could plausibly still match on:
    /// its own last index when it is simply short, one below the start of the
    /// conflicting term when it holds a different entry, and its snapshot
    /// point when it is ahead of the leader's guess. The leader takes it
    /// directly, which is why a follower a thousand entries behind is caught
    /// up in two messages rather than a thousand
    pub hint_index: u64,
    /// Echoed from the request
    pub read_round: u64,
    pub follower_id: NodeId,
}

impl AppendEntriesReply {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_bool(buf, self.success);
        put_u64(buf, self.match_index);
        put_u64(buf, self.hint_index);
        put_u64(buf, self.read_round);
        put_u64(buf, self.follower_id);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            term: c.u64()?,
            success: c.bool()?,
            match_index: c.u64()?,
            hint_index: c.u64()?,
            read_round: c.u64()?,
            follower_id: c.u64()?,
        })
    }
}

/// A follower asking the leader how far it must have applied before a read of
/// its own local state is linearizable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadIndexRequest {
    /// The asking node's term, so a leader from an older term is not trusted
    pub term: u64,
    pub from: NodeId,
}

impl ReadIndexRequest {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_u64(buf, self.from);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            term: c.u64()?,
            from: c.u64()?,
        })
    }
}

/// The index a follower must reach before answering the read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadIndexReply {
    pub term: u64,
    pub success: bool,
    /// The leader's commit index at the moment it confirmed it still led
    pub read_index: u64,
    /// Who the answering node believes leads, so a misdirected read can be
    /// retried at the right place rather than failing
    pub leader_id: Option<NodeId>,
}

impl ReadIndexReply {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_bool(buf, self.success);
        put_u64(buf, self.read_index);
        crate::codec::put_opt_u64(buf, self.leader_id);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            term: c.u64()?,
            success: c.bool()?,
            read_index: c.u64()?,
            leader_id: c.opt_u64()?,
        })
    }
}

/// Where the leader takes `next_index` after a follower refuses a batch.
///
/// The follower's hint is used directly rather than blended with a
/// one-at-a-time step, because the hint is always an index the follower could
/// still match on and the leader re-runs the consistency check against it
/// anyway. The floor is what the follower has already confirmed: moving
/// `next_index` below a confirmed `match_index` would resend entries the
/// follower is known to hold
#[inline]
pub fn next_index_after_reject(hint_index: u64, match_index: u64) -> u64 {
    hint_index.saturating_add(1).max(match_index + 1).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::RaftCommand;

    fn entry(term: u64, index: u64) -> Arc<RaftLogEntry> {
        Arc::new(RaftLogEntry::new(
            term,
            index,
            RaftCommand::Put {
                key: format!("k{index}").into_bytes(),
                value: vec![9u8; 16],
            },
        ))
    }

    #[test]
    fn append_entries_round_trips_with_a_batch() {
        let req = AppendEntriesRequest {
            term: 7,
            leader_id: 1,
            prev_log_index: 10,
            prev_log_term: 6,
            entries: (11..=20).map(|i| entry(7, i)).collect(),
            leader_commit: 9,
            read_round: 44,
        };
        let mut buf = Vec::new();
        req.encode(&mut buf);
        let back = AppendEntriesRequest::decode(&mut Cursor::new(&buf)).expect("decode");
        assert_eq!(back, req);
        assert_eq!(buf.len(), req.encoded_len());
    }

    #[test]
    fn a_heartbeat_is_the_same_message_without_entries() {
        let req = AppendEntriesRequest {
            term: 7,
            leader_id: 1,
            prev_log_index: 10,
            prev_log_term: 6,
            entries: Vec::new(),
            leader_commit: 10,
            read_round: 3,
        };
        assert!(req.is_heartbeat());
        let mut buf = Vec::new();
        req.encode(&mut buf);
        assert_eq!(buf.len(), 52);
        assert_eq!(
            AppendEntriesRequest::decode(&mut Cursor::new(&buf)).expect("decode"),
            req
        );
    }

    #[test]
    fn replies_round_trip() {
        let reply = AppendEntriesReply {
            term: 8,
            success: false,
            match_index: 0,
            hint_index: 42,
            read_round: 5,
            follower_id: 3,
        };
        let mut buf = Vec::new();
        reply.encode(&mut buf);
        assert_eq!(
            AppendEntriesReply::decode(&mut Cursor::new(&buf)).expect("decode"),
            reply
        );

        let r = ReadIndexReply {
            term: 8,
            success: true,
            read_index: 900,
            leader_id: Some(2),
        };
        let mut buf = Vec::new();
        r.encode(&mut buf);
        assert_eq!(
            ReadIndexReply::decode(&mut Cursor::new(&buf)).expect("decode"),
            r
        );
    }

    #[test]
    fn a_hint_moves_the_guess_straight_to_where_the_follower_is() {
        // A follower a long way behind is caught in one step
        assert_eq!(next_index_after_reject(40, 0), 41);
        // A follower ahead of the guess pulls it forward
        assert_eq!(next_index_after_reject(9000, 0), 9001);
        // And the guess never falls below what the follower confirmed
        assert_eq!(next_index_after_reject(5, 50), 51);
        // Never below one
        assert_eq!(next_index_after_reject(0, 0), 1);
    }

    #[test]
    fn an_absurd_entry_count_is_refused_before_allocating() {
        let mut buf = Vec::new();
        for _ in 0..6 {
            put_u64(&mut buf, 1);
        }
        put_u32(&mut buf, u32::MAX);
        assert!(AppendEntriesRequest::decode(&mut Cursor::new(&buf)).is_err());
    }
}
