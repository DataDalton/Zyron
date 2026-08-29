//! Choosing a leader, and not choosing one more often than necessary.
//!
//! ## Randomized timers
//!
//! Followers that all give up on a leader at the same instant all campaign at
//! the same instant, split the vote, and repeat. The fix is that each follower
//! draws its own timeout from a range, so one of them is first by a margin
//! wide enough to collect a majority before the next one starts. The draw
//! comes from a per-node generator seeded at startup rather than from a shared
//! source, so two nodes that boot together still diverge.
//!
//! Deadlines are [`std::time::Instant`], which is monotonic. A system clock
//! stepped backwards by an operator or by NTP would otherwise freeze every
//! election timer in the group at once.
//!
//! ## Pre-vote, and what it is actually for
//!
//! A node cut off from the group times out, raises its term, and campaigns.
//! Nobody answers, so it times out again and raises its term again. When the
//! partition heals it rejoins carrying a term far above everyone else's, and
//! the sitting leader, which is perfectly healthy, is deposed by a node that
//! has been talking to nothing for a minute. The group then holds an election
//! it did not need.
//!
//! Pre-vote asks the question first: a candidate sends a vote request stamped
//! with the term it *would* use, without raising its own. Peers answer whether
//! they would grant it, using the same up-to-date test as a real vote, and
//! additionally refuse while they can still hear a leader. A node with no
//! peers gets no answers, so its term never moves, and rejoining costs the
//! group nothing.

use std::time::{Duration, Instant};

use zyron_common::error::Result;
use zyron_common::prng::{Xoshiro256pp, splitMix64};

use crate::NodeId;
use crate::codec::{Cursor, put_bool, put_u64};

/// A randomized deadline for hearing from a leader.
pub struct ElectionTimer {
    rng: Xoshiro256pp,
    min: Duration,
    max: Duration,
    current: Duration,
    deadline: Instant,
}

impl ElectionTimer {
    /// Builds a timer whose draws are this node's alone.
    ///
    /// The seed folds the node id into a startup value so that two nodes
    /// launched in the same millisecond still draw different timeouts
    pub fn new(node_id: NodeId, min: Duration, max: Duration, now: Instant) -> Self {
        let mut seed = node_id
            ^ std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15);
        let s = splitMix64(&mut seed);
        let mut timer = Self {
            rng: Xoshiro256pp::fromSeed(s),
            min,
            max,
            current: min,
            deadline: now,
        };
        timer.reset(now);
        timer
    }

    /// Draws a new timeout and restarts the countdown
    pub fn reset(&mut self, now: Instant) {
        self.current = self.draw();
        self.deadline = now + self.current;
    }

    fn draw(&mut self) -> Duration {
        let lo = self.min.as_nanos() as u64;
        let hi = self.max.as_nanos() as u64;
        if hi <= lo {
            return self.min;
        }
        let span = hi - lo + 1;
        Duration::from_nanos(lo + self.rng.nextU64() % span)
    }

    #[inline]
    pub fn expired(&self, now: Instant) -> bool {
        now >= self.deadline
    }

    #[inline]
    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    /// The timeout currently drawn, reported by the metrics view
    #[inline]
    pub fn current_timeout(&self) -> Duration {
        self.current
    }
}

/// Asks a peer for its vote, or asks whether it would give one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteRequest {
    /// The term being campaigned for. Under pre-vote this is one past the
    /// candidate's own term and the candidate has not adopted it
    pub term: u64,
    pub candidate_id: NodeId,
    pub last_log_index: u64,
    pub last_log_term: u64,
    /// True while the candidate is only asking whether it could win
    pub pre_vote: bool,
}

impl RequestVoteRequest {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_u64(buf, self.candidate_id);
        put_u64(buf, self.last_log_index);
        put_u64(buf, self.last_log_term);
        put_bool(buf, self.pre_vote);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            term: c.u64()?,
            candidate_id: c.u64()?,
            last_log_index: c.u64()?,
            last_log_term: c.u64()?,
            pre_vote: c.bool()?,
        })
    }
}

/// The answer to one vote request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteReply {
    /// The voter's term, so a candidate behind the group learns it and stands
    /// down. Under pre-vote this is the voter's real term, which the candidate
    /// uses to catch up without having campaigned
    pub term: u64,
    pub vote_granted: bool,
    /// Echoed so a pre-vote answer is never counted as a real vote, even if it
    /// arrives after the candidate has moved on to the real election
    pub pre_vote: bool,
    /// Who answered, so the candidate can tally without tracking request ids
    pub voter_id: NodeId,
}

impl RequestVoteReply {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_bool(buf, self.vote_granted);
        put_bool(buf, self.pre_vote);
        put_u64(buf, self.voter_id);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            term: c.u64()?,
            vote_granted: c.bool()?,
            pre_vote: c.bool()?,
            voter_id: c.u64()?,
        })
    }
}

/// Whether a candidate's log is at least as complete as the voter's.
///
/// The later term wins outright, because a longer log from an older term can
/// hold entries that were never committed. Only when the terms match does
/// length decide. This is the test that keeps a node missing committed entries
/// from ever being elected, and it is the whole of Raft's leader completeness
/// property
#[inline]
pub fn log_is_up_to_date(
    candidate_last_term: u64,
    candidate_last_index: u64,
    voter_last_term: u64,
    voter_last_index: u64,
) -> bool {
    match candidate_last_term.cmp(&voter_last_term) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => candidate_last_index >= voter_last_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_stay_inside_the_range_and_vary() {
        let now = Instant::now();
        let min = Duration::from_millis(150);
        let max = Duration::from_millis(300);
        let mut timer = ElectionTimer::new(42, min, max, now);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            timer.reset(now);
            let d = timer.current_timeout();
            assert!(d >= min && d <= max, "{d:?} outside the range");
            seen.insert(d.as_micros());
        }
        assert!(seen.len() > 100, "draws are not varying: {}", seen.len());
    }

    #[test]
    fn two_nodes_do_not_draw_the_same_sequence() {
        let now = Instant::now();
        let min = Duration::from_millis(150);
        let max = Duration::from_millis(300);
        let mut a = ElectionTimer::new(1, min, max, now);
        let mut b = ElectionTimer::new(2, min, max, now);
        let mut same = 0;
        for _ in 0..64 {
            a.reset(now);
            b.reset(now);
            if a.current_timeout() == b.current_timeout() {
                same += 1;
            }
        }
        assert!(same < 8, "two nodes drew the same timeout {same} times");
    }

    #[test]
    fn expiry_follows_the_deadline() {
        let now = Instant::now();
        let timer =
            ElectionTimer::new(1, Duration::from_millis(10), Duration::from_millis(10), now);
        assert!(!timer.expired(now));
        assert!(timer.expired(now + Duration::from_millis(11)));
    }

    #[test]
    fn a_later_term_beats_a_longer_log() {
        // Candidate has fewer entries but from a later term
        assert!(log_is_up_to_date(5, 2, 4, 900));
        // And a longer log from an older term loses
        assert!(!log_is_up_to_date(4, 900, 5, 2));
    }

    #[test]
    fn equal_terms_compare_by_length() {
        assert!(log_is_up_to_date(4, 10, 4, 10));
        assert!(log_is_up_to_date(4, 11, 4, 10));
        assert!(!log_is_up_to_date(4, 9, 4, 10));
    }

    #[test]
    fn vote_messages_round_trip() {
        let req = RequestVoteRequest {
            term: 9,
            candidate_id: 3,
            last_log_index: 40,
            last_log_term: 8,
            pre_vote: true,
        };
        let mut buf = Vec::new();
        req.encode(&mut buf);
        assert_eq!(
            RequestVoteRequest::decode(&mut Cursor::new(&buf)).expect("decode"),
            req
        );

        let reply = RequestVoteReply {
            term: 9,
            vote_granted: true,
            pre_vote: true,
            voter_id: 2,
        };
        let mut buf = Vec::new();
        reply.encode(&mut buf);
        assert_eq!(
            RequestVoteReply::decode(&mut Cursor::new(&buf)).expect("decode"),
            reply
        );
    }
}
