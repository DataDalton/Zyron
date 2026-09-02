//! Categories and policy for user-authored object rewrites.
//!
//! The rewriters themselves walk a parsed statement, so they live with the
//! parser. What they are classified as, and what a tenant has said should
//! happen to each class, is decided here, because the upgrade orchestrator
//! and the catalog views both read it without wanting a parser dependency

use std::fmt;

/// A kind of user-authored object a rewriter can target
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectKind {
    View,
    MaterializedView,
    Procedure,
    Function,
    Aggregate,
    Trigger,
    Workflow,
    Pipeline,
    Schedule,
    Dashboard,
    Prompt,
    RagConfig,
    DataQualityRule,
    StreamingJob,
    Endpoint,
}

/// Every object kind a rewriter can target, in catalog order
pub const ALL_OBJECT_KINDS: &[ObjectKind] = &[
    ObjectKind::View,
    ObjectKind::MaterializedView,
    ObjectKind::Procedure,
    ObjectKind::Function,
    ObjectKind::Aggregate,
    ObjectKind::Trigger,
    ObjectKind::Workflow,
    ObjectKind::Pipeline,
    ObjectKind::Schedule,
    ObjectKind::Dashboard,
    ObjectKind::Prompt,
    ObjectKind::RagConfig,
    ObjectKind::DataQualityRule,
    ObjectKind::StreamingJob,
    ObjectKind::Endpoint,
];

impl ObjectKind {
    pub const fn catalog_name(self) -> &'static str {
        match self {
            ObjectKind::View => "view",
            ObjectKind::MaterializedView => "materialized_view",
            ObjectKind::Procedure => "procedure",
            ObjectKind::Function => "function",
            ObjectKind::Aggregate => "aggregate",
            ObjectKind::Trigger => "trigger",
            ObjectKind::Workflow => "workflow",
            ObjectKind::Pipeline => "pipeline",
            ObjectKind::Schedule => "schedule",
            ObjectKind::Dashboard => "dashboard",
            ObjectKind::Prompt => "prompt",
            ObjectKind::RagConfig => "rag_config",
            ObjectKind::DataQualityRule => "data_quality_rule",
            ObjectKind::StreamingJob => "streaming_job",
            ObjectKind::Endpoint => "endpoint",
        }
    }

    pub fn parse(name: &str) -> Option<ObjectKind> {
        ALL_OBJECT_KINDS
            .iter()
            .copied()
            .find(|kind| kind.catalog_name().eq_ignore_ascii_case(name))
    }
}

impl fmt::Display for ObjectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.catalog_name())
    }
}

/// How much judgment a rewrite needs
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RewriteCategory {
    /// A mechanical rename or a syntactic reshuffle that preserves meaning
    Safe,
    /// A change that could alter behavior in narrow cases, such as widening
    /// a signature or adding a defaultable parameter
    Ambiguous,
    /// A removed feature or a behavior change with no defensible automatic
    /// mapping
    Unsafe,
}

impl RewriteCategory {
    pub const fn label(self) -> &'static str {
        match self {
            RewriteCategory::Safe => "safe",
            RewriteCategory::Ambiguous => "ambiguous",
            RewriteCategory::Unsafe => "unsafe",
        }
    }

    pub fn parse(name: &str) -> Option<RewriteCategory> {
        match name.to_ascii_lowercase().as_str() {
            "safe" => Some(RewriteCategory::Safe),
            "ambiguous" => Some(RewriteCategory::Ambiguous),
            "unsafe" => Some(RewriteCategory::Unsafe),
            _ => None,
        }
    }
}

impl fmt::Display for RewriteCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What a tenant has said should happen to each rewrite class
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UserObjectRewritePolicy {
    /// Safe applies on its own, ambiguous needs an acknowledgment, unsafe
    /// blocks the upgrade
    #[default]
    AutoSafe,
    /// Every class needs an acknowledgment, safe included
    NotifyAll,
    /// Nothing applies on its own, every rewrite is done by hand
    ManualOnly,
}

impl UserObjectRewritePolicy {
    pub const fn label(self) -> &'static str {
        match self {
            UserObjectRewritePolicy::AutoSafe => "auto_safe",
            UserObjectRewritePolicy::NotifyAll => "notify_all",
            UserObjectRewritePolicy::ManualOnly => "manual_only",
        }
    }

    pub fn parse(name: &str) -> Option<UserObjectRewritePolicy> {
        match name.to_ascii_lowercase().as_str() {
            "auto_safe" => Some(UserObjectRewritePolicy::AutoSafe),
            "notify_all" => Some(UserObjectRewritePolicy::NotifyAll),
            "manual_only" => Some(UserObjectRewritePolicy::ManualOnly),
            _ => None,
        }
    }

    /// What happens to a rewrite of this category under this policy
    pub const fn disposition(self, category: RewriteCategory) -> RewriteDisposition {
        match (self, category) {
            (UserObjectRewritePolicy::AutoSafe, RewriteCategory::Safe) => {
                RewriteDisposition::AutoApply
            }
            (UserObjectRewritePolicy::AutoSafe, RewriteCategory::Ambiguous) => {
                RewriteDisposition::AwaitAck
            }
            (UserObjectRewritePolicy::AutoSafe, RewriteCategory::Unsafe) => {
                RewriteDisposition::Block
            }
            (UserObjectRewritePolicy::NotifyAll, RewriteCategory::Unsafe) => {
                RewriteDisposition::Block
            }
            (UserObjectRewritePolicy::NotifyAll, _) => RewriteDisposition::AwaitAck,
            (UserObjectRewritePolicy::ManualOnly, _) => RewriteDisposition::Manual,
        }
    }
}

impl fmt::Display for UserObjectRewritePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// What the substrate does with one proposed rewrite
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteDisposition {
    /// Applied without asking
    AutoApply,
    /// Queued until an admin acknowledges it
    AwaitAck,
    /// Blocks the upgrade until the object is rewritten by hand or the
    /// breakage is explicitly accepted
    Block,
    /// Left alone, the operator applies it themselves
    Manual,
}

impl RewriteDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            RewriteDisposition::AutoApply => "auto_apply",
            RewriteDisposition::AwaitAck => "await_ack",
            RewriteDisposition::Block => "block",
            RewriteDisposition::Manual => "manual",
        }
    }

    /// Whether this disposition stops an upgrade from proceeding
    #[inline]
    pub const fn blocks_upgrade(self) -> bool {
        matches!(self, RewriteDisposition::Block)
    }

    /// Whether this disposition needs an admin before the upgrade proceeds
    #[inline]
    pub const fn needs_ack(self) -> bool {
        matches!(self, RewriteDisposition::AwaitAck)
    }
}

impl fmt::Display for RewriteDisposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where one queued rewrite has got to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteStatus {
    Pending,
    Applied,
    Acknowledged,
    Blocked,
    AcceptedBroken,
    Failed,
}

impl RewriteStatus {
    pub const fn label(self) -> &'static str {
        match self {
            RewriteStatus::Pending => "pending",
            RewriteStatus::Applied => "applied",
            RewriteStatus::Acknowledged => "acknowledged",
            RewriteStatus::Blocked => "blocked",
            RewriteStatus::AcceptedBroken => "accepted_broken",
            RewriteStatus::Failed => "failed",
        }
    }
}

impl fmt::Display for RewriteStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// The classification of every object the compat gate looked at
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RewriteClassification {
    pub safe: u64,
    pub ambiguous: u64,
    pub unsafe_count: u64,
    /// Objects no registered rewriter touched
    pub unaffected: u64,
    /// Names of the objects in the unsafe class, which the gate has to print
    pub unsafe_objects: Vec<String>,
    /// Names of the objects in the ambiguous class
    pub ambiguous_objects: Vec<String>,
}

impl RewriteClassification {
    pub fn record(&mut self, object: &str, category: RewriteCategory) {
        match category {
            RewriteCategory::Safe => self.safe += 1,
            RewriteCategory::Ambiguous => {
                self.ambiguous += 1;
                self.ambiguous_objects.push(object.to_string());
            }
            RewriteCategory::Unsafe => {
                self.unsafe_count += 1;
                self.unsafe_objects.push(object.to_string());
            }
        }
    }

    pub fn record_unaffected(&mut self) {
        self.unaffected += 1;
    }

    pub fn total(&self) -> u64 {
        self.safe + self.ambiguous + self.unsafe_count + self.unaffected
    }

    /// Whether this classification stops the upgrade under a policy
    pub fn blocks_upgrade(&self, policy: UserObjectRewritePolicy) -> bool {
        (self.unsafe_count > 0 && policy.disposition(RewriteCategory::Unsafe).blocks_upgrade())
            || (self.safe > 0 && policy.disposition(RewriteCategory::Safe).blocks_upgrade())
            || (self.ambiguous > 0
                && policy
                    .disposition(RewriteCategory::Ambiguous)
                    .blocks_upgrade())
    }

    /// Whether this classification needs an acknowledgment under a policy
    pub fn needs_ack(&self, policy: UserObjectRewritePolicy) -> bool {
        (self.safe > 0 && policy.disposition(RewriteCategory::Safe).needs_ack())
            || (self.ambiguous > 0 && policy.disposition(RewriteCategory::Ambiguous).needs_ack())
            || (self.unsafe_count > 0 && policy.disposition(RewriteCategory::Unsafe).needs_ack())
    }

    /// The line the compat gate prints
    pub fn summary(&self) -> String {
        format!(
            "{} objects examined, {} safe, {} ambiguous, {} unsafe, {} unaffected",
            self.total(),
            self.safe,
            self.ambiguous,
            self.unsafe_count,
            self.unaffected
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_safe_applies_safe_and_blocks_unsafe() {
        let policy = UserObjectRewritePolicy::AutoSafe;
        assert_eq!(
            policy.disposition(RewriteCategory::Safe),
            RewriteDisposition::AutoApply
        );
        assert_eq!(
            policy.disposition(RewriteCategory::Ambiguous),
            RewriteDisposition::AwaitAck
        );
        assert_eq!(
            policy.disposition(RewriteCategory::Unsafe),
            RewriteDisposition::Block
        );
    }

    #[test]
    fn test_notify_all_asks_even_for_safe() {
        let policy = UserObjectRewritePolicy::NotifyAll;
        assert_eq!(
            policy.disposition(RewriteCategory::Safe),
            RewriteDisposition::AwaitAck
        );
        assert_eq!(
            policy.disposition(RewriteCategory::Unsafe),
            RewriteDisposition::Block
        );
    }

    #[test]
    fn test_manual_only_applies_nothing() {
        for category in [
            RewriteCategory::Safe,
            RewriteCategory::Ambiguous,
            RewriteCategory::Unsafe,
        ] {
            assert_eq!(
                UserObjectRewritePolicy::ManualOnly.disposition(category),
                RewriteDisposition::Manual
            );
        }
    }

    #[test]
    fn test_policy_and_category_parse_from_text() {
        assert_eq!(
            UserObjectRewritePolicy::parse("AUTO_SAFE"),
            Some(UserObjectRewritePolicy::AutoSafe)
        );
        assert_eq!(UserObjectRewritePolicy::parse("nope"), None);
        assert_eq!(
            RewriteCategory::parse("Ambiguous"),
            Some(RewriteCategory::Ambiguous)
        );
        assert_eq!(RewriteCategory::parse("nope"), None);
    }

    #[test]
    fn test_classification_summarizes_and_gates() {
        let mut classification = RewriteClassification::default();
        for i in 0..90 {
            classification.record(&format!("v{i}"), RewriteCategory::Safe);
        }
        for i in 0..5 {
            classification.record(&format!("f{i}"), RewriteCategory::Ambiguous);
        }
        for i in 0..5 {
            classification.record(&format!("p{i}"), RewriteCategory::Unsafe);
        }
        assert_eq!(classification.total(), 100);
        assert_eq!(classification.unsafe_objects.len(), 5);
        assert!(classification.summary().contains("90 safe"));
        assert!(classification.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
        assert!(classification.needs_ack(UserObjectRewritePolicy::AutoSafe));

        let mut safe_only = RewriteClassification::default();
        safe_only.record("v", RewriteCategory::Safe);
        safe_only.record_unaffected();
        assert!(!safe_only.blocks_upgrade(UserObjectRewritePolicy::AutoSafe));
        assert!(!safe_only.needs_ack(UserObjectRewritePolicy::AutoSafe));
        assert!(safe_only.needs_ack(UserObjectRewritePolicy::NotifyAll));
        assert_eq!(safe_only.total(), 2);
    }

    #[test]
    fn test_object_kinds_round_trip_their_names() {
        for kind in ALL_OBJECT_KINDS {
            assert_eq!(ObjectKind::parse(kind.catalog_name()), Some(*kind));
        }
        assert_eq!(ObjectKind::parse("nothing"), None);
    }
}
