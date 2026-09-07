//! What the parts of a node that cannot call each other leave for the
//! upgrade service.
//!
//! The DDL surface, the mesh handlers, and the server's run loop all run on
//! their own tasks. Each writes an intent here and wakes the service, which
//! carries it out on its next pass, and the run loop waits here for the one
//! intent that ends the process: a restart

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::stager::StagedRelease;

/// How long a drain is honored when the caller named no deadline
pub const DEFAULT_PEER_DRAIN_SECS: u64 = 300;

/// Added to the caller's own deadline before this node serves again, so the
/// node outlives the wait the coordinator is doing and a restart that
/// arrives on time is never cut short
pub const PEER_DRAIN_MARGIN: Duration = Duration::from_secs(30);

/// A restart the service armed, read by the run loop after it has drained
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartIntent {
    /// Restart on the binary activated for this version
    Upgrade { version: String },
    /// Restart on the binary that ran before
    Rollback { version: String },
}

impl RestartIntent {
    pub fn version(&self) -> &str {
        match self {
            RestartIntent::Upgrade { version } | RestartIntent::Rollback { version } => version,
        }
    }
}

/// A request an operator made through the DDL surface
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualRequest {
    pub version: String,
    pub actor: String,
}

/// A restart another node's coordinator asked this node for over the mesh
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinatedRestart {
    /// Activate the staged binary for this version and restart on it
    IntoStaged { version: String },
    /// Put the previous binary back and restart on it
    ToPrevious,
}

#[derive(Default)]
struct Intents {
    /// Versions the mesh asked this node to stage
    stage: Vec<String>,
    manual_upgrade: Option<ManualRequest>,
    manual_rollback: Option<String>,
    coordinated: Option<CoordinatedRestart>,
    restart: Option<RestartIntent>,
    staged: Option<StagedRelease>,
    /// Why the last staging of each version failed, read by the coordinator
    /// through the node's status detail
    stage_failures: HashMap<String, String>,
    /// Upgrade settings waiting to reach the replicated log, config key and
    /// value, oldest first
    cluster_settings: Vec<(String, String)>,
}

/// The intents one node holds
pub struct NodeControl {
    intents: parking_lot::Mutex<Intents>,
    /// Wakes the service
    wake: tokio::sync::Notify,
    /// Wakes the run loop for a restart
    restart_wake: tokio::sync::Notify,
    started_at: Instant,
    /// When a drain another node asked for stops being honored.
    ///
    /// A drain is entered on a peer's word and left on the same peer's
    /// word, so a coordinator that stops talking would otherwise hold this
    /// node out of service for as long as the process lives. The deadline
    /// is the one the caller itself named, refreshed every time that caller
    /// asks how far the drain has got, so a slow coordinator keeps the node
    /// drained and a dead one does not
    peer_drain_deadline: parking_lot::Mutex<Option<Instant>>,
}

impl Default for NodeControl {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeControl {
    pub fn new() -> Self {
        Self {
            intents: parking_lot::Mutex::new(Intents::default()),
            wake: tokio::sync::Notify::new(),
            restart_wake: tokio::sync::Notify::new(),
            started_at: Instant::now(),
            peer_drain_deadline: parking_lot::Mutex::new(None),
        }
    }

    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// Seconds this process has been running
    pub fn uptime_secs(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }

    /// Records how long a peer's drain is honored for.
    ///
    /// A caller that names no deadline gets [`DEFAULT_PEER_DRAIN_SECS`],
    /// because a drain no one ever lifts takes the node out of service for
    /// good. The margin is added so the node outlives the coordinator's own
    /// wait and a restart that arrives on time is never cut short
    pub fn note_peer_drain(&self, deadline_ms: u32, now: Instant) {
        let asked = if deadline_ms == 0 {
            Duration::from_secs(DEFAULT_PEER_DRAIN_SECS)
        } else {
            Duration::from_millis(u64::from(deadline_ms))
        };
        *self.peer_drain_deadline.lock() = Some(now + asked + PEER_DRAIN_MARGIN);
    }

    /// Whether a drain a peer asked for has outlived its deadline
    pub fn peer_drain_expired(&self, now: Instant) -> bool {
        self.peer_drain_deadline
            .lock()
            .is_some_and(|deadline| now >= deadline)
    }

    /// Forgets the deadline, for a drain that ended the way it should
    pub fn clear_peer_drain(&self) {
        *self.peer_drain_deadline.lock() = None;
    }

    /// Wakes the service so it runs a pass now
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    /// Waits for a wake or the given time, whichever comes first
    pub async fn wait_wake(&self, at_most: Duration) {
        let _ = tokio::time::timeout(at_most, self.wake.notified()).await;
    }

    /// Asks this node to stage a release. Returns false when that version
    /// is already staged or already asked for
    pub fn request_stage(&self, version: &str) -> bool {
        let mut intents = self.intents.lock();
        if intents
            .staged
            .as_ref()
            .is_some_and(|staged| staged.version == version)
        {
            return false;
        }
        if intents.stage.iter().any(|v| v == version) {
            return false;
        }
        intents.stage_failures.remove(version);
        intents.stage.push(version.to_string());
        drop(intents);
        self.wake();
        true
    }

    /// Takes every staging request made since the last call
    pub fn take_stage_requests(&self) -> Vec<String> {
        std::mem::take(&mut self.intents.lock().stage)
    }

    /// Asks for one upgrade setting to reach the replicated log. The service
    /// appends it when this node leads and hands it to the leader otherwise.
    /// A newer value for a key replaces one still waiting
    pub fn request_cluster_setting(&self, key: &str, value: &str) {
        let mut intents = self.intents.lock();
        intents
            .cluster_settings
            .retain(|(waiting, _)| waiting != key);
        intents
            .cluster_settings
            .push((key.to_string(), value.to_string()));
        drop(intents);
        self.wake();
    }

    /// Takes every cluster setting request made since the last call
    pub fn take_cluster_settings(&self) -> Vec<(String, String)> {
        std::mem::take(&mut self.intents.lock().cluster_settings)
    }

    /// Puts back requests the log could not take yet, ahead of any made
    /// since, unless a newer value for the same key arrived meanwhile
    pub fn requeue_cluster_settings(&self, waiting: Vec<(String, String)>) {
        let mut intents = self.intents.lock();
        let mut merged: Vec<(String, String)> = waiting
            .into_iter()
            .filter(|(key, _)| !intents.cluster_settings.iter().any(|(k, _)| k == key))
            .collect();
        merged.append(&mut intents.cluster_settings);
        intents.cluster_settings = merged;
    }

    /// Records why staging a version failed
    pub fn record_stage_failure(&self, version: &str, reason: String) {
        self.intents
            .lock()
            .stage_failures
            .insert(version.to_string(), reason);
    }

    /// Why the last staging of a version failed, if it did
    pub fn stage_failure(&self, version: &str) -> Option<String> {
        self.intents.lock().stage_failures.get(version).cloned()
    }

    /// Records what is staged now
    pub fn set_staged(&self, staged: Option<StagedRelease>) {
        self.intents.lock().staged = staged;
    }

    pub fn staged(&self) -> Option<StagedRelease> {
        self.intents.lock().staged.clone()
    }

    pub fn staged_version(&self) -> Option<String> {
        self.intents
            .lock()
            .staged
            .as_ref()
            .map(|staged| staged.version.clone())
    }

    /// Records an operator's request to upgrade. Refused while one is
    /// already waiting, so two operators do not race for the same pass
    pub fn request_manual_upgrade(&self, request: ManualRequest) -> Result<(), String> {
        let mut intents = self.intents.lock();
        if let Some(pending) = &intents.manual_upgrade {
            return Err(format!(
                "an upgrade to {} requested by {} is already waiting for the next pass",
                pending.version, pending.actor
            ));
        }
        intents.manual_upgrade = Some(request);
        drop(intents);
        self.wake();
        Ok(())
    }

    pub fn take_manual_upgrade(&self) -> Option<ManualRequest> {
        self.intents.lock().manual_upgrade.take()
    }

    /// Records an operator's request to roll back
    pub fn request_manual_rollback(&self, actor: &str) -> Result<(), String> {
        let mut intents = self.intents.lock();
        if intents.manual_rollback.is_some() {
            return Err("a rollback is already waiting for the next pass".to_string());
        }
        intents.manual_rollback = Some(actor.to_string());
        drop(intents);
        self.wake();
        Ok(())
    }

    pub fn take_manual_rollback(&self) -> Option<String> {
        self.intents.lock().manual_rollback.take()
    }

    /// Records a restart a coordinator asked for over the mesh. Refused
    /// while one is already waiting or a restart is already armed, and
    /// refused for a version this node has not staged
    pub fn request_coordinated(&self, request: CoordinatedRestart) -> Result<(), String> {
        let mut intents = self.intents.lock();
        if intents.restart.is_some() {
            return Err("this node is already restarting".to_string());
        }
        if let Some(pending) = &intents.coordinated {
            return Err(format!("a restart is already waiting, {pending:?}"));
        }
        if let CoordinatedRestart::IntoStaged { version } = &request {
            let staged = intents.staged.as_ref().map(|s| s.version.clone());
            if staged.as_deref() != Some(version.as_str()) {
                return Err(match staged {
                    Some(staged) => format!("this node has {staged} staged, not {version}"),
                    None => format!("no release {version} is staged here"),
                });
            }
        }
        intents.coordinated = Some(request);
        drop(intents);
        self.wake();
        Ok(())
    }

    pub fn take_coordinated(&self) -> Option<CoordinatedRestart> {
        self.intents.lock().coordinated.take()
    }

    /// Arms a restart. The run loop drains the node and exits with it.
    /// Returns false when one is already armed
    pub fn arm_restart(&self, intent: RestartIntent) -> bool {
        let mut intents = self.intents.lock();
        if intents.restart.is_some() {
            return false;
        }
        intents.restart = Some(intent);
        drop(intents);
        self.restart_wake.notify_one();
        true
    }

    pub fn restart_intent(&self) -> Option<RestartIntent> {
        self.intents.lock().restart.clone()
    }

    /// Waits until a restart is armed
    pub async fn wait_restart(&self) {
        loop {
            if self.intents.lock().restart.is_some() {
                return;
            }
            self.restart_wake.notified().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_setting_requests_keep_the_newest_value_per_key() {
        let control = NodeControl::new();
        control.request_cluster_setting("upgrade.paused", "true");
        control.request_cluster_setting("upgrade.channel", "beta");
        control.request_cluster_setting("upgrade.paused", "false");
        let taken = control.take_cluster_settings();
        assert_eq!(
            taken,
            vec![
                ("upgrade.channel".to_string(), "beta".to_string()),
                ("upgrade.paused".to_string(), "false".to_string()),
            ]
        );
        assert!(control.take_cluster_settings().is_empty());

        // A request that could not reach the log goes back ahead of newer
        // ones, unless its key was asked for again meanwhile
        control.request_cluster_setting("upgrade.paused", "true");
        control.requeue_cluster_settings(taken);
        assert_eq!(
            control.take_cluster_settings(),
            vec![
                ("upgrade.channel".to_string(), "beta".to_string()),
                ("upgrade.paused".to_string(), "true".to_string()),
            ]
        );
    }

    #[test]
    fn test_a_stage_request_is_taken_once_and_not_repeated_for_a_staged_version() {
        let control = NodeControl::new();
        assert!(control.request_stage("0.13.0"));
        assert!(!control.request_stage("0.13.0"), "already asked");
        assert_eq!(control.take_stage_requests(), vec!["0.13.0"]);
        assert!(control.take_stage_requests().is_empty());
        control.set_staged(Some(StagedRelease {
            version: "0.13.0".into(),
            path: std::path::PathBuf::from("staging/zyron-server-0.13.0"),
            sha256: String::new(),
            signature_scheme: "Ed25519".into(),
            size_bytes: 1,
        }));
        assert!(!control.request_stage("0.13.0"), "already staged");
        assert_eq!(control.staged_version().as_deref(), Some("0.13.0"));
    }

    #[test]
    fn test_a_coordinated_restart_needs_the_version_staged() {
        let control = NodeControl::new();
        let err = control
            .request_coordinated(CoordinatedRestart::IntoStaged {
                version: "0.13.0".into(),
            })
            .expect_err("nothing staged");
        assert!(err.contains("no release 0.13.0 is staged"), "{err}");
        control.set_staged(Some(StagedRelease {
            version: "0.13.0".into(),
            path: std::path::PathBuf::from("staging/zyron-server-0.13.0"),
            sha256: String::new(),
            signature_scheme: "Ed25519".into(),
            size_bytes: 1,
        }));
        let err = control
            .request_coordinated(CoordinatedRestart::IntoStaged {
                version: "0.14.0".into(),
            })
            .expect_err("another version is staged");
        assert!(err.contains("has 0.13.0 staged"), "{err}");
        control
            .request_coordinated(CoordinatedRestart::IntoStaged {
                version: "0.13.0".into(),
            })
            .expect("staged");
        let err = control
            .request_coordinated(CoordinatedRestart::ToPrevious)
            .expect_err("one waits at a time");
        assert!(err.contains("already waiting"), "{err}");
        assert_eq!(
            control.take_coordinated(),
            Some(CoordinatedRestart::IntoStaged {
                version: "0.13.0".into()
            })
        );
    }

    #[test]
    fn test_one_manual_request_waits_at_a_time() {
        let control = NodeControl::new();
        control
            .request_manual_upgrade(ManualRequest {
                version: "0.13.0".into(),
                actor: "ana".into(),
            })
            .expect("first");
        let err = control
            .request_manual_upgrade(ManualRequest {
                version: "0.14.0".into(),
                actor: "bo".into(),
            })
            .expect_err("second waits");
        assert!(err.contains("ana"), "{err}");
        assert_eq!(
            control.take_manual_upgrade().map(|r| r.version),
            Some("0.13.0".into())
        );
        assert!(control.take_manual_upgrade().is_none());
    }

    #[tokio::test]
    async fn test_arming_a_restart_wakes_the_waiter_once() {
        let control = Arc::new(NodeControl::new());
        let waiter = {
            let control = Arc::clone(&control);
            tokio::spawn(async move { control.wait_restart().await })
        };
        assert!(control.arm_restart(RestartIntent::Upgrade {
            version: "0.13.0".into()
        }));
        assert!(
            !control.arm_restart(RestartIntent::Rollback {
                version: "0.12.0".into()
            }),
            "a second restart cannot be armed over the first"
        );
        tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("woken")
            .expect("joined");
        assert_eq!(
            control.restart_intent().map(|i| i.version().to_string()),
            Some("0.13.0".into())
        );
    }

    /// A drain a peer asked for lapses on the deadline that peer named, so
    /// a coordinator that stopped between draining a node and restarting it
    /// cannot hold that node out of service for the life of the process.
    /// Asking how far the drain has got is what buys it more time
    #[test]
    fn test_a_peer_drain_lapses_on_the_deadline_the_caller_named() {
        let control = NodeControl::new();
        let at = Instant::now();
        assert!(
            !control.peer_drain_expired(at),
            "nothing has asked this node to drain"
        );

        control.note_peer_drain(30_000, at);
        assert!(!control.peer_drain_expired(at + Duration::from_secs(30)));
        assert!(
            !control.peer_drain_expired(
                at + Duration::from_secs(30) + PEER_DRAIN_MARGIN - Duration::from_millis(1)
            ),
            "the margin outlives the wait the caller itself is doing"
        );
        assert!(control.peer_drain_expired(at + Duration::from_secs(30) + PEER_DRAIN_MARGIN));

        // A later poll from a caller that is still there moves it out again
        let polled = at + Duration::from_secs(20);
        control.note_peer_drain(30_000, polled);
        assert!(!control.peer_drain_expired(at + Duration::from_secs(30) + PEER_DRAIN_MARGIN));

        control.clear_peer_drain();
        assert!(
            !control.peer_drain_expired(polled + Duration::from_secs(86_400)),
            "a drain that ended properly leaves no deadline behind"
        );

        // A deadline left by a coordinator that went away must not reach
        // into a drain this node later begins for itself, which is what
        // `drain_here` clears it for
        control.note_peer_drain(30_000, at);
        control.clear_peer_drain();
        assert!(
            !control.peer_drain_expired(at + Duration::from_secs(3_600)),
            "a peer's deadline outlived the drain it belonged to"
        );

        // A caller that names no deadline still gets a bounded one
        control.note_peer_drain(0, at);
        assert!(!control.peer_drain_expired(at + Duration::from_secs(DEFAULT_PEER_DRAIN_SECS)));
        assert!(control.peer_drain_expired(
            at + Duration::from_secs(DEFAULT_PEER_DRAIN_SECS) + PEER_DRAIN_MARGIN
        ));
    }
}
