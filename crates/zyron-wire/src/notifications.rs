//! LISTEN/NOTIFY pub/sub notification channels.
//!
//! Shared across all connections via ServerState. Each channel name maps
//! to a tokio broadcast sender. Connections subscribe by calling listen()
//! and receive notifications asynchronously.

use std::collections::HashMap;
use std::sync::Mutex;

use tokio::sync::broadcast;

/// Shared notification channel registry for LISTEN/NOTIFY support.
pub struct NotificationChannels {
    /// Channel name -> broadcast sender. Each sender has a buffer of 256 messages.
    channels: Mutex<HashMap<String, broadcast::Sender<Notification>>>,
}

/// A notification message sent through a channel.
#[derive(Clone, Debug)]
pub struct Notification {
    /// The channel name.
    pub channel: String,
    /// Optional payload string.
    pub payload: String,
    /// Process ID of the sender.
    pub sender_pid: i32,
}

impl NotificationChannels {
    /// Creates a new empty notification channel registry.
    pub fn new() -> Self {
        Self {
            channels: Mutex::new(HashMap::new()),
        }
    }

    /// Subscribes to a named channel. Returns a receiver that will get
    /// all future notifications sent to this channel.
    pub fn listen(&self, channel: &str) -> broadcast::Receiver<Notification> {
        let mut channels = self.channels.lock().unwrap();
        let sender = channels
            .entry(channel.to_string())
            .or_insert_with(|| broadcast::channel(256).0);
        sender.subscribe()
    }

    /// Removes a subscription channel if no receivers remain.
    pub fn unlisten(&self, channel: &str) {
        let mut channels = self.channels.lock().unwrap();
        if let Some(sender) = channels.get(channel) {
            if sender.receiver_count() == 0 {
                channels.remove(channel);
            }
        }
    }

    /// Sends a notification to all listeners on the named channel.
    /// Returns the number of receivers that received the message.
    /// Returns 0 if no listeners exist for the channel.
    pub fn notify(&self, channel: &str, payload: &str, sender_pid: i32) -> usize {
        let channels = self.channels.lock().unwrap();
        if let Some(sender) = channels.get(channel) {
            sender
                .send(Notification {
                    channel: channel.to_string(),
                    payload: payload.to_string(),
                    sender_pid,
                })
                .unwrap_or(0)
        } else {
            0
        }
    }
}

impl Default for NotificationChannels {
    fn default() -> Self {
        Self::new()
    }
}
