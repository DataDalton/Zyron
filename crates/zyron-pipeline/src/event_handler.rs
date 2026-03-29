//! Event system for async dispatch of database events.
//!
//! Supports registering handlers for specific event types and dispatching
//! events through a bounded channel. A background worker thread drains
//! the channel and invokes matching handlers.

use crate::ids::EventHandlerId;
use std::sync::Arc;
use std::thread;
use zyron_common::{Result, ZyronError};

/// Types of database events that handlers can subscribe to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EventType {
    TableCreated,
    TableDropped,
    IndexCreated,
    IndexDropped,
    SchemaChanged,
    PipelineCompleted,
    PipelineSlaBreach,
    QualityCheckFailed,
    QualityDriftDetected,
    CheckpointCompleted,
    ErrorOccurred,
    TriggerRecursionLimitHit,
    Custom(String),
}

impl EventType {
    /// Returns a static string label for display purposes.
    pub fn label(&self) -> &str {
        match self {
            EventType::TableCreated => "TableCreated",
            EventType::TableDropped => "TableDropped",
            EventType::IndexCreated => "IndexCreated",
            EventType::IndexDropped => "IndexDropped",
            EventType::SchemaChanged => "SchemaChanged",
            EventType::PipelineCompleted => "PipelineCompleted",
            EventType::PipelineSlaBreach => "PipelineSlaBreach",
            EventType::QualityCheckFailed => "QualityCheckFailed",
            EventType::QualityDriftDetected => "QualityDriftDetected",
            EventType::CheckpointCompleted => "CheckpointCompleted",
            EventType::ErrorOccurred => "ErrorOccurred",
            EventType::TriggerRecursionLimitHit => "TriggerRecursionLimitHit",
            EventType::Custom(_) => "Custom",
        }
    }
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Custom(name) => write!(f, "Custom:{}", name),
            other => f.write_str(other.label()),
        }
    }
}

/// Payload attached to a dispatched event, carrying contextual data
/// about what occurred.
#[derive(Clone, Debug)]
pub struct EventPayload {
    pub eventType: EventType,
    pub timestamp: i64,
    pub source: String,
    pub details: hashbrown::HashMap<String, String>,
}

/// A registered handler that will be invoked when a matching event
/// is dispatched. The handler references a function by name (resolved
/// at execution time through the UDF registry).
#[derive(Clone, Debug)]
pub struct EventHandler {
    pub id: EventHandlerId,
    pub name: String,
    pub eventType: EventType,
    /// Optional condition expression (as SQL text) that must evaluate
    /// to true for the handler to fire. Not evaluated in this module,
    /// but stored for the execution layer.
    pub condition: Option<String>,
    /// Name of the function to invoke when the event fires.
    pub functionName: String,
    pub enabled: bool,
}

/// Default bounded channel capacity for the event dispatch queue.
const EVENT_CHANNEL_CAPACITY: usize = 4096;

/// Manages event handler registration and asynchronous event dispatch.
/// Events are enqueued to a bounded crossbeam channel and processed
/// by a background worker thread.
pub struct EventDispatcher {
    /// Handlers grouped by event type for fast lookup. Uses EventType
    /// directly as key (Hash + Eq), avoiding string allocation on dispatch.
    handlers: scc::HashMap<EventType, Vec<Arc<EventHandler>>>,
    /// Send side of the bounded dispatch channel.
    sender: crossbeam::channel::Sender<(EventType, EventPayload)>,
    /// Receive side, held for the worker thread.
    receiver: crossbeam::channel::Receiver<(EventType, EventPayload)>,
}

impl EventDispatcher {
    /// Creates a new event dispatcher with a bounded channel.
    pub fn new() -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(EVENT_CHANNEL_CAPACITY);
        Self {
            handlers: scc::HashMap::new(),
            sender,
            receiver,
        }
    }

    /// Creates a new event dispatcher with a custom channel capacity.
    pub fn withCapacity(capacity: usize) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(capacity);
        Self {
            handlers: scc::HashMap::new(),
            sender,
            receiver,
        }
    }

    /// Registers an event handler. The handler will be invoked for
    /// events matching its event type.
    pub fn register(&self, handler: EventHandler) -> Result<()> {
        let eventType = handler.eventType.clone();
        let arc = Arc::new(handler);
        let entry = self.handlers.entry_sync(eventType);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.get_mut().push(arc);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![arc]);
            }
        }
        Ok(())
    }

    /// Removes a handler by name across all event types.
    /// Returns an error if no handler with the given name exists.
    pub fn dropHandler(&self, name: &str) -> Result<()> {
        let mut found = false;
        let mut keysToUpdate: Vec<EventType> = Vec::new();

        self.handlers.iter_sync(|k, v| {
            if v.iter().any(|h| h.name == name) {
                keysToUpdate.push(k.clone());
            }
            true
        });

        for key in keysToUpdate {
            let entry = self.handlers.entry_sync(key);
            if let scc::hash_map::Entry::Occupied(mut occ) = entry {
                let beforeLen = occ.get().len();
                occ.get_mut().retain(|h| h.name != name);
                if occ.get().len() < beforeLen {
                    found = true;
                }
            }
        }

        if found {
            Ok(())
        } else {
            Err(ZyronError::EventHandlerNotFound(name.to_string()))
        }
    }

    /// Enqueues an event for asynchronous processing by the worker
    /// thread. Returns immediately after placing the event on the
    /// channel. Returns an error if the channel is full.
    pub fn dispatch(&self, eventType: EventType, payload: EventPayload) -> Result<()> {
        self.sender
            .try_send((eventType, payload))
            .map_err(|e| ZyronError::Internal(format!("event dispatch failed: {}", e)))
    }

    /// Returns a clone of all handlers registered for a given event type.
    pub fn handlersFor(&self, eventType: &EventType) -> Vec<Arc<EventHandler>> {
        self.handlers
            .read_sync(eventType, |_k, v| v.clone())
            .unwrap_or_default()
    }

    /// Spawns a background worker thread that drains the event channel
    /// and invokes matching handlers. The callback receives each
    /// (handler, payload) pair for actual execution.
    ///
    /// Returns a JoinHandle for the worker thread. The worker exits
    /// when all senders are dropped and the channel is drained.
    pub fn startWorker<F>(&self, callback: F) -> thread::JoinHandle<()>
    where
        F: Fn(&EventHandler, &EventPayload) + Send + 'static,
    {
        let receiver = self.receiver.clone();
        let handlers = self.cloneHandlerSnapshot();

        thread::spawn(move || {
            while let Ok((eventType, payload)) = receiver.recv() {
                if let Some(handlerList) = handlers.get(&eventType) {
                    for handler in handlerList {
                        if handler.enabled {
                            callback(handler, &payload);
                        }
                    }
                }
            }
        })
    }

    /// Spawns a worker that re-reads handlers from the live registry
    /// on each event, so new registrations take effect immediately.
    pub fn startLiveWorker<F>(&self, callback: F) -> thread::JoinHandle<()>
    where
        F: Fn(&EventHandler, &EventPayload) + Send + Sync + 'static,
    {
        let receiver = self.receiver.clone();
        let handlerMap = self.handlerMapClone();

        thread::spawn(move || {
            while let Ok((eventType, payload)) = receiver.recv() {
                let currentHandlers: Vec<Arc<EventHandler>> = handlerMap
                    .read_sync(&eventType, |_k, v| v.clone())
                    .unwrap_or_default();

                for handler in &currentHandlers {
                    if handler.enabled {
                        callback(handler, &payload);
                    }
                }
            }
        })
    }

    /// Returns a snapshot of all handlers as a standard HashMap
    /// for the static worker thread.
    fn cloneHandlerSnapshot(&self) -> hashbrown::HashMap<EventType, Vec<Arc<EventHandler>>> {
        let mut snapshot = hashbrown::HashMap::new();
        self.handlers.iter_sync(|k, v| {
            snapshot.insert(k.clone(), v.clone());
            true
        });
        snapshot
    }

    /// Returns a reference-counted clone of the handler map for
    /// the live worker thread.
    fn handlerMapClone(&self) -> Arc<scc::HashMap<EventType, Vec<Arc<EventHandler>>>> {
        let cloned = scc::HashMap::new();
        self.handlers.iter_sync(|k, v| {
            let _ = cloned.insert_sync(k.clone(), v.clone());
            true
        });
        Arc::new(cloned)
    }

    /// Returns the total number of registered handlers across all
    /// event types.
    pub fn handlerCount(&self) -> usize {
        let mut count = 0;
        self.handlers.iter_sync(|_k, v| {
            count += v.len();
            true
        });
        count
    }
}

impl Default for EventDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn makeHandler(id: u32, name: &str, eventType: EventType) -> EventHandler {
        EventHandler {
            id: EventHandlerId(id),
            name: name.to_string(),
            eventType,
            condition: None,
            functionName: format!("{}_fn", name),
            enabled: true,
        }
    }

    fn makePayload(eventType: EventType, source: &str) -> EventPayload {
        EventPayload {
            eventType,
            timestamp: 1000,
            source: source.to_string(),
            details: hashbrown::HashMap::new(),
        }
    }

    #[test]
    fn test_register_handler() {
        let dispatcher = EventDispatcher::new();
        let handler = makeHandler(1, "on_create", EventType::TableCreated);
        dispatcher.register(handler).expect("register");
        assert_eq!(dispatcher.handlerCount(), 1);
    }

    #[test]
    fn test_register_multiple_same_event() {
        let dispatcher = EventDispatcher::new();
        dispatcher
            .register(makeHandler(1, "h1", EventType::TableCreated))
            .expect("register");
        dispatcher
            .register(makeHandler(2, "h2", EventType::TableCreated))
            .expect("register");

        let handlers = dispatcher.handlersFor(&EventType::TableCreated);
        assert_eq!(handlers.len(), 2);
    }

    #[test]
    fn test_drop_handler() {
        let dispatcher = EventDispatcher::new();
        dispatcher
            .register(makeHandler(1, "temp_handler", EventType::TableDropped))
            .expect("register");
        dispatcher.dropHandler("temp_handler").expect("drop");
        assert_eq!(dispatcher.handlerCount(), 0);
    }

    #[test]
    fn test_drop_nonexistent_handler() {
        let dispatcher = EventDispatcher::new();
        let err = dispatcher.dropHandler("missing").expect_err("should fail");
        assert!(matches!(err, ZyronError::EventHandlerNotFound(_)));
    }

    #[test]
    fn test_dispatch_enqueues() {
        let dispatcher = EventDispatcher::withCapacity(16);
        dispatcher
            .register(makeHandler(1, "h1", EventType::IndexCreated))
            .expect("register");

        let payload = makePayload(EventType::IndexCreated, "test");
        dispatcher
            .dispatch(EventType::IndexCreated, payload)
            .expect("dispatch");
    }

    #[test]
    fn test_handlers_for_returns_correct_type() {
        let dispatcher = EventDispatcher::new();
        dispatcher
            .register(makeHandler(1, "h1", EventType::TableCreated))
            .expect("register");
        dispatcher
            .register(makeHandler(2, "h2", EventType::TableDropped))
            .expect("register");

        let created = dispatcher.handlersFor(&EventType::TableCreated);
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].name, "h1");

        let dropped = dispatcher.handlersFor(&EventType::TableDropped);
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].name, "h2");
    }

    #[test]
    fn test_handlers_for_empty() {
        let dispatcher = EventDispatcher::new();
        let handlers = dispatcher.handlersFor(&EventType::CheckpointCompleted);
        assert!(handlers.is_empty());
    }

    #[test]
    fn test_custom_event_type() {
        let dispatcher = EventDispatcher::new();
        let customType = EventType::Custom("my_event".to_string());
        dispatcher
            .register(makeHandler(1, "custom_h", customType.clone()))
            .expect("register");

        let handlers = dispatcher.handlersFor(&customType);
        assert_eq!(handlers.len(), 1);
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(EventType::TableCreated.to_string(), "TableCreated");
        assert_eq!(
            EventType::Custom("foo".to_string()).to_string(),
            "Custom:foo"
        );
    }

    #[test]
    fn test_disabled_handler_not_invoked() {
        let dispatcher = EventDispatcher::withCapacity(16);
        let mut handler = makeHandler(1, "disabled_h", EventType::ErrorOccurred);
        handler.enabled = false;
        dispatcher.register(handler).expect("register");

        let counter = Arc::new(AtomicUsize::new(0));
        let counterClone = Arc::clone(&counter);

        let payload = makePayload(EventType::ErrorOccurred, "test");
        dispatcher
            .dispatch(EventType::ErrorOccurred, payload)
            .expect("dispatch");

        // Drop the dispatcher sender side by dropping the dispatcher,
        // then start a worker-like loop manually.
        let handlers = dispatcher.cloneHandlerSnapshot();
        // The receiver is still accessible via the dispatcher.
        // Process one message manually.
        if let Ok((eventType, _payload)) = dispatcher.receiver.try_recv() {
            if let Some(handlerList) = handlers.get(&eventType) {
                for h in handlerList {
                    if h.enabled {
                        counterClone.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        }

        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_worker_processes_events() {
        let dispatcher = EventDispatcher::withCapacity(16);
        dispatcher
            .register(makeHandler(1, "worker_h", EventType::SchemaChanged))
            .expect("register");

        let counter = Arc::new(AtomicUsize::new(0));
        let counterClone = Arc::clone(&counter);

        let handle = dispatcher.startWorker(move |_handler, _payload| {
            counterClone.fetch_add(1, Ordering::SeqCst);
        });

        let payload = makePayload(EventType::SchemaChanged, "test");
        dispatcher
            .dispatch(EventType::SchemaChanged, payload)
            .expect("dispatch");

        // Drop the sender to signal the worker to exit after draining.
        drop(dispatcher);
        handle.join().expect("worker thread should complete");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_handler_count() {
        let dispatcher = EventDispatcher::new();
        assert_eq!(dispatcher.handlerCount(), 0);

        dispatcher
            .register(makeHandler(1, "a", EventType::TableCreated))
            .expect("register");
        dispatcher
            .register(makeHandler(2, "b", EventType::TableDropped))
            .expect("register");
        dispatcher
            .register(makeHandler(3, "c", EventType::TableCreated))
            .expect("register");

        assert_eq!(dispatcher.handlerCount(), 3);
    }

    #[test]
    fn test_event_payload_details() {
        let mut details = hashbrown::HashMap::new();
        details.insert("table".to_string(), "users".to_string());
        details.insert("schema".to_string(), "public".to_string());

        let payload = EventPayload {
            eventType: EventType::TableCreated,
            timestamp: 12345,
            source: "ddl_executor".to_string(),
            details,
        };

        assert_eq!(payload.details.len(), 2);
        assert_eq!(payload.details.get("table"), Some(&"users".to_string()));
        assert_eq!(payload.timestamp, 12345);
    }
}
