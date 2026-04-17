//! Finite state machine type with transition validation.
//!
//! Defines a workflow with states, events, and transitions. Validates
//! state changes at function call time to prevent invalid transitions.

use crate::diff::JsonValue;
use std::collections::{HashSet, VecDeque};
use zyron_common::{Result, ZyronError};

/// A transition from one state to another, triggered by an event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub from_state: String,
    pub event: String,
    pub to_state: String,
    pub guard: Option<String>,
}

/// A state machine definition.
#[derive(Debug, Clone)]
pub struct StateMachineDef {
    pub states: Vec<String>,
    pub initial_state: String,
    pub transitions: Vec<Transition>,
}

/// Parses a state machine definition from JSON.
/// Expected format:
/// {
///   "states": ["pending", "active", "completed"],
///   "initial": "pending",
///   "transitions": [
///     {"from": "pending", "event": "start", "to": "active"},
///     {"from": "active", "event": "finish", "to": "completed"}
///   ]
/// }
pub fn sm_parse(json_def: &str) -> Result<StateMachineDef> {
    let val = JsonValue::parse(json_def)?;
    let obj = match val {
        JsonValue::Object(items) => items,
        _ => {
            return Err(ZyronError::ExecutionError(
                "State machine must be an object".into(),
            ));
        }
    };

    let states = obj
        .iter()
        .find(|(k, _)| k == "states")
        .map(|(_, v)| v)
        .ok_or_else(|| ZyronError::ExecutionError("Missing 'states'".into()))?;
    let states_arr = match states {
        JsonValue::Array(a) => a,
        _ => {
            return Err(ZyronError::ExecutionError(
                "'states' must be an array".into(),
            ));
        }
    };
    let states: Vec<String> = states_arr
        .iter()
        .filter_map(|v| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .collect();

    if states.is_empty() {
        return Err(ZyronError::ExecutionError(
            "State machine must have at least one state".into(),
        ));
    }

    let initial = obj
        .iter()
        .find(|(k, _)| k == "initial")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| states[0].clone());

    if !states.contains(&initial) {
        return Err(ZyronError::ExecutionError(format!(
            "Initial state '{}' not in states list",
            initial
        )));
    }

    let transitions_val = obj.iter().find(|(k, _)| k == "transitions").map(|(_, v)| v);
    let transitions: Vec<Transition> = match transitions_val {
        Some(JsonValue::Array(arr)) => arr
            .iter()
            .filter_map(|t| parse_transition(t).ok())
            .collect(),
        _ => Vec::new(),
    };

    // Validate transitions reference known states
    for t in &transitions {
        if !states.contains(&t.from_state) {
            return Err(ZyronError::ExecutionError(format!(
                "Transition from unknown state: {}",
                t.from_state
            )));
        }
        if !states.contains(&t.to_state) {
            return Err(ZyronError::ExecutionError(format!(
                "Transition to unknown state: {}",
                t.to_state
            )));
        }
    }

    Ok(StateMachineDef {
        states,
        initial_state: initial,
        transitions,
    })
}

fn parse_transition(val: &JsonValue) -> Result<Transition> {
    let obj = match val {
        JsonValue::Object(items) => items,
        _ => {
            return Err(ZyronError::ExecutionError(
                "Transition must be an object".into(),
            ));
        }
    };

    let from = obj
        .iter()
        .find(|(k, _)| k == "from")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| ZyronError::ExecutionError("Missing 'from'".into()))?;

    let event = obj
        .iter()
        .find(|(k, _)| k == "event")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| ZyronError::ExecutionError("Missing 'event'".into()))?;

    let to = obj
        .iter()
        .find(|(k, _)| k == "to")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| ZyronError::ExecutionError("Missing 'to'".into()))?;

    let guard = obj
        .iter()
        .find(|(k, _)| k == "guard")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        });

    Ok(Transition {
        from_state: from,
        event,
        to_state: to,
        guard,
    })
}

/// Validates a state machine definition (checks for unreachable states, etc.).
pub fn sm_validate_definition(def: &StateMachineDef) -> Result<()> {
    if !def.states.contains(&def.initial_state) {
        return Err(ZyronError::ExecutionError(format!(
            "Initial state not in states: {}",
            def.initial_state
        )));
    }

    // Check reachability via BFS
    let reachable = sm_reachable_states(def, &def.initial_state);
    let unreachable: Vec<&String> = def
        .states
        .iter()
        .filter(|s| !reachable.contains(s))
        .collect();

    if !unreachable.is_empty() {
        return Err(ZyronError::ExecutionError(format!(
            "Unreachable states: {:?}",
            unreachable
        )));
    }

    Ok(())
}

/// Attempts to transition the state machine from current_state via event.
/// Returns the new state, or an error if the transition is invalid.
pub fn sm_transition(def: &StateMachineDef, current_state: &str, event: &str) -> Result<String> {
    if !def.states.contains(&current_state.to_string()) {
        return Err(ZyronError::ExecutionError(format!(
            "Unknown current state: {}",
            current_state
        )));
    }

    for t in &def.transitions {
        if t.from_state == current_state && t.event == event {
            return Ok(t.to_state.clone());
        }
    }

    Err(ZyronError::ExecutionError(format!(
        "No transition from '{}' on event '{}'",
        current_state, event
    )))
}

/// Returns true if a transition is possible from current_state via event.
pub fn sm_can_transition(def: &StateMachineDef, current_state: &str, event: &str) -> bool {
    def.transitions
        .iter()
        .any(|t| t.from_state == current_state && t.event == event)
}

/// Returns all events that can be triggered from the current state.
pub fn sm_available_events(def: &StateMachineDef, current_state: &str) -> Vec<String> {
    def.transitions
        .iter()
        .filter(|t| t.from_state == current_state)
        .map(|t| t.event.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}

/// Returns true if the given state has no outgoing transitions.
pub fn sm_is_terminal(def: &StateMachineDef, state: &str) -> bool {
    !def.transitions.iter().any(|t| t.from_state == state)
}

/// Returns all states reachable from the given starting state via BFS.
pub fn sm_reachable_states(def: &StateMachineDef, from: &str) -> Vec<String> {
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(from.to_string());
    reachable.insert(from.to_string());

    while let Some(current) = queue.pop_front() {
        for t in &def.transitions {
            if t.from_state == current && !reachable.contains(&t.to_state) {
                reachable.insert(t.to_state.clone());
                queue.push_back(t.to_state.clone());
            }
        }
    }

    reachable.into_iter().collect()
}

/// Returns the shortest sequence of events to transition from `from` to `to`.
/// Returns None if no path exists.
pub fn sm_shortest_path(def: &StateMachineDef, from: &str, to: &str) -> Option<Vec<String>> {
    if from == to {
        return Some(Vec::new());
    }

    // BFS tracking the event used to reach each state
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();

    visited.insert(from.to_string());
    queue.push_back((from.to_string(), Vec::new()));

    while let Some((current, path)) = queue.pop_front() {
        for t in &def.transitions {
            if t.from_state == current && !visited.contains(&t.to_state) {
                let mut new_path = path.clone();
                new_path.push(t.event.clone());
                if t.to_state == to {
                    return Some(new_path);
                }
                visited.insert(t.to_state.clone());
                queue.push_back((t.to_state.clone(), new_path));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_def() -> StateMachineDef {
        StateMachineDef {
            states: vec!["pending".into(), "active".into(), "completed".into()],
            initial_state: "pending".into(),
            transitions: vec![
                Transition {
                    from_state: "pending".into(),
                    event: "start".into(),
                    to_state: "active".into(),
                    guard: None,
                },
                Transition {
                    from_state: "active".into(),
                    event: "finish".into(),
                    to_state: "completed".into(),
                    guard: None,
                },
                Transition {
                    from_state: "pending".into(),
                    event: "cancel".into(),
                    to_state: "completed".into(),
                    guard: None,
                },
            ],
        }
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "states":["pending","active","completed"],
            "initial":"pending",
            "transitions":[
                {"from":"pending","event":"start","to":"active"},
                {"from":"active","event":"finish","to":"completed"}
            ]
        }"#;
        let def = sm_parse(json).unwrap();
        assert_eq!(def.states.len(), 3);
        assert_eq!(def.initial_state, "pending");
        assert_eq!(def.transitions.len(), 2);
    }

    #[test]
    fn test_parse_invalid_initial() {
        let json = r#"{
            "states":["a"],
            "initial":"b"
        }"#;
        assert!(sm_parse(json).is_err());
    }

    #[test]
    fn test_parse_missing_states() {
        let json = r#"{"initial":"a"}"#;
        assert!(sm_parse(json).is_err());
    }

    #[test]
    fn test_validate_reachable() {
        let def = sample_def();
        assert!(sm_validate_definition(&def).is_ok());
    }

    #[test]
    fn test_validate_unreachable() {
        let def = StateMachineDef {
            states: vec!["a".into(), "b".into(), "c".into()],
            initial_state: "a".into(),
            transitions: vec![Transition {
                from_state: "a".into(),
                event: "x".into(),
                to_state: "b".into(),
                guard: None,
            }],
        };
        assert!(sm_validate_definition(&def).is_err()); // c unreachable
    }

    #[test]
    fn test_transition() {
        let def = sample_def();
        let next = sm_transition(&def, "pending", "start").unwrap();
        assert_eq!(next, "active");
    }

    #[test]
    fn test_invalid_transition() {
        let def = sample_def();
        // Cannot finish from pending
        assert!(sm_transition(&def, "pending", "finish").is_err());
    }

    #[test]
    fn test_can_transition() {
        let def = sample_def();
        assert!(sm_can_transition(&def, "pending", "start"));
        assert!(!sm_can_transition(&def, "pending", "finish"));
    }

    #[test]
    fn test_available_events() {
        let def = sample_def();
        let events = sm_available_events(&def, "pending");
        assert_eq!(events.len(), 2);
        assert!(events.contains(&"start".to_string()));
        assert!(events.contains(&"cancel".to_string()));
    }

    #[test]
    fn test_is_terminal() {
        let def = sample_def();
        assert!(!sm_is_terminal(&def, "pending"));
        assert!(sm_is_terminal(&def, "completed"));
    }

    #[test]
    fn test_reachable_states() {
        let def = sample_def();
        let reachable = sm_reachable_states(&def, "pending");
        assert!(reachable.contains(&"pending".to_string()));
        assert!(reachable.contains(&"active".to_string()));
        assert!(reachable.contains(&"completed".to_string()));
    }

    #[test]
    fn test_reachable_from_terminal() {
        let def = sample_def();
        let reachable = sm_reachable_states(&def, "completed");
        assert_eq!(reachable.len(), 1); // Only itself
    }

    #[test]
    fn test_shortest_path_direct() {
        let def = sample_def();
        let path = sm_shortest_path(&def, "pending", "active").unwrap();
        assert_eq!(path, vec!["start"]);
    }

    #[test]
    fn test_shortest_path_multi_step() {
        let def = sample_def();
        let path = sm_shortest_path(&def, "pending", "completed").unwrap();
        // Could be ["cancel"] (1 step) or ["start", "finish"] (2 steps)
        assert_eq!(path, vec!["cancel"]);
    }

    #[test]
    fn test_shortest_path_none() {
        let def = sample_def();
        // Cannot go from completed to anywhere
        assert!(sm_shortest_path(&def, "completed", "pending").is_none());
    }

    #[test]
    fn test_shortest_path_same_state() {
        let def = sample_def();
        let path = sm_shortest_path(&def, "pending", "pending").unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_unknown_state_transition() {
        let def = sample_def();
        assert!(sm_transition(&def, "unknown", "start").is_err());
    }
}
