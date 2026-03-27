// Query history persistence. Loads and saves command history from a file
// in the user home directory.

use std::path::{Path, PathBuf};

/// Returns the path to the history file (~/.zyron_history).
pub fn history_path() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        PathBuf::from(home).join(".zyron_history")
    } else {
        PathBuf::from(".zyron_history")
    }
}

/// Loads history entries from the given file path.
/// Returns an empty vector if the file does not exist or cannot be read.
pub fn load_history(path: &Path) -> Vec<String> {
    match std::fs::read_to_string(path) {
        Ok(contents) => contents
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Saves the last max_entries history entries to the given file path.
pub fn save_history(path: &Path, history: &[String], max_entries: usize) -> std::io::Result<()> {
    let start = if history.len() > max_entries {
        history.len() - max_entries
    } else {
        0
    };
    let entries = &history[start..];
    let content = entries.join("\n");
    std::fs::write(path, content)
}

/// Adds an entry to the history list. Skips empty entries and consecutive duplicates.
pub fn add_to_history(history: &mut Vec<String>, entry: &str) {
    let trimmed = entry.trim();
    if trimmed.is_empty() {
        return;
    }
    if let Some(last) = history.last() {
        if last == trimmed {
            return;
        }
    }
    history.push(trimmed.to_string());
}
