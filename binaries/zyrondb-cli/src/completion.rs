//! Tab completion, inline hints, and input-buffer coloring.
//!
//! Two Helper implementations are provided because the top-level REPL is
//! a command shell, not a SQL editor:
//!
//! - `CommandHelper` — used for the command-mode prompt. Completes the
//!   top-level command vocabulary and subcommand verbs (tables/indexes/
//!   users/stats/etc). Does not color SQL keywords because command mode
//!   input is not SQL.
//!
//! - `SqlHelper` — used inside the `sql` sub-REPL. Completes the full
//!   SQL keyword vocabulary sourced from
//!   `zyron_parser::token::KEYWORD_STRINGS` (auto-generated from the
//!   parser's match arms so the two can never drift). Colors keywords in
//!   the input buffer as the user types.
//!
//! Both helpers honor the process-wide `color::enabled()` decision so
//! their output is safe in pipes and `NO_COLOR` sessions.

use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Context, Helper};
use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::OnceLock;
use zyron_parser::token::KEYWORD_STRINGS;

use crate::color;
use crate::command::TOP_LEVEL_COMMANDS;

// ---------------------------------------------------------------------------
// Command-mode completion
// ---------------------------------------------------------------------------

/// Per-command subcommand catalog. Stage-1 scope: the nouns with fixed
/// verb vocabulary. Object names (table names, user names, index names)
/// are NOT fetched from the catalog; that's a Stage-2 enhancement that
/// requires an in-process metadata cache.
const COMMAND_SUBCOMMANDS: &[(&str, &[&str])] = &[
    ("tables", &["list", "describe", "drop", "truncate"]),
    ("indexes", &["list", "describe", "drop", "reindex"]),
    ("users", &["list", "create", "drop", "grant", "revoke"]),
    (
        "stats",
        &[
            "summary",
            "wal",
            "tables",
            "indexes",
            "buffer",
            "connections",
        ],
    ),
    ("wal", &["status"]),
    ("slots", &["list", "create", "drop"]),
    ("cdc", &["streams", "feeds", "ingests", "drop"]),
    ("config", &["show", "set"]),
    ("sessions", &["list"]),
];

/// Rustyline helper for the top-level command prompt.
pub struct CommandHelper;

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, fragment) = current_word(line, pos);
        if fragment.is_empty() {
            return Ok((start, Vec::new()));
        }

        // Determine whether the cursor sits on the first token or a later
        // one. First-token matches against top-level command names; later
        // tokens match against the subcommand verbs for the leading noun.
        let leading = first_token(line);
        let is_first_token = leading.map(|(_, end)| pos <= end).unwrap_or(true);

        let mut out = Vec::new();
        if is_first_token {
            for name in TOP_LEVEL_COMMANDS {
                if name.starts_with(fragment) {
                    out.push(Pair {
                        display: name.to_string(),
                        replacement: name.to_string(),
                    });
                }
            }
        } else if let Some((head, _)) = leading {
            let head_lower = head.to_ascii_lowercase();
            if let Some((_, subs)) = COMMAND_SUBCOMMANDS
                .iter()
                .find(|(n, _)| *n == head_lower.as_str())
            {
                for sub in *subs {
                    if sub.starts_with(fragment) {
                        out.push(Pair {
                            display: sub.to_string(),
                            replacement: sub.to_string(),
                        });
                    }
                }
            }
        }
        Ok((start, out))
    }
}

impl Hinter for CommandHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if pos != line.len() {
            return None;
        }
        let (_, fragment) = current_word(line, pos);
        if fragment.len() < 2 {
            return None;
        }
        let leading = first_token(line);
        let is_first_token = leading.map(|(_, end)| pos <= end).unwrap_or(true);
        if is_first_token {
            for name in TOP_LEVEL_COMMANDS {
                if name.starts_with(fragment) && name.len() > fragment.len() {
                    return Some(name[fragment.len()..].to_string());
                }
            }
        } else if let Some((head, _)) = leading {
            let head_lower = head.to_ascii_lowercase();
            if let Some((_, subs)) = COMMAND_SUBCOMMANDS
                .iter()
                .find(|(n, _)| *n == head_lower.as_str())
            {
                for sub in *subs {
                    if sub.starts_with(fragment) && sub.len() > fragment.len() {
                        return Some(sub[fragment.len()..].to_string());
                    }
                }
            }
        }
        None
    }
}

impl Highlighter for CommandHelper {
    /// Colors the leading command noun bold cyan so the user can see at
    /// a glance whether the first word is recognized. Unrecognized first
    /// words stay plain; subsequent tokens are never styled since we
    /// don't know their grammatical role without parsing.
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if !color::enabled() || line.is_empty() {
            return Cow::Borrowed(line);
        }
        let bytes = line.as_bytes();
        // Count leading whitespace.
        let mut i = 0;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return Cow::Borrowed(line);
        }
        let word_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let word = &line[word_start..i];
        let lower = word.to_ascii_lowercase();
        if TOP_LEVEL_COMMANDS.iter().any(|n| *n == lower.as_str())
            || matches!(
                lower.as_str(),
                "?" | "h" | "q" | "ver" | "info" | "dbs" | "db" | "ls" | "cls" | "bye" | "logout"
            )
        {
            let mut out = String::with_capacity(line.len() + 16);
            out.push_str(&line[..word_start]);
            out.push_str(&color::bold_cyan(word));
            out.push_str(&line[i..]);
            Cow::Owned(out)
        } else {
            Cow::Borrowed(line)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if !color::enabled() {
            return Cow::Borrowed(hint);
        }
        Cow::Owned(color::gray(hint))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        color::enabled()
    }
}

impl Validator for CommandHelper {}
impl Helper for CommandHelper {}

// ---------------------------------------------------------------------------
// SQL-sub-mode completion
// ---------------------------------------------------------------------------

fn keyword_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| KEYWORD_STRINGS.iter().copied().collect())
}

/// Rustyline helper for the SQL sub-REPL. Reuses the auto-generated
/// keyword list and highlights recognized keywords in the input buffer.
pub struct SqlHelper;

impl Completer for SqlHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let (start, fragment) = current_word(line, pos);
        if fragment.is_empty() {
            return Ok((start, Vec::new()));
        }
        let upper = fragment.to_ascii_uppercase();
        let mut out = Vec::new();
        for kw in KEYWORD_STRINGS {
            if kw.starts_with(&upper) {
                out.push(Pair {
                    display: kw.to_string(),
                    replacement: kw.to_string(),
                });
            }
        }
        Ok((start, out))
    }
}

impl Hinter for SqlHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<String> {
        if pos != line.len() {
            return None;
        }
        let (_, fragment) = current_word(line, pos);
        if fragment.len() < 2 {
            return None;
        }
        let upper = fragment.to_ascii_uppercase();
        for kw in KEYWORD_STRINGS {
            if kw.starts_with(&upper) && kw.len() > upper.len() {
                return Some(kw[upper.len()..].to_string());
            }
        }
        None
    }
}

impl Highlighter for SqlHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if !color::enabled() || line.is_empty() {
            return Cow::Borrowed(line);
        }
        let kws = keyword_set();
        let bytes = line.as_bytes();
        let mut out = String::with_capacity(line.len() + 16);
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if !is_ident_byte(b) {
                out.push(b as char);
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && is_ident_byte(bytes[i]) {
                i += 1;
            }
            let word = &line[start..i];
            let upper = word.to_ascii_uppercase();
            if kws.contains(upper.as_str()) {
                out.push_str(&color::bold_cyan(word));
            } else {
                out.push_str(word);
            }
        }
        Cow::Owned(out)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if !color::enabled() {
            return Cow::Borrowed(hint);
        }
        Cow::Owned(color::gray(hint))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        color::enabled()
    }
}

impl Validator for SqlHelper {}
impl Helper for SqlHelper {}

// ---------------------------------------------------------------------------
// Shared token-boundary helpers
// ---------------------------------------------------------------------------

/// Returns `(start_byte, fragment)` for the word the cursor sits inside.
/// Words are separated by ASCII whitespace; the fragment may be empty
/// when the cursor is at a whitespace boundary.
fn current_word(line: &str, pos: usize) -> (usize, &str) {
    let bytes = line.as_bytes();
    let mut start = pos;
    while start > 0 && !bytes[start - 1].is_ascii_whitespace() {
        start -= 1;
    }
    (start, &line[start..pos])
}

/// Returns the first whitespace-delimited token's byte span, or `None`
/// when the line has no non-whitespace content.
fn first_token(line: &str) -> Option<(&str, usize)> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if start == i {
        None
    } else {
        Some((&line[start..i], i))
    }
}

#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}
