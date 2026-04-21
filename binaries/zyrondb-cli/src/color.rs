//! ANSI color and style helpers. All wrappers fall back to plain text when
//! colors are disabled (non-tty stdout, `NO_COLOR` env var, or `--no-color`
//! flag in the future). This keeps the REPL readable in pipes, log files,
//! and accessibility modes.
//!
//! On Windows, rustyline enables virtual-terminal mode during editor init,
//! which also applies to our direct stdout writes. Any banner printed
//! before the editor is constructed must still respect that guarantee,
//! so callers should construct the `rustyline::Editor` before emitting
//! colored output.
//!
//! The public API returns owned `String`s. ANSI escapes are ~10 bytes per
//! wrapping call, negligible compared to the surrounding text layout work.

use std::io::IsTerminal;
use std::sync::OnceLock;

/// ANSI SGR reset.
pub const RESET: &str = "\x1b[0m";

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";

const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const CYAN: &str = "\x1b[36m";
const GRAY: &str = "\x1b[90m";

/// Cached color-enablement decision. Evaluated once on first access.
static COLOR_ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns whether ANSI styling should be emitted. True when stdout is a
/// terminal and `NO_COLOR` is not set. Cached on first call because it
/// cannot change over the life of the process.
#[inline]
pub fn enabled() -> bool {
    *COLOR_ENABLED.get_or_init(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        std::io::stdout().is_terminal()
    })
}

/// Wraps `s` with the given SGR sequence when colors are enabled.
#[inline]
fn wrap(code: &'static str, s: &str) -> String {
    if enabled() {
        format!("{}{}{}", code, s, RESET)
    } else {
        s.to_string()
    }
}

pub fn bold(s: &str) -> String {
    wrap(BOLD, s)
}
pub fn dim(s: &str) -> String {
    wrap(DIM, s)
}
pub fn italic(s: &str) -> String {
    wrap(ITALIC, s)
}
pub fn cyan(s: &str) -> String {
    wrap(CYAN, s)
}
pub fn gray(s: &str) -> String {
    wrap(GRAY, s)
}

/// Compound: bold + color. Two SGR codes collapsed into one sequence so
/// paragraphs don't accumulate stray reset escapes.
#[inline]
fn wrap2(a: &'static str, b: &'static str, s: &str) -> String {
    if enabled() {
        format!("{}{}{}{}", a, b, s, RESET)
    } else {
        s.to_string()
    }
}

pub fn bold_blue(s: &str) -> String {
    wrap2(BOLD, BLUE, s)
}
pub fn bold_red(s: &str) -> String {
    wrap2(BOLD, RED, s)
}
pub fn bold_yellow(s: &str) -> String {
    wrap2(BOLD, YELLOW, s)
}
pub fn bold_cyan(s: &str) -> String {
    wrap2(BOLD, CYAN, s)
}
pub fn bold_green(s: &str) -> String {
    wrap2(BOLD, GREEN, s)
}

/// Prints the ASCII logo in bold blue. Kept as a module-level constant so
/// both the startup banner and `\conninfo` can reprint it without
/// duplicating the string.
pub const ZYRON_LOGO: &str = "\
 _____
|__  /_   _ _ __ ___  _ __
  / /| | | | '__/ _ \\| '_ \\
 / /_| |_| | | | (_) | | | |
/____|\\__, |_|  \\___/|_| |_|
      |___/                 ";
