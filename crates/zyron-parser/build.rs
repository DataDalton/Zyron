//! Extracts the list of recognized SQL keyword strings from the
//! `lookup_keyword` function in `src/token.rs` and writes them out as
//! a single `pub static KEYWORD_STRINGS: &[&str]`. Downstream tooling
//! (CLI tab completion, syntax highlighters, docs) can then enumerate
//! the vocabulary without maintaining a parallel list.
//!
//! The scraper finds every `"FOO"` literal inside the `pub fn
//! lookup_keyword` match block. That block is the single source of
//! truth: adding a keyword arm there automatically updates
//! `KEYWORD_STRINGS` on the next build.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let src_path = PathBuf::from("src/token.rs");
    println!("cargo:rerun-if-changed={}", src_path.display());

    let source = fs::read_to_string(&src_path).expect("read src/token.rs");

    let body = extract_lookup_keyword_body(&source)
        .expect("could not locate `pub fn lookup_keyword` body in src/token.rs");

    let strings: BTreeSet<&str> = scrape_keyword_literals(body).collect();

    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(out_dir).join("keyword_strings.rs");

    let mut rendered = String::new();
    rendered.push_str("/// Auto-generated at build time from the `lookup_keyword` match arms in\n");
    rendered.push_str("/// `src/token.rs`. Do not edit; edit the match and rebuild.\n");
    rendered.push_str("pub static KEYWORD_STRINGS: &[&str] = &[\n");
    for s in &strings {
        rendered.push_str("    \"");
        rendered.push_str(s);
        rendered.push_str("\",\n");
    }
    rendered.push_str("];\n");

    fs::write(&out_path, rendered).expect("write keyword_strings.rs");
}

/// Returns the substring between the opening and matching closing brace of
/// the `pub fn lookup_keyword` function. Counts braces so nested `{}` inside
/// string literals don't trip the scan (not that they appear today, but the
/// scraper should still be robust).
fn extract_lookup_keyword_body(source: &str) -> Option<&str> {
    let sig_pos = source.find("pub fn lookup_keyword")?;
    let body_start = source[sig_pos..].find('{')? + sig_pos + 1;

    let bytes = source.as_bytes();
    let mut depth: i32 = 1;
    let mut i = body_start;
    let mut in_string = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut escape = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_line_comment {
            if b == b'\n' {
                in_line_comment = false;
            }
        } else if in_block_comment {
            if b == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                in_block_comment = false;
                i += 1;
            }
        } else if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                    in_line_comment = true;
                    i += 1;
                }
                b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                    in_block_comment = true;
                    i += 1;
                }
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&source[body_start..i]);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Yields every `"..."` literal in `body` whose contents are the keyword
/// string form (uppercase ASCII letters or underscores, at least one char).
/// Comments and escape sequences are already stripped by the caller having
/// passed the function body rather than the whole file.
fn scrape_keyword_literals(body: &str) -> impl Iterator<Item = &str> {
    let bytes = body.as_bytes();
    let mut positions: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    let mut in_line_comment = false;
    let mut in_block_comment = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_line_comment {
            if b == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if in_block_comment {
            if b == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() {
            if bytes[i + 1] == b'/' {
                in_line_comment = true;
                i += 2;
                continue;
            }
            if bytes[i + 1] == b'*' {
                in_block_comment = true;
                i += 2;
                continue;
            }
        }
        if b == b'"' {
            let start = i + 1;
            let mut j = start;
            let mut escape = false;
            while j < bytes.len() {
                let cb = bytes[j];
                if escape {
                    escape = false;
                } else if cb == b'\\' {
                    escape = true;
                } else if cb == b'"' {
                    break;
                }
                j += 1;
            }
            positions.push((start, j));
            i = j + 1;
            continue;
        }
        i += 1;
    }

    positions.into_iter().filter_map(move |(s, e)| {
        let slice = &body[s..e];
        if slice.is_empty() {
            return None;
        }
        if slice.bytes().all(|c| c.is_ascii_uppercase() || c == b'_') {
            Some(slice)
        } else {
            None
        }
    })
}
