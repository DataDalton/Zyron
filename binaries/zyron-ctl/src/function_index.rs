//! The complete list of built-in functions, read out of the code that
//! dispatches them.
//!
//! There is no table of functions to read: dispatch is a match on the lowered
//! name, and Rust cannot enumerate a match's arms. So the list is extracted
//! from the dispatch sources themselves, which is the only enumeration that
//! cannot fall behind what the engine answers to. A gate holds the extraction
//! to a floor, so a pattern that stops matching fails loudly instead of
//! quietly producing a short list.
//!
//! A name here is a function the engine has. A name that also has a registry
//! entry gets a page of its own and is linked from the index, and the rest are
//! listed so a reader knows they exist and what they are called.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Fewest functions the extraction has to find before it is believed.
///
/// The surface was 481 when this was written. A floor well below that catches
/// a pattern that broke without failing every time a function is removed.
pub const EXTRACTION_FLOOR: usize = 400;

/// Where the match arms that name a built-in function live.
///
/// `types_bridge` dispatches from its own module and from each submodule, so
/// the whole directory is read rather than one file of it.
const DISPATCH_SOURCES: &[&str] = &[
    "zyron-executor/src/types_bridge",
    "zyron-executor/src/array_functions.rs",
];

/// Every built-in function the dispatch sources name, lowercased and sorted.
///
/// `crates_root` is the directory holding the crates, which is where the
/// dispatch sources are found.
pub fn extract(crates_root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for relative in DISPATCH_SOURCES {
        read_sources(&crates_root.join(relative), &mut names);
    }
    names
}

/// The subject a dispatch file covers, for grouping the index.
///
/// A dispatch file holds the functions of one domain, so its name is the
/// grouping. `mod.rs` dispatches what belongs to no single domain.
fn subject_of(file_stem: &str) -> &'static str {
    match file_stem {
        "array_functions" => "Arrays",
        "bitfield_crypto" => "Bitfields and hashing",
        "color_fingerprint" => "Colour and fingerprinting",
        "cron_range_time" => "Time, ranges and schedules",
        "expectation_metrics" => "Data quality metrics",
        "financial_ts" => "Finance and time series",
        "idgen_identifier" => "Identifiers and generated keys",
        "ltree_bridge" => "Label trees",
        "masking_bridge" => "Masking",
        "matrix_geo" => "Matrices and geometry",
        "media_bridge" => "Media",
        "misc_domains" => "Text, encoding and markup",
        "money_quantity" => "Money and quantities",
        "network_url" => "Network and URL",
        "prob_statistics" => "Probability and statistics",
        "regex_strings" => "Regular expressions and strings",
        "state_rate_tree" => "State machines and rate limits",
        "vector_money_semver" => "Vectors and versions",
        _ => "General",
    }
}

/// Every function, with the subject of the file that dispatches it.
pub fn extract_subjects(crates_root: &Path) -> BTreeMap<String, &'static str> {
    let mut out = BTreeMap::new();
    for relative in DISPATCH_SOURCES {
        read_subjects(&crates_root.join(relative), &mut out);
    }
    out
}

fn read_subjects(path: &Path, out: &mut BTreeMap<String, &'static str>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            read_subjects(&entry.path(), out);
        }
        return;
    }
    if path.extension().and_then(|e| e.to_str()) != Some("rs") {
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let file_subject = subject_of(stem);
    for (name, section) in names_with_sections(&text) {
        // A file holding one domain names it. A file that dispatches several
        // marks them with section comments, which name a narrower subject
        // than the file does
        let subject = match (file_subject, section) {
            ("General", Some(section)) => section,
            ("General", None) => subject_by_name(&name),
            _ => file_subject,
        };
        match out.get(&name) {
            Some(held) if *held != "General" => {}
            _ => {
                out.insert(name, subject);
            }
        }
    }
}

/// The subject a name implies, for a function the file and its sections leave
/// ungrouped.
///
/// Read last, so a file or a section that names a subject always wins. A name
/// matching nothing here stays under the general heading rather than being
/// placed by guesswork.
fn subject_by_name(name: &str) -> &'static str {
    if name.starts_with("json") || name.starts_with("jsonb") {
        return "JSON and VARIANT";
    }
    if name.starts_with("bloom")
        || name.starts_with("cms_")
        || name.starts_with("hll_")
        || name.starts_with("tdigest")
    {
        return "Probabilistic sketches";
    }
    if name.ends_with("_diff") || name.ends_with("_patch") {
        return "Diff and patch";
    }
    "General"
}

/// The subject a section comment names, for the sections the general
/// dispatcher divides itself into.
fn section_subject(heading: &str) -> Option<&'static str> {
    let heading = heading.trim().trim_matches('-').trim();
    Some(match heading {
        h if h.starts_with("fuzzy") => "Fuzzy text matching",
        "string_ops" => "Regular expressions and strings",
        "formatting" => "Money and quantities",
        "color" => "Colour and fingerprinting",
        h if h.starts_with("data_quality") => "Data quality metrics",
        "encoding" => "Text, encoding and markup",
        "semver" => "Vectors and versions",
        h if h.starts_with("id_gen") => "Identifiers and generated keys",
        "checksums" => "Bitfields and hashing",
        "natural sort" => "Regular expressions and strings",
        "file detection" => "Media",
        "document processing" => "Media",
        "barcode/QR" => "Media",
        _ => return None,
    })
}

/// Each dispatched name and the section comment it sits under, where the file
/// divides itself into sections.
///
/// Only the arms of a match on the function name count. A dispatch file also
/// holds helper matches over option words, such as the resize modes and the
/// barcode symbologies, and those words are values a function reads rather
/// than functions the engine answers to.
fn names_with_sections(text: &str) -> Vec<(String, Option<&'static str>)> {
    let mut out = Vec::new();
    let mut section: Option<&'static str> = None;
    let mut depth: i32 = 0;
    // Brace depth of the open match on the function name, when one is open
    let mut dispatch_depth: Option<i32> = None;
    let lines: Vec<&str> = text.lines().collect();
    for (at, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("// ----------") {
            section = section_subject(rest);
            continue;
        }
        // An arm sits one brace inside its match, at the depth the line opens at
        if dispatch_depth == Some(depth) {
            // A long alternation wraps, leaving the bar that follows the last
            // name of a line on the next line, so the scan reads both and
            // keeps only the names this line opens
            let mut window = String::with_capacity(line.len() * 2);
            window.push_str(line);
            if let Some(next) = lines.get(at + 1) {
                window.push('\n');
                window.push_str(next);
            }
            let mut names = BTreeSet::new();
            collect_names_before(&window, line.len(), &mut names);
            for name in names {
                out.push((name, section));
            }
        }
        let opens = dispatch_depth.is_none() && opens_name_match(line);
        depth += brace_delta(line);
        if opens {
            dispatch_depth = Some(depth);
        } else if dispatch_depth.is_some_and(|open_at| depth < open_at) {
            dispatch_depth = None;
        }
    }
    out
}

/// True when the line opens a match whose subject is the function name.
///
/// Dispatch reads the name, under that spelling or lowered into `lower`.
/// A match on anything else is a helper reading an option word.
fn opens_name_match(line: &str) -> bool {
    let Some(at) = line.find("match ") else {
        return false;
    };
    let Some(brace) = line[at..].find('{') else {
        return false;
    };
    let subject = &line[at + "match ".len()..at + brace];
    subject
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .any(|word| word == "name" || word == "lower")
}

fn brace_delta(line: &str) -> i32 {
    line.bytes().filter(|b| *b == b'{').count() as i32
        - line.bytes().filter(|b| *b == b'}').count() as i32
}

/// Each function's argument list, where the code states one.
///
/// Argument checks name the function and its parameters in the error they
/// raise, as `name(arg, arg [, optional])`. Those strings are the only
/// statement of a signature in the codebase, so they are read rather than
/// restated. A function whose check states no signature has no entry here.
pub fn extract_signatures(crates_root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for relative in DISPATCH_SOURCES {
        read_signatures(&crates_root.join(relative), &mut out);
    }
    out
}

/// Reads signatures from one file, or from every Rust file under a directory.
fn read_signatures(path: &Path, out: &mut BTreeMap<String, String>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            read_signatures(&entry.path(), out);
        }
        return;
    }
    if path.extension().and_then(|e| e.to_str()) != Some("rs") {
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    collect_signatures(&text, out);
}

/// Reads `name(args)` out of the quoted strings a file holds.
///
/// The string continues past the closing parenthesis in most cases, because
/// the check appends what it expected, so the signature is the span up to the
/// parenthesis that closes the argument list.
fn collect_signatures(text: &str, out: &mut BTreeMap<String, String>) {
    let mut at = 0;
    while let Some(open) = text[at..].find('"') {
        let start = at + open + 1;
        let Some(len) = text[start..].find('"') else {
            break;
        };
        let quoted = &text[start..start + len];
        at = start + len + 1;
        let Some((name, signature)) = split_signature(quoted) else {
            continue;
        };
        // The first statement of a signature wins, so a later message that
        // phrases it differently does not overwrite a good one
        out.entry(name).or_insert(signature);
    }
}

/// Splits a quoted string into a function name and its whole signature, when
/// the string opens with one.
fn split_signature(quoted: &str) -> Option<(String, String)> {
    let paren = quoted.find('(')?;
    let name = &quoted[..paren];
    if !is_function_name(name) {
        return None;
    }
    // Nesting is one level deep at most in these strings, so the matching
    // parenthesis is the first one that brings the depth back to zero
    let mut depth = 0usize;
    for (offset, byte) in quoted.bytes().enumerate().skip(paren) {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((name.to_string(), quoted[..=offset].to_string()));
                }
            }
            _ => {}
        }
    }
    None
}

/// Reads one file, or every Rust file under one directory.
fn read_sources(path: &Path, out: &mut BTreeSet<String>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            read_sources(&entry.path(), out);
        }
        return;
    }
    if path.extension().and_then(|e| e.to_str()) != Some("rs") {
        return;
    }
    if let Ok(text) = std::fs::read_to_string(path) {
        // Read through the same walk the grouping uses, so the list of names
        // and the list of subjects cannot disagree about what is a function
        out.extend(names_with_sections(&text).into_iter().map(|(name, _)| name));
    }
}

/// Reads the quoted names a dispatch file matches on, for the names that open
/// before `cutoff`.
///
/// A match arm reads `"name" =>` and an alternative reads `"name" | "name" =>`,
/// so a quoted lowercase word followed by `=>` or `|` is a dispatched name. A
/// word with a space or an uppercase letter is prose rather than a function.
/// The scan reads past the cutoff for the bar or arrow that follows the last
/// name, because a long alternation wraps and leaves that bar on the next line.
fn collect_names_before(text: &str, cutoff: usize, out: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut at = 0;
    while let Some(open) = text[at..].find('"') {
        let start = at + open + 1;
        let Some(len) = text[start..].find('"') else {
            break;
        };
        if start > cutoff {
            break;
        }
        let word = &text[start..start + len];
        let after = start + len + 1;
        at = after;
        if !is_function_name(word) {
            continue;
        }
        // Only a word a match arm reads counts, which is one followed by the
        // arrow or by another alternative
        let mut cursor = after;
        while cursor < bytes.len() && (bytes[cursor] == b' ' || bytes[cursor] == b'\n') {
            cursor += 1;
        }
        let tail = &text[cursor..];
        if tail.starts_with("=>") || tail.starts_with('|') {
            out.insert(word.to_string());
        }
    }
}

/// True when a quoted word looks like a function name rather than prose.
fn is_function_name(word: &str) -> bool {
    !word.is_empty()
        && word.len() <= 40
        && word
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        && word.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crates_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("crates"))
            .expect("the crates directory")
    }

    #[test]
    fn the_extraction_finds_the_function_surface() {
        let names = extract(&crates_root());
        assert!(
            names.len() >= EXTRACTION_FLOOR,
            "the extraction found {} functions, below the floor of {}. A dispatch \
             file moved or the arm pattern changed, and the index would be short \
             without saying so",
            names.len(),
            EXTRACTION_FLOOR
        );
    }

    #[test]
    fn the_extraction_finds_functions_it_should_and_no_prose() {
        let names = extract(&crates_root());
        // lag and percent_rank end a wrapped alternation, where the bar that
        // follows them sits on the next line
        for expected in [
            "levenshtein",
            "array_length",
            "base64url_encode",
            "lag",
            "percent_rank",
        ] {
            assert!(
                names.contains(expected),
                "{expected} is dispatched and the extraction missed it"
            );
        }
        for prose in ["argument", "arguments", "takes"] {
            assert!(
                !names.contains(prose),
                "'{prose}' is prose and the extraction read it as a function"
            );
        }
        // Option words a helper match reads. Each is a value an argument takes,
        // so listing one would send a reader looking for a function that is
        // not there
        for option in ["fit", "cover", "stretch", "code39", "upca", "first", "last"] {
            assert!(
                !names.contains(option),
                "'{option}' is a word an argument takes, and the extraction read                  it as a function the engine dispatches"
            );
        }
    }

    #[test]
    fn every_documented_function_is_one_the_engine_dispatches() {
        let dispatched = extract(&crates_root());
        for entry in zyron_parser::grammar::GRAMMAR {
            if entry.position != zyron_parser::grammar::GrammarPosition::Function {
                continue;
            }
            assert!(
                dispatched.contains(&entry.name.to_ascii_lowercase()),
                "{} has a page and the engine dispatches no function by that                  name, so the reference documents something a caller cannot use",
                entry.name
            );
        }
    }

    #[test]
    fn a_function_page_says_what_it_returns() {
        for entry in zyron_parser::grammar::GRAMMAR {
            if entry.position != zyron_parser::grammar::GrammarPosition::Function {
                continue;
            }
            assert!(
                entry.returns.is_some(),
                "{} is a function and its page would not say what a call yields",
                entry.name
            );
        }
    }
}
