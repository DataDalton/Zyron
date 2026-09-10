//! Writes the SQL statement reference from the grammar registry.
//!
//! Every page under the reference tree is output. Nothing there is edited by
//! hand, and `docs check` regenerates into a temporary directory and compares,
//! so a page edited in place or a registry that moved without the pages being
//! rewritten both fail rather than going unnoticed.
//!
//! Generation is deterministic: the same registry produces byte-identical
//! output, so a difference in the comparison means the registry moved rather
//! than that the generator did.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use zyron_parser::grammar::{Category, GRAMMAR, GrammarEntry};
use zyron_wire::connection::{ReplicationClass, replication_class};

/// What a generation run produced, for the caller to report.
pub struct Generated {
    /// Each page's path under the reference root, and its whole content
    pub pages: BTreeMap<PathBuf, String>,
}

/// Renders the whole reference tree in memory.
pub fn render() -> Generated {
    let mut pages = BTreeMap::new();
    for category in Category::all() {
        let mut entries: Vec<&GrammarEntry> =
            GRAMMAR.iter().filter(|e| e.category == *category).collect();
        entries.sort_by_key(|e| e.name);
        if entries.is_empty() {
            continue;
        }
        for entry in &entries {
            pages.insert(
                PathBuf::from(category.directory()).join(format!("{}.md", slug(entry.name))),
                page(entry),
            );
        }
        // The function index is the Functions category's own index, and it
        // lists every function the engine dispatches rather than only the ones
        // with a page, so the generic index would say less and be wrong
        if *category != Category::Functions {
            pages.insert(
                PathBuf::from(category.directory()).join("README.md"),
                category_index(*category, &entries),
            );
        }
    }
    if let Some(root) = crates_root() {
        let dispatched = crate::function_index::extract(&root);
        if dispatched.len() >= crate::function_index::EXTRACTION_FLOOR {
            let signatures = crate::function_index::extract_signatures(&root);
            let subjects = crate::function_index::extract_subjects(&root);
            pages.insert(
                PathBuf::from("functions").join("README.md"),
                function_index(&dispatched, &signatures, &subjects),
            );
        }
    }
    pages.insert(PathBuf::from("README.md"), top_index());
    Generated { pages }
}

/// The crates directory, found from this binary's own manifest.
///
/// The function list is read out of the dispatch sources, so generation needs
/// to find them. A tree built somewhere without them writes no function index
/// rather than writing an empty one.
fn crates_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("crates"))?;
    root.is_dir().then_some(root)
}

/// The complete list of built-in functions.
///
/// Every function the engine dispatches is listed, because a reader looking
/// for one needs to know whether it exists before anything else. The ones with
/// a registry entry link to their page, and the rest are named, so the list
/// never claims the engine has fewer functions than it does.
fn function_index(
    dispatched: &std::collections::BTreeSet<String>,
    signatures: &std::collections::BTreeMap<String, String>,
    subjects: &std::collections::BTreeMap<String, &'static str>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Built-in Functions");
    let _ = writeln!(out);
    // The note has to match what the index actually writes, so a function
    // without a page keeps the longer sentence
    let all_documented = dispatched.iter().all(|name| {
        GRAMMAR
            .iter()
            .any(|e| e.name.eq_ignore_ascii_case(name.as_str()))
    });
    let note = if all_documented {
        FUNCTION_INDEX_NOTE
    } else {
        FUNCTION_INDEX_PARTIAL_NOTE
    };
    let _ = writeln!(out, "{} functions, {note}", dispatched.len());

    // Grouped by the domain of the file that dispatches each one, so a reader
    // looking for a money function reads one section rather than the whole list
    let mut by_subject: BTreeMap<&'static str, Vec<&String>> = BTreeMap::new();
    for name in dispatched {
        let subject = subjects.get(name).copied().unwrap_or("General");
        by_subject.entry(subject).or_default().push(name);
    }

    let mut documented = 0usize;
    for (subject, names) in &by_subject {
        let _ = writeln!(out);
        let _ = writeln!(out, "## {subject}");
        let _ = writeln!(out);
        for name in names {
            match GRAMMAR
                .iter()
                .find(|e| e.name.eq_ignore_ascii_case(name.as_str()))
            {
                Some(entry) => {
                    documented += 1;
                    // A page in the Functions category is a sibling of this
                    // index, and one anywhere else is reached through the parent
                    let path = if entry.category == Category::Functions {
                        format!("{}.md", slug(entry.name))
                    } else {
                        format!("../{}/{}.md", entry.category.directory(), slug(entry.name))
                    };
                    let _ = writeln!(out, "- [{}]({}), {}", name, path, entry.summary);
                }
                // A function with no page is named with its argument list
                // where the code states one, which a reader needs in order to
                // call it
                None => match signatures.get(name.as_str()) {
                    Some(signature) => {
                        let _ = writeln!(out, "- `{signature}`");
                    }
                    None => {
                        let _ = writeln!(out, "- `{name}`");
                    }
                },
            }
        }
    }

    let signed = dispatched
        .iter()
        .filter(|n| signatures.contains_key(*n))
        .count();
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{documented} have a page. {signed} state an argument list."
    );
    out
}

/// The file name a construct's page takes.
fn slug(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for byte in name.bytes() {
        match byte {
            b'A'..=b'Z' => out.push(byte.to_ascii_lowercase() as char),
            b'a'..=b'z' | b'0'..=b'9' => out.push(byte as char),
            _ => {
                if !out.ends_with('-') {
                    out.push('-');
                }
            }
        }
    }
    out.trim_matches('-').to_string()
}

/// One construct's page.
fn page(entry: &GrammarEntry) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {}", entry.name);
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", entry.description);
    let _ = writeln!(out);
    let _ = writeln!(out, "## Syntax");
    let _ = writeln!(out);
    let _ = writeln!(out, "```sql");
    let _ = writeln!(out, "{}", entry.syntax);
    let _ = writeln!(out, "```");

    if let Some(returns) = entry.returns {
        let _ = writeln!(out);
        let _ = writeln!(out, "## Returns");
        let _ = writeln!(out);
        let _ = writeln!(out, "{returns}");
    }

    if !entry.clauses.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "## Clauses");
        let _ = writeln!(out);
        let _ = writeln!(out, "| Clause | What it does | Left out |");
        let _ = writeln!(out, "| --- | --- | --- |");
        for clause in entry.clauses {
            let _ = writeln!(
                out,
                "| `{}` | {} | {} |",
                clause.syntax,
                clause.what,
                clause.default.unwrap_or("Not applicable.")
            );
        }
    }

    if !entry.examples.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "## Examples");
        for example in entry.examples {
            let _ = writeln!(out);
            let _ = writeln!(out, "```sql");
            let _ = writeln!(out, "{}", example.statement);
            let _ = writeln!(out, "```");
            let _ = writeln!(out);
            let _ = writeln!(out, "{}", example.yields);
        }
    }

    if !entry.refusals.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "## Refused");
        let _ = writeln!(out);
        for refusal in entry.refusals {
            let _ = writeln!(out, "- {}.", refusal.when);
        }
    }

    if let Some(class) = class_of(entry) {
        let _ = writeln!(out);
        let _ = writeln!(out, "## On a cluster");
        let _ = writeln!(out);
        let _ = writeln!(out, "{}", class);
    }

    if !entry.see_also.is_empty() {
        let _ = writeln!(out);
        let _ = writeln!(out, "## See also");
        let _ = writeln!(out);
        for related in entry.see_also {
            let target = GRAMMAR.iter().find(|e| e.name == *related);
            match target {
                Some(target) => {
                    let path = if target.category == entry.category {
                        format!("{}.md", slug(target.name))
                    } else {
                        format!(
                            "../{}/{}.md",
                            target.category.directory(),
                            slug(target.name)
                        )
                    };
                    let _ = writeln!(out, "- [{}]({})", target.name, path);
                }
                // The coverage gate refuses a see_also naming no entry, so
                // this arm is reached only by a registry that has not been
                // through it
                None => {
                    let _ = writeln!(out, "- {related}");
                }
            }
        }
    }
    out
}

/// One category's index.
fn category_index(category: Category, entries: &[&GrammarEntry]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# {}", category.title());
    let _ = writeln!(out);
    let _ = writeln!(out, "{} statements and constructs.", entries.len());
    let _ = writeln!(out);
    for entry in entries {
        let _ = writeln!(
            out,
            "- [{}]({}.md), {}",
            entry.name,
            slug(entry.name),
            entry.summary
        );
    }
    out
}

/// The tree's own index.
fn top_index() -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# SQL Statement Reference");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "One page per statement and construct, written from the parser's own grammar registry by `zyron-ctl docs generate`. Nothing here is edited by hand: a page is rewritten by changing the registry the parser reads."
    );
    let _ = writeln!(out);
    for category in Category::all() {
        let count = GRAMMAR.iter().filter(|e| e.category == *category).count();
        if count == 0 {
            continue;
        }
        let _ = writeln!(
            out,
            "- [{}]({}/README.md), {} entries.",
            category.title(),
            category.directory(),
            count
        );
    }
    if let Some(root) = crates_root() {
        let dispatched = crate::function_index::extract(&root);
        if dispatched.len() >= crate::function_index::EXTRACTION_FLOOR {
            let _ = writeln!(
                out,
                "- [Built-in Functions](functions/README.md), {} functions.",
                dispatched.len()
            );
        }
    }
    out
}

/// What a statement does on a consensus group, read from the classifier the
/// wire protocol actually uses.
///
/// Derived rather than written down: the classifier is an exhaustive match
/// over every statement, so a page cannot claim something the code disagrees
/// with. Returned as None for a construct that is not a statement of its own,
/// such as a FROM item or a function, which has no class.
fn class_of(entry: &GrammarEntry) -> Option<String> {
    let example = entry.examples.first()?;
    let parsed = zyron_parser::parse(example.statement).ok()?;
    let statement = parsed.first()?;
    Some(match replication_class(statement) {
        ReplicationClass::Rows => ROWS_NOTE.to_string(),
        ReplicationClass::Statement => STATEMENT_NOTE.to_string(),
        ReplicationClass::Local => LOCAL_NOTE.to_string(),
        // The reason the classifier carries is the one the refusal shows, so
        // the page and the error say the same thing
        ReplicationClass::Unsupported { reason } => {
            format!("This is refused on a member of a consensus group. {reason}")
        }
    })
}

/// What the function index says about itself, under the count.
const FUNCTION_INDEX_NOTE: &str =
    "read from the code that dispatches them. Every name links to a page of its own.";

/// What the function index says when some function has no page yet.
const FUNCTION_INDEX_PARTIAL_NOTE: &str = "read from the code that dispatches them. A name linked here has a page of its own, and the rest are listed so a reader knows they exist.";

/// What a page says about a statement whose rows are replicated.
const ROWS_NOTE: &str = "The rows this statement writes are captured and replicated, so every member ends up with what it produced rather than running it again. A value like now() or a sequence draw therefore reads the same on every member.";

/// What a page says about a statement that is replicated and run everywhere.
const STATEMENT_NOTE: &str = "The statement itself is replicated and every member runs it. It moves no rows and is deterministic given the same catalog, so running it everywhere puts the files and indexes it describes on every member.";

/// What a page says about a statement that reaches no other member.
const LOCAL_NOTE: &str = "This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.";

/// Writes the tree, replacing whatever was there.
///
/// The directory is emptied first, so a page whose construct was renamed or
/// removed does not survive as a file nothing links to.
pub fn generate(root: &Path) -> std::io::Result<usize> {
    let rendered = render();
    if root.exists() {
        std::fs::remove_dir_all(root)?;
    }
    for (relative, content) in &rendered.pages {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, content)?;
    }
    Ok(rendered.pages.len())
}

/// Compares the tree on disk against what the registry would write now.
///
/// Returns the pages that differ, are missing, or are there and should not be.
pub fn check(root: &Path) -> std::io::Result<Vec<String>> {
    let rendered = render();
    let mut findings = Vec::new();
    for (relative, expected) in &rendered.pages {
        let path = root.join(relative);
        match std::fs::read_to_string(&path) {
            Ok(found) if found == *expected => {}
            Ok(_) => findings.push(format!(
                "{} differs from what the registry writes",
                relative.display()
            )),
            Err(_) => findings.push(format!("{} is missing", relative.display())),
        }
    }
    let mut on_disk = Vec::new();
    collect_pages(root, root, &mut on_disk);
    for relative in on_disk {
        if !rendered.pages.contains_key(&relative) {
            findings.push(format!(
                "{} is on disk and the registry does not write it",
                relative.display()
            ));
        }
    }
    findings.sort();
    Ok(findings)
}

fn collect_pages(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_pages(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md")
            && let Ok(relative) = path.strip_prefix(root)
        {
            out.push(relative.to_path_buf());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let once = render();
        let twice = render();
        assert_eq!(
            once.pages, twice.pages,
            "the same registry produced two different trees, so a check could \
             not tell a registry change from a generator one"
        );
    }

    #[test]
    fn the_generator_own_prose_reads_as_reference_text() {
        // The registry is held to this list by the parser's coverage gate. The
        // sentences the generator writes itself appear on every page, so they
        // are held to it here
        let written: &[(&str, &str)] = &[
            ("function index note", FUNCTION_INDEX_NOTE),
            ("function index partial note", FUNCTION_INDEX_PARTIAL_NOTE),
            ("rows note", ROWS_NOTE),
            ("statement note", STATEMENT_NOTE),
            ("local note", LOCAL_NOTE),
        ];
        let mut findings: Vec<String> = Vec::new();
        for (what, prose) in written {
            let lower = prose.to_lowercase();
            for banned in zyron_parser::grammar::NOT_REFERENCE_VOICE {
                if lower.contains(banned) {
                    findings.push(format!("{what}: '{}'", banned.trim()));
                }
            }
        }
        assert!(
            findings.is_empty(),
            "the generator writes {} phrase(s) that do not read as reference              text: {:?}",
            findings.len(),
            findings
        );
    }

    #[test]
    fn every_page_links_to_pages_the_tree_holds() {
        let rendered = render();
        for (relative, content) in &rendered.pages {
            let dir = relative.parent().unwrap_or(Path::new(""));
            for line in content.lines() {
                let Some(open) = line.find("](") else {
                    continue;
                };
                let rest = &line[open + 2..];
                let Some(close) = rest.find(')') else {
                    continue;
                };
                let target = &rest[..close];
                // A page may quote an outside address, in an example or in a
                // reference to a standard. Only a link into the tree is ours
                // to resolve
                if target.contains("://") || target.starts_with("mailto:") {
                    continue;
                }
                let resolved = normalize(&dir.join(target));
                assert!(
                    rendered.pages.contains_key(&resolved),
                    "{} links to {}, which the tree does not hold",
                    relative.display(),
                    target
                );
            }
        }
    }

    /// Resolves a relative link's `..` segments, which a link between two
    /// categories carries.
    fn normalize(path: &Path) -> PathBuf {
        let mut out = PathBuf::new();
        for part in path.components() {
            match part {
                std::path::Component::ParentDir => {
                    out.pop();
                }
                std::path::Component::CurDir => {}
                other => out.push(other.as_os_str()),
            }
        }
        out
    }

    #[test]
    fn a_slug_is_a_file_name() {
        assert_eq!(slug("CREATE TEMPORARY TABLE"), "create-temporary-table");
        assert_eq!(slug("array_to_string"), "array-to-string");
        assert_eq!(slug("ASOF JOIN"), "asof-join");
    }
}
