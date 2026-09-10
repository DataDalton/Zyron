//! Machine-readable description of the SQL surface.
//!
//! One entry per construct, carrying the words that open it, the syntax it
//! accepts, what it does, what each of its clauses does, what it refuses and
//! an example of it working. The language server reads this table for
//! completion and hover, and `zyron-ctl docs generate` writes the statement
//! reference from it, so a construct added to the parser is covered in both
//! places by registering it here rather than by editing either.
//!
//! The table is checked against the code from four directions: every statement
//! the parser produces has an entry, every word an entry lists is one the
//! parser reads, every example parses and unparses back to itself, and every
//! refusal quotes text the source that emits it still contains. A registry
//! that drifts from the parser fails the build rather than misleading a reader.

pub mod entries;

use crate::token::{Keyword, lookup_keyword};

pub use entries::GRAMMAR;

/// Where in a statement a construct may be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarPosition {
    /// Opens a statement
    Statement,
    /// A FROM clause item
    FromItem,
    /// Written after a relation in a FROM clause
    FromPostfix,
    /// Joins two relations
    Join,
    /// A clause inside a statement that another word opened
    Clause,
    /// Callable in an expression
    Function,
}

/// The part of the surface a construct belongs to, which is the directory the
/// reference writes its page into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Defining and changing objects
    Ddl,
    /// Reading and writing rows
    Dml,
    /// Shaping a result: FROM items, joins, and the clauses that read rows
    Queries,
    /// Transaction boundaries, savepoints and locking
    Transactions,
    /// State that belongs to one connection
    Session,
    /// ZyronLake tables and the operations only they accept
    Lake,
    /// Streaming jobs, subscriptions and change feeds
    Streaming,
    /// Callable in an expression
    Functions,
}

impl Category {
    /// The directory name the reference writes this category's pages into.
    pub fn directory(&self) -> &'static str {
        match self {
            Category::Ddl => "ddl",
            Category::Dml => "dml",
            Category::Queries => "queries",
            Category::Transactions => "transactions",
            Category::Session => "session",
            Category::Lake => "lake",
            Category::Streaming => "streaming",
            Category::Functions => "functions",
        }
    }

    /// The heading the category's index page carries.
    pub fn title(&self) -> &'static str {
        match self {
            Category::Ddl => "Defining Objects",
            Category::Dml => "Reading and Writing Rows",
            Category::Queries => "Shaping a Result",
            Category::Transactions => "Transactions",
            Category::Session => "Session State",
            Category::Lake => "ZyronLake",
            Category::Streaming => "Streaming",
            Category::Functions => "Built-in Functions",
        }
    }

    /// Every category, in the order the top index lists them.
    pub fn all() -> &'static [Category] {
        &[
            Category::Ddl,
            Category::Dml,
            Category::Queries,
            Category::Transactions,
            Category::Session,
            Category::Lake,
            Category::Streaming,
            Category::Functions,
        ]
    }
}

/// One optional or alternative part of a construct.
#[derive(Debug, Clone, Copy)]
pub struct Clause {
    /// The clause as it is written
    pub syntax: &'static str,
    /// The AST field this clause sets, which is what lets the coverage gate
    /// tell a documented option from one nobody wrote down. None for a clause
    /// that shapes the statement without landing in a field of its own
    pub field: Option<&'static str>,
    /// What writing it does
    pub what: &'static str,
    /// What happens when it is left out, for a clause that has a default
    pub default: Option<&'static str>,
}

/// One combination the parser accepts and something later refuses.
#[derive(Debug, Clone, Copy)]
pub struct Refusal {
    /// The condition that reaches the refusal
    pub when: &'static str,
    /// A distinctive fragment of the message the code emits, which the
    /// coverage gate looks for in the source so the reference and the error
    /// cannot disagree
    pub message: &'static str,
}

/// One statement that works, and what it produces.
#[derive(Debug, Clone, Copy)]
pub struct Example {
    /// A statement that parses and unparses back to itself
    pub statement: &'static str,
    /// What running it gives
    pub yields: &'static str,
}

/// One construct the parser accepts.
#[derive(Debug, Clone, Copy)]
pub struct GrammarEntry {
    /// The construct's name, as a completion offers it
    pub name: &'static str,
    /// The words that open it, in the order they are written
    pub keywords: &'static [&'static str],
    pub position: GrammarPosition,
    pub category: Category,
    /// The form the parser accepts, for a hover to show
    pub syntax: &'static str,
    /// What the construct does, in one sentence
    pub summary: &'static str,
    /// What the construct does, at the length a reference page wants
    pub description: &'static str,
    /// Each optional or alternative part, and what writing it does
    pub clauses: &'static [Clause],
    /// Each combination that is refused, and the message that refuses it
    pub refusals: &'static [Refusal],
    /// Statements that work, for a reader to copy
    pub examples: &'static [Example],
    /// Related constructs, by their registry name
    pub see_also: &'static [&'static str],
    /// What a call yields, for a function. None for a statement, which
    /// returns rows or a count rather than a value
    pub returns: Option<&'static str>,
}

/// Every entry whose name starts with `prefix`, case insensitively. A
/// completion offers these.
pub fn entries_matching(prefix: &str) -> impl Iterator<Item = &'static GrammarEntry> + '_ {
    // Compared byte by byte against the entry's own name. Folding either side
    // would allocate a String per entry, and a completion runs this for every
    // entry on every keystroke
    GRAMMAR
        .iter()
        .filter(move |e| starts_with_fold(e.name, prefix))
}

/// True when `name` begins with `prefix`, ignoring case, without folding
/// either side into a buffer.
fn starts_with_fold(name: &str, prefix: &str) -> bool {
    let name = name.as_bytes();
    let prefix = prefix.as_bytes();
    if prefix.len() > name.len() {
        return false;
    }
    name.iter()
        .zip(prefix)
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// The entry a word names, for a hover to render. Matches on the construct's
/// name and on its first opening word, so hovering PIVOT and hovering the
/// whole construct both resolve.
pub fn entry_for_word(word: &str) -> Option<&'static GrammarEntry> {
    GRAMMAR.iter().find(|e| {
        e.name.eq_ignore_ascii_case(word)
            || e.keywords
                .first()
                .is_some_and(|k| k.eq_ignore_ascii_case(word))
    })
}

/// Every entry in one category, in the order the registry lists them.
pub fn entries_in(category: Category) -> impl Iterator<Item = &'static GrammarEntry> {
    GRAMMAR.iter().filter(move |e| e.category == category)
}

/// The keyword a grammar word maps to, or None when the word is a function
/// name the lexer reads as an identifier.
pub fn keyword_of(word: &str) -> Option<Keyword> {
    // `lookup_keyword` folds into a stack buffer of its own, so folding here
    // first would allocate for nothing
    lookup_keyword(word)
}

/// Constructions that do not belong in reference text.
///
/// Each is a way of explaining why something is so, or of addressing the
/// reader, rather than stating what the thing does. A reference states
/// behaviour, its defaults and its errors. The dashes and the semicolon are
/// banned in documentation throughout this repository.
///
/// Held here rather than in a test so the registry and the generator that
/// renders it are held to one list.
pub const NOT_REFERENCE_VOICE: &[&str] = &[
    "which is what",
    "that is what",
    "the point is",
    "is the point",
    "whole point",
    "its point is",
    "that is the",
    "this is how",
    "this is the",
    "what makes",
    "is what keeps",
    "is what lets",
    "is what separates",
    "is what stops",
    "is the difference",
    "turns out",
    "usually wants",
    " you ",
    " your ",
    "\u{2014}",
    "\u{2013}",
    ";",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_function_entry_names_a_word_a_call_can_open() {
        // A function name has to reach a function call. A word the lexer
        // produces as a keyword still reaches one when the parser reads that
        // keyword as an identifier, which is how rate() is callable
        for entry in GRAMMAR {
            if entry.position != GrammarPosition::Function {
                continue;
            }
            let word = entry.keywords[0];
            let Some(keyword) = keyword_of(word) else {
                continue;
            };
            assert!(
                crate::parser::keyword_to_ident_str(keyword).is_some(),
                "{} is a keyword the parser does not read as an identifier, so \
                 a call to it would parse as that keyword instead",
                entry.name
            );
        }
    }

    #[test]
    fn a_word_resolves_to_its_entry_for_hover() {
        assert_eq!(entry_for_word("pivot").map(|e| e.name), Some("PIVOT"));
        assert_eq!(
            entry_for_word("MATCH_CONDITION").map(|e| e.name),
            Some("MATCH_CONDITION")
        );
        assert_eq!(
            entry_for_word("array_transform").map(|e| e.name),
            Some("array_transform")
        );
        assert!(entry_for_word("not_a_construct").is_none());
    }

    #[test]
    fn a_prefix_offers_the_constructs_it_opens() {
        let names: Vec<&str> = entries_matching("UNP").map(|e| e.name).collect();
        assert_eq!(names, vec!["UNPIVOT"]);
        let names: Vec<&str> = entries_matching("array_s").map(|e| e.name).collect();
        assert_eq!(names, vec!["array_sort", "array_slice"]);
    }

    #[test]
    fn every_entry_carries_what_a_reference_page_needs() {
        for entry in GRAMMAR {
            assert!(
                !entry.description.is_empty(),
                "{} has no description, so its page would say only its summary",
                entry.name
            );
            assert!(
                !entry.examples.is_empty(),
                "{} has no example, so a reader has nothing to copy",
                entry.name
            );
            for clause in entry.clauses {
                assert!(
                    !clause.what.is_empty(),
                    "{} has a clause '{}' that says nothing about what it does",
                    entry.name,
                    clause.syntax
                );
            }
        }
    }

    #[test]
    fn every_see_also_names_an_entry_that_exists() {
        for entry in GRAMMAR {
            for related in entry.see_also {
                assert!(
                    GRAMMAR.iter().any(|e| e.name == *related),
                    "{} points at '{}', which is not in the registry",
                    entry.name,
                    related
                );
            }
        }
    }

    #[test]
    fn no_two_entries_share_a_name() {
        for (i, entry) in GRAMMAR.iter().enumerate() {
            assert!(
                !GRAMMAR[..i].iter().any(|e| e.name == entry.name),
                "{} is registered twice, so a hover would resolve to whichever came first",
                entry.name
            );
        }
    }

    #[test]
    fn every_category_holds_at_least_one_entry() {
        for category in Category::all() {
            assert!(
                entries_in(*category).next().is_some(),
                "{} has no entries, so the reference would write an empty index",
                category.title()
            );
        }
    }
}
