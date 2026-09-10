//! The registry is held to the parser.
//!
//! Three gates. Every statement the parser produces has a registry entry, so
//! a statement documented nowhere fails here rather than being discovered
//! missing by a reader. Every example parses and unparses back to itself, so
//! a syntax change that invalidates an example fails here rather than
//! printing something the parser would reject. Every refusal quotes text the
//! source that emits it still contains, so the reference and the error cannot
//! drift apart.
//!
//! Run: cargo test -p zyron-parser --test grammar_coverage_test -- --nocapture

use zyron_parser::grammar::{
    Category, GRAMMAR, GrammarEntry, GrammarPosition, NOT_REFERENCE_VOICE,
};
use zyron_parser::{Statement, parse, statement_to_sql};

/// The registry entry that documents one statement.
///
/// The match is exhaustive over `Statement`, which is the gate: a variant
/// added to the parser stops this file compiling until it is given an arm,
/// and the arm names the registry entry that has to exist for the test below
/// to pass. Neither half can be satisfied by writing nothing.
fn documented_as(statement: &Statement) -> &'static str {
    match statement {
        Statement::Select(_) => "SELECT",
        Statement::Insert(_) => "INSERT",
        Statement::Update(_) => "UPDATE",
        Statement::Delete(_) => "DELETE",
        Statement::Merge(_) => "MERGE",
        Statement::Truncate(_) => "TRUNCATE",
        Statement::Copy(_) => "COPY",
        Statement::ValuesQuery(_) => "VALUES",

        // One variant carries both forms, and the temporary form has its own
        // page because its lifetime, its visibility and its id space differ
        Statement::CreateTable(s) => {
            if s.temporary {
                "CREATE TEMPORARY TABLE"
            } else {
                "CREATE TABLE"
            }
        }
        Statement::DropTable(_) => "DROP TABLE",
        Statement::AlterTable(_) => "ALTER TABLE",
        Statement::UndropTable(_) => "UNDROP TABLE",
        Statement::CreateIndex(_) => "CREATE INDEX",
        Statement::DropIndex(_) => "DROP INDEX",
        Statement::AlterIndex(_) => "ALTER INDEX",
        Statement::Reindex(_) => "REINDEX",
        Statement::CreateView(_) => "CREATE VIEW",
        Statement::DropView(_) => "DROP VIEW",
        Statement::AlterView(_) => "ALTER VIEW",
        Statement::CreateMaterializedView(_) => "CREATE MATERIALIZED VIEW",
        Statement::DropMaterializedView(_) => "DROP MATERIALIZED VIEW",
        Statement::RefreshMaterializedView(_) => "REFRESH MATERIALIZED VIEW",
        Statement::CreateSchema(_) => "CREATE SCHEMA",
        Statement::DropSchema(_) => "DROP SCHEMA",
        Statement::CreateSequence(_) => "CREATE SEQUENCE",
        Statement::DropSequence(_) => "DROP SEQUENCE",
        Statement::AlterSequence(_) => "ALTER SEQUENCE",
        Statement::CreateType(_) => "CREATE TYPE",
        Statement::DropType(_) => "DROP TYPE",
        Statement::CreateCollation(_) => "CREATE COLLATION",
        Statement::DropCollation(_) => "DROP COLLATION",
        Statement::CommentOn(_) => "COMMENT ON",
        Statement::CreateTrigger(_) => "CREATE TRIGGER",
        Statement::DropTrigger(_) => "DROP TRIGGER",
        Statement::CreateFunction(_) => "CREATE FUNCTION",
        Statement::DropFunction(_) => "DROP FUNCTION",
        Statement::CreateAggregate(_) => "CREATE AGGREGATE",
        Statement::DropAggregate(_) => "DROP AGGREGATE",
        Statement::CreateProcedure(_) => "CREATE PROCEDURE",
        Statement::DropProcedure(_) => "DROP PROCEDURE",
        Statement::Call(_) => "CALL",
        Statement::DoBlock(_) => "DO",
        Statement::CreateFulltextIndex(_) => "CREATE FULLTEXT INDEX",
        Statement::CreateVectorIndex(_) => "CREATE VECTOR INDEX",
        Statement::CreateSpatialIndex(_) => "CREATE SPATIAL INDEX",
        Statement::CreateHybridIndex(_) => "CREATE HYBRID INDEX",
        Statement::CreateGraphSchema(_) => "CREATE GRAPH SCHEMA",
        Statement::DropGraphSchema(_) => "DROP GRAPH SCHEMA",
        Statement::CreateAnalyzer(_) => "CREATE ANALYZER",
        Statement::AlterAnalyzer(_) => "ALTER ANALYZER",
        Statement::DropAnalyzer(_) => "DROP ANALYZER",
        Statement::CreateSynonymDictionary(_) => "CREATE SYNONYM DICTIONARY",
        Statement::AlterSynonymDictionary(_) => "ALTER SYNONYM DICTIONARY",
        Statement::DropSynonymDictionary(_) => "DROP SYNONYM DICTIONARY",
        Statement::CreateForeignTable(_) => "CREATE FOREIGN TABLE",
        Statement::DropForeignTable(_) => "DROP FOREIGN TABLE",

        Statement::Begin(_) => "BEGIN",
        Statement::Commit(_) => "COMMIT",
        Statement::Rollback(_) => "ROLLBACK",
        Statement::Savepoint(_) => "SAVEPOINT",
        Statement::ReleaseSavepoint(_) => "RELEASE SAVEPOINT",

        Statement::SetVariable(_) => "SET",
        Statement::Show(_) => "SHOW",
        Statement::Prepare(_) => "PREPARE",
        Statement::Execute(_) => "EXECUTE",
        Statement::Deallocate(_) => "DEALLOCATE",
        Statement::DeclareCursor(_) => "DECLARE",
        Statement::FetchCursor(_) => "FETCH",
        Statement::CloseCursor(_) => "CLOSE",
        Statement::Listen(_) => "LISTEN",
        Statement::Notify(_) => "NOTIFY",
        Statement::Explain(_) => "EXPLAIN",
        Statement::ExplainRewrite(_) => "EXPLAIN REWRITE",

        Statement::Grant(_) => "GRANT",
        Statement::Revoke(_) => "REVOKE",
        Statement::CreateUser(_) => "CREATE USER",
        Statement::AlterUser(_) => "ALTER USER",
        Statement::DropUser(_) => "DROP USER",
        Statement::CreateRole(_) => "CREATE ROLE",
        Statement::AlterRole(_) => "ALTER ROLE",
        Statement::DropRole(_) => "DROP ROLE",
        Statement::CreateAbacPolicy(_) => "CREATE ABAC POLICY",
        Statement::AlterSecurityMap(_) => "ALTER SECURITY MAP",
        Statement::DropSecurityMap(_) => "DROP SECURITY MAP",
        Statement::AlterColumnClassification(_) => "ALTER COLUMN CLASSIFICATION",
        Statement::LegalHold(_) => "LEGAL HOLD",
        Statement::ForgetUser(_) => "FORGET USER",
        Statement::ExportUser(_) => "EXPORT USER",
        Statement::SetSignatureScheme(_) => "SET SIGNATURE SCHEME",
        Statement::RotateSignatureScheme(_) => "ROTATE SIGNATURE SCHEME",
        Statement::RotateServicePrincipalKey(_) => "ROTATE SERVICE PRINCIPAL KEY",

        Statement::Vacuum(_) => "VACUUM",
        Statement::Analyze(_) => "ANALYZE",
        Statement::Checkpoint(_) => "CHECKPOINT",
        Statement::OptimizeTable(_) => "OPTIMIZE TABLE",
        Statement::CancelBackend { .. } => "CANCEL",
        Statement::AlterSystemSet(_) => "ALTER SYSTEM SET",
        Statement::AlterCluster(_) => "ALTER CLUSTER",
        Statement::ListRegistry(_) => "LIST REGISTRY",
        Statement::TriggerUpgrade(_) => "TRIGGER UPGRADE",
        Statement::AcknowledgeUpgradeRewrites(_) => "ACKNOWLEDGE UPGRADE REWRITES",
        Statement::ShowUpgrade(_) => "SHOW UPGRADE",
        Statement::CreatePeer(_) => "CREATE PEER",
        Statement::DropPeer(_) => "DROP PEER",
        Statement::CreateBulkhead(_) => "CREATE BULKHEAD",
        Statement::DropBulkhead(_) => "DROP BULKHEAD",
        Statement::CreateRetryPolicy(_) => "CREATE RETRY POLICY",
        Statement::DropRetryPolicy(_) => "DROP RETRY POLICY",
        Statement::EnableFeature(_) => "ENABLE FEATURE",
        Statement::DisableFeature(_) => "DISABLE FEATURE",
        Statement::CreateSchedule(_) => "CREATE SCHEDULE",
        Statement::DropSchedule(_) => "DROP SCHEDULE",
        Statement::PauseSchedule(_) => "PAUSE SCHEDULE",
        Statement::ResumeSchedule(_) => "RESUME SCHEDULE",
        Statement::CreatePipeline(_) => "CREATE PIPELINE",
        Statement::RunPipeline(_) => "RUN PIPELINE",
        Statement::DropPipeline(_) => "DROP PIPELINE",
        Statement::AddExpectation(_) => "ADD EXPECTATION",
        Statement::DropExpectation(_) => "DROP EXPECTATION",
        Statement::CreateEventHandler(_) => "CREATE EVENT HANDLER",
        Statement::DropEventHandler(_) => "DROP EVENT HANDLER",
        Statement::CreateFeatureGroup(_) => "CREATE FEATURE GROUP",
        Statement::DropFeatureGroup(_) => "DROP FEATURE GROUP",
        Statement::CreateModel(_) => "CREATE MODEL",
        Statement::DropModel(_) => "DROP MODEL",

        Statement::AlterTableTtl(_) => "ALTER TABLE SET TTL",
        Statement::AlterTableOptions(_) => "ALTER TABLE SET OPTIONS",
        Statement::AlterTableSetUsing(_) => "ALTER TABLE SET USING",
        Statement::AlterTableClusterBy(_) => "ALTER TABLE CLUSTER BY",
        Statement::AlterTableClusteringSchedule(_) => "ALTER TABLE CLUSTERING SCHEDULE",
        Statement::AlterTableMove(_) => "ALTER TABLE MOVE",
        Statement::AlterTableFollow(_) => "ALTER TABLE FOLLOW",
        Statement::ArchiveTable(_) => "ARCHIVE TABLE",
        Statement::RestoreTable(_) => "RESTORE TABLE",
        Statement::RestoreTableVersion(_) => "RESTORE TABLE VERSION",
        Statement::RestoreSoftDelete(_) => "RESTORE SOFT DELETE",
        Statement::RunRetentionJob(_) => "RUN RETENTION JOB",
        Statement::CreateBranch(_) => "CREATE BRANCH",
        Statement::MergeBranch(_) => "MERGE BRANCH",
        Statement::DropBranch(_) => "DROP BRANCH",
        Statement::UseBranch(_) => "USE BRANCH",
        Statement::CreateVersion(_) => "CREATE VERSION",
        Statement::DropVersion(_) => "DROP VERSION",

        Statement::CreateReplicationSlot(_) => "CREATE REPLICATION SLOT",
        Statement::DropReplicationSlot(_) => "DROP REPLICATION SLOT",
        Statement::CreateCdcStream(_) => "CREATE CDC STREAM",
        Statement::DropCdcStream(_) => "DROP CDC STREAM",
        Statement::CreateCdcIngest(_) => "CREATE CDC INGEST",
        Statement::DropCdcIngest(_) => "DROP CDC INGEST",
        Statement::CreateStreamingJob(_) => "CREATE STREAMING JOB",
        Statement::AlterStreamingJob(_) => "ALTER STREAMING JOB",
        Statement::DropStreamingJob(_) => "DROP STREAMING JOB",
        Statement::CreatePublication(_) => "CREATE PUBLICATION",
        Statement::AlterPublication(_) => "ALTER PUBLICATION",
        Statement::DropPublication(_) => "DROP PUBLICATION",
        Statement::TagPublication(_) => "TAG PUBLICATION",
        Statement::UntagPublication(_) => "UNTAG PUBLICATION",
        Statement::CreateExternalSource(_) => "CREATE EXTERNAL SOURCE",
        Statement::AlterExternalSource(_) => "ALTER EXTERNAL SOURCE",
        Statement::DropExternalSource(_) => "DROP EXTERNAL SOURCE",
        Statement::CreateExternalSink(_) => "CREATE EXTERNAL SINK",
        Statement::AlterExternalSink(_) => "ALTER EXTERNAL SINK",
        Statement::DropExternalSink(_) => "DROP EXTERNAL SINK",
        Statement::CreateEndpoint(_) => "CREATE ENDPOINT",
        Statement::CreateStreamingEndpoint(_) => "CREATE STREAMING ENDPOINT",
        Statement::AlterEndpoint(_) => "ALTER ENDPOINT",
        Statement::DropEndpoint(_) => "DROP ENDPOINT",
    }
}

/// Every registry name the match above can return.
///
/// Kept beside the match so the test can walk it. The match is what makes it
/// complete: a name here that no arm returns is dead, and a variant with no
/// arm does not compile.
const DOCUMENTED_STATEMENTS: &[&str] = &[
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "TRUNCATE",
    "COPY",
    "VALUES",
    "CREATE TABLE",
    "CREATE TEMPORARY TABLE",
    "DROP TABLE",
    "ALTER TABLE",
    "UNDROP TABLE",
    "CREATE INDEX",
    "DROP INDEX",
    "ALTER INDEX",
    "REINDEX",
    "CREATE VIEW",
    "DROP VIEW",
    "ALTER VIEW",
    "CREATE MATERIALIZED VIEW",
    "DROP MATERIALIZED VIEW",
    "REFRESH MATERIALIZED VIEW",
    "CREATE SCHEMA",
    "DROP SCHEMA",
    "CREATE SEQUENCE",
    "DROP SEQUENCE",
    "ALTER SEQUENCE",
    "CREATE TYPE",
    "DROP TYPE",
    "CREATE COLLATION",
    "DROP COLLATION",
    "COMMENT ON",
    "CREATE TRIGGER",
    "DROP TRIGGER",
    "CREATE FUNCTION",
    "DROP FUNCTION",
    "CREATE AGGREGATE",
    "DROP AGGREGATE",
    "CREATE PROCEDURE",
    "DROP PROCEDURE",
    "CALL",
    "DO",
    "CREATE FULLTEXT INDEX",
    "CREATE VECTOR INDEX",
    "CREATE SPATIAL INDEX",
    "CREATE HYBRID INDEX",
    "CREATE GRAPH SCHEMA",
    "DROP GRAPH SCHEMA",
    "CREATE ANALYZER",
    "ALTER ANALYZER",
    "DROP ANALYZER",
    "CREATE SYNONYM DICTIONARY",
    "ALTER SYNONYM DICTIONARY",
    "DROP SYNONYM DICTIONARY",
    "CREATE FOREIGN TABLE",
    "DROP FOREIGN TABLE",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "RELEASE SAVEPOINT",
    "SET",
    "SHOW",
    "PREPARE",
    "EXECUTE",
    "DEALLOCATE",
    "DECLARE",
    "FETCH",
    "CLOSE",
    "LISTEN",
    "NOTIFY",
    "EXPLAIN",
    "EXPLAIN REWRITE",
    "GRANT",
    "REVOKE",
    "CREATE USER",
    "ALTER USER",
    "DROP USER",
    "CREATE ROLE",
    "ALTER ROLE",
    "DROP ROLE",
    "CREATE ABAC POLICY",
    "ALTER SECURITY MAP",
    "DROP SECURITY MAP",
    "ALTER COLUMN CLASSIFICATION",
    "LEGAL HOLD",
    "FORGET USER",
    "EXPORT USER",
    "SET SIGNATURE SCHEME",
    "ROTATE SIGNATURE SCHEME",
    "ROTATE SERVICE PRINCIPAL KEY",
    "VACUUM",
    "ANALYZE",
    "CHECKPOINT",
    "OPTIMIZE TABLE",
    "CANCEL",
    "ALTER SYSTEM SET",
    "ALTER CLUSTER",
    "LIST REGISTRY",
    "TRIGGER UPGRADE",
    "ACKNOWLEDGE UPGRADE REWRITES",
    "SHOW UPGRADE",
    "CREATE PEER",
    "DROP PEER",
    "CREATE BULKHEAD",
    "DROP BULKHEAD",
    "CREATE RETRY POLICY",
    "DROP RETRY POLICY",
    "ENABLE FEATURE",
    "DISABLE FEATURE",
    "CREATE SCHEDULE",
    "DROP SCHEDULE",
    "PAUSE SCHEDULE",
    "RESUME SCHEDULE",
    "CREATE PIPELINE",
    "RUN PIPELINE",
    "DROP PIPELINE",
    "ADD EXPECTATION",
    "DROP EXPECTATION",
    "CREATE EVENT HANDLER",
    "DROP EVENT HANDLER",
    "CREATE FEATURE GROUP",
    "DROP FEATURE GROUP",
    "CREATE MODEL",
    "DROP MODEL",
    "ALTER TABLE SET TTL",
    "ALTER TABLE SET OPTIONS",
    "ALTER TABLE SET USING",
    "ALTER TABLE CLUSTER BY",
    "ALTER TABLE CLUSTERING SCHEDULE",
    "ALTER TABLE MOVE",
    "ALTER TABLE FOLLOW",
    "ARCHIVE TABLE",
    "RESTORE TABLE",
    "RESTORE TABLE VERSION",
    "RESTORE SOFT DELETE",
    "RUN RETENTION JOB",
    "CREATE BRANCH",
    "MERGE BRANCH",
    "DROP BRANCH",
    "USE BRANCH",
    "CREATE VERSION",
    "DROP VERSION",
    "CREATE REPLICATION SLOT",
    "DROP REPLICATION SLOT",
    "CREATE CDC STREAM",
    "DROP CDC STREAM",
    "CREATE CDC INGEST",
    "DROP CDC INGEST",
    "CREATE STREAMING JOB",
    "ALTER STREAMING JOB",
    "DROP STREAMING JOB",
    "CREATE PUBLICATION",
    "ALTER PUBLICATION",
    "DROP PUBLICATION",
    "TAG PUBLICATION",
    "UNTAG PUBLICATION",
    "CREATE EXTERNAL SOURCE",
    "ALTER EXTERNAL SOURCE",
    "DROP EXTERNAL SOURCE",
    "CREATE EXTERNAL SINK",
    "ALTER EXTERNAL SINK",
    "DROP EXTERNAL SINK",
    "CREATE ENDPOINT",
    "CREATE STREAMING ENDPOINT",
    "ALTER ENDPOINT",
    "DROP ENDPOINT",
];

fn entry(name: &str) -> Option<&'static GrammarEntry> {
    GRAMMAR.iter().find(|e| e.name == name)
}

#[test]
fn every_statement_the_parser_produces_has_a_registry_entry() {
    let missing: Vec<&str> = DOCUMENTED_STATEMENTS
        .iter()
        .copied()
        .filter(|name| entry(name).is_none())
        .collect();
    assert!(
        missing.is_empty(),
        "{} of {} statements have no registry entry, so the reference would \
         not document them: {:?}",
        missing.len(),
        DOCUMENTED_STATEMENTS.len(),
        missing
    );
}

#[test]
fn the_statement_list_and_the_match_name_the_same_entries() {
    // Each arm is read through the match itself, by parsing the example the
    // entry already carries and asking which entry documents the statement
    // that comes back. An arm naming something the registry does not hold
    // fails here, so a new variant cannot be satisfied by inventing a name
    let mut checked = 0usize;
    for item in GRAMMAR {
        if item.position != GrammarPosition::Statement {
            continue;
        }
        let Some(example) = item.examples.first() else {
            continue;
        };
        let parsed = parse(example.statement)
            .unwrap_or_else(|e| panic!("{}'s example does not parse: {}", item.name, e));
        let first = parsed
            .first()
            .unwrap_or_else(|| panic!("{}'s example parsed to nothing", item.name));
        // Several surfaces share one variant, so the name that comes back is
        // not always this entry's own. What it must always be is a page the
        // registry holds and the statement list names
        let named = documented_as(first);
        assert!(
            entry(named).is_some(),
            "the arm reached by {}'s example names {}, which no registry entry \
             holds, so the reference would not write that page",
            item.name,
            named
        );
        assert!(
            DOCUMENTED_STATEMENTS.contains(&named),
            "the arm reached by {}'s example names {}, which the statement list \
             does not, so the two halves disagree",
            item.name,
            named
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no arm was read through the match, so this gate checked nothing"
    );
    let mut seen: Vec<&str> = DOCUMENTED_STATEMENTS.to_vec();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!(
        before,
        seen.len(),
        "a statement is listed twice, so two arms would document one page"
    );
}

#[test]
fn every_example_parses_and_survives_the_unparser_where_it_reaches() {
    // Parsing is required of every example: an example the parser rejects is
    // a page telling a reader to write something that does not work.
    //
    // The round trip is required wherever the unparser handles the statement.
    // It covers a subset of the surface, so demanding it everywhere would
    // make this gate a claim about the unparser rather than about the
    // reference, and would fail for statements whose page is correct.
    let mut round_tripped = 0usize;
    let mut beyond_unparser = 0usize;
    for entry in GRAMMAR {
        for example in entry.examples {
            let parsed = parse(example.statement).unwrap_or_else(|e| {
                panic!(
                    "{}'s example does not parse: {}
  {}",
                    entry.name, e, example.statement
                )
            });
            let first = parsed
                .first()
                .unwrap_or_else(|| panic!("{}'s example parsed to nothing", entry.name));
            let Ok(rendered) = statement_to_sql(first) else {
                beyond_unparser += 1;
                continue;
            };
            let reparsed = parse(&rendered).unwrap_or_else(|e| {
                panic!(
                    "{}'s example does not parse after unparsing: {}
  {}",
                    entry.name, e, rendered
                )
            });
            assert_eq!(
                reparsed.first(),
                Some(first),
                "{}'s example does not survive a round trip:
  wrote {}
  read  {}",
                entry.name,
                example.statement,
                rendered
            );
            round_tripped += 1;
        }
    }
    println!(
        "{round_tripped} examples round-tripped, {beyond_unparser} parsed but are          beyond what the unparser writes back"
    );
    assert!(
        round_tripped > 0,
        "no example round-tripped, so this gate checked nothing"
    );
}

#[test]
fn every_refusal_quotes_text_the_source_still_contains() {
    // Read once, because a refusal fragment may be emitted from any crate
    let mut sources = String::new();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crates directory");
    collect_sources(root, &mut sources);
    assert!(
        !sources.is_empty(),
        "no source was read, so this gate would pass without checking anything"
    );
    for entry in GRAMMAR {
        for refusal in entry.refusals {
            assert!(
                sources.contains(refusal.message),
                "{} documents a refusal reading '{}', which no source emits; \
                 the reference and the error have drifted apart",
                entry.name,
                refusal.message
            );
        }
    }
}

fn collect_sources(dir: &std::path::Path, out: &mut String) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Build output holds copies of the sources and dwarfs them
            if name == "target" || name.starts_with('.') {
                continue;
            }
            collect_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && let Ok(text) = std::fs::read_to_string(&path)
        {
            out.push_str(&text);
        }
    }
}

#[test]
fn every_category_the_reference_writes_has_pages_in_it() {
    for category in Category::all() {
        let count = GRAMMAR.iter().filter(|e| e.category == *category).count();
        assert!(
            count > 0,
            "{} has no entries, so the reference would write an empty index for it",
            category.title()
        );
    }
}

#[test]
fn every_word_an_entry_lists_is_one_the_parser_reads() {
    // A word is one the parser reads when the lexer produces it as a keyword,
    // or when the parser matches it as a case-insensitive identifier. The
    // second kind is how a statement opens with a word that is not reserved,
    // such as LIST or ROTATE, and it is still a word the parser sees. What
    // this refuses is a word nothing in the parser looks for, which is what a
    // typo in the registry produces.
    let mut sources = String::new();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crates directory");
    collect_sources(root, &mut sources);
    for entry in GRAMMAR {
        for word in entry.keywords {
            if zyron_parser::grammar::keyword_of(word).is_some() {
                continue;
            }
            let soft = format!("\"{}\"", word.to_ascii_lowercase());
            assert!(
                sources.contains(&soft),
                "{} lists '{}', which is neither a keyword the lexer produces                  nor a word the parser matches as an identifier",
                entry.name,
                word
            );
        }
    }
}

#[test]
fn every_entry_reads_as_reference_text() {
    let mut findings: Vec<String> = Vec::new();
    for entry in GRAMMAR {
        let mut fields: Vec<(&str, &str)> = vec![
            ("summary", entry.summary),
            ("description", entry.description),
        ];
        if let Some(returns) = entry.returns {
            fields.push(("returns", returns));
        }
        for clause in entry.clauses {
            fields.push(("clause.what", clause.what));
            if let Some(default) = clause.default {
                fields.push(("clause.default", default));
            }
        }
        for example in entry.examples {
            fields.push(("example.yields", example.yields));
        }
        for (field, text) in fields {
            let lower = text.to_lowercase();
            for banned in NOT_REFERENCE_VOICE {
                if lower.contains(banned) {
                    findings.push(format!("{} {field}: '{}'", entry.name, banned.trim()));
                }
            }
        }
    }
    assert!(
        findings.is_empty(),
        "{} field(s) explain rather than state, or use a construction \
         documentation here does not:\n  {}",
        findings.len(),
        findings.join("\n  ")
    );
}
