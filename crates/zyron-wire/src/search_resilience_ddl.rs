//! DDL handlers for analyzers, synonym dictionaries, hybrid indexes,
//! resilience policies (bulkheads and retry policies), user defined types,
//! and collations.
//!
//! Every object is catalog backed. Analyzer and synonym configuration is
//! additionally snapshotted into FTS index parameters at index creation so
//! a restart rebuilds the exact pipeline each index was created with.

use std::collections::HashMap;
use std::sync::Arc;

use zyron_catalog::index_params::{
    FtsIndexParams, HybridIndexParams, decode_fts_params, decode_hybrid_params,
};
use zyron_common::ZyronError;
use zyron_parser::ast::{self, TableOption, TableOptionValue};

use crate::connection::ServerState;
use crate::ddl_dispatch::{DdlResult, check_ddl_privilege, resolve_qualified_name};
use crate::messages::ProtocolError;
use crate::session::Session;

// ---------------------------------------------------------------------------
// Option readers
// ---------------------------------------------------------------------------

fn opt_find<'a>(options: &'a [TableOption], key: &str) -> Option<&'a TableOptionValue> {
    options
        .iter()
        .find(|o| o.key.eq_ignore_ascii_case(key))
        .map(|o| &o.value)
}

fn opt_str(options: &[TableOption], key: &str) -> Option<String> {
    opt_find(options, key).map(|v| match v {
        TableOptionValue::String(s) | TableOptionValue::Identifier(s) => s.clone(),
        TableOptionValue::Integer(n) => n.to_string(),
        TableOptionValue::Float(f) => f.to_string(),
        TableOptionValue::Boolean(b) => b.to_string(),
        TableOptionValue::StringList(l) => l.join(","),
    })
}

fn opt_list(options: &[TableOption], key: &str) -> Option<Vec<String>> {
    opt_find(options, key).map(|v| match v {
        TableOptionValue::StringList(l) => l.clone(),
        TableOptionValue::String(s) | TableOptionValue::Identifier(s) => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        TableOptionValue::Integer(n) => vec![n.to_string()],
        TableOptionValue::Float(f) => vec![f.to_string()],
        TableOptionValue::Boolean(b) => vec![b.to_string()],
    })
}

fn opt_u64(options: &[TableOption], key: &str) -> Result<Option<u64>, ProtocolError> {
    match opt_find(options, key) {
        None => Ok(None),
        Some(TableOptionValue::Integer(n)) if *n >= 0 => Ok(Some(*n as u64)),
        Some(other) => Err(ddl_error(format!(
            "option {key} must be a non negative integer, got {other:?}"
        ))),
    }
}

fn opt_f64(options: &[TableOption], key: &str) -> Result<Option<f64>, ProtocolError> {
    match opt_find(options, key) {
        None => Ok(None),
        Some(TableOptionValue::Float(f)) => Ok(Some(*f)),
        Some(TableOptionValue::Integer(n)) => Ok(Some(*n as f64)),
        Some(other) => Err(ddl_error(format!(
            "option {key} must be a number, got {other:?}"
        ))),
    }
}

fn opt_bool(options: &[TableOption], key: &str) -> Result<Option<bool>, ProtocolError> {
    match opt_find(options, key) {
        None => Ok(None),
        Some(TableOptionValue::Boolean(b)) => Ok(Some(*b)),
        Some(other) => Err(ddl_error(format!(
            "option {key} must be true or false, got {other:?}"
        ))),
    }
}

fn ddl_error(message: String) -> ProtocolError {
    ProtocolError::Database(ZyronError::ExecutionError(message))
}

fn reject_unknown_options(
    options: &[TableOption],
    allowed: &[&str],
    statement: &str,
) -> Result<(), ProtocolError> {
    for opt in options {
        if !allowed.iter().any(|a| opt.key.eq_ignore_ascii_case(a)) {
            return Err(ddl_error(format!(
                "unknown {statement} option {}, expected one of {}",
                opt.key,
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

/// Parses a duration literal like '100ms', '5s', '2m', '1h', or a bare
/// integer of milliseconds
pub(crate) fn parse_duration_ms(text: &str) -> Result<u64, ProtocolError> {
    let t = text.trim();
    let (digits, scale) = if let Some(rest) = t.strip_suffix("ms") {
        (rest, 1u64)
    } else if let Some(rest) = t.strip_suffix('s') {
        (rest, 1_000)
    } else if let Some(rest) = t.strip_suffix('m') {
        (rest, 60_000)
    } else if let Some(rest) = t.strip_suffix('h') {
        (rest, 3_600_000)
    } else {
        (t, 1)
    };
    digits
        .trim()
        .parse::<u64>()
        .map(|n| n.saturating_mul(scale))
        .map_err(|_| {
            ddl_error(format!(
                "invalid duration '{text}', expected forms like 100ms, 5s, 2m, 1h"
            ))
        })
}

fn encode_params<T: serde::Serialize>(params: &T) -> Result<Vec<u8>, ProtocolError> {
    serde_json::to_vec(params)
        .map_err(|e| ddl_error(format!("failed to encode index parameters: {e}")))
}

// ---------------------------------------------------------------------------
// Analyzer resolution and construction
// ---------------------------------------------------------------------------

const BUILTIN_ANALYZERS: &[&str] = &["standard", "simple", "whitespace", "cjk"];

/// Resolves an analyzer name to the pipeline snapshot an index stores.
/// Builtin names snapshot as builtin, catalog analyzers snapshot their full
/// configuration so later ALTER or DROP of the analyzer object never
/// changes an existing index
fn resolve_analyzer_params(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    name: &str,
    synonyms_dictionary: String,
) -> Result<FtsIndexParams, ProtocolError> {
    let lower = name.to_lowercase();
    if BUILTIN_ANALYZERS.contains(&lower.as_str()) {
        return Ok(FtsIndexParams {
            analyzer_name: lower,
            builtin: true,
            tokenizer: String::new(),
            char_filters: Vec::new(),
            token_filters: Vec::new(),
            synonyms_dictionary,
        });
    }
    let (schema_id, bare) = resolve_qualified_name(name, server, session)?;
    let entry = server
        .catalog
        .get_analyzer(schema_id, &bare)
        .ok_or_else(|| {
            ddl_error(format!(
                "analyzer {name} does not exist, expected a builtin ({}) or a CREATE ANALYZER name",
                BUILTIN_ANALYZERS.join(", ")
            ))
        })?;
    Ok(FtsIndexParams {
        analyzer_name: entry.name.clone(),
        builtin: false,
        tokenizer: entry.tokenizer.clone(),
        char_filters: entry.char_filters.clone(),
        token_filters: entry.token_filters.clone(),
        synonyms_dictionary,
    })
}

/// Expands a synonym dictionary entry into the expansion map a
/// SynonymFilter consumes, honoring bidirectional and one way rules
pub fn dictionary_expansions(
    entry: &zyron_catalog::SynonymDictionaryEntry,
) -> HashMap<String, Vec<String>> {
    let mut set = zyron_search::SynonymSet::new(&entry.name);
    for rule in &entry.rules {
        let terms: Vec<&str> = rule.terms.iter().map(String::as_str).collect();
        if rule.targets.is_empty() {
            set.add_group(&terms);
        } else {
            let targets: Vec<&str> = rule.targets.iter().map(String::as_str).collect();
            for term in &terms {
                set.add_mapping(term, &targets);
            }
        }
    }
    set.all_expansions().clone()
}

/// Loads the expansions for a dictionary name through the session
/// namespace, erroring when it does not exist
fn load_dictionary_expansions(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    name: &str,
) -> Result<HashMap<String, Vec<String>>, ProtocolError> {
    let (schema_id, bare) = resolve_qualified_name(name, server, session)?;
    let entry = server
        .catalog
        .get_synonym_dictionary(schema_id, &bare)
        .ok_or_else(|| ddl_error(format!("synonym dictionary {name} does not exist")))?;
    Ok(dictionary_expansions(&entry))
}

/// Builds the runnable analyzer for a stored parameter snapshot. Returns
/// the analyzer and whether its pipeline encodes terms phonetically
pub fn build_index_analyzer(
    params: &FtsIndexParams,
    synonyms: Option<HashMap<String, Vec<String>>>,
) -> Result<(Arc<dyn zyron_search::Analyzer>, bool), ZyronError> {
    let phonetic = params
        .token_filters
        .iter()
        .any(|f| f.to_lowercase().starts_with("phonetic"));
    if params.builtin && synonyms.is_none() {
        let boxed = zyron_search::text::analyzer::analyzer_from_name(&params.analyzer_name)?;
        return Ok((Arc::from(boxed), false));
    }
    let config = if params.builtin {
        // A builtin with a synonym dictionary is rebuilt as the equivalent
        // configured pipeline so the synonym filter has a place in the chain
        match params.analyzer_name.as_str() {
            "standard" => zyron_search::AnalyzerConfig {
                tokenizer: "standard".to_string(),
                char_filters: vec![],
                token_filters: vec!["lowercase".into(), "stop".into(), "stem".into()],
            },
            "simple" => zyron_search::AnalyzerConfig {
                tokenizer: "standard".to_string(),
                char_filters: vec![],
                token_filters: vec!["lowercase".into()],
            },
            "whitespace" => zyron_search::AnalyzerConfig {
                tokenizer: "whitespace".to_string(),
                char_filters: vec![],
                token_filters: vec![],
            },
            _ => zyron_search::AnalyzerConfig {
                tokenizer: "cjk".to_string(),
                char_filters: vec![],
                token_filters: vec!["lowercase".into()],
            },
        }
    } else {
        zyron_search::AnalyzerConfig {
            tokenizer: params.tokenizer.clone(),
            char_filters: params.char_filters.clone(),
            token_filters: params.token_filters.clone(),
        }
    };
    let analyzer = zyron_search::build_analyzer(&params.analyzer_name, &config, synonyms)?;
    Ok((Arc::new(analyzer), phonetic))
}

/// Builds and installs the analyzer for a live FTS index from its stored
/// snapshot. Used at index creation and at server startup. A dictionary
/// that no longer exists degrades to no synonym expansion with a warning
/// rather than blocking the index
pub fn install_index_analyzer(
    catalog: &zyron_catalog::catalog::Catalog,
    fts: &zyron_search::FtsManager,
    index_id: u32,
    params: &FtsIndexParams,
) -> Result<(), ZyronError> {
    let synonyms = if params.synonyms_dictionary.is_empty() {
        None
    } else {
        let mut found = None;
        for dict in catalog.list_synonym_dictionaries() {
            if dict.name == params.synonyms_dictionary {
                found = Some(dictionary_expansions(&dict));
                break;
            }
        }
        if found.is_none() {
            tracing::warn!(
                index_id,
                dictionary = %params.synonyms_dictionary,
                "synonym dictionary referenced by the index no longer exists, \
                 the index analyzes without synonym expansion"
            );
        }
        found
    };
    let (analyzer, phonetic) = build_index_analyzer(params, synonyms)?;
    fts.set_index_analyzer(index_id, analyzer, phonetic);
    Ok(())
}

/// Reads the analyzer configuration of a CREATE FULLTEXT INDEX WITH clause.
/// Returns None when neither an analyzer nor a synonym dictionary is named,
/// which keeps the legacy unconfigured pipeline
pub(crate) fn fts_params_from_options(
    options: &[TableOption],
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<Option<FtsIndexParams>, ProtocolError> {
    reject_unknown_options(options, &["analyzer", "synonyms"], "FULLTEXT INDEX")?;
    let analyzer = opt_str(options, "analyzer");
    let synonyms_dictionary = opt_str(options, "synonyms").unwrap_or_default();
    if analyzer.is_none() && synonyms_dictionary.is_empty() {
        return Ok(None);
    }
    if !synonyms_dictionary.is_empty() {
        // Resolves now so a missing dictionary fails the CREATE loudly
        let _ = load_dictionary_expansions(server, session, &synonyms_dictionary)?;
    }
    let name = analyzer.unwrap_or_else(|| "simple".to_string());
    let params = resolve_analyzer_params(server, session, &name, synonyms_dictionary)?;
    Ok(Some(params))
}

// ---------------------------------------------------------------------------
// CREATE / ALTER / DROP ANALYZER
// ---------------------------------------------------------------------------

fn analyzer_config_from_options(
    options: &[TableOption],
    existing: Option<&zyron_catalog::AnalyzerEntry>,
) -> Result<(String, Vec<String>, Vec<String>), ProtocolError> {
    reject_unknown_options(
        options,
        &["tokenizer", "char_filters", "token_filters"],
        "ANALYZER",
    )?;
    let tokenizer = opt_str(options, "tokenizer")
        .or_else(|| existing.map(|e| e.tokenizer.clone()))
        .unwrap_or_else(|| "standard".to_string());
    let char_filters = opt_list(options, "char_filters")
        .or_else(|| existing.map(|e| e.char_filters.clone()))
        .unwrap_or_default();
    let token_filters = opt_list(options, "token_filters")
        .or_else(|| existing.map(|e| e.token_filters.clone()))
        .unwrap_or_default();
    // Validate the pipeline actually builds. A synonym filter entry gets a
    // placeholder map, the dictionary attaches at index creation
    let config = zyron_search::AnalyzerConfig {
        tokenizer: tokenizer.clone(),
        char_filters: char_filters.clone(),
        token_filters: token_filters.clone(),
    };
    let placeholder = if token_filters
        .iter()
        .any(|f| f.eq_ignore_ascii_case("synonym"))
    {
        Some(HashMap::new())
    } else {
        None
    };
    zyron_search::build_analyzer("validation", &config, placeholder)
        .map_err(ProtocolError::Database)?;
    Ok((tokenizer, char_filters, token_filters))
}

pub(crate) async fn handle_create_analyzer(
    stmt: &ast::CreateAnalyzerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server.catalog.get_analyzer(schema_id, &name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE ANALYZER".to_string()));
        }
        return Err(ddl_error(format!("analyzer {name} already exists")));
    }
    let (tokenizer, char_filters, token_filters) =
        analyzer_config_from_options(&stmt.options, None)?;
    let entry = zyron_catalog::AnalyzerEntry {
        id: 0,
        schema_id,
        name,
        tokenizer,
        char_filters,
        token_filters,
    };
    server
        .catalog
        .create_analyzer(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE ANALYZER".to_string()))
}

pub(crate) async fn handle_alter_analyzer(
    stmt: &ast::AlterAnalyzerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    let existing = server
        .catalog
        .get_analyzer(schema_id, &name)
        .ok_or_else(|| ddl_error(format!("analyzer {name} does not exist")))?;
    let (tokenizer, char_filters, token_filters) =
        analyzer_config_from_options(&stmt.options, Some(&existing))?;
    let entry = zyron_catalog::AnalyzerEntry {
        id: existing.id,
        schema_id,
        name,
        tokenizer,
        char_filters,
        token_filters,
    };
    server
        .catalog
        .update_analyzer(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("ALTER ANALYZER".to_string()))
}

pub(crate) async fn handle_drop_analyzer(
    stmt: &ast::DropAnalyzerStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server.catalog.get_analyzer(schema_id, &name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP ANALYZER".to_string()));
        }
        return Err(ddl_error(format!("analyzer {name} does not exist")));
    }
    server
        .catalog
        .drop_analyzer(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP ANALYZER".to_string()))
}

// ---------------------------------------------------------------------------
// CREATE / ALTER / DROP SYNONYM DICTIONARY
// ---------------------------------------------------------------------------

fn rules_from_ast(
    rules: &[ast::SynonymRule],
) -> Result<Vec<zyron_catalog::SynonymRuleEntry>, ProtocolError> {
    let mut entries = Vec::with_capacity(rules.len());
    for rule in rules {
        if rule.terms.iter().all(|t| t.trim().is_empty()) {
            return Err(ddl_error(
                "a synonym rule requires at least one term".to_string(),
            ));
        }
        if rule.targets.is_empty() && rule.terms.len() < 2 {
            return Err(ddl_error(format!(
                "bidirectional synonym group ('{}') needs at least two terms",
                rule.terms.join("', '")
            )));
        }
        entries.push(zyron_catalog::SynonymRuleEntry {
            terms: rule.terms.iter().map(|t| t.to_lowercase()).collect(),
            targets: rule.targets.iter().map(|t| t.to_lowercase()).collect(),
        });
    }
    Ok(entries)
}

/// Rebuilds the analyzer of every live FTS index whose snapshot references
/// this dictionary, so an ALTER takes effect for future writes and queries
fn refresh_indexes_for_dictionary(server: &ServerState, dictionary: &str) {
    for table in server.catalog.list_all_tables() {
        for idx in server.catalog.get_indexes_for_table(table.id) {
            let params = match idx.index_type {
                zyron_catalog::IndexType::Fulltext => decode_fts_params(&idx.parameters),
                zyron_catalog::IndexType::Hybrid => {
                    decode_hybrid_params(&idx.parameters).map(|h| h.fulltext)
                }
                _ => None,
            };
            if let Some(p) = params
                && p.synonyms_dictionary == dictionary
                && let Some(fts) = server.fts_manager.as_ref()
                && let Err(e) = install_index_analyzer(&server.catalog, fts, idx.id.0, &p)
            {
                tracing::warn!(
                    index = %idx.name,
                    "failed to refresh index analyzer after synonym dictionary change: {e}"
                );
            }
        }
    }
}

pub(crate) async fn handle_create_synonym_dictionary(
    stmt: &ast::CreateSynonymDictionaryStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server
        .catalog
        .get_synonym_dictionary(schema_id, &name)
        .is_some()
    {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE SYNONYM DICTIONARY".to_string()));
        }
        return Err(ddl_error(format!(
            "synonym dictionary {name} already exists"
        )));
    }
    let entry = zyron_catalog::SynonymDictionaryEntry {
        id: 0,
        schema_id,
        name,
        rules: rules_from_ast(&stmt.rules)?,
    };
    server
        .catalog
        .create_synonym_dictionary(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE SYNONYM DICTIONARY".to_string()))
}

pub(crate) async fn handle_alter_synonym_dictionary(
    stmt: &ast::AlterSynonymDictionaryStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    let existing = server
        .catalog
        .get_synonym_dictionary(schema_id, &name)
        .ok_or_else(|| ddl_error(format!("synonym dictionary {name} does not exist")))?;
    let mut rules = existing.rules.clone();
    match &stmt.action {
        ast::AlterSynonymDictionaryAction::Add(new_rules) => {
            rules.extend(rules_from_ast(new_rules)?);
        }
        ast::AlterSynonymDictionaryAction::Drop(terms) => {
            let dropped: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();
            for rule in &mut rules {
                rule.terms.retain(|t| !dropped.contains(t));
                rule.targets.retain(|t| !dropped.contains(t));
            }
            // A rule that lost its meaning goes away entirely: a one way
            // rule with no source or no target, or a group of fewer than two
            rules.retain(|r| {
                if r.targets.is_empty() {
                    r.terms.len() >= 2
                } else {
                    !r.terms.is_empty() && !r.targets.is_empty()
                }
            });
        }
    }
    let entry = zyron_catalog::SynonymDictionaryEntry {
        id: existing.id,
        schema_id,
        name: name.clone(),
        rules,
    };
    server
        .catalog
        .update_synonym_dictionary(entry)
        .await
        .map_err(ProtocolError::Database)?;
    refresh_indexes_for_dictionary(server, &name);
    Ok(DdlResult::Tag("ALTER SYNONYM DICTIONARY".to_string()))
}

pub(crate) async fn handle_drop_synonym_dictionary(
    stmt: &ast::DropSynonymDictionaryStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server
        .catalog
        .get_synonym_dictionary(schema_id, &name)
        .is_none()
    {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP SYNONYM DICTIONARY".to_string()));
        }
        return Err(ddl_error(format!(
            "synonym dictionary {name} does not exist"
        )));
    }
    // An index that snapshotted this dictionary keeps working from its
    // snapshot, but a restart could no longer rebuild the expansion, so a
    // referenced dictionary refuses to drop
    for table in server.catalog.list_all_tables() {
        for idx in server.catalog.get_indexes_for_table(table.id) {
            let referenced = match idx.index_type {
                zyron_catalog::IndexType::Fulltext => decode_fts_params(&idx.parameters)
                    .is_some_and(|p| p.synonyms_dictionary == name),
                zyron_catalog::IndexType::Hybrid => decode_hybrid_params(&idx.parameters)
                    .is_some_and(|p| p.fulltext.synonyms_dictionary == name),
                _ => false,
            };
            if referenced {
                return Err(ddl_error(format!(
                    "synonym dictionary {name} is used by index {} on table {}, drop the index first",
                    idx.name, table.name
                )));
            }
        }
    }
    server
        .catalog
        .drop_synonym_dictionary(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP SYNONYM DICTIONARY".to_string()))
}

// ---------------------------------------------------------------------------
// CREATE HYBRID INDEX
// ---------------------------------------------------------------------------

pub(crate) async fn handle_create_hybrid_index(
    stmt: &ast::CreateHybridIndexStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, table_name) = resolve_qualified_name(&stmt.table, server, session)?;
    let table = server
        .catalog
        .get_table(schema_id, &table_name)
        .map_err(ProtocolError::Database)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Table,
        table.id.0,
    )?;
    reject_unknown_options(
        &stmt.options,
        &[
            "fulltext_analyzer",
            "vector_distance",
            "fusion_method",
            "rrf_k",
            "synonyms",
        ],
        "HYBRID INDEX",
    )?;

    let text_col = table
        .columns
        .iter()
        .find(|c| c.name == stmt.text_column)
        .ok_or_else(|| {
            ddl_error(format!(
                "column {} not found in table {}",
                stmt.text_column, table_name
            ))
        })?;
    if !text_col.type_id.is_string() {
        return Err(ddl_error(format!(
            "hybrid index text column {} must be a text type, it is {}",
            text_col.name, text_col.type_id
        )));
    }
    let vector_col = table
        .columns
        .iter()
        .find(|c| c.name == stmt.vector_column)
        .ok_or_else(|| {
            ddl_error(format!(
                "column {} not found in table {}",
                stmt.vector_column, table_name
            ))
        })?;
    if vector_col.type_id != zyron_common::TypeId::Vector {
        return Err(ddl_error(format!(
            "hybrid index vector column {} must be a VECTOR column, it is {}",
            vector_col.name, vector_col.type_id
        )));
    }
    let dims = vector_col
        .max_length
        .map(|l| l as u16)
        .filter(|d| *d > 0)
        .ok_or_else(|| {
            ddl_error(format!(
                "vector column {} declares no dimension, declare it as VECTOR(n)",
                vector_col.name
            ))
        })?;

    let analyzer_name =
        opt_str(&stmt.options, "fulltext_analyzer").unwrap_or_else(|| "standard".to_string());
    let synonyms_dictionary = opt_str(&stmt.options, "synonyms").unwrap_or_default();
    let vector_distance =
        opt_str(&stmt.options, "vector_distance").unwrap_or_else(|| "cosine".to_string());
    let fusion_method =
        opt_str(&stmt.options, "fusion_method").unwrap_or_else(|| "rrf".to_string());
    if !matches!(fusion_method.as_str(), "rrf" | "linear") {
        return Err(ddl_error(format!(
            "fusion_method must be rrf or linear, got {fusion_method}"
        )));
    }
    let rrf_k = opt_u64(&stmt.options, "rrf_k")?.unwrap_or(60);
    if rrf_k == 0 {
        return Err(ddl_error("rrf_k must be at least 1".to_string()));
    }

    let fulltext =
        resolve_analyzer_params(server, session, &analyzer_name, synonyms_dictionary.clone())?;
    if !synonyms_dictionary.is_empty() {
        // Resolves now so a missing dictionary fails the CREATE loudly
        let _ = load_dictionary_expansions(server, session, &synonyms_dictionary)?;
    }
    let params = HybridIndexParams {
        fulltext: fulltext.clone(),
        vector_distance: vector_distance.clone(),
        fusion_method,
        rrf_k: rrf_k as u32,
        text_column_id: text_col.id.0,
        vector_column_id: vector_col.id.0,
        vector_dims: dims,
    };

    // Building until the rows that predate the index have been read into both
    // halves, so neither answers a query with less than the table holds
    let active_at_publication = server.txn_manager.proc_array().active_txn_ids();
    let index_id = server
        .catalog
        .create_index_with_params(
            table.id,
            schema_id,
            &stmt.name,
            &[stmt.text_column.clone(), stmt.vector_column.clone()],
            false,
            zyron_catalog::IndexType::Hybrid,
            Some(encode_params(&params)?),
            zyron_catalog::IndexState::Building,
        )
        .await
        .map_err(ProtocolError::Database)?;

    // Both engine halves register under the hybrid index id. Failure of
    // either rolls the whole index back so no half lives alone
    if let Some(ref fts) = server.fts_manager {
        if let Err(e) = fts.create_index(index_id.0, table.id.0, vec![text_col.id.0]) {
            let _ = server.catalog.drop_index(table.id, &stmt.name).await;
            return Err(ProtocolError::Database(e));
        }
        if let Err(e) = install_index_analyzer(&server.catalog, fts, index_id.0, &params.fulltext) {
            let _ = fts.drop_index(index_id.0);
            let _ = server.catalog.drop_index(table.id, &stmt.name).await;
            return Err(ProtocolError::Database(e));
        }
    }
    if let Some(ref vec_mgr) = server.vector_manager {
        let metric = match vector_distance.as_str() {
            "euclidean" | "l2" => zyron_search::vector::DistanceMetric::Euclidean,
            "dot_product" | "dot" => zyron_search::vector::DistanceMetric::DotProduct,
            "manhattan" | "l1" => zyron_search::vector::DistanceMetric::Manhattan,
            _ => zyron_search::vector::DistanceMetric::Cosine,
        };
        let config = zyron_search::vector::HnswConfig {
            m: 16,
            efConstruction: 200,
            efSearch: 64,
            metric,
        };
        if let Err(e) = vec_mgr.create_index(index_id.0, table.id.0, vector_col.id.0, dims, config)
        {
            if let Some(ref fts) = server.fts_manager {
                let _ = fts.drop_index(index_id.0);
            }
            let _ = server.catalog.drop_index(table.id, &stmt.name).await;
            return Err(ProtocolError::Database(e));
        }
    }

    let issuer = session
        .as_ref()
        .map(|s| s.user.clone())
        .unwrap_or_else(|| "unknown".to_string());
    if let Err(e) = crate::ddl_dispatch::fill_search_index_from_rows(
        server,
        table.id,
        &stmt.name,
        &active_at_publication,
        &crate::ddl_dispatch::transactions_held_open(session),
        &issuer,
    )
    .await
    {
        if let Some(m) = &server.fts_manager {
            let _ = m.drop_index(index_id.0);
        }
        if let Some(m) = &server.vector_manager {
            let _ = m.drop_index(index_id.0);
        }
        let _ = server.catalog.drop_index(table.id, &stmt.name).await;
        return Err(ProtocolError::Database(e));
    }
    Ok(DdlResult::Tag("CREATE INDEX".to_string()))
}

// ---------------------------------------------------------------------------
// CREATE / DROP BULKHEAD and RETRY POLICY
// ---------------------------------------------------------------------------

pub(crate) async fn handle_create_bulkhead(
    stmt: &ast::CreateBulkheadStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server
        .catalog
        .get_resilience_policy(schema_id, &name)
        .is_some()
    {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE BULKHEAD".to_string()));
        }
        return Err(ddl_error(format!(
            "resilience policy {name} already exists"
        )));
    }
    reject_unknown_options(
        &stmt.options,
        &["max_concurrent", "max_wait", "queue_size"],
        "BULKHEAD",
    )?;
    let max_concurrent = opt_u64(&stmt.options, "max_concurrent")?
        .ok_or_else(|| ddl_error("CREATE BULKHEAD requires max_concurrent".to_string()))?;
    if max_concurrent == 0 || max_concurrent > u32::MAX as u64 {
        return Err(ddl_error(
            "max_concurrent must be between 1 and 4294967295".to_string(),
        ));
    }
    let max_wait_ms = match opt_str(&stmt.options, "max_wait") {
        Some(text) => parse_duration_ms(&text)?,
        None => 0,
    };
    let queue_size = opt_u64(&stmt.options, "queue_size")?.unwrap_or(0);
    if queue_size > u32::MAX as u64 {
        return Err(ddl_error("queue_size does not fit in 32 bits".to_string()));
    }
    let entry = zyron_catalog::ResiliencePolicyEntry {
        id: 0,
        schema_id,
        name,
        kind: zyron_catalog::ResiliencePolicyKind::Bulkhead,
        max_concurrent: max_concurrent as u32,
        queue_size: queue_size as u32,
        max_wait_ms,
        max_attempts: 0,
        backoff: String::new(),
        base_delay_ms: 0,
        max_delay_ms: 0,
        jitter: 0.0,
        retryable_errors: Vec::new(),
    };
    server
        .catalog
        .create_resilience_policy(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE BULKHEAD".to_string()))
}

pub(crate) async fn handle_create_retry_policy(
    stmt: &ast::CreateRetryPolicyStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server
        .catalog
        .get_resilience_policy(schema_id, &name)
        .is_some()
    {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE RETRY POLICY".to_string()));
        }
        return Err(ddl_error(format!(
            "resilience policy {name} already exists"
        )));
    }
    reject_unknown_options(
        &stmt.options,
        &[
            "max_attempts",
            "backoff",
            "base_delay",
            "max_delay",
            "jitter",
            "retryable_errors",
        ],
        "RETRY POLICY",
    )?;
    let max_attempts = opt_u64(&stmt.options, "max_attempts")?
        .ok_or_else(|| ddl_error("CREATE RETRY POLICY requires max_attempts".to_string()))?;
    if max_attempts == 0 || max_attempts > 1000 {
        return Err(ddl_error(
            "max_attempts must be between 1 and 1000".to_string(),
        ));
    }
    let backoff = opt_str(&stmt.options, "backoff").unwrap_or_else(|| "exponential".to_string());
    if !matches!(backoff.as_str(), "fixed" | "linear" | "exponential") {
        return Err(ddl_error(format!(
            "backoff must be fixed, linear, or exponential, got {backoff}"
        )));
    }
    let base_delay_ms = match opt_str(&stmt.options, "base_delay") {
        Some(text) => parse_duration_ms(&text)?,
        None => 100,
    };
    let max_delay_ms = match opt_str(&stmt.options, "max_delay") {
        Some(text) => parse_duration_ms(&text)?,
        None => 10_000,
    };
    if max_delay_ms < base_delay_ms {
        return Err(ddl_error(
            "max_delay must be at least base_delay".to_string(),
        ));
    }
    let jitter = opt_f64(&stmt.options, "jitter")?.unwrap_or(0.0);
    if !(0.0..=1.0).contains(&jitter) {
        return Err(ddl_error("jitter must be between 0.0 and 1.0".to_string()));
    }
    let retryable_errors = opt_list(&stmt.options, "retryable_errors").unwrap_or_default();
    let entry = zyron_catalog::ResiliencePolicyEntry {
        id: 0,
        schema_id,
        name,
        kind: zyron_catalog::ResiliencePolicyKind::Retry,
        max_concurrent: 0,
        queue_size: 0,
        max_wait_ms: 0,
        max_attempts: max_attempts as u32,
        backoff,
        base_delay_ms,
        max_delay_ms,
        jitter,
        retryable_errors,
    };
    server
        .catalog
        .create_resilience_policy(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE RETRY POLICY".to_string()))
}

async fn drop_resilience_policy(
    name_ast: &str,
    if_exists: bool,
    expected: zyron_catalog::ResiliencePolicyKind,
    tag: &str,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(name_ast, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    let Some(entry) = server.catalog.get_resilience_policy(schema_id, &name) else {
        if if_exists {
            return Ok(DdlResult::Tag(tag.to_string()));
        }
        return Err(ddl_error(format!(
            "{} {name} does not exist",
            kind_word(expected)
        )));
    };
    if entry.kind != expected {
        return Err(ddl_error(format!(
            "{name} is a {}, not a {}",
            kind_word(entry.kind),
            kind_word(expected)
        )));
    }
    server
        .catalog
        .drop_resilience_policy(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    zyron_executor::resilience_exec::invalidate_policy(entry.id);
    Ok(DdlResult::Tag(tag.to_string()))
}

fn kind_word(kind: zyron_catalog::ResiliencePolicyKind) -> &'static str {
    match kind {
        zyron_catalog::ResiliencePolicyKind::Bulkhead => "bulkhead",
        zyron_catalog::ResiliencePolicyKind::Retry => "retry policy",
    }
}

pub(crate) async fn handle_drop_bulkhead(
    stmt: &ast::DropBulkheadStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    drop_resilience_policy(
        &stmt.name,
        stmt.if_exists,
        zyron_catalog::ResiliencePolicyKind::Bulkhead,
        "DROP BULKHEAD",
        server,
        session,
    )
    .await
}

pub(crate) async fn handle_drop_retry_policy(
    stmt: &ast::DropRetryPolicyStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    drop_resilience_policy(
        &stmt.name,
        stmt.if_exists,
        zyron_catalog::ResiliencePolicyKind::Retry,
        "DROP RETRY POLICY",
        server,
        session,
    )
    .await
}

// ---------------------------------------------------------------------------
// CREATE / DROP TYPE
// ---------------------------------------------------------------------------

fn validate_cast_expr(expr: &str, option_name: &str) -> Result<(), ProtocolError> {
    let sql = format!("SELECT ({expr})");
    zyron_parser::parse(&sql).map_err(|e| {
        ddl_error(format!(
            "CREATE TYPE {option_name} expression does not parse: {e}"
        ))
    })?;
    Ok(())
}

pub(crate) async fn handle_create_type(
    stmt: &ast::CreateTypeStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server.catalog.get_user_type(schema_id, &name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE TYPE".to_string()));
        }
        return Err(ddl_error(format!("type {name} already exists")));
    }
    let storage_type_id = stmt.storage.to_type_id();
    if storage_type_id == zyron_common::TypeId::Composite {
        return Err(ddl_error(
            "a user defined type cannot use another composite type as storage".to_string(),
        ));
    }
    if let Some(expr) = &stmt.check_expr {
        validate_cast_expr(expr, "check")?;
    }
    if let Some(expr) = &stmt.input_cast_expr {
        validate_cast_expr(expr, "input_cast")?;
    }
    if let Some(expr) = &stmt.output_cast_expr {
        validate_cast_expr(expr, "output_cast")?;
    }
    let entry = zyron_catalog::UserTypeEntry {
        id: 0,
        schema_id,
        name,
        storage_type_id,
        storage_max_length: stmt.storage.declared_max_length().map(|l| l as u32),
        storage_fractional_digits: stmt.storage.fractional_digits(),
        check_expr: stmt.check_expr.clone(),
        input_cast_expr: stmt.input_cast_expr.clone(),
        output_cast_expr: stmt.output_cast_expr.clone(),
    };
    server
        .catalog
        .create_user_type(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE TYPE".to_string()))
}

pub(crate) async fn handle_drop_type(
    stmt: &ast::DropTypeStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    let Some(entry) = server.catalog.get_user_type(schema_id, &name) else {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP TYPE".to_string()));
        }
        return Err(ddl_error(format!("type {name} does not exist")));
    };
    for table in server.catalog.list_all_tables() {
        for column in &table.columns {
            if column.attrs.user_type_id == Some(entry.id) {
                return Err(ddl_error(format!(
                    "type {name} is used by column {}.{}, drop or alter the column first",
                    table.name, column.name
                )));
            }
        }
    }
    server
        .catalog
        .drop_user_type(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP TYPE".to_string()))
}

// ---------------------------------------------------------------------------
// CREATE / DROP COLLATION
// ---------------------------------------------------------------------------

fn locale_is_wellformed(locale: &str) -> bool {
    let mut parts = locale.split(['_', '-']);
    let Some(lang) = parts.next() else {
        return false;
    };
    if !(2..=3).contains(&lang.len()) || !lang.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(region) => {
            region.len() == 2
                && region.chars().all(|c| c.is_ascii_alphabetic())
                && parts.next().is_none()
        }
    }
}

pub(crate) async fn handle_create_collation(
    stmt: &ast::CreateCollationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server.catalog.get_collation(schema_id, &name).is_some() {
        if stmt.if_not_exists {
            return Ok(DdlResult::Tag("CREATE COLLATION".to_string()));
        }
        return Err(ddl_error(format!("collation {name} already exists")));
    }
    reject_unknown_options(
        &stmt.options,
        &["locale", "provider", "deterministic", "case_sensitive"],
        "COLLATION",
    )?;
    let locale = opt_str(&stmt.options, "locale")
        .ok_or_else(|| ddl_error("CREATE COLLATION requires a locale option".to_string()))?;
    if !locale_is_wellformed(&locale) {
        return Err(ddl_error(format!(
            "locale {locale} is not a well formed language tag, expected forms like de or de_DE"
        )));
    }
    let provider = opt_str(&stmt.options, "provider").unwrap_or_else(|| "icu".to_string());
    if !matches!(provider.as_str(), "icu" | "binary") {
        return Err(ddl_error(format!(
            "provider must be icu or binary, got {provider}"
        )));
    }
    let deterministic = opt_bool(&stmt.options, "deterministic")?.unwrap_or(true);
    let case_sensitive = opt_bool(&stmt.options, "case_sensitive")?.unwrap_or(true);
    // Prove the collator actually constructs for this locale before the
    // entry lands in the catalog
    zyron_types::collation::validate_collation(&locale, &provider, case_sensitive)
        .map_err(ProtocolError::Database)?;
    let entry = zyron_catalog::CollationEntry {
        id: 0,
        schema_id,
        name,
        locale,
        provider,
        deterministic,
        case_sensitive,
    };
    server
        .catalog
        .create_collation(entry)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("CREATE COLLATION".to_string()))
}

pub(crate) async fn handle_drop_collation(
    stmt: &ast::DropCollationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (schema_id, name) = resolve_qualified_name(&stmt.name, server, session)?;
    check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Create,
        zyron_auth::ObjectType::Schema,
        schema_id.0,
    )?;
    if server.catalog.get_collation(schema_id, &name).is_none() {
        if stmt.if_exists {
            return Ok(DdlResult::Tag("DROP COLLATION".to_string()));
        }
        return Err(ddl_error(format!("collation {name} does not exist")));
    }
    for table in server.catalog.list_all_tables() {
        for column in &table.columns {
            if column.attrs.collation.as_deref() == Some(name.as_str()) {
                return Err(ddl_error(format!(
                    "collation {name} is used by column {}.{}, drop or alter the column first",
                    table.name, column.name
                )));
            }
        }
    }
    server
        .catalog
        .drop_collation(schema_id, &name)
        .await
        .map_err(ProtocolError::Database)?;
    Ok(DdlResult::Tag("DROP COLLATION".to_string()))
}
