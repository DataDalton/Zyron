//! Typed views of `IndexEntry.parameters` for the index kinds that store
//! structured configuration: fulltext analyzer snapshots and hybrid fusion
//! settings. The wire layer writes these at CREATE INDEX and the executor
//! reads them at query time, so the schema lives here where both can see it.

/// Analyzer pipeline snapshot stored in a fulltext IndexEntry's parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FtsIndexParams {
    pub analyzer_name: String,
    /// True when analyzer_name is a builtin pipeline rather than a config
    pub builtin: bool,
    #[serde(default)]
    pub tokenizer: String,
    #[serde(default)]
    pub char_filters: Vec<String>,
    #[serde(default)]
    pub token_filters: Vec<String>,
    /// Synonym dictionary name attached to the index, empty when none
    #[serde(default)]
    pub synonyms_dictionary: String,
}

/// Fusion configuration stored in a hybrid IndexEntry's parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HybridIndexParams {
    pub fulltext: FtsIndexParams,
    pub vector_distance: String,
    pub fusion_method: String,
    pub rrf_k: u32,
    pub text_column_id: u16,
    pub vector_column_id: u16,
    pub vector_dims: u16,
}

pub fn decode_fts_params(parameters: &Option<Vec<u8>>) -> Option<FtsIndexParams> {
    parameters
        .as_ref()
        .and_then(|bytes| serde_json::from_slice(bytes).ok())
}

pub fn decode_hybrid_params(parameters: &Option<Vec<u8>>) -> Option<HybridIndexParams> {
    parameters
        .as_ref()
        .and_then(|bytes| serde_json::from_slice(bytes).ok())
}
