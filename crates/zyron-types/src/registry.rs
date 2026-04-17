//! Function registry: maps SQL function names to return types and classifies
//! them as scalar, aggregate, or window functions.
//!
//! The planner uses this to resolve return types during binding.
//! The executor uses this to dispatch calls to the correct module.

use zyron_common::TypeId;

/// Returns true if the given name is a known scalar function.
pub fn is_types_scalar_function(name: &str) -> bool {
    infer_types_scalar_return_type(name, &[]).is_some()
}

/// Returns true if the given name is a known aggregate function.
pub fn is_types_aggregate_function(name: &str) -> bool {
    infer_types_aggregate_return_type(name, &[]).is_some()
}

/// Returns true if the given name is a known window function.
pub fn is_types_window_function(name: &str) -> bool {
    infer_types_window_return_type(name, &[]).is_some()
}

/// Infers the return type of a scalar function given argument types.
pub fn infer_types_scalar_return_type(name: &str, arg_types: &[TypeId]) -> Option<TypeId> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        // ---------- bitfield ----------
        "bitfield_set"
        | "bitfield_clear"
        | "bitfield_toggle"
        | "bitfield_and"
        | "bitfield_or"
        | "bitfield_xor"
        | "bitfield_not"
        | "bitfield_from_positions" => Some(TypeId::Bitfield),
        "bitfield_test" | "bitfield_all" | "bitfield_any" => Some(TypeId::Boolean),
        "bitfield_count" => Some(TypeId::Int32),
        "bitfield_to_positions" => Some(TypeId::Array),

        // ---------- color ----------
        "color_rgb" | "color_rgba" | "color_hex" | "color_hsl" | "color_from_rgb"
        | "color_from_rgba" | "color_from_hex" | "color_from_hsl" | "color_blend"
        | "color_lighten" | "color_darken" => Some(TypeId::Color),
        "color_to_hex" => Some(TypeId::Varchar),
        "color_to_hsl" => Some(TypeId::Array),
        "wcag_contrast_ratio" => Some(TypeId::Float64),
        "wcag_compliant" => Some(TypeId::Boolean),
        "color_palette" => Some(TypeId::Array),

        // ---------- crypto ----------
        "sha256" | "sha384" | "sha512" | "blake3" | "hmac_sha256" => Some(TypeId::Bytea),
        "hash_combine" | "consistent_hash" => Some(TypeId::Int64),

        // ---------- encoding ----------
        "hex_encode" | "base58_encode" | "base32_encode" | "base64url_encode" => {
            Some(TypeId::Varchar)
        }
        "hex_decode" | "base58_decode" | "base32_decode" | "base64url_decode" => {
            Some(TypeId::Bytea)
        }
        "crc32" | "crc32c" | "murmur3_32" => Some(TypeId::Int32),
        "xxhash64" => Some(TypeId::Int64),
        "murmur3_128" => Some(TypeId::Int128),

        // ---------- fuzzy ----------
        "levenshtein" | "damerau_levenshtein" | "hamming" => Some(TypeId::Int32),
        "levenshtein_similarity" | "jaro_similarity" | "jaro_winkler" => Some(TypeId::Float64),
        "soundex" | "metaphone" | "nysiis" => Some(TypeId::Varchar),
        "double_metaphone" => Some(TypeId::Array),

        // ---------- similarity ----------
        "jaccard_similarity"
        | "sorensen_dice"
        | "cosine_similarity"
        | "overlap_coefficient"
        | "ngram_similarity" => Some(TypeId::Float64),
        "qgram_distance" => Some(TypeId::Int32),
        "fuzzy_join_candidates" => Some(TypeId::Array),

        // ---------- entity_resolution ----------
        "address_similarity" | "name_similarity" | "company_similarity" => Some(TypeId::Float64),
        "entity_resolve" | "merge_records" => Some(TypeId::Array),

        // ---------- fingerprint ----------
        "minhash_signature" | "minhash_encode" => Some(TypeId::Bytea),
        "minhash_decode" => Some(TypeId::Array),
        "minhash_similarity" => Some(TypeId::Float64),
        "simhash" => Some(TypeId::Int64),
        "simhash_distance" => Some(TypeId::Int32),
        "simhash_similar" => Some(TypeId::Boolean),
        "shingle" | "word_shingle" => Some(TypeId::Array),

        // ---------- id_gen ----------
        "gen_uuid_v4" | "uuid_v4" => Some(TypeId::Uuid),
        "gen_uuid_v7" | "uuid_v7" => Some(TypeId::Uuid),
        "uuid_to_string" => Some(TypeId::Varchar),
        "gen_ulid" | "ulid" => Some(TypeId::Varchar),
        "gen_snowflake" | "snowflake" | "gen_tsid" | "tsid" => Some(TypeId::Int64),
        "gen_cuid2" | "cuid2" => Some(TypeId::Varchar),
        "gen_nanoid" | "nanoid" => Some(TypeId::Varchar),
        "gen_ksuid" | "ksuid" => Some(TypeId::Varchar),

        // ---------- identifier ----------
        "validate_isbn" | "validate_iban" | "validate_ean" | "validate_vin" | "validate_issn"
        | "validate_swift" | "validate_ssn" => Some(TypeId::Boolean),
        "isbn_format" | "isbn_to_13" | "iban_country" | "iban_bban" | "vin_country"
        | "vin_manufacturer" => Some(TypeId::Varchar),
        "vin_year" => Some(TypeId::Int32),

        // ---------- rating ----------
        "elo_expected" | "elo_update" | "bayesian_average" | "win_rate" | "wilson_score" => {
            Some(TypeId::Float64)
        }
        "glicko2_update" | "trueskill_update" => Some(TypeId::Array),

        // ---------- semver ----------
        "semver_parse" => Some(TypeId::SemVer),
        "semver_format" => Some(TypeId::Varchar),
        "semver_major" | "semver_minor" | "semver_patch" => Some(TypeId::Int32),
        "semver_compare" => Some(TypeId::Int32),
        "semver_satisfies" | "semver_is_prerelease" => Some(TypeId::Boolean),
        "semver_increment_major" | "semver_increment_minor" | "semver_increment_patch" => {
            Some(TypeId::SemVer)
        }

        // ---------- network ----------
        "inet_parse" | "cidr_parse" => Some(TypeId::Inet),
        "inet_format" | "macaddr_format" => Some(TypeId::Varchar),
        "inet_family" => Some(TypeId::Int32),
        "inet_prefix" => Some(TypeId::Int32),
        "inet_contains" | "inet_is_private" | "inet_is_loopback" => Some(TypeId::Boolean),
        "inet_network" | "inet_broadcast" | "inet_netmask" | "inet_host" => Some(TypeId::Inet),
        "macaddr_parse" => Some(TypeId::MacAddr),
        "macaddr_oui" => Some(TypeId::Bytea),

        // ---------- money ----------
        "money_create" | "money_add" | "money_subtract" | "money_multiply" | "money_convert" => {
            Some(TypeId::Money)
        }
        "money_format" | "money_currency_code" | "money_currency_symbol" => Some(TypeId::Varchar),
        "money_minor_digits" => Some(TypeId::Int32),
        "currency_lookup" | "currency_by_numeric" => Some(TypeId::Composite),

        // ---------- quantity ----------
        "quantity_create" | "quantity_add" | "quantity_subtract" | "quantity_multiply"
        | "quantity_scale" => Some(TypeId::Quantity),
        "quantity_convert" => Some(TypeId::Float64),
        "quantity_format" | "quantity_dimension" | "quantity_unit_name" => Some(TypeId::Varchar),

        // ---------- cron ----------
        "cron_parse" => Some(TypeId::Bytea),
        "cron_next" | "cron_prev" => Some(TypeId::TimestampTz),
        "cron_matches" => Some(TypeId::Boolean),
        "cron_between" => Some(TypeId::Array),
        "cron_human_readable" => Some(TypeId::Varchar),

        // ---------- range ----------
        "range_create" | "range_union" | "range_intersection" => Some(TypeId::Range),
        "range_contains_value"
        | "range_contains_range"
        | "range_overlaps"
        | "range_adjacent"
        | "range_is_empty"
        | "range_lower_inclusive"
        | "range_upper_inclusive" => Some(TypeId::Boolean),
        "range_lower" | "range_upper" => Some(TypeId::Bytea),

        // ---------- regex_type ----------
        "regex_compile" => Some(TypeId::Bytea),
        "regex_match" | "regex_match_compiled" => Some(TypeId::Boolean),
        "regex_find" | "regex_find_compiled" => Some(TypeId::Composite),
        "regex_find_all" | "regex_capture" | "regex_split" => Some(TypeId::Array),
        "regex_replace" | "regex_replace_all" => Some(TypeId::Varchar),
        "regex_count" => Some(TypeId::Int32),

        // ---------- business_time ----------
        "fiscal_quarter" | "week_of_fiscal_year" => Some(TypeId::Int32),
        "fiscal_year" => Some(TypeId::Int32),
        "day_of_week" => Some(TypeId::Int32),
        "is_business_day" => Some(TypeId::Boolean),
        "next_business_day" | "add_business_days" | "parse_natural_date" => Some(TypeId::Date),
        "business_days_between" => Some(TypeId::Int32),
        "parse_natural_duration" => Some(TypeId::Interval),

        // ---------- json_schema ----------
        "json_schema_validate" | "validate_json_schema" => Some(TypeId::Boolean),
        "json_schema_errors" => Some(TypeId::Array),

        // ---------- diff ----------
        "text_diff" | "json_diff" | "text_patch" | "json_patch" | "json_merge_patch" => {
            Some(TypeId::Text)
        }
        "text_diff_words" => Some(TypeId::Array),

        // ---------- state_machine ----------
        "sm_parse" => Some(TypeId::Bytea),
        "sm_transition" => Some(TypeId::Varchar),
        "sm_can_transition" | "sm_is_terminal" => Some(TypeId::Boolean),
        "sm_available_events" | "sm_reachable_states" | "sm_shortest_path" => Some(TypeId::Array),

        // ---------- rate_limit ----------
        "token_bucket_create"
        | "token_bucket_consume"
        | "leaky_bucket_create"
        | "leaky_bucket_add" => Some(TypeId::Bytea),
        "token_bucket_available" => Some(TypeId::Float64),
        "sliding_window_count" | "fixed_window_count" => Some(TypeId::Int64),
        "sliding_window_check" => Some(TypeId::Boolean),

        // ---------- string_ops ----------
        "initcap" | "camel_case" | "snake_case" | "kebab_case" | "pascal_case" | "title_case"
        | "slug" | "truncate_words" | "truncate_chars" | "strip_html" => Some(TypeId::Varchar),
        "extract_emails" | "extract_urls" | "extract_phone_numbers" => Some(TypeId::Array),

        // ---------- formatting ----------
        "format_number" | "format_currency" | "format_bytes" | "format_duration"
        | "format_percentage" | "format_ordinal" => Some(TypeId::Varchar),
        "parse_number" | "convert_units" => Some(TypeId::Float64),

        // ---------- hierarchy ----------
        "materialized_path" => Some(TypeId::Varchar),
        "path_ancestors"
        | "closure_table_ancestors"
        | "closure_table_descendants"
        | "nested_set_subtree"
        | "nested_set_rebuild"
        | "closure_table_insert" => Some(TypeId::Array),
        "path_depth" | "closure_table_depth" => Some(TypeId::Int32),
        "is_ancestor" => Some(TypeId::Boolean),

        // ---------- data_quality ----------
        "validate_email"
        | "validate_url"
        | "validate_json"
        | "validate_uuid"
        | "validate_credit_card"
        | "is_valid_date" => Some(TypeId::Boolean),
        "data_profile" | "profile_column" | "data_contract" => Some(TypeId::Array),

        // ---------- probabilistic ----------
        "hll_create" | "hll_add" | "hll_merge" => Some(TypeId::HyperLogLog),
        "hll_count" => Some(TypeId::Int64),
        "hll_error" => Some(TypeId::Float64),
        "bloom_create" | "bloom_add" | "bloom_merge" => Some(TypeId::BloomFilter),
        "bloom_contains" => Some(TypeId::Boolean),
        "bloom_false_positive_rate" => Some(TypeId::Float64),
        "tdigest_create" | "tdigest_add" | "tdigest_merge" => Some(TypeId::TDigest),
        "tdigest_quantile" | "tdigest_cdf" => Some(TypeId::Float64),
        "cms_create" | "cms_add" | "cms_merge" => Some(TypeId::CountMinSketch),
        "cms_estimate" => Some(TypeId::Int64),

        // ---------- matrix ----------
        "matrix_create"
        | "matrix_identity"
        | "matrix_multiply"
        | "matrix_transpose"
        | "matrix_inverse"
        | "matrix_add"
        | "matrix_subtract"
        | "matrix_scalar_multiply" => Some(TypeId::Matrix),
        "matrix_determinant" | "matrix_trace" | "matrix_norm" | "dot_product" => {
            Some(TypeId::Float64)
        }
        "cross_product" | "eigenvalues" => Some(TypeId::Array),
        "svd" | "pca" => Some(TypeId::Composite),

        // ---------- financial ----------
        "npv" | "irr" | "xnpv" | "xirr" | "pmt" | "fv" | "pv" | "compound_interest"
        | "depreciation_sl" | "depreciation_db" | "depreciation_syd" | "bond_price"
        | "bond_yield" => Some(TypeId::Float64),
        "amortization_schedule" => Some(TypeId::Array),

        // ---------- statistics ----------
        "correlation" | "covariance" | "zscore" | "percentile" | "stddev_pop" | "stddev_sample"
        | "variance_pop" | "variance_sample" | "skewness" | "kurtosis" => Some(TypeId::Float64),
        "linear_regression" => Some(TypeId::Array),
        "exponential_smoothing"
        | "forecast_linear"
        | "moving_average"
        | "weighted_moving_average" => Some(TypeId::Array),
        "outlier_detect_zscore" | "outlier_detect_iqr" => Some(TypeId::Array),

        // ---------- window functions (routed through BoundExpr::WindowFunction) ----------
        // These can also appear as bare function calls; return types are declared here
        // so the binder can resolve them when nested inside a WindowFunction wrapper.
        "row_number" | "rank" | "dense_rank" | "ntile" => Some(TypeId::Int64),
        "percent_rank"
        | "cume_dist"
        | "ema"
        | "rate"
        | "delta"
        | "derivative"
        | "moving_average"
        | "moving_avg"
        | "exponential_smoothing" => Some(TypeId::Float64),
        "lag" | "lead" | "first_value" | "last_value" | "nth_value" => {
            arg_types.first().copied().or(Some(TypeId::Null))
        }

        // ---------- timeseries ----------
        "time_bucket" => Some(TypeId::Timestamp),
        "time_bucket_calendar" => Some(TypeId::Timestamp),
        "time_bucket_gapfill" => Some(TypeId::Array),
        "time_bucket_gapfill_calendar" => Some(TypeId::Array),
        "locf" | "interpolate" | "lttb" => Some(TypeId::Array),

        // ---------- geospatial ----------
        "st_make_point"
        | "st_geom_from_text"
        | "st_geom_from_geojson"
        | "st_buffer"
        | "st_union"
        | "st_centroid"
        | "h3_to_boundary" => Some(TypeId::Geometry),
        "st_as_text" | "st_as_geojson" => Some(TypeId::Text),
        "st_distance" | "st_area" => Some(TypeId::Float64),
        "st_dwithin" | "st_contains" | "st_intersects" => Some(TypeId::Boolean),
        "h3_from_point" => Some(TypeId::Int64),
        "h3_distance" => Some(TypeId::Int32),

        // ---------- url_type ----------
        "url_parse" => Some(TypeId::Composite),
        "url_scheme" | "url_host" | "url_path" | "url_normalize" | "url_domain" | "url_tld"
        | "url_resolve" | "url_fragment" | "url_query_param" => Some(TypeId::Varchar),
        "url_port" => Some(TypeId::Int32),
        "url_query_params" => Some(TypeId::Array),
        "url_is_absolute" => Some(TypeId::Boolean),

        _ => None,
    }
    .or_else(|| {
        // For pass-through functions, fall back to the first argument type
        // when the name suggests identity-like behavior.
        let _ = arg_types;
        None
    })
}

/// Infers the return type of an aggregate function given argument types.
pub fn infer_types_aggregate_return_type(name: &str, arg_types: &[TypeId]) -> Option<TypeId> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        // first/last return the type of their first argument (value)
        "first" | "last" => arg_types.first().copied().or(Some(TypeId::Null)),
        // time_weight returns a weighted average (f64)
        "time_weight" => Some(TypeId::Float64),
        // Probabilistic aggregates
        "hll_merge_agg" => Some(TypeId::HyperLogLog),
        "bloom_merge_agg" => Some(TypeId::BloomFilter),
        "tdigest_merge_agg" => Some(TypeId::TDigest),
        "cms_merge_agg" => Some(TypeId::CountMinSketch),
        // Statistical aggregates
        "stddev_agg" | "variance_agg" | "correlation_agg" | "covariance_agg" => {
            Some(TypeId::Float64)
        }
        _ => None,
    }
}

/// Infers the return type of a window function given argument types.
pub fn infer_types_window_return_type(name: &str, arg_types: &[TypeId]) -> Option<TypeId> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        // Float-returning window functions
        "ema"
        | "rate"
        | "delta"
        | "derivative"
        | "moving_average"
        | "moving_avg"
        | "exponential_smoothing"
        | "percent_rank"
        | "cume_dist" => Some(TypeId::Float64),
        // Integer ranking functions
        "row_number" | "rank" | "dense_rank" | "ntile" => Some(TypeId::Int64),
        // Navigation functions return the type of the first argument
        "lag" | "lead" | "first_value" | "last_value" | "nth_value" => {
            arg_types.first().copied().or(Some(TypeId::Null))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_known() {
        assert!(is_types_scalar_function("levenshtein"));
        assert!(is_types_scalar_function("format_bytes"));
        assert!(is_types_scalar_function("st_distance"));
        assert!(is_types_scalar_function("gen_uuid_v7"));
        assert!(is_types_scalar_function("npv"));
    }

    #[test]
    fn test_scalar_case_insensitive() {
        assert!(is_types_scalar_function("LEVENSHTEIN"));
        assert!(is_types_scalar_function("Format_Bytes"));
    }

    #[test]
    fn test_scalar_unknown() {
        assert!(!is_types_scalar_function("nonexistent_function"));
    }

    #[test]
    fn test_aggregate_known() {
        assert!(is_types_aggregate_function("first"));
        assert!(is_types_aggregate_function("last"));
        assert!(is_types_aggregate_function("time_weight"));
    }

    #[test]
    fn test_window_known() {
        assert!(is_types_window_function("ema"));
        assert!(is_types_window_function("rate"));
        assert!(is_types_window_function("delta"));
        assert!(is_types_window_function("derivative"));
    }

    #[test]
    fn test_infer_scalar_type() {
        assert_eq!(
            infer_types_scalar_return_type("levenshtein", &[]),
            Some(TypeId::Int32)
        );
        assert_eq!(
            infer_types_scalar_return_type("format_bytes", &[]),
            Some(TypeId::Varchar)
        );
        assert_eq!(
            infer_types_scalar_return_type("validate_email", &[]),
            Some(TypeId::Boolean)
        );
        assert_eq!(
            infer_types_scalar_return_type("gen_uuid_v7", &[]),
            Some(TypeId::Uuid)
        );
    }

    #[test]
    fn test_infer_aggregate_type() {
        assert_eq!(
            infer_types_aggregate_return_type("first", &[TypeId::Int64]),
            Some(TypeId::Int64)
        );
        assert_eq!(
            infer_types_aggregate_return_type("time_weight", &[]),
            Some(TypeId::Float64)
        );
    }

    #[test]
    fn test_infer_window_type() {
        assert_eq!(
            infer_types_window_return_type("ema", &[]),
            Some(TypeId::Float64)
        );
    }

    #[test]
    fn test_comprehensive_coverage() {
        // Sample of functions from every module - should all be registered
        let samples = [
            ("bitfield_set", true),
            ("color_rgb", true),
            ("sha256", true),
            ("hex_encode", true),
            ("levenshtein", true),
            ("jaccard_similarity", true),
            ("name_similarity", true),
            ("simhash", true),
            ("gen_uuid_v7", true),
            ("validate_isbn", true),
            ("elo_update", true),
            ("semver_parse", true),
            ("inet_parse", true),
            ("money_create", true),
            ("quantity_convert", true),
            ("cron_next", true),
            ("range_contains_value", true),
            ("regex_match", true),
            ("fiscal_quarter", true),
            ("json_schema_validate", true),
            ("text_diff", true),
            ("sm_transition", true),
            ("token_bucket_consume", true),
            ("slug", true),
            ("format_currency", true),
            ("path_depth", true),
            ("validate_email", true),
            ("hll_count", true),
            ("matrix_multiply", true),
            ("npv", true),
            ("correlation", true),
            ("time_bucket", true),
            ("st_distance", true),
            ("url_parse", true),
            ("definitely_not_a_function", false),
        ];

        for (name, expected) in samples {
            assert_eq!(
                is_types_scalar_function(name)
                    || is_types_aggregate_function(name)
                    || is_types_window_function(name),
                expected,
                "function '{}' registration mismatch",
                name
            );
        }
    }
}
