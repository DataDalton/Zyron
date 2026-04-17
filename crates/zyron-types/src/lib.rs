//! Native data types and operations for ZyronDB.
//!
//! This crate provides domain-specific functions and type operations.
//! All functions accept primitive Rust types and return primitives.
//! The executor bridges Column types to these APIs.

pub mod registry;

pub use registry::{
    infer_types_aggregate_return_type, infer_types_scalar_return_type,
    infer_types_window_return_type, is_types_aggregate_function, is_types_scalar_function,
    is_types_window_function,
};

pub mod bitfield;
pub mod business_time;
pub mod color;
pub mod cron;
pub mod crypto;
pub mod data_quality;
pub mod diff;
pub mod encoding;
pub mod entity_resolution;
pub mod financial;
pub mod fingerprint;
pub mod formatting;
pub mod fuzzy;
pub mod geospatial;
pub mod hierarchy;
pub mod id_gen;
pub mod identifier;
pub mod json_schema;
pub mod matrix;
pub mod money;
pub mod network;
pub mod probabilistic;
pub mod quantity;
pub mod range;
pub mod rate_limit;
pub mod rating;
pub mod regex_type;
pub mod semver;
pub mod similarity;
pub mod state_machine;
pub mod statistics;
pub mod string_ops;
pub mod timeseries;
pub mod url_type;
