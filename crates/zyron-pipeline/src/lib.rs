#![allow(non_snake_case)]
//! Pipeline, trigger, UDF, and stored procedure engine for ZyronDB.

pub mod aggregate;
pub mod dependency;
pub mod event_handler;
pub mod ids;
pub mod lineage;
pub mod materialized_view;
pub mod mv_advisor;
pub mod pipeline;
pub mod quality;
pub mod quality_drift;
pub mod refresh;
pub mod schedule;
pub mod sla;
pub mod storage;
pub mod stored_procedure;
pub mod trigger;
pub mod trigger_trace;
pub mod udf;
pub mod watermark;
