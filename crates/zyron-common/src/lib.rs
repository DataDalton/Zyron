//! ZyronDB common types, errors, and utilities.
//!
//! This crate provides shared definitions used across all ZyronDB components.

pub mod config;
pub mod error;
pub mod page;
pub mod types;

pub use config::{ServerConfig, StorageConfig};
pub use error::{Result, ZyronError};
pub use page::{PageHeader, PageId, PAGE_SIZE};
pub use types::TypeId;
