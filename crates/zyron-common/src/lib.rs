//! ZyronDB common types, errors, and utilities.
//!
//! This crate provides shared definitions used across all ZyronDB components.

pub mod config;
pub mod error;
pub mod page;
pub mod types;
pub mod zerocopy;

pub use config::{ServerConfig, StorageConfig};
pub use error::{Result, ZyronError};
pub use page::{PAGE_SIZE, PageHeader, PageId};
pub use types::TypeId;
