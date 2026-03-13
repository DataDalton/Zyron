//! Operator trait and execution batch types for the volcano-style pull executor.
//!
//! Each operator implements the Operator trait, producing ExecutionBatch results
//! one batch at a time. Operators form a tree where each node pulls data from
//! its children on demand.

pub mod aggregate;
pub mod distinct;
pub mod filter;
pub mod join;
pub mod limit;
pub mod modify;
pub mod project;
pub mod scan;
pub mod setop;
pub mod sort;

use std::future::Future;
use std::pin::Pin;

use zyron_common::Result;
use zyron_storage::TupleId;

use crate::batch::DataBatch;

/// Boxed future returned by Operator::next().
pub type OperatorResult<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ExecutionBatch>>> + Send + 'a>>;

/// A batch of rows produced by an operator, optionally carrying tuple IDs
/// for DML operators that need to identify specific rows in heap storage.
pub struct ExecutionBatch {
    /// Columnar batch containing the row data.
    pub batch: DataBatch,
    /// Optional tuple IDs for each row, used by UPDATE and DELETE operators
    /// to locate the source tuples in heap pages.
    pub tuple_ids: Option<Vec<TupleId>>,
}

impl ExecutionBatch {
    /// Creates a new ExecutionBatch without tuple IDs.
    pub fn new(batch: DataBatch) -> Self {
        Self {
            batch,
            tuple_ids: None,
        }
    }

    /// Creates a new ExecutionBatch with associated tuple IDs.
    pub fn with_tuple_ids(batch: DataBatch, tuple_ids: Vec<TupleId>) -> Self {
        Self {
            batch,
            tuple_ids: Some(tuple_ids),
        }
    }

    /// Returns the number of rows in this batch.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows
    }
}

/// Pull-based operator trait for the volcano execution model.
///
/// Each call to next() returns the next batch of rows, or None when exhausted.
/// Operators are composed into a tree, with leaf operators (scans) reading from
/// storage and interior operators (filter, project, join) transforming data
/// from their children.
///
/// Uses boxed futures for dyn-compatible async dispatch.
pub trait Operator: Send {
    /// Returns the next batch of rows, or None if the operator is exhausted.
    fn next(&mut self) -> OperatorResult<'_>;
}
