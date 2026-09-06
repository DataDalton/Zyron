//! Steps that move a columnar file from one format version to the next.
//!
//! One file per step, each carrying the transformation, its registration,
//! and the fixture of the version it reads. Retiring a version deletes its
//! step file and its fixture from `../fixtures/` and narrows the reader
//! window, nothing else in the tree refers to them

mod v1_0_to_v1_1;
