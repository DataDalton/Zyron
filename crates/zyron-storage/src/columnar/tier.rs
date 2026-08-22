//! Where a columnar segment file lives once it has been relocated to a
//! colder storage tier.
//!
//! A tier is a directory. The fold writes into the columnar root, and a
//! relocation moves the file into `<root>/tiers/<name>/` without rewriting a
//! byte of it. Pointing that directory at a slower or cheaper mount is what
//! makes the tier cheaper, so the engine needs no per-tier transport and a
//! read of a cold segment is the same positioned read as a hot one.
//!
//! Every segment of a table resolves back to the same columnar root whatever
//! tier it sits on, which is what keeps one patch log per table.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Directory under the columnar root that holds the per-tier directories.
pub const TIER_DIR_NAME: &str = "tiers";

/// The columnar root a segment file belongs to.
///
/// A hot segment sits directly in the root. A relocated one sits in
/// `<root>/tiers/<name>/`, so the root is recovered by stripping that pair.
/// Returns None only for a path with no parent at all.
pub fn columnar_root_for_segment(segment_path: &Path) -> Option<&Path> {
    let parent = segment_path.parent()?;
    match parent.parent() {
        Some(grandparent) if grandparent.file_name() == Some(OsStr::new(TIER_DIR_NAME)) => {
            grandparent.parent()
        }
        _ => Some(parent),
    }
}

/// The directory holding a tier's segment files. Hot is the columnar root
/// itself, which is where the fold writes and where a rehydrated segment
/// returns to.
pub fn tier_segment_dir(columnar_root: &Path, tier_name: &str) -> PathBuf {
    if tier_name.eq_ignore_ascii_case("hot") {
        return columnar_root.to_path_buf();
    }
    columnar_root.join(TIER_DIR_NAME).join(tier_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_hot_segment_resolves_to_its_own_directory() {
        let p = Path::new("/data/columnar/table_1_10_20_0.zyr");
        assert_eq!(
            columnar_root_for_segment(p),
            Some(Path::new("/data/columnar"))
        );
    }

    #[test]
    fn test_a_relocated_segment_resolves_to_the_same_root() {
        let hot = Path::new("/data/columnar/table_1_10_20_0.zyr");
        let cold = Path::new("/data/columnar/tiers/cold/table_1_10_20_0.zyr");
        assert_eq!(
            columnar_root_for_segment(cold),
            columnar_root_for_segment(hot),
            "one patch log per table however its segments are spread"
        );
    }

    /// A directory that merely happens to be two deep is not a tier, so its
    /// segments keep their own parent as the root.
    #[test]
    fn test_a_directory_named_something_else_is_not_a_tier() {
        let p = Path::new("/data/columnar/archive_backup/x.zyr");
        assert_eq!(
            columnar_root_for_segment(p),
            Some(Path::new("/data/columnar/archive_backup"))
        );
    }

    #[test]
    fn test_tier_directories_nest_under_the_root() {
        let root = Path::new("/data/columnar");
        assert_eq!(
            tier_segment_dir(root, "hot"),
            PathBuf::from("/data/columnar")
        );
        assert_eq!(
            tier_segment_dir(root, "cold"),
            PathBuf::from("/data/columnar/tiers/cold")
        );
        // Round trip: a file placed in the tier directory resolves back
        let placed = tier_segment_dir(root, "warm").join("f.zyr");
        assert_eq!(columnar_root_for_segment(&placed), Some(root));
    }
}
