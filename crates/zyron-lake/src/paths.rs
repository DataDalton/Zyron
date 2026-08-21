//! On-disk layout for lake tables.
//!
//! Every path the format touches is derived here so the writer, reader,
//! recovery and maintenance converge on one layout:
//!
//! ```text
//! <data_dir>/lake/t<table_id>/
//!   _zyron_log/
//!     _latest _last_checkpoint
//!     <version:020>.zyl            one binary version file per commit
//!     <version:020>.zym            binary manifest checkpoint
//!     branches/<name>/<v:020>.zyl  alternate log head, zero data copied
//!     _refs/<table_id:09>.ref      clone pin held by another table
//!     clustering/pass_<id>.zycluster
//!   stats/                         table and per-file statistics
//!   lineage/                       row-level lineage, optional
//!   data/p-<partition_id:016x>.zyr
//!   _staging/pass_<id>/            speculative candidates, never in a manifest
//!   _tmp/
//! <data_dir>/lake/_txn/<seq:08>.intent   cross-table commit intent
//! ```

use std::path::{Path, PathBuf};

/// Removes a file this process staged and no longer references, reporting
/// it if the removal fails.
///
/// Every caller is already on an abandon or error path, so a failure here
/// cannot be returned anywhere that would act on it. What it must not do is
/// happen silently. The file is named by no manifest, so nothing reads it
/// and no answer changes, but it holds disk until a REPAIR sweep reclaims
/// it, and an operator with no line in the log has no reason to run one.
///
/// A file that is already gone is the intended state, not a failure
pub(crate) fn discard_staged_file(path: &Path) {
    if let Err(e) = std::fs::remove_file(path) {
        if e.kind() == std::io::ErrorKind::NotFound {
            return;
        }
        tracing::warn!(
            target: "zyron::lake",
            path = %path.display(),
            error = %e,
            "a staged lake file could not be removed and is left unreferenced on disk, REPAIR reclaims it"
        );
    }
}

/// Removes a staging directory and everything under it, reporting it if the
/// removal fails. As [`discard_staged_file`], for a whole pass directory
pub(crate) fn discard_staged_dir(path: &Path) {
    if let Err(e) = std::fs::remove_dir_all(path) {
        if e.kind() == std::io::ErrorKind::NotFound {
            return;
        }
        tracing::warn!(
            target: "zyron::lake",
            path = %path.display(),
            error = %e,
            "a lake staging directory could not be removed and is left on disk, REPAIR reclaims it"
        );
    }
}

/// Width of the zero-padded decimal version component in .zyl and .zym
/// file names. 20 digits covers the full u64 range so names sort
/// lexicographically in version order
pub const VERSION_NAME_WIDTH: usize = 20;

/// Path derivations for one lake table. Constructed once per open table
/// and cheap to clone, every method is pure allocation-only arithmetic
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakePaths {
    root: PathBuf,
    /// Where data files live, when that is not under `root`.
    ///
    /// A follower on shared storage keeps its own log and reads the
    /// leader's data files in place. That is what makes syncing a table
    /// metadata only: the entries name immutable files both sides can
    /// already open, so nothing is copied
    shared_data: Option<PathBuf>,
}

impl LakePaths {
    /// Roots the layout at `<data_dir>/lake/t<table_id>`
    pub fn new(data_dir: &Path, table_id: u32) -> Self {
        Self {
            root: data_dir.join("lake").join(format!("t{}", table_id)),
            shared_data: None,
        }
    }

    /// Resolves data files under another table's root while keeping this
    /// table's own log, which is how a follower reads a leader's files
    /// without copying them.
    pub fn with_shared_data(mut self, leader: &LakePaths) -> Self {
        self.shared_data = Some(leader.data_dir());
        self
    }

    /// True when data files live outside this table's own root
    pub fn has_shared_data(&self) -> bool {
        self.shared_data.is_some()
    }

    /// The table id encoded in the root directory name, `t<id>`. None when
    /// the root was not produced by `new`
    pub fn table_id(&self) -> Option<u32> {
        self.root
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix('t'))
            .and_then(|n| n.parse().ok())
    }

    /// The table's root directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The transaction log directory
    pub fn log_dir(&self) -> PathBuf {
        self.root.join("_zyron_log")
    }

    /// Hint file holding the newest checkpoint version number
    pub fn last_checkpoint_hint(&self) -> PathBuf {
        self.log_dir().join("_last_checkpoint")
    }

    /// One committed version: `<version:020>.zyl` under the log dir
    pub fn version_file(&self, version: u64) -> PathBuf {
        self.log_dir().join(format!(
            "{:0width$}.zyl",
            version,
            width = VERSION_NAME_WIDTH
        ))
    }

    /// One manifest checkpoint: `<version:020>.zym` under the log dir
    pub fn checkpoint_file(&self, version: u64) -> PathBuf {
        self.log_dir().join(format!(
            "{:0width$}.zym",
            version,
            width = VERSION_NAME_WIDTH
        ))
    }

    /// A branch's log directory under the main log
    pub fn branch_dir(&self, branch: &str) -> PathBuf {
        self.log_dir().join("branches").join(branch)
    }

    /// One committed version on a branch head
    pub fn branch_version_file(&self, branch: &str, version: u64) -> PathBuf {
        self.branch_dir(branch).join(format!(
            "{:0width$}.zyl",
            version,
            width = VERSION_NAME_WIDTH
        ))
    }

    /// Clone pin held by another table, blocks vacuum of shared files
    pub fn clone_ref(&self, holder_table_id: u32) -> PathBuf {
        self.log_dir()
            .join("_refs")
            .join(format!("{:09}.ref", holder_table_id))
    }

    /// Resumable clustering pass checkpoint
    pub fn clustering_pass(&self, pass_id: u64) -> PathBuf {
        self.log_dir()
            .join("clustering")
            .join(format!("pass_{}.zycluster", pass_id))
    }

    /// Table and per-file statistics directory
    pub fn stats_dir(&self) -> PathBuf {
        self.root.join("stats")
    }

    /// Row-level lineage directory
    pub fn lineage_dir(&self) -> PathBuf {
        self.root.join("lineage")
    }

    /// Data file directory
    pub fn data_dir(&self) -> PathBuf {
        match &self.shared_data {
            Some(dir) => dir.clone(),
            None => self.root.join("data"),
        }
    }

    /// One immutable data file: `p-<partition_id:016x>.zyr`
    pub fn data_file(&self, partition_id: u64) -> PathBuf {
        self.data_dir().join(data_file_name(partition_id))
    }

    /// One immutable index file: `x-<index_id:08x>-<partition_id:016x>.zyr`
    ///
    /// Index files sit beside the data files they point into, so a
    /// follower reading a leader's data over shared storage reads its
    /// indexes in place too and the sync stays metadata only
    pub fn index_file(&self, index_id: u32, partition_id: u64) -> PathBuf {
        self.data_dir()
            .join(index_file_name(index_id, partition_id))
    }

    /// Staging directory for one speculative clustering pass, its files are
    /// renamed into data/ on accept and unlinked on reject
    pub fn staging_dir(&self, pass_id: u64) -> PathBuf {
        self.root.join("_staging").join(format!("pass_{}", pass_id))
    }

    /// Scratch directory for in-flight writes
    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("_tmp")
    }
}

/// File name one data file carries, wherever it currently sits. A
/// clustering pass writes candidates under `_staging/pass_<id>/` and
/// renames them into `data/` on accept, so the name has to be derivable
/// from the partition id alone rather than from the final directory
pub fn data_file_name(partition_id: u64) -> String {
    let mut name = String::with_capacity(DATA_FILE_NAME_LEN);
    write_data_file_name(&mut name, partition_id);
    name
}

/// Bytes a data file name occupies, which is what a caller building one
/// into a buffer reserves
pub const DATA_FILE_NAME_LEN: usize = "p-0000000000000000.zyr".len();

/// Bytes an index file name occupies
pub const INDEX_FILE_NAME_LEN: usize = "x-00000000-0000000000000000.zyr".len();

/// Writes a data file name into an existing buffer.
///
/// A caller naming a whole file set reuses one buffer rather than allocating
/// a string per file. Formatting into a `String` cannot fail, so the writer's
/// result carries no information
pub fn write_data_file_name(buf: &mut String, partition_id: u64) {
    use std::fmt::Write;
    let _ = write!(buf, "p-{:016x}.zyr", partition_id);
}

/// Writes an index file name into an existing buffer, as
/// [`write_data_file_name`] does for a data file
pub fn write_index_file_name(buf: &mut String, index_id: u32, partition_id: u64) {
    use std::fmt::Write;
    let _ = write!(buf, "x-{:08x}-{:016x}.zyr", index_id, partition_id);
}

/// File name one index file carries: `x-<index_id:08x>-<partition:016x>.zyr`.
///
/// The two ids are separate components rather than one so a sweep can tell
/// which index a stray file belongs to without opening it
pub fn index_file_name(index_id: u32, partition_id: u64) -> String {
    format!("x-{:08x}-{:016x}.zyr", index_id, partition_id)
}

/// Parses the ids out of an `x-<index:08x>-<partition:016x>.zyr` name.
/// Returns None for data files and foreign names, so a directory sweep can
/// separate the two kinds without opening either
pub fn parse_index_file_name(name: &str) -> Option<(u32, u64)> {
    let stem = name.strip_prefix("x-")?.strip_suffix(".zyr")?;
    let (index, partition) = stem.split_once('-')?;
    if index.len() != 8 || !index.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    if partition.len() != 16 || !partition.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some((
        u32::from_str_radix(index, 16).ok()?,
        u64::from_str_radix(partition, 16).ok()?,
    ))
}

/// Cross-table commit intent: `<data_dir>/lake/_txn/<seq:08>.intent`
pub fn txn_intent_file(data_dir: &Path, seq: u64) -> PathBuf {
    data_dir
        .join("lake")
        .join("_txn")
        .join(format!("{:08}.intent", seq))
}

/// Parses the version number out of a `<version:020>.zyl` or `.zym` file
/// name. Returns None for hint files, foreign names, or non-numeric stems,
/// so a directory scan can fold over version files and skip everything else
pub fn parse_version_file_name(name: &str) -> Option<(u64, VersionFileKind)> {
    let (stem, kind) = if let Some(s) = name.strip_suffix(".zyl") {
        (s, VersionFileKind::Version)
    } else if let Some(s) = name.strip_suffix(".zym") {
        (s, VersionFileKind::Checkpoint)
    } else {
        return None;
    };
    if stem.len() != VERSION_NAME_WIDTH || !stem.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    stem.parse::<u64>().ok().map(|v| (v, kind))
}

/// What a parsed log-directory file name refers to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionFileKind {
    Version,
    Checkpoint,
}

/// Parses the partition id out of a `p-<id:016x>.zyr` data file name.
/// Returns None for staging leftovers and foreign names, so a directory
/// sweep can fold over data files and skip everything else
pub fn parse_data_file_name(name: &str) -> Option<u64> {
    let stem = name.strip_prefix("p-")?.strip_suffix(".zyr")?;
    if stem.len() != 16 || !stem.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(stem, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_paths_are_stable() {
        let p = LakePaths::new(Path::new("/data"), 77);
        assert!(p.root().ends_with("lake/t77") || p.root().ends_with("lake\\t77"));
        assert!(
            p.version_file(41)
                .to_string_lossy()
                .ends_with("00000000000000000041.zyl")
        );
        assert!(
            p.checkpoint_file(40)
                .to_string_lossy()
                .ends_with("00000000000000000040.zym")
        );
        assert!(
            p.branch_version_file("staging", 41)
                .to_string_lossy()
                .contains("branches")
        );
        assert!(p.clone_ref(77).to_string_lossy().ends_with("000000077.ref"));
        assert!(
            p.data_file(0x2a)
                .to_string_lossy()
                .ends_with("p-000000000000002a.zyr")
        );
        assert!(
            txn_intent_file(Path::new("/data"), 123)
                .to_string_lossy()
                .ends_with("00000123.intent")
        );
    }

    #[test]
    fn test_version_names_sort_in_version_order() {
        let p = LakePaths::new(Path::new("/data"), 1);
        let a = p.version_file(9);
        let b = p.version_file(10);
        let c = p.version_file(u64::MAX);
        assert!(a.to_string_lossy() < b.to_string_lossy());
        assert!(b.to_string_lossy() < c.to_string_lossy());
    }

    #[test]
    fn test_parse_version_file_name() {
        assert_eq!(
            parse_version_file_name("00000000000000000041.zyl"),
            Some((41, VersionFileKind::Version))
        );
        assert_eq!(
            parse_version_file_name("00000000000000000040.zym"),
            Some((40, VersionFileKind::Checkpoint))
        );
        assert_eq!(parse_version_file_name("_latest"), None);
        assert_eq!(parse_version_file_name("_last_checkpoint"), None);
        assert_eq!(parse_version_file_name("41.zyl"), None);
        assert_eq!(parse_version_file_name("0000000000000000004x.zyl"), None);
        assert_eq!(parse_version_file_name("p-000000000000002a.zyr"), None);
    }

    #[test]
    fn test_parse_data_file_name() {
        assert_eq!(parse_data_file_name("p-000000000000002a.zyr"), Some(0x2a));
        assert_eq!(
            parse_data_file_name("p-ffffffffffffffff.zyr"),
            Some(u64::MAX)
        );
        assert_eq!(parse_data_file_name("p-2a.zyr"), None);
        assert_eq!(parse_data_file_name("p-000000000000002a.zyr.tmp"), None);
        assert_eq!(parse_data_file_name("00000000000000000041.zyl"), None);
    }

    #[test]
    fn test_index_and_data_file_names_never_collide() {
        let p = LakePaths::new(Path::new("/data"), 5);
        let index = p.index_file(3, 0x2a);
        let name = index.file_name().expect("name").to_string_lossy();
        assert_eq!(&*name, "x-00000003-000000000000002a.zyr");
        assert_eq!(parse_index_file_name(&name), Some((3, 0x2a)));
        // Neither parser accepts the other's names, so a directory sweep
        // classifies every file exactly once
        assert_eq!(parse_data_file_name(&name), None);
        assert_eq!(parse_index_file_name("p-000000000000002a.zyr"), None);
        assert_eq!(parse_index_file_name("x-3-2a.zyr"), None);
        assert_eq!(
            parse_index_file_name("x-0000000g-000000000000002a.zyr"),
            None
        );
    }

    #[test]
    fn test_parse_roundtrips_every_generated_name() {
        let p = LakePaths::new(Path::new("/data"), 3);
        for v in [0u64, 1, 41, 1_000_000, u64::MAX] {
            let name = p
                .version_file(v)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .into_iter()
                .next()
                .and_then(|n| parse_version_file_name(&n));
            assert_eq!(name, Some((v, VersionFileKind::Version)));
        }
    }
}
