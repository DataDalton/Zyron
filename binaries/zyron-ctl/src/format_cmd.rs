#![allow(non_snake_case)]
//! Format, upgrade, deprecation, and release subcommands.
//!
//! Three of these run against files on disk with no server involved, which
//! is what makes them usable on a node that will not start: `format
//! inspect`, `format migrate`, and `format verify` read the format registry
//! this binary carries and nothing else. The rest go to a running server,
//! because upgrade state and deprecation warnings live there

use std::path::{Path, PathBuf};

use zyron_common::format::{
    FormatKind, FormatRegistry, FormatVersion, Framing, envelope, migration, stamp::FormatStamp,
};

/// What one file turned out to be
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inspection {
    pub path: PathBuf,
    pub kind: Option<FormatKind>,
    pub version: Option<FormatVersion>,
    pub framing: &'static str,
    pub size_bytes: u64,
    /// Whether both checksums verified, where the framing carries them
    pub integrity_ok: bool,
    pub detail: String,
}

impl Inspection {
    /// The line `format inspect` prints
    pub fn render(&self) -> String {
        match (self.kind, self.version) {
            (Some(kind), Some(version)) => format!(
                "{}\n  magic     {}\n  kind      {kind}\n  version   {version}\n  \
                 framing   {}\n  size      {} bytes\n  integrity {}\n  {}",
                self.path.display(),
                kind.magic_str(),
                self.framing,
                self.size_bytes,
                if self.integrity_ok { "ok" } else { "FAILED" },
                self.detail
            ),
            _ => format!(
                "{}\n  kind      unrecognized\n  size      {} bytes\n  {}",
                self.path.display(),
                self.size_bytes,
                self.detail
            ),
        }
    }
}

/// Identifies one file.
///
/// A file-framed format is decoded whole so both checksums are checked. A
/// stamped record is identified from its nine-byte stamp, whose integrity
/// belongs to the container that holds it, which the detail line says
pub fn inspect(path: &Path) -> Result<Inspection, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size_bytes = bytes.len() as u64;

    // An envelope and a stamp share their first eight bytes by design, so
    // which one a file carries is decided by the format's own framing rather
    // than by which parse happens to succeed
    if let Ok((kind, version)) = envelope::peek(&bytes) {
        let framing = kind.framing();
        let (integrity_ok, detail) = match framing {
            Framing::Envelope => match envelope::decode(&bytes) {
                Ok(parsed) => (
                    true,
                    format!(
                        "header and body checksums verified, {} body bytes",
                        parsed.body.len()
                    ),
                ),
                Err(e) => (false, e.to_string()),
            },
            Framing::Stamp => match FormatStamp::from_bytes(&bytes) {
                Ok(_) => (
                    true,
                    "stamped record, integrity belongs to the container that holds it".to_string(),
                ),
                Err(e) => (false, e.to_string()),
            },
            Framing::RecordTag => (
                true,
                "versioned record, integrity belongs to the stream that holds it".to_string(),
            ),
            Framing::Text => (
                true,
                "declared [format] section, the file is hand editable".to_string(),
            ),
            Framing::OwnTrailer => match envelope::decode_header(&bytes) {
                Ok(_) => (
                    true,
                    "header verified, the trailer's integrity belongs to the format's own reader"
                        .to_string(),
                ),
                Err(e) => (false, e.to_string()),
            },
            Framing::EnvelopeChain => match inspect_chain(&bytes) {
                Ok(records) => (
                    true,
                    format!(
                        "{records} envelope(s) in the chain, every header and body checksum \
                         verified"
                    ),
                ),
                Err((records, detail)) => (
                    false,
                    format!("{records} envelope(s) verified, then {detail}"),
                ),
            },
        };
        return Ok(Inspection {
            path: path.to_path_buf(),
            kind: Some(kind),
            version: Some(version),
            framing: framing.label(),
            size_bytes,
            integrity_ok,
            detail,
        });
    }

    if let Ok(text) = std::str::from_utf8(&bytes) {
        if let Ok((kind, version)) = zyron_common::format::text_envelope::parse(text) {
            return Ok(Inspection {
                path: path.to_path_buf(),
                kind: Some(kind),
                version: Some(version),
                framing: "text",
                size_bytes,
                integrity_ok: true,
                detail: "declared [format] section, the file is hand editable".to_string(),
            });
        }
    }

    Ok(Inspection {
        path: path.to_path_buf(),
        kind: None,
        version: None,
        framing: "unknown",
        size_bytes,
        integrity_ok: false,
        detail: "the first bytes name no registered format".to_string(),
    })
}

/// Walks a chain of envelopes, each carrying its body length in its header
/// extension, verifying every one. Answers with the record count, or with
/// the records verified before the one that fails and what fails in it. A
/// chain whose last record is cut short is what a stop mid append leaves,
/// which the format's own reader cuts off on its next open
fn inspect_chain(bytes: &[u8]) -> Result<u64, (u64, String)> {
    let mut offset = 0usize;
    let mut records = 0u64;
    while offset < bytes.len() {
        let (header, extension) = envelope::decode_header(&bytes[offset..]).map_err(|e| {
            (
                records,
                format!("the record at byte {offset} does not decode, {e}"),
            )
        })?;
        if extension.len() != 4 {
            return Err((
                records,
                format!(
                    "the record at byte {offset} carries a {} byte header extension rather \
                     than its body length",
                    extension.len()
                ),
            ));
        }
        let body_len =
            u32::from_le_bytes([extension[0], extension[1], extension[2], extension[3]]) as usize;
        let total = header.body_offset() + body_len + envelope::ENVELOPE_FOOTER_LEN;
        if offset + total > bytes.len() {
            return Err((
                records,
                format!(
                    "the record at byte {offset} declares {total} bytes and {} remain",
                    bytes.len() - offset
                ),
            ));
        }
        envelope::decode_as(&bytes[offset..offset + total], header.kind).map_err(|e| {
            (
                records,
                format!("the record at byte {offset} does not verify, {e}"),
            )
        })?;
        offset += total;
        records += 1;
    }
    Ok(records)
}

/// What `format verify` found across a directory
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyReport {
    pub files_checked: u64,
    pub files_ok: u64,
    pub files_unrecognized: u64,
    pub failures: Vec<String>,
    /// Per format kind, how many files and at which versions
    pub by_kind: Vec<(FormatKind, FormatVersion, u64)>,
}

impl VerifyReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "{} file(s) checked, {} ok, {} unrecognized, {} failure(s)\n",
            self.files_checked,
            self.files_ok,
            self.files_unrecognized,
            self.failures.len()
        );
        for (kind, version, count) in &self.by_kind {
            out.push_str(&format!("  {kind} at {version}: {count} file(s)\n"));
        }
        for failure in &self.failures {
            out.push_str(&format!("  FAILED {failure}\n"));
        }
        out
    }
}

/// Verifies every Zyron file under a directory against the registry
pub fn verify(registry: &FormatRegistry, directory: &Path) -> Result<VerifyReport, String> {
    let mut report = VerifyReport::default();
    for path in walk(directory) {
        let inspection = match inspect(&path) {
            Ok(inspection) => inspection,
            Err(e) => {
                report.failures.push(e);
                continue;
            }
        };
        let (Some(kind), Some(version)) = (inspection.kind, inspection.version) else {
            report.files_unrecognized += 1;
            continue;
        };
        report.files_checked += 1;
        if !inspection.integrity_ok {
            report
                .failures
                .push(format!("{}, {}", path.display(), inspection.detail));
            continue;
        }
        match registry.get(kind) {
            Some(entry) => {
                if !entry
                    .registration
                    .reader_supported_versions
                    .contains(version)
                {
                    report.failures.push(format!(
                        "{} is a {kind} file at {version}, outside the reader window {}",
                        path.display(),
                        entry.registration.reader_supported_versions
                    ));
                    continue;
                }
            }
            None => {
                report.failures.push(format!(
                    "{} is a {kind} file, which this binary does not register",
                    path.display()
                ));
                continue;
            }
        }
        report.files_ok += 1;
        match report
            .by_kind
            .iter_mut()
            .find(|(k, v, _)| *k == kind && *v == version)
        {
            Some(slot) => slot.2 += 1,
            None => report.by_kind.push((kind, version, 1)),
        }
    }
    report
        .by_kind
        .sort_by_key(|(kind, version, _)| (*kind, *version));
    Ok(report)
}

/// What `format migrate` did
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MigrateReport {
    pub kind: Option<FormatKind>,
    pub target: Option<FormatVersion>,
    pub files_scanned: u64,
    pub files_migrated: u64,
    pub files_already_current: u64,
    pub failures: Vec<String>,
}

impl MigrateReport {
    pub fn render(&self) -> String {
        let target = self
            .target
            .map(|v| v.to_string())
            .unwrap_or_else(|| "the current version".to_string());
        let mut out = format!(
            "{} file(s) scanned, {} migrated to {target}, {} already current, {} failure(s)\n",
            self.files_scanned,
            self.files_migrated,
            self.files_already_current,
            self.failures.len()
        );
        for failure in &self.failures {
            out.push_str(&format!("  FAILED {failure}\n"));
        }
        out
    }
}

/// Moves every file of one kind under a directory to a target version.
///
/// The target has to be the version this binary writes: moving a file to a
/// version this binary does not produce would leave bytes nothing here can
/// read back, so it is refused rather than attempted
pub fn migrate(
    registry: &FormatRegistry,
    kind: FormatKind,
    target: Option<FormatVersion>,
    directory: &Path,
) -> Result<MigrateReport, String> {
    let entry = registry
        .get(kind)
        .ok_or_else(|| format!("format `{kind}` is not registered in this binary"))?;
    let current = entry.registration.writer_current_version;
    if let Some(requested) = target {
        if requested != current {
            return Err(format!(
                "this binary writes {kind} at {current}, not {requested}. Run the release \
                 that writes {requested} to move files there"
            ));
        }
    }

    let mut report = MigrateReport {
        kind: Some(kind),
        target: Some(current),
        ..MigrateReport::default()
    };
    for path in walk(directory) {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        match envelope::peek(&bytes) {
            Ok((found, _)) if found == kind => {}
            _ => continue,
        }
        report.files_scanned += 1;
        let opened = match migration::open_as(registry, &bytes, kind) {
            Ok(opened) => opened,
            Err(e) => {
                report.failures.push(format!("{}, {e}", path.display()));
                continue;
            }
        };
        if !opened.path.needs_migration() {
            report.files_already_current += 1;
            continue;
        }
        let rewritten = opened.reencode();
        let tmp = path.with_extension("zymig.tmp");
        if let Err(e) = std::fs::write(&tmp, &rewritten) {
            report.failures.push(format!("{}, {e}", tmp.display()));
            continue;
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            report.failures.push(format!("{}, {e}", path.display()));
            continue;
        }
        report.files_migrated += 1;
    }
    Ok(report)
}

/// Every file under a directory, deepest last
fn walk(directory: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The SQL each server-side subcommand runs.
///
/// Keeping them here rather than inline in the dispatch means the CLI and
/// the SQL surface cannot drift: the command is the statement
pub mod statements {
    /// `zyron-ctl upgrade show-state`
    pub const SHOW_STATE: &str = "SHOW UPGRADE STATE";
    /// `zyron-ctl upgrade rollback`
    pub const ROLLBACK: &str = "TRIGGER MANUAL ROLLBACK";
    /// `zyron-ctl upgrade pause`
    pub const PAUSE: &str = "ALTER SYSTEM SET auto_upgrade_paused = true";
    /// `zyron-ctl upgrade resume`
    pub const RESUME: &str = "ALTER SYSTEM SET auto_upgrade_paused = false";
    /// `zyron-ctl deprecation report`
    pub const DEPRECATION_REPORT: &str = "SELECT * FROM zyron_sys.upgrade.deprecation_warnings";
    /// The format registry, which `upgrade check` reads to report what the
    /// cluster has on disk
    pub const FORMAT_REGISTRY: &str = "SELECT * FROM zyron_sys.storage.format_registry";
    /// In-flight format migrations
    pub const FORMAT_MIGRATIONS: &str = "SHOW FORMAT MIGRATIONS";
    /// The rewrite queue, which is the other half of what the gate found
    pub const REWRITES: &str = "SELECT * FROM zyron_sys.upgrade.user_object_rewrites";

    /// `zyron-ctl upgrade trigger --version X.Y.Z`
    pub fn trigger(version: &str) -> String {
        format!("TRIGGER MANUAL UPGRADE TO '{version}'")
    }

    /// `zyron-ctl upgrade acknowledge --category <ambiguous|unsafe>`
    pub fn acknowledge(category: &str) -> String {
        format!(
            "ACKNOWLEDGE UPGRADE REWRITES {}",
            category.to_ascii_uppercase()
        )
    }

    /// `zyron-ctl deprecation guides <item>`
    pub fn guide(item: &str) -> String {
        format!(
            "SELECT item_id, title, url, body FROM zyron_sys.deprecation.migration_guides \
             WHERE item_id = '{}'",
            item.replace('\'', "''")
        )
    }
}

/// What `release stage` put in place
#[derive(Debug)]
pub struct StagedForFeed {
    pub version: String,
    pub channel: String,
    pub manifestPath: PathBuf,
    pub binaryPath: PathBuf,
}

/// Puts a signed manifest and a release binary where a node's local feed
/// reads them, the way an operator delivers a release to an air-gapped
/// cluster.
///
/// The manifest goes under `releases/<channel>.manifest` in the data
/// directory and the binary beside it as `zyron-server-<version>`. The
/// binary is checked against the digest the manifest declares before it is
/// copied, so a wrong file is refused here rather than by the node. The
/// manifest's signature is the node's to check, with the release key it is
/// configured with
pub fn stage_release(
    manifest: &Path,
    binary: &Path,
    data_dir: &Path,
    version: Option<&str>,
) -> Result<StagedForFeed, String> {
    use zyron_server::upgrade::feed::{LocalFeedSource, parse_manifest, verify_sha256};
    use zyron_server::upgrade::stager::LocalArtifactSource;

    let body = std::fs::read_to_string(manifest)
        .map_err(|e| format!("reading {}: {e}", manifest.display()))?;
    let document = parse_manifest(&body).map_err(|e| e.to_string())?;
    let manifest_value: zyron_common::format::ReleaseManifest = document.into();
    if manifest_value.releases.is_empty() {
        return Err("the manifest carries no releases".to_string());
    }
    let version = match version {
        Some(version) => version.to_string(),
        None if manifest_value.releases.len() == 1 => manifest_value.releases[0].version.clone(),
        None => {
            let versions: Vec<&str> = manifest_value
                .releases
                .iter()
                .map(|r| r.version.as_str())
                .collect();
            return Err(format!(
                "the manifest carries {} releases ({}), say which one the binary is with \
                 --version",
                versions.len(),
                versions.join(", ")
            ));
        }
    };
    let release = manifest_value.release(&version).ok_or_else(|| {
        format!(
            "the manifest for the {} channel carries no release {version}",
            manifest_value.channel
        )
    })?;
    verify_sha256(binary, &release.sha256).map_err(|e| e.to_string())?;

    let feed_dir = data_dir.join("releases");
    let feed = LocalFeedSource::new(feed_dir.clone());
    let manifest_path = feed.publish(&manifest_value).map_err(|e| e.to_string())?;
    let artifacts = LocalArtifactSource::new(feed_dir);
    let binary_path = artifacts.path_for(&version);
    let partial = binary_path.with_extension("partial");
    std::fs::copy(binary, &partial).map_err(|e| format!("copying the binary: {e}"))?;
    std::fs::rename(&partial, &binary_path).map_err(|e| {
        let _ = std::fs::remove_file(&partial);
        format!("placing the binary: {e}")
    })?;
    Ok(StagedForFeed {
        version,
        channel: manifest_value.channel.clone(),
        manifestPath: manifest_path,
        binaryPath: binary_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
    use zyron_common::format::version::VersionWindow;
    use zyron_common::format::{ALL_FORMAT_KINDS, FormatMigrator};

    #[test]
    fn test_stage_release_places_the_manifest_and_a_matching_binary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let binary = dir.path().join("zyron-server-next");
        std::fs::write(&binary, b"a release binary").expect("writes");
        let sha = zyron_server::upgrade::feed::sha256_hex(b"a release binary");
        let manifest = dir.path().join("stable.json");
        std::fs::write(
            &manifest,
            format!(
                r#"{{"channel":"stable","generated_at_secs":1,"releases":[{{"version":"9.9.9","artifact_url":"","sha256":"{sha}","signature_scheme":"Ed25519","signature":"","upgrade_chain":[],"carries_format_bump":false,"notes_url":""}}],"signature_scheme":"Ed25519","signature":""}}"#
            ),
        )
        .expect("writes");
        let data_dir = dir.path().join("data");

        let staged = stage_release(&manifest, &binary, &data_dir, None).expect("stages");
        assert_eq!(staged.version, "9.9.9");
        assert_eq!(staged.channel, "stable");
        assert_eq!(
            staged.manifestPath,
            data_dir.join("releases").join(format!(
                "{}.manifest",
                zyron_server::upgrade::feed::feed_name("stable")
            ))
        );
        assert_eq!(
            std::fs::read(&staged.binaryPath).expect("copied"),
            b"a release binary"
        );

        // A binary that is not the one the manifest describes is refused
        std::fs::write(&binary, b"something else").expect("writes");
        let err = stage_release(&manifest, &binary, &data_dir, Some("9.9.9")).expect_err("refused");
        assert!(err.contains("the manifest declares"), "{err}");
        let err = stage_release(&manifest, &binary, &data_dir, Some("1.0.0")).expect_err("refused");
        assert!(err.contains("no release 1.0.0"), "{err}");
    }

    fn append_marker(body: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = body.to_vec();
        out.extend_from_slice(b"-v2");
        Ok(out)
    }

    fn strip_marker(body: &[u8]) -> Result<Vec<u8>, String> {
        match body.strip_suffix(b"-v2") {
            Some(rest) => Ok(rest.to_vec()),
            None => Err("no marker".to_string()),
        }
    }

    /// A registry where the heap page format has two versions, so migration
    /// has something to do
    fn registry() -> FormatRegistry {
        let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|kind| {
                let bumped = kind == FormatKind::StatisticsFile;
                FormatRegistration {
                    kind,
                    writer_current_version: if bumped {
                        FormatVersion::new(1, 1)
                    } else {
                        FormatVersion::V1
                    },
                    reader_supported_versions: if bumped {
                        VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                    } else {
                        VersionWindow::single(FormatVersion::V1)
                    },
                    migration_policy: MigrationPolicy::Lazy,
                    migration_reversible: true,
                    binary_version_gate: "0.11.0",
                    deprecation_status: DeprecationStatus::Active,
                    retirement_date: if bumped { Some("2027-01-01") } else { None },
                    downgrade_write_supported: false,
                    notes: "test",
                }
            })
            .collect();
        let migrators = [FormatMigrator {
            kind: FormatKind::StatisticsFile,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: true,
            forward: append_marker,
            backward: Some(strip_marker),
            no_body_change: false,
            description: "appends the v2 marker",
        }];
        let fixtures = [zyron_common::format::FormatFixture {
            kind: FormatKind::StatisticsFile,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1.bin",
        }];
        FormatRegistry::from_parts(&registrations, &migrators, &fixtures).expect("loads")
    }

    #[test]
    fn test_inspect_reports_an_envelope_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("page.bin");
        std::fs::write(
            &path,
            envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"the body"),
        )
        .expect("writes");
        let inspection = inspect(&path).expect("inspects");
        assert_eq!(inspection.kind, Some(FormatKind::StatisticsFile));
        assert_eq!(inspection.version, Some(FormatVersion::V1));
        assert_eq!(inspection.framing, "envelope");
        assert!(inspection.integrity_ok);
        let text = inspection.render();
        assert!(text.contains("ZSTS"), "{text}");
        assert!(text.contains("statistics_file"), "{text}");
        assert!(text.contains("integrity ok"), "{text}");
    }

    #[test]
    fn test_inspect_reports_a_corrupted_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("page.bin");
        let mut bytes =
            envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"the body");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&path, bytes).expect("writes");
        let inspection = inspect(&path).expect("inspects");
        assert_eq!(inspection.kind, Some(FormatKind::StatisticsFile));
        assert!(!inspection.integrity_ok);
        assert!(inspection.render().contains("integrity FAILED"));
    }

    #[test]
    fn test_inspect_reports_a_stamped_record_and_a_text_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stamped = dir.path().join("bloom.bin");
        let mut bytes = FormatStamp::new(FormatKind::BloomFilter, FormatVersion::V1)
            .to_bytes()
            .to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        std::fs::write(&stamped, bytes).expect("writes");
        let inspection = inspect(&stamped).expect("inspects");
        assert_eq!(inspection.kind, Some(FormatKind::BloomFilter));
        assert_eq!(inspection.framing, "stamp");

        let text_file = dir.path().join("zyron.toml");
        std::fs::write(
            &text_file,
            zyron_common::format::text_envelope::with_header(
                FormatKind::ZyronTomlConfig,
                FormatVersion::V1,
                "[server]\nport = 5433\n",
            ),
        )
        .expect("writes");
        let inspection = inspect(&text_file).expect("inspects");
        assert_eq!(inspection.kind, Some(FormatKind::ZyronTomlConfig));
        assert_eq!(inspection.framing, "text");
    }

    #[test]
    fn test_inspect_reports_a_file_that_is_not_ours() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"just some bytes").expect("writes");
        let inspection = inspect(&path).expect("inspects");
        assert!(inspection.kind.is_none());
        assert!(inspection.render().contains("unrecognized"));
    }

    #[test]
    fn test_verify_counts_by_kind_and_names_failures() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..3 {
            std::fs::write(
                dir.path().join(format!("{i}.zysts")),
                envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body"),
            )
            .expect("writes");
        }
        std::fs::write(
            dir.path().join("wal.bin"),
            envelope::encode(FormatKind::WalSegment, FormatVersion::V1, b"body"),
        )
        .expect("writes");
        let mut broken = envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body");
        let last = broken.len() - 1;
        broken[last] ^= 0xFF;
        std::fs::write(dir.path().join("broken.zysts"), broken).expect("writes");
        std::fs::write(dir.path().join("readme.txt"), b"not ours").expect("writes");

        let report = verify(&registry(), dir.path()).expect("verifies");
        assert!(!report.passed());
        assert_eq!(report.files_ok, 4);
        assert_eq!(report.files_unrecognized, 1);
        assert_eq!(report.failures.len(), 1);
        let text = report.render();
        assert!(text.contains("statistics_file at 1.0: 3 file(s)"), "{text}");
        assert!(text.contains("wal_segment at 1.0: 1 file(s)"), "{text}");
        assert!(text.contains("FAILED"), "{text}");
    }

    #[test]
    fn test_migrate_moves_old_files_and_leaves_current_ones() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..3 {
            std::fs::write(
                dir.path().join(format!("{i}.zysts")),
                envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body"),
            )
            .expect("writes");
        }
        std::fs::write(
            dir.path().join("current.zysts"),
            envelope::encode(
                FormatKind::StatisticsFile,
                FormatVersion::new(1, 1),
                b"body-v2",
            ),
        )
        .expect("writes");

        let registry = registry();
        let report = migrate(
            &registry,
            FormatKind::StatisticsFile,
            Some(FormatVersion::new(1, 1)),
            dir.path(),
        )
        .expect("migrates");
        assert_eq!(report.files_scanned, 4);
        assert_eq!(report.files_migrated, 3);
        assert_eq!(report.files_already_current, 1);
        assert!(report.failures.is_empty());

        // Every file now inspects at the current version
        let after = verify(&registry, dir.path()).expect("verifies");
        assert!(after.passed());
        assert_eq!(
            after.by_kind,
            vec![(FormatKind::StatisticsFile, FormatVersion::new(1, 1), 4)]
        );
    }

    #[test]
    fn test_migrate_refuses_a_target_this_binary_does_not_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = migrate(
            &registry(),
            FormatKind::StatisticsFile,
            Some(FormatVersion::new(9, 0)),
            dir.path(),
        )
        .expect_err("refused");
        assert!(err.contains("writes statistics_file at 1.1"), "{err}");
    }

    fn append_whole_file(file: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = file.to_vec();
        out.extend_from_slice(b"-trailer2");
        Ok(out)
    }

    /// A registry where the columnar file, which owns its trailer, has two
    /// versions so an own-trailer migration has something to move
    fn own_trailer_registry() -> FormatRegistry {
        let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|kind| {
                let bumped = kind == FormatKind::ZyrColumnar;
                FormatRegistration {
                    kind,
                    writer_current_version: if bumped {
                        FormatVersion::new(1, 1)
                    } else {
                        FormatVersion::V1
                    },
                    reader_supported_versions: if bumped {
                        VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                    } else {
                        VersionWindow::single(FormatVersion::V1)
                    },
                    migration_policy: MigrationPolicy::Eager,
                    migration_reversible: false,
                    binary_version_gate: "0.12.0",
                    deprecation_status: DeprecationStatus::Active,
                    retirement_date: if bumped { Some("2027-03-01") } else { None },
                    downgrade_write_supported: false,
                    notes: "test",
                }
            })
            .collect();
        let migrators = [FormatMigrator {
            kind: FormatKind::ZyrColumnar,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: false,
            forward: append_whole_file,
            backward: None,
            no_body_change: false,
            description: "appends a whole-file marker",
        }];
        let fixtures = [zyron_common::format::FormatFixture {
            kind: FormatKind::ZyrColumnar,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1_0.bin",
        }];
        FormatRegistry::from_parts(&registrations, &migrators, &fixtures).expect("loads")
    }

    #[test]
    fn test_migrate_rewrites_an_own_trailer_file_whole_not_re_enveloped() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A file that owns its trailer is handed to its migrator whole, and
        // the migrated bytes are written back verbatim rather than wrapped in
        // a second envelope
        let original = envelope::encode(
            FormatKind::ZyrColumnar,
            FormatVersion::V1,
            b"columnar-trailer",
        );
        let mut expected = original.clone();
        expected.extend_from_slice(b"-trailer2");
        std::fs::write(dir.path().join("data.zyr"), &original).expect("writes");

        let registry = own_trailer_registry();
        let report = migrate(
            &registry,
            FormatKind::ZyrColumnar,
            Some(FormatVersion::new(1, 1)),
            dir.path(),
        )
        .expect("migrates");
        assert_eq!(report.files_migrated, 1);

        let written = std::fs::read(dir.path().join("data.zyr")).expect("reads");
        assert_eq!(
            written, expected,
            "an own-trailer file must be rewritten whole, the buggy path wrapped it in a fresh envelope"
        );
    }

    #[test]
    fn test_the_statements_match_the_sql_surface() {
        assert_eq!(statements::SHOW_STATE, "SHOW UPGRADE STATE");
        assert_eq!(
            statements::trigger("2.3.1"),
            "TRIGGER MANUAL UPGRADE TO '2.3.1'"
        );
        assert!(statements::PAUSE.contains("auto_upgrade_paused = true"));
        assert!(statements::RESUME.contains("auto_upgrade_paused = false"));
        // A quote in an item name is escaped rather than closing the literal
        assert!(statements::guide("it's").contains("'it''s'"));
    }
}
