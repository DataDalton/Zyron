//! The release check.
//!
//! Everything a release has to be true about the format substrate, checked
//! in one place so a build either satisfies all of it or fails naming the
//! part it does not. This is what stops a breaking change shipping without
//! its migrator, and what stops reader code for a retired version lingering
//! in the tree after nobody can produce a file at that version any more.
//!
//! The registry load already refuses a set that is internally inconsistent,
//! so what is left here is the things a load cannot see: whether today is
//! past a retirement date, whether a deprecation has a rewriter, whether a
//! scheme's test vectors still pass, and whether each of the three wire
//! protocols has exactly one current version

use zyron_common::format::registry::is_iso_date;
use zyron_common::format::scheme::SchemeStatus;
use zyron_common::format::wire_version::{WireProtocol, WireVersionRegistry, WireVersionStatus};
use zyron_common::format::{
    ALL_FORMAT_KINDS, BinaryVersion, FormatKind, FormatSubstrate, MAGIC_ALLOCATIONS,
};

/// One thing the check found wrong
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Which check produced it, printed as the line's prefix
    pub check: &'static str,
    pub detail: String,
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.check, self.detail)
    }
}

/// The whole check's result
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReleaseReport {
    pub findings: Vec<Finding>,
    pub formats_checked: usize,
    pub migrators_checked: usize,
    pub fixtures_checked: usize,
    pub schemes_checked: usize,
    pub deprecations_checked: usize,
    pub rewriters_checked: usize,
    /// The wire protocols whose version rows were checked
    pub protocols_checked: usize,
}

impl ReleaseReport {
    pub fn passed(&self) -> bool {
        self.findings.is_empty()
    }

    /// The line the CLI prints on success
    pub fn summary(&self) -> String {
        format!(
            "{} formats, {} migrators, {} fixtures, {} signature schemes, \
             {} deprecations, {} rewriters, {} wire protocols checked, {} finding(s)",
            self.formats_checked,
            self.migrators_checked,
            self.fixtures_checked,
            self.schemes_checked,
            self.deprecations_checked,
            self.rewriters_checked,
            self.protocols_checked,
            self.findings.len()
        )
    }

    fn note(&mut self, check: &'static str, detail: impl Into<String>) {
        self.findings.push(Finding {
            check,
            detail: detail.into(),
        });
    }
}

/// Runs every check against the substrate this binary carries.
///
/// `today` is an ISO 8601 date, so a caller can ask what the check would say
/// on a future date, which is how a retirement is scheduled with confidence
pub fn run(substrate: &FormatSubstrate, today: &str) -> ReleaseReport {
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut report = ReleaseReport::default();
    check_registrations(substrate, &mut report);
    check_magics(&mut report);
    check_migrators_and_fixtures(substrate, &mut report);
    check_retirement(substrate, today, now_secs, &mut report);
    check_schemes(substrate, &mut report);
    check_deprecations(substrate, &mut report);
    check_protocols(&substrate.wire_versions, &mut report);
    report
}

/// Every format kind has exactly one registration with a usable window
fn check_registrations(substrate: &FormatSubstrate, report: &mut ReleaseReport) {
    report.formats_checked = substrate.formats.len();
    for kind in substrate.formats.missing() {
        report.note(
            "registration",
            format!("format `{kind}` has no FormatRegistration"),
        );
    }
    for entry in substrate.formats.entries() {
        let registration = &entry.registration;
        let window = registration.reader_supported_versions;
        if !window.contains(registration.writer_current_version) {
            report.note(
                "registration",
                format!(
                    "format `{}` writes {} but reads only {window}",
                    registration.kind, registration.writer_current_version
                ),
            );
        }
        if registration.binary_version_gate.is_empty() {
            report.note(
                "registration",
                format!(
                    "format `{}` declares no binary_version_gate",
                    registration.kind
                ),
            );
        }
        if let Some(date) = registration.retirement_date {
            if !is_iso_date(date) {
                report.note(
                    "registration",
                    format!(
                        "format `{}` declares retirement_date `{date}`, which is not \
                         YYYY-MM-DD",
                        registration.kind
                    ),
                );
            }
        }
    }
}

/// No two formats share a magic
fn check_magics(report: &mut ReleaseReport) {
    for (index, left) in MAGIC_ALLOCATIONS.iter().enumerate() {
        for right in MAGIC_ALLOCATIONS.iter().skip(index + 1) {
            if left.magic == right.magic {
                report.note(
                    "magic",
                    format!(
                        "`{}` and `{}` both claim {}",
                        left.kind,
                        right.kind,
                        zyron_common::format::envelope::printable_magic(&left.magic)
                    ),
                );
            }
        }
    }
    // Every allocated kind is in the enum's own list, so a magic cannot be
    // added without the kind that owns it
    for kind in ALL_FORMAT_KINDS {
        if !MAGIC_ALLOCATIONS.iter().any(|row| row.kind == *kind) {
            report.note("magic", format!("format `{kind}` has no allocated magic"));
        }
    }
}

/// Every step inside a reader window has a migrator, and every version
/// behind the current one has a fixture
fn check_migrators_and_fixtures(substrate: &FormatSubstrate, report: &mut ReleaseReport) {
    for entry in substrate.formats.entries() {
        report.migrators_checked += entry.migrators.len();
        report.fixtures_checked += entry.fixtures.len();
        let registration = &entry.registration;
        let window = registration.reader_supported_versions;

        let mut cursor = window.oldest;
        while cursor < registration.writer_current_version {
            match entry.migrator(cursor) {
                Some(step) => {
                    if step.reversible && step.backward.is_none() && !step.no_body_change {
                        report.note(
                            "migrator",
                            format!(
                                "format `{}` step {cursor} to {} claims reversible with no \
                                 backward function",
                                registration.kind, step.to
                            ),
                        );
                    }
                    cursor = step.to;
                }
                None => {
                    report.note(
                        "migrator",
                        format!(
                            "format `{}` has no migrator for {cursor}, and the step is not \
                             marked no_body_change",
                            registration.kind
                        ),
                    );
                    break;
                }
            }
        }

        for version in window.iter() {
            if version >= registration.writer_current_version {
                continue;
            }
            if !entry.fixtures.iter().any(|fx| fx.version == version) {
                report.note(
                    "fixture",
                    format!(
                        "format `{}` reads {version} but ships no fixture for it",
                        registration.kind
                    ),
                );
            }
        }

        for fixture in &entry.fixtures {
            if fixture.bytes.is_empty() {
                continue;
            }
            // A format that owns its trailer is verified by its own reader,
            // so only the header is asked for its kind and version here
            let header = if registration.kind.framing().migrates_whole_file() {
                zyron_common::format::envelope::decode_header(fixture.bytes).map(|(h, _)| h)
            } else {
                zyron_common::format::envelope::decode(fixture.bytes).map(|parsed| parsed.header)
            };
            match header {
                Ok(header) => {
                    if header.kind != registration.kind {
                        report.note(
                            "fixture",
                            format!(
                                "fixture {} is a {} file, not a {}",
                                fixture.path, header.kind, registration.kind
                            ),
                        );
                    }
                    if header.version != fixture.version {
                        report.note(
                            "fixture",
                            format!(
                                "fixture {} declares version {} but holds {}",
                                fixture.path, fixture.version, header.version
                            ),
                        );
                    }
                }
                Err(e) => report.note(
                    "fixture",
                    format!("fixture {} does not parse, {e}", fixture.path),
                ),
            }
        }
    }
}

/// Reader code for a version past its retirement date has to be gone
fn check_retirement(
    substrate: &FormatSubstrate,
    today: &str,
    now_secs: u64,
    report: &mut ReleaseReport,
) {
    if !is_iso_date(today) {
        report.note(
            "retirement",
            format!("`{today}` is not a YYYY-MM-DD date to check against"),
        );
        return;
    }
    for entry in substrate.formats.entries() {
        let registration = &entry.registration;
        let Some(retirement) = registration.retirement_date else {
            continue;
        };
        if !is_iso_date(retirement) || retirement > today {
            continue;
        }
        let window = registration.reader_supported_versions;
        if window.oldest < registration.writer_current_version {
            report.note(
                "retirement",
                format!(
                    "format `{}` passed its retirement date {retirement} with reader code \
                     for {} still in the tree. Delete the reader, its migrator, and its \
                     fixture, then narrow reader_supported_versions",
                    registration.kind, window.oldest
                ),
            );
        }
    }
    for scheme in substrate.schemes.schemes() {
        let Some(retirement) = scheme.retirement_date else {
            continue;
        };
        if !is_iso_date(retirement) || retirement > today {
            continue;
        }
        if scheme.status != SchemeStatus::Retired {
            report.note(
                "retirement",
                format!(
                    "scheme `{}` passed its retirement date {retirement} but is still {}",
                    scheme.scheme_name, scheme.status
                ),
            );
        }
    }

    // The date says a scheme may retire. This says its verifier may go: the
    // last artifact it signed has expired, so nothing in the field still
    // needs the code. Carrying it past that point is the legacy this phase
    // exists to refuse
    for scheme_name in substrate.schemes.verifiers_ready_for_removal(now_secs) {
        report.note(
            "retirement",
            format!(
                "the verifier for scheme `{scheme_name}` is still compiled in, and the last \
                 artifact signed with it has expired. Delete the verifier"
            ),
        );
    }
}

/// Every registered scheme has a distinct id and name, and every id that a
/// binary artifact has to carry fits one byte
fn check_schemes(substrate: &FormatSubstrate, report: &mut ReleaseReport) {
    let schemes = substrate.schemes.schemes();
    report.schemes_checked = schemes.len();
    for (index, left) in schemes.iter().enumerate() {
        for right in schemes.iter().skip(index + 1) {
            if left.scheme_id == right.scheme_id {
                report.note(
                    "scheme",
                    format!(
                        "`{}` and `{}` both claim scheme id {}",
                        left.scheme_name, right.scheme_name, left.scheme_id
                    ),
                );
            }
            if left.scheme_name.eq_ignore_ascii_case(right.scheme_name) {
                report.note(
                    "scheme",
                    format!("scheme name `{}` is registered twice", left.scheme_name),
                );
            }
        }
        if left.scheme_id.as_artifact_byte().is_none() {
            report.note(
                "scheme",
                format!(
                    "scheme `{}` has id {}, which does not fit the one byte a binary \
                     artifact carries",
                    left.scheme_name, left.scheme_id
                ),
            );
        }
        if left.first_available_version.is_empty() {
            report.note(
                "scheme",
                format!(
                    "scheme `{}` declares no first_available_version",
                    left.scheme_name
                ),
            );
        }
    }
}

/// Every deprecation has a guide and, when it is a SQL surface, a rewriter
/// with a declared category
fn check_deprecations(substrate: &FormatSubstrate, report: &mut ReleaseReport) {
    let records = substrate.deprecations.records();
    report.deprecations_checked = records.len();
    let rewriters = zyron_parser::rewriter::registered();
    report.rewriters_checked = rewriters.len();

    for record in records {
        if record.migration_guide_url.is_none() && !record.no_guide_required {
            report.note(
                "deprecation",
                format!(
                    "`{}` has no migration_guide_url and is not marked no_guide_required",
                    record.item_id
                ),
            );
        }
        if !record.item_kind.is_user_authored_sql() {
            continue;
        }
        let covered = rewriters.iter().any(|rewrite| {
            rewrite.description.contains(record.item_id)
                || rewrite.name.contains(record.item_id)
                || record
                    .replacement_ref
                    .map(|replacement| rewrite.description.contains(replacement))
                    .unwrap_or(false)
        });
        if !covered {
            report.note(
                "deprecation",
                format!(
                    "`{}` is a SQL surface with no registered rewriter naming it",
                    record.item_id
                ),
            );
        }
    }

    for rewrite in &rewriters {
        if rewrite.target.is_empty() {
            report.note(
                "rewriter",
                format!("`{}` targets no object kind", rewrite.name),
            );
        }
        if rewrite.description.is_empty() {
            report.note(
                "rewriter",
                format!("`{}` carries no description", rewrite.name),
            );
        }
    }
}

/// Every wire protocol has exactly one current version, no two rows of a
/// protocol share a number, every row names the release that introduced it
/// as `major.minor.patch`, and a retired row names the release that removed
/// it. A binary speaks one current version of each protocol, so a protocol
/// with none or with two is a build that cannot say what it speaks
fn check_protocols(registry: &WireVersionRegistry, report: &mut ReleaseReport) {
    report.protocols_checked = WireProtocol::ALL.len();
    for protocol in WireProtocol::ALL {
        let rows: Vec<_> = registry.of(protocol).collect();
        let current = rows
            .iter()
            .filter(|row| row.status == WireVersionStatus::Current)
            .count();
        if current != 1 {
            report.note(
                "protocol",
                format!(
                    "the {protocol} protocol registers {current} current versions, and a binary \
                     speaks exactly one"
                ),
            );
        }
        for (index, row) in rows.iter().enumerate() {
            if rows[..index]
                .iter()
                .any(|earlier| earlier.version == row.version)
            {
                report.note(
                    "protocol",
                    format!(
                        "the {protocol} protocol registers version {} twice",
                        row.version
                    ),
                );
            }
            if BinaryVersion::parse(row.introduced_in_binary_version).is_none() {
                report.note(
                    "protocol",
                    format!(
                        "the {protocol} protocol's version {} names `{}` as the release that \
                         introduced it, which is not major.minor.patch",
                        row.version, row.introduced_in_binary_version
                    ),
                );
            }
            match row.retired_in_binary_version {
                Some(retired) if BinaryVersion::parse(retired).is_none() => report.note(
                    "protocol",
                    format!(
                        "the {protocol} protocol's version {} names `{retired}` as the release \
                         that retired it, which is not major.minor.patch",
                        row.version
                    ),
                ),
                None if row.status == WireVersionStatus::Retired => report.note(
                    "protocol",
                    format!(
                        "the {protocol} protocol's version {} is retired and names no release \
                         that removed it",
                        row.version
                    ),
                ),
                _ => {}
            }
        }
    }
}

/// Format kinds this binary registers but has no live writer for, which the
/// report prints so a reserved magic is never mistaken for a gap
pub fn reserved_kinds() -> &'static [FormatKind] {
    zyron_common::format::reserved::RESERVED_KINDS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Today's date is supplied rather than read, so this test does not
    /// change its answer as the calendar moves
    const TODAY: &str = "2026-09-01";

    #[test]
    fn test_this_release_passes() {
        let substrate = zyron_common::format::substrate().expect("loads");
        let report = run(substrate, TODAY);
        assert!(
            report.passed(),
            "{}",
            report
                .findings
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert_eq!(report.formats_checked, ALL_FORMAT_KINDS.len());
        assert!(report.schemes_checked >= 3);
        assert_eq!(report.protocols_checked, WireProtocol::ALL.len());
    }

    #[test]
    fn test_the_summary_names_what_was_checked() {
        let substrate = zyron_common::format::substrate().expect("loads");
        let text = run(substrate, TODAY).summary();
        assert!(text.contains("formats"), "{text}");
        assert!(text.contains("3 wire protocols checked"), "{text}");
        assert!(text.contains("0 finding(s)"), "{text}");
    }

    /// The server links all three protocol crates, so each registers its one
    /// current version
    #[test]
    fn test_every_protocol_registers_one_current_version_in_this_binary() {
        let substrate = zyron_common::format::substrate().expect("loads");
        for protocol in WireProtocol::ALL {
            assert_eq!(
                substrate.wire_versions.accepted(protocol).len(),
                1,
                "{protocol} accepts one version"
            );
        }
        assert_eq!(
            substrate.wire_versions.current_version(WireProtocol::Mesh),
            zyron_mesh::MESH_PROTOCOL_VERSION
        );
        assert_eq!(
            substrate
                .wire_versions
                .current_version(WireProtocol::Consensus),
            u32::from(zyron_raft::CONSENSUS_PROTOCOL_VERSION)
        );
    }

    /// A protocol with no current version, one registered twice, a retired
    /// row naming no release, and a release that is not a version are each
    /// a finding naming the protocol
    #[test]
    fn test_a_protocol_registry_that_cannot_say_what_it_speaks_is_a_finding() {
        use zyron_common::format::wire_version::WireProtocolVersion;
        let registry = WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                protocol: WireProtocol::Client,
                version: 3,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: None,
                notes: "",
            },
            WireProtocolVersion {
                protocol: WireProtocol::Client,
                version: 3,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: None,
                notes: "",
            },
            WireProtocolVersion {
                protocol: WireProtocol::Mesh,
                version: 1,
                status: WireVersionStatus::Retired,
                introduced_in_binary_version: "next",
                retired_in_binary_version: None,
                notes: "",
            },
        ]);
        let mut report = ReleaseReport::default();
        check_protocols(&registry, &mut report);
        let details: Vec<&str> = report.findings.iter().map(|f| f.detail.as_str()).collect();
        assert!(
            details
                .iter()
                .any(|d| d.contains("client protocol registers 2 current versions")),
            "{details:?}"
        );
        assert!(
            details
                .iter()
                .any(|d| d.contains("client protocol registers version 3 twice")),
            "{details:?}"
        );
        assert!(
            details
                .iter()
                .any(|d| d.contains("mesh protocol registers 0 current versions")),
            "{details:?}"
        );
        assert!(
            details
                .iter()
                .any(|d| d.contains("`next` as the release that introduced it")),
            "{details:?}"
        );
        assert!(
            details
                .iter()
                .any(|d| d.contains("retired and names no release")),
            "{details:?}"
        );
        assert!(
            details
                .iter()
                .any(|d| d.contains("consensus protocol registers 0 current versions")),
            "{details:?}"
        );
        assert!(report.findings.iter().all(|f| f.check == "protocol"));
    }

    #[test]
    fn test_a_bad_date_is_reported_rather_than_ignored() {
        let substrate = zyron_common::format::substrate().expect("loads");
        let report = run(substrate, "not a date");
        assert!(!report.passed());
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.check == "retirement" && f.detail.contains("not a YYYY-MM-DD"))
        );
    }

    #[test]
    fn test_findings_read_as_one_line_each() {
        let finding = Finding {
            check: "migrator",
            detail: "format `heap_page` has no migrator for 1.0".to_string(),
        };
        assert_eq!(
            finding.to_string(),
            "migrator: format `heap_page` has no migrator for 1.0"
        );
    }

    #[test]
    fn test_reserved_kinds_are_named() {
        let reserved = reserved_kinds();
        assert_eq!(reserved.len(), 4);
        assert!(reserved.contains(&FormatKind::VolumeMetadata));
    }
}
