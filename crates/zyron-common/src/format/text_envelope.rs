//! The envelope a text file carries.
//!
//! Two Zyron files are meant to be read and edited by people: `zyron.toml`
//! and a backup manifest. A binary header in either would defeat the point,
//! so they carry the same identity in a declared section instead of in
//! bytes.
//!
//! ```toml
//! [format]
//! kind = "ZCFG"
//! version = "1.0"
//! ```
//!
//! The section says the same three things the binary header's first eight
//! bytes say, and is read the same way: the kind decides which parser runs,
//! the version decides which reader, and a version outside the window fails
//! closed. There is no checksum because the file is hand editable, so an
//! edit that changes a byte is expected rather than corruption

use std::fmt;

use super::kind::FormatKind;
use super::version::FormatVersion;

/// The section name that carries the envelope
pub const TEXT_ENVELOPE_SECTION: &str = "format";

/// Why a text envelope would not read
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextEnvelopeError {
    /// The file has no `[format]` section
    Missing,
    /// The section has no `kind` key
    MissingKind,
    /// The section has no `version` key
    MissingVersion,
    /// The `kind` names no registered format
    UnknownKind { named: String },
    /// The `kind` names a different format than the caller expected
    KindMismatch {
        expected: FormatKind,
        found: FormatKind,
    },
    /// The `version` is not `major.minor`
    BadVersion { value: String },
    /// The version is outside the reader window
    UnknownVersion {
        kind: FormatKind,
        found: FormatVersion,
        supported: FormatVersion,
    },
}

impl fmt::Display for TextEnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextEnvelopeError::Missing => write!(
                f,
                "the file has no [{TEXT_ENVELOPE_SECTION}] section naming its kind and version"
            ),
            TextEnvelopeError::MissingKind => {
                write!(f, "the [{TEXT_ENVELOPE_SECTION}] section has no `kind` key")
            }
            TextEnvelopeError::MissingVersion => write!(
                f,
                "the [{TEXT_ENVELOPE_SECTION}] section has no `version` key"
            ),
            TextEnvelopeError::UnknownKind { named } => write!(
                f,
                "the [{TEXT_ENVELOPE_SECTION}] section names kind `{named}`, which addresses \
                 no registered format"
            ),
            TextEnvelopeError::KindMismatch { expected, found } => {
                write!(f, "expected a {} file, found a {} file", expected, found)
            }
            TextEnvelopeError::BadVersion { value } => write!(
                f,
                "the [{TEXT_ENVELOPE_SECTION}] section declares version `{value}`, which is \
                 not `<major>.<minor>`"
            ),
            TextEnvelopeError::UnknownVersion {
                kind,
                found,
                supported,
            } => write!(
                f,
                "the {kind} file is at format version {found}, this binary writes and reads \
                 {supported}. Upgrade through a release that still reads {found} to move the \
                 file forward first"
            ),
        }
    }
}

impl std::error::Error for TextEnvelopeError {}

/// The section a writer puts at the top of the file, trailing newline
/// included
pub fn header(kind: FormatKind, version: FormatVersion) -> String {
    format!(
        "[{TEXT_ENVELOPE_SECTION}]\nkind = \"{}\"\nversion = \"{}\"\n",
        kind.magic_str(),
        version
    )
}

/// Reads the kind and version out of a TOML document.
///
/// Parses the section directly rather than through a TOML value tree, so a
/// document whose body will not parse still reports which format and version
/// it claims to be. That is what lets a config migrator run on a file the
/// current parser rejects
pub fn parse(text: &str) -> Result<(FormatKind, FormatVersion), TextEnvelopeError> {
    let mut in_section = false;
    let mut kind: Option<String> = None;
    let mut version: Option<String> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with('[') {
            in_section = trimmed == format!("[{TEXT_ENVELOPE_SECTION}]");
            if !in_section && kind.is_some() && version.is_some() {
                break;
            }
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"').trim_matches('\'');
        match key.trim() {
            "kind" => kind = Some(value.to_string()),
            "version" => version = Some(value.to_string()),
            _ => {}
        }
    }

    if kind.is_none() && version.is_none() {
        return Err(TextEnvelopeError::Missing);
    }
    let kind = kind.ok_or(TextEnvelopeError::MissingKind)?;
    let version = version.ok_or(TextEnvelopeError::MissingVersion)?;

    let mut magic = [0u8; 4];
    let bytes = kind.as_bytes();
    if bytes.len() != 4 {
        return Err(TextEnvelopeError::UnknownKind { named: kind });
    }
    magic.copy_from_slice(bytes);
    let kind = FormatKind::from_magic(magic).ok_or(TextEnvelopeError::UnknownKind {
        named: kind.clone(),
    })?;
    let version: FormatVersion = version
        .parse()
        .map_err(|_| TextEnvelopeError::BadVersion { value: version })?;
    Ok((kind, version))
}

/// Reads the envelope and checks it against what the caller expects
pub fn parse_as(
    text: &str,
    expected: FormatKind,
    supported: FormatVersion,
) -> Result<FormatVersion, TextEnvelopeError> {
    let (kind, version) = parse(text)?;
    if kind != expected {
        return Err(TextEnvelopeError::KindMismatch {
            expected,
            found: kind,
        });
    }
    if version != supported {
        return Err(TextEnvelopeError::UnknownVersion {
            kind,
            found: version,
            supported,
        });
    }
    Ok(version)
}

/// Puts the envelope at the top of a document, replacing one already there.
///
/// For a document made entirely of sections, which is what `zyron.toml` is.
/// A document with top-level keys needs [`with_footer`] instead, because a
/// leading section would swallow those keys
pub fn with_header(kind: FormatKind, version: FormatVersion, body: &str) -> String {
    let mut out = header(kind, version);
    out.push('\n');
    out.push_str(strip_header(body).trim_start_matches('\n'));
    out
}

/// Puts the envelope at the end of a document, replacing one already there.
///
/// For a document that opens with top-level keys, which is what a backup
/// manifest is. The section is read wherever it sits, so where it goes is
/// only a question of not changing what the keys around it belong to
pub fn with_footer(kind: FormatKind, version: FormatVersion, body: &str) -> String {
    let mut out = strip_header(body);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push('\n');
    out.push_str(&header(kind, version));
    out
}

/// Removes an existing `[format]` section, returning the rest unchanged.
///
/// A section runs until the next section header, so that is what ends it
/// here rather than a blank line, which TOML allows inside a table
pub fn strip_header(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_section = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed == format!("[{TEXT_ENVELOPE_SECTION}]");
            if in_section {
                continue;
            }
        }
        if in_section {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_round_trips() {
        let text = header(FormatKind::ZyronTomlConfig, FormatVersion::new(1, 2));
        let (kind, version) = parse(&text).expect("parses");
        assert_eq!(kind, FormatKind::ZyronTomlConfig);
        assert_eq!(version, FormatVersion::new(1, 2));
    }

    #[test]
    fn test_header_is_found_ahead_of_a_body() {
        let text = with_header(
            FormatKind::BackupArchive,
            FormatVersion::V1,
            "[server]\nport = 5433\n",
        );
        let (kind, version) = parse(&text).expect("parses");
        assert_eq!(kind, FormatKind::BackupArchive);
        assert_eq!(version, FormatVersion::V1);
        assert!(text.contains("port = 5433"));
        assert!(text.starts_with("[format]"));
    }

    #[test]
    fn test_a_document_without_the_section_is_refused() {
        assert_eq!(
            parse("[server]\nport = 5433\n"),
            Err(TextEnvelopeError::Missing)
        );
    }

    #[test]
    fn test_a_partial_section_names_what_is_missing() {
        assert_eq!(
            parse("[format]\nkind = \"ZCFG\"\n"),
            Err(TextEnvelopeError::MissingVersion)
        );
        assert_eq!(
            parse("[format]\nversion = \"1.0\"\n"),
            Err(TextEnvelopeError::MissingKind)
        );
    }

    #[test]
    fn test_an_unknown_kind_is_refused() {
        assert!(matches!(
            parse("[format]\nkind = \"QQQQ\"\nversion = \"1.0\"\n"),
            Err(TextEnvelopeError::UnknownKind { .. })
        ));
        assert!(matches!(
            parse("[format]\nkind = \"TOOLONG\"\nversion = \"1.0\"\n"),
            Err(TextEnvelopeError::UnknownKind { .. })
        ));
    }

    #[test]
    fn test_a_bad_version_is_refused() {
        assert!(matches!(
            parse("[format]\nkind = \"ZCFG\"\nversion = \"one\"\n"),
            Err(TextEnvelopeError::BadVersion { .. })
        ));
    }

    #[test]
    fn test_parse_as_checks_kind_and_version() {
        let text = header(FormatKind::ZyronTomlConfig, FormatVersion::V1);
        assert_eq!(
            parse_as(&text, FormatKind::ZyronTomlConfig, FormatVersion::V1).expect("matches"),
            FormatVersion::V1
        );
        assert!(matches!(
            parse_as(&text, FormatKind::BackupArchive, FormatVersion::V1),
            Err(TextEnvelopeError::KindMismatch { .. })
        ));
        let err = parse_as(&text, FormatKind::ZyronTomlConfig, FormatVersion::new(2, 0))
            .expect_err("version mismatch");
        assert!(err.to_string().contains("Upgrade through"), "{err}");
    }

    #[test]
    fn test_replacing_an_existing_header_leaves_one() {
        let first = with_header(
            FormatKind::ZyronTomlConfig,
            FormatVersion::V1,
            "[server]\nport = 1\n",
        );
        let second = with_header(
            FormatKind::ZyronTomlConfig,
            FormatVersion::new(1, 1),
            &first,
        );
        assert_eq!(second.matches("[format]").count(), 1, "{second}");
        let (_, version) = parse(&second).expect("parses");
        assert_eq!(version, FormatVersion::new(1, 1));
        assert!(second.contains("port = 1"));
    }

    #[test]
    fn test_footer_placement_leaves_top_level_keys_alone() {
        let body = "version = 1\nserverVersion = \"0.11.0\"\n\n[[files]]\npath = \"a\"\n";
        let text = with_footer(FormatKind::BackupArchive, FormatVersion::V1, body);
        assert!(text.starts_with("version = 1"), "{text}");
        assert!(text.trim_end().ends_with("version = \"1.0\""), "{text}");
        let (kind, version) = parse(&text).expect("parses");
        assert_eq!(kind, FormatKind::BackupArchive);
        assert_eq!(version, FormatVersion::V1);

        // Replacing the footer leaves exactly one, and the body intact
        let again = with_footer(FormatKind::BackupArchive, FormatVersion::new(1, 1), &text);
        assert_eq!(again.matches("[format]").count(), 1, "{again}");
        assert!(again.contains("serverVersion = \"0.11.0\""), "{again}");
        assert!(again.contains("[[files]]"), "{again}");
        assert_eq!(parse(&again).expect("parses").1, FormatVersion::new(1, 1));
    }

    #[test]
    fn test_stripping_ends_a_section_at_the_next_header_not_a_blank_line() {
        let text = "[format]\nkind = \"ZCFG\"\n\nversion = \"1.0\"\n\n[server]\nport = 1\n";
        let stripped = strip_header(text);
        assert!(!stripped.contains("kind ="), "{stripped}");
        assert!(!stripped.contains("version ="), "{stripped}");
        assert!(stripped.contains("port = 1"), "{stripped}");
    }

    #[test]
    fn test_comments_are_skipped() {
        let text = "# a note\n\n[format]\n# another\nkind = \"ZCFG\"\nversion = \"1.0\"\n";
        assert_eq!(
            parse(text).expect("parses"),
            (FormatKind::ZyronTomlConfig, FormatVersion::V1)
        );
    }
}
