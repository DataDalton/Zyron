//! What a binary can read, asked of the binary itself.
//!
//! The compatibility gate needs the target's reader floors and the config
//! keys it still accepts. Those live inside the target binary's registries,
//! so the staged binary is run with `--capabilities` and answers with a JSON
//! document. The running binary answers the same question the same way,
//! which is what makes the document one shape rather than two

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use zyron_common::format::registry::FormatRegistry;
use zyron_common::format::{FormatKind, FormatSubstrate, FormatVersion};
use zyron_common::{Result, ZyronError};

use super::compat_gate::TargetCapabilities;

/// The flag a binary answers its capabilities to
pub const CAPABILITIES_FLAG: &str = "--capabilities";

/// What a binary reports about itself
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitiesDocument {
    pub version: String,
    /// Per format, the oldest version this binary reads, the version as its
    /// integer form
    pub reader_floor: Vec<(String, u32)>,
    /// Every config key this binary accepts
    pub config_keys: Vec<String>,
}

/// The document for the registry this process runs with
pub fn describe(
    registry: &FormatRegistry,
    version: &str,
    config_keys: Vec<String>,
) -> CapabilitiesDocument {
    let mut reader_floor: Vec<(String, u32)> = registry
        .entries()
        .map(|entry| {
            (
                entry.registration.kind.catalog_name().to_string(),
                entry.registration.reader_supported_versions.oldest.as_u32(),
            )
        })
        .collect();
    reader_floor.sort();
    CapabilitiesDocument {
        version: version.to_string(),
        reader_floor,
        config_keys,
    }
}

/// Every config key this binary accepts, which is every key `SHOW ALL`
/// reports for a default config
pub fn running_config_keys() -> Vec<String> {
    let mut keys: Vec<String> = crate::config::ZyronConfig::default()
        .all_config_entries()
        .into_iter()
        .map(|(key, _, _)| key)
        .collect();
    keys.sort();
    keys.dedup();
    keys
}

/// Renders this binary's document, which `zyron-server --capabilities`
/// prints
pub fn render_running(substrate: &FormatSubstrate) -> Result<String> {
    let document = describe(
        &substrate.formats,
        env!("CARGO_PKG_VERSION"),
        running_config_keys(),
    );
    serde_json::to_string_pretty(&document)
        .map_err(|e| ZyronError::Internal(format!("capabilities encode, {e}")))
}

/// Turns a document into what the gate consumes.
///
/// A format name the running binary does not know is left out rather than
/// refused, because a target that carries a format this binary has never
/// heard of is not a problem for the data this binary wrote. A removed
/// config key is one the running binary accepts and the target does not
pub fn target_from_document(
    document: &CapabilitiesDocument,
    running_keys: &[String],
) -> Result<TargetCapabilities> {
    if zyron_common::format::BinaryVersion::parse(&document.version).is_none() {
        return Err(ZyronError::UpgradeRefused(format!(
            "the staged binary reports `{}` as its version, which is not major.minor.patch",
            document.version
        )));
    }
    let reader_floor = document
        .reader_floor
        .iter()
        .filter_map(|(name, version)| {
            FormatKind::from_catalog_name(name)
                .map(|kind| (kind, FormatVersion::from_u32(*version)))
        })
        .collect();
    let removed_config_keys = running_keys
        .iter()
        .filter(|key| {
            !document
                .config_keys
                .iter()
                .any(|accepted| accepted.eq_ignore_ascii_case(key))
        })
        .cloned()
        .collect();
    Ok(TargetCapabilities {
        version: document.version.clone(),
        reader_floor,
        removed_config_keys,
    })
}

/// Parses the document a binary printed
pub fn parse_document(text: &str) -> Result<CapabilitiesDocument> {
    serde_json::from_str(text).map_err(|e| {
        ZyronError::UpgradeRefused(format!(
            "the staged binary's capabilities are not readable, {e}. It answered: {}",
            text.chars().take(200).collect::<String>()
        ))
    })
}

/// Runs a staged binary and reads what it can do.
///
/// The binary was verified against the release key before it was written,
/// so running it is running a release. It is asked only for its
/// capabilities and given the deadline, after which it is killed
pub async fn from_staged(
    binary: &Path,
    running_keys: &[String],
    timeout: Duration,
) -> Result<TargetCapabilities> {
    let child = tokio::process::Command::new(binary)
        .arg(CAPABILITIES_FLAG)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| {
            ZyronError::UpgradeRefused(format!(
                "the staged binary at {} would not start, {e}",
                binary.display()
            ))
        })?;
    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return Err(ZyronError::UpgradeRefused(format!(
                "the staged binary at {} did not answer, {e}",
                binary.display()
            )));
        }
        Err(_) => {
            return Err(ZyronError::UpgradeRefused(format!(
                "the staged binary at {} did not answer {CAPABILITIES_FLAG} inside {}s",
                binary.display(),
                timeout.as_secs()
            )));
        }
    };
    if !output.status.success() {
        return Err(ZyronError::UpgradeRefused(format!(
            "the staged binary at {} exited with {} answering {CAPABILITIES_FLAG}: {}",
            binary.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let document = parse_document(&String::from_utf8_lossy(&output.stdout))?;
    target_from_document(&document, running_keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn substrate() -> &'static FormatSubstrate {
        zyron_common::format::substrate().expect("the server registers every format")
    }

    #[test]
    fn test_the_running_binary_describes_every_format_and_config_key() {
        let text = render_running(substrate()).expect("renders");
        let document = parse_document(&text).expect("parses back");
        assert_eq!(document.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(
            document.reader_floor.len(),
            zyron_common::format::ALL_FORMAT_KINDS.len()
        );
        assert!(document.config_keys.iter().any(|k| k == "upgrade.channel"));
        assert!(document.config_keys.iter().any(|k| k == "server.port"));
    }

    #[test]
    fn test_a_target_reports_the_keys_it_dropped_and_its_floors() {
        let mut document = describe(
            &substrate().formats,
            "0.13.0",
            vec!["server.port".to_string(), "upgrade.channel".to_string()],
        );
        document
            .reader_floor
            .push(("something_new".to_string(), 65536));
        let running = vec![
            "server.port".to_string(),
            "upgrade.channel".to_string(),
            "query.old_knob".to_string(),
        ];
        let target = target_from_document(&document, &running).expect("converts");
        assert_eq!(target.version, "0.13.0");
        assert_eq!(target.removed_config_keys, vec!["query.old_knob"]);
        assert_eq!(
            target.reader_floor.len(),
            zyron_common::format::ALL_FORMAT_KINDS.len(),
            "an unknown format is left out, not refused"
        );
        assert_eq!(
            target.floor_for(FormatKind::UpgradeJournal),
            Some(FormatVersion::V1)
        );
    }

    #[test]
    fn test_a_document_with_a_bad_version_is_refused() {
        let document = CapabilitiesDocument {
            version: "latest".into(),
            reader_floor: Vec::new(),
            config_keys: Vec::new(),
        };
        let err = target_from_document(&document, &[]).expect_err("refused");
        assert!(err.to_string().contains("major.minor.patch"), "{err}");
    }

    #[tokio::test]
    async fn test_a_binary_that_is_not_there_is_refused_not_hung() {
        let missing = std::env::temp_dir().join("zyron-server-does-not-exist");
        let err = from_staged(&missing, &[], Duration::from_secs(5))
            .await
            .expect_err("refused");
        assert!(err.to_string().contains("would not start"), "{err}");
    }
}
