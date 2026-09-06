//! Wire protocol version lifecycle.
//!
//! Three protocols cross a node boundary. The client protocol is what a
//! driver negotiates in its startup message. The mesh protocol is the calls
//! one node makes to another over HTTP. The consensus protocol is the frames
//! between members of a group. Each is registered here with the version the
//! binary speaks, so the release check, the startup gate, and the
//! `zyron_sys.wire.protocol_versions` view report all three from one place.
//!
//! A server speaks the current version of a protocol and the one before it
//! while a transition is open, and nothing else. When the transition closes,
//! the older codec is deleted from the tree rather than kept as a shim, and
//! the registry row records that it is gone.
//!
//! ## How a peer protocol moves
//!
//! The members of one group run two adjacent releases for the length of
//! every rolling upgrade, so the two peer protocols evolve inside a version
//! by adding fields that carry a default. A mesh body decodes with any field
//! absent and ignores any field the reader does not know. A consensus
//! message appends a new field after the ones that shipped, and its decoder
//! reads an absent trailing field as the default. A change that cannot be
//! expressed that way, a removed field, a new enum variant, a different
//! encoding, is a new protocol version, and a node speaks it only once every
//! member of its group runs a binary that reads it

use std::fmt;

/// One of the protocols a node speaks across a boundary
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WireProtocol {
    /// The client protocol, negotiated in the startup message
    Client,
    /// The node-to-node calls under `/internal/mesh/`
    Mesh,
    /// The consensus frames between members of a group
    Consensus,
}

impl WireProtocol {
    /// Every protocol, in the order the registry lists them
    pub const ALL: [WireProtocol; 3] = [
        WireProtocol::Client,
        WireProtocol::Mesh,
        WireProtocol::Consensus,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            WireProtocol::Client => "client",
            WireProtocol::Mesh => "mesh",
            WireProtocol::Consensus => "consensus",
        }
    }
}

impl fmt::Display for WireProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where a wire protocol version sits in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireVersionStatus {
    /// The version new connections negotiate
    Current,
    /// Still accepted while the transition is open
    Transitional,
    /// No longer accepted, the codec is gone
    Retired,
}

impl WireVersionStatus {
    pub const fn label(self) -> &'static str {
        match self {
            WireVersionStatus::Current => "current",
            WireVersionStatus::Transitional => "transitional",
            WireVersionStatus::Retired => "retired",
        }
    }

    #[inline]
    pub const fn is_accepted(self) -> bool {
        matches!(
            self,
            WireVersionStatus::Current | WireVersionStatus::Transitional
        )
    }
}

impl fmt::Display for WireVersionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One registered wire protocol version
#[derive(Debug, Clone, Copy)]
pub struct WireProtocolVersion {
    pub protocol: WireProtocol,
    /// The number a peer sends, in the startup message, the URL path, or the
    /// frame header
    pub version: u32,
    pub status: WireVersionStatus,
    /// The Zyron version that introduced it
    pub introduced_in_binary_version: &'static str,
    /// The Zyron version that stops accepting it, when one is scheduled
    pub retired_in_binary_version: Option<&'static str>,
    /// One line naming what changed at this version
    pub notes: &'static str,
}

inventory::collect!(WireProtocolVersion);

/// Why a negotiation failed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireVersionError {
    /// The peer asked for a version the server has never spoken
    Unknown {
        protocol: WireProtocol,
        requested: u32,
        current: u32,
    },
    /// The peer asked for a version whose transition has closed
    Retired {
        protocol: WireProtocol,
        requested: u32,
        current: u32,
        retired_in: Option<&'static str>,
    },
    /// The peer asked for a version newer than this server speaks
    TooNew {
        protocol: WireProtocol,
        requested: u32,
        current: u32,
    },
}

impl fmt::Display for WireVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireVersionError::Unknown {
                protocol,
                requested,
                current,
            } => write!(
                f,
                "{protocol} protocol version {requested} is not one this server speaks. \
                 The current version is {current}"
            ),
            WireVersionError::Retired {
                protocol,
                requested,
                current,
                retired_in,
            } => match retired_in {
                Some(version) => write!(
                    f,
                    "{protocol} protocol version {requested} was retired in Zyron {version}. \
                     Update the peer to speak version {current}"
                ),
                None => write!(
                    f,
                    "{protocol} protocol version {requested} is retired. Update the peer to \
                     speak version {current}"
                ),
            },
            WireVersionError::TooNew {
                protocol,
                requested,
                current,
            } => write!(
                f,
                "{protocol} protocol version {requested} is newer than this server speaks. \
                 The current version is {current}"
            ),
        }
    }
}

impl std::error::Error for WireVersionError {}

/// The loaded wire protocol registry, every protocol in one list ordered by
/// protocol and then by version
#[derive(Debug, Default)]
pub struct WireVersionRegistry {
    versions: Vec<WireProtocolVersion>,
}

impl WireVersionRegistry {
    pub fn load() -> WireVersionRegistry {
        let versions: Vec<WireProtocolVersion> = inventory::iter::<WireProtocolVersion>
            .into_iter()
            .copied()
            .collect();
        WireVersionRegistry::from_versions(versions)
    }

    pub fn from_versions(mut versions: Vec<WireProtocolVersion>) -> WireVersionRegistry {
        versions.sort_by_key(|v| (v.protocol, v.version));
        WireVersionRegistry { versions }
    }

    pub fn versions(&self) -> &[WireProtocolVersion] {
        &self.versions
    }

    /// The registered versions of one protocol, lowest first
    pub fn of(&self, protocol: WireProtocol) -> impl Iterator<Item = &WireProtocolVersion> {
        self.versions.iter().filter(move |v| v.protocol == protocol)
    }

    /// The version new connections of a protocol negotiate
    pub fn current(&self, protocol: WireProtocol) -> Option<&WireProtocolVersion> {
        self.of(protocol)
            .find(|v| v.status == WireVersionStatus::Current)
    }

    /// The version number new connections of a protocol negotiate, 0 when
    /// none is registered
    pub fn current_version(&self, protocol: WireProtocol) -> u32 {
        self.current(protocol).map(|v| v.version).unwrap_or(0)
    }

    /// Every version of a protocol a peer may ask for right now
    pub fn accepted(&self, protocol: WireProtocol) -> Vec<u32> {
        self.of(protocol)
            .filter(|v| v.status.is_accepted())
            .map(|v| v.version)
            .collect()
    }

    /// Resolves what a connection of a protocol actually speaks
    pub fn negotiate(
        &self,
        protocol: WireProtocol,
        requested: u32,
    ) -> Result<u32, WireVersionError> {
        let current = self.current_version(protocol);
        match self.of(protocol).find(|v| v.version == requested) {
            Some(entry) if entry.status.is_accepted() => Ok(entry.version),
            Some(entry) => Err(WireVersionError::Retired {
                protocol,
                requested,
                current,
                retired_in: entry.retired_in_binary_version,
            }),
            None if requested > current => Err(WireVersionError::TooNew {
                protocol,
                requested,
                current,
            }),
            None => Err(WireVersionError::Unknown {
                protocol,
                requested,
                current,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> WireVersionRegistry {
        WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                protocol: WireProtocol::Client,
                version: 1,
                status: WireVersionStatus::Transitional,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: None,
                notes: "the first wire version",
            },
            WireProtocolVersion {
                protocol: WireProtocol::Client,
                version: 2,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.11.0",
                retired_in_binary_version: None,
                notes: "adds the negotiated feature list",
            },
            WireProtocolVersion {
                protocol: WireProtocol::Mesh,
                version: 1,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.6.0",
                retired_in_binary_version: None,
                notes: "the mesh calls",
            },
        ])
    }

    #[test]
    fn test_both_versions_negotiate_during_the_transition() {
        let registry = registry();
        assert_eq!(registry.current_version(WireProtocol::Client), 2);
        assert_eq!(registry.accepted(WireProtocol::Client), vec![1, 2]);
        assert_eq!(
            registry
                .negotiate(WireProtocol::Client, 1)
                .expect("accepts v1"),
            1
        );
        assert_eq!(
            registry
                .negotiate(WireProtocol::Client, 2)
                .expect("accepts v2"),
            2
        );
    }

    #[test]
    fn test_retired_version_is_refused_after_cutover() {
        let registry = WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                protocol: WireProtocol::Client,
                version: 1,
                status: WireVersionStatus::Retired,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: Some("0.12.0"),
                notes: "retired at the cutover",
            },
            WireProtocolVersion {
                protocol: WireProtocol::Client,
                version: 2,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.11.0",
                retired_in_binary_version: None,
                notes: "current",
            },
        ]);
        assert_eq!(registry.accepted(WireProtocol::Client), vec![2]);
        let err = registry
            .negotiate(WireProtocol::Client, 1)
            .expect_err("refuses v1");
        assert!(err.to_string().contains("retired in Zyron 0.12.0"), "{err}");
        assert!(err.to_string().contains("version 2"), "{err}");
    }

    #[test]
    fn test_unknown_and_too_new_versions_are_named() {
        let registry = registry();
        assert!(matches!(
            registry.negotiate(WireProtocol::Client, 99),
            Err(WireVersionError::TooNew { requested: 99, .. })
        ));
        let sparse = WireVersionRegistry::from_versions(vec![WireProtocolVersion {
            protocol: WireProtocol::Client,
            version: 5,
            status: WireVersionStatus::Current,
            introduced_in_binary_version: "0.11.0",
            retired_in_binary_version: None,
            notes: "current",
        }]);
        assert!(matches!(
            sparse.negotiate(WireProtocol::Client, 3),
            Err(WireVersionError::Unknown { requested: 3, .. })
        ));
    }

    /// Each protocol negotiates against its own rows, so a mesh version
    /// number says nothing about the client protocol and the other way
    /// round
    #[test]
    fn test_protocols_negotiate_independently() {
        let registry = registry();
        assert_eq!(registry.current_version(WireProtocol::Mesh), 1);
        assert_eq!(registry.accepted(WireProtocol::Mesh), vec![1]);
        assert_eq!(registry.current_version(WireProtocol::Consensus), 0);
        assert!(matches!(
            registry.negotiate(WireProtocol::Mesh, 2),
            Err(WireVersionError::TooNew {
                protocol: WireProtocol::Mesh,
                requested: 2,
                current: 1
            })
        ));
        let err = registry
            .negotiate(WireProtocol::Mesh, 2)
            .expect_err("refuses")
            .to_string();
        assert!(err.starts_with("mesh protocol version 2"), "{err}");
        assert_eq!(registry.of(WireProtocol::Client).count(), 2);
        assert_eq!(
            registry
                .versions()
                .iter()
                .map(|v| v.protocol)
                .collect::<Vec<_>>(),
            vec![
                WireProtocol::Client,
                WireProtocol::Client,
                WireProtocol::Mesh
            ],
            "the list is ordered by protocol and then by version"
        );
    }
}
