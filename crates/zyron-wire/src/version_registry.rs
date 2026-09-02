//! The wire protocol version registry.
//!
//! A client names the protocol version it speaks in its startup message.
//! The registry decides whether the server still speaks it: the current
//! version always, the one before it while a transition is open, and nothing
//! else. When a transition closes the older codec is deleted from the tree
//! rather than carried as a shim, and the registry row records the release
//! that removed it so the refusal names it

use zyron_common::format::wire_version::{WireProtocolVersion, WireVersionStatus};

/// Protocol major 3, the version every client speaks today.
///
/// Written as the major number rather than the packed `major << 16 | minor`
/// form the startup message carries, because the registry is about which
/// codec runs and only the major decides that
pub const WIRE_PROTOCOL_V3: u32 = 3;

/// The packed value a v3 startup message carries, `3.0`
pub const WIRE_PROTOCOL_V3_PACKED: i32 = 196_608;

inventory::submit! {
    WireProtocolVersion {
        version: WIRE_PROTOCOL_V3,
        status: WireVersionStatus::Current,
        introduced_in_binary_version: "0.1.0",
        retired_in_binary_version: None,
        notes: "message framing, extended query protocol, and the COPY subprotocol",
    }
}

/// Resolves the protocol major a startup message asked for.
///
/// The packed version is split the way the startup message packs it, and
/// only the major is negotiated: a minor this server does not know is not a
/// different codec, it is a client asking for an extension it will not get
pub fn negotiate_packed(packed: i32) -> Result<u32, String> {
    let major = ((packed >> 16) & 0xFFFF) as u32;
    let substrate = zyron_common::format::substrate().map_err(|e| e.to_string())?;
    substrate
        .wire_versions
        .negotiate(major)
        .map_err(|e| e.to_string())
}

/// Every protocol major this server accepts right now
pub fn accepted() -> Vec<u32> {
    zyron_common::format::substrate()
        .map(|s| s.wire_versions.accepted())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::wire_version::WireVersionRegistry;

    #[test]
    fn test_v3_is_the_current_version() {
        let substrate = zyron_common::format::substrate().expect("loads");
        assert_eq!(substrate.wire_versions.current_version(), WIRE_PROTOCOL_V3);
        assert_eq!(accepted(), vec![WIRE_PROTOCOL_V3]);
    }

    #[test]
    fn test_a_v3_startup_negotiates() {
        assert_eq!(
            negotiate_packed(WIRE_PROTOCOL_V3_PACKED).expect("accepts v3"),
            WIRE_PROTOCOL_V3
        );
    }

    #[test]
    fn test_an_unknown_major_is_refused_naming_the_current_one() {
        let err = negotiate_packed(2 << 16).expect_err("refuses v2");
        assert!(err.contains("The current version is 3"), "{err}");
        let err = negotiate_packed(9 << 16).expect_err("refuses v9");
        assert!(err.contains("newer than this server speaks"), "{err}");
    }

    /// A transition accepts both, and the cutover refuses the older one with
    /// the release that removed it
    #[test]
    fn test_transition_then_cutover() {
        let transition = WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                version: 3,
                status: WireVersionStatus::Transitional,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: None,
                notes: "the previous version",
            },
            WireProtocolVersion {
                version: 4,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.12.0",
                retired_in_binary_version: None,
                notes: "the current version",
            },
        ]);
        assert_eq!(transition.accepted(), vec![3, 4]);
        assert!(transition.negotiate(3).is_ok());
        assert!(transition.negotiate(4).is_ok());

        let after_cutover = WireVersionRegistry::from_versions(vec![
            WireProtocolVersion {
                version: 3,
                status: WireVersionStatus::Retired,
                introduced_in_binary_version: "0.1.0",
                retired_in_binary_version: Some("0.13.0"),
                notes: "retired at the cutover",
            },
            WireProtocolVersion {
                version: 4,
                status: WireVersionStatus::Current,
                introduced_in_binary_version: "0.12.0",
                retired_in_binary_version: None,
                notes: "the current version",
            },
        ]);
        assert_eq!(after_cutover.accepted(), vec![4]);
        let err = after_cutover
            .negotiate(3)
            .expect_err("refuses v3")
            .to_string();
        assert!(err.contains("retired in Zyron 0.13.0"), "{err}");
        assert!(err.contains("Update the driver"), "{err}");
    }
}
