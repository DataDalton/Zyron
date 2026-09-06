//! The release signing key built into this binary.
//!
//! Every release manifest and every release binary the feed carries is
//! signed with the vendor's release key, and a node verifies both against
//! the public half before it installs anything. That half ships inside the
//! binary, read from `release-signing.pub` beside the crate at build time,
//! so a node checks releases from the moment it starts with nothing to
//! configure. A deployment that signs its own releases points
//! `upgrade.release_signing_key` at its own public key instead.
//!
//! Rotating the vendor key is replacing that file and shipping the binary
//! that carries the new one. A release signed with the old key is then
//! refused by the new binary, which is the point of the rotation

use zyron_auth::signature::VerifyingMaterial;
use zyron_common::{Result, ZyronError};

/// The scheme the built-in key belongs to
pub const BUILT_IN_SCHEME: &str = "Ed25519";

/// The public half of the vendor's release key as hex, one line
const BUILT_IN_KEY_HEX: &str = include_str!("../../release-signing.pub");

/// The built-in key as hex, for the config surface to report
pub fn built_in_key_hex() -> &'static str {
    BUILT_IN_KEY_HEX.trim()
}

/// The built-in key as verifying material
pub fn built_in() -> Result<VerifyingMaterial> {
    let bytes = super::feed::decode_hex(built_in_key_hex()).ok_or_else(|| {
        ZyronError::Internal(
            "the release signing key built into this binary is not hex".to_string(),
        )
    })?;
    let key: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
        ZyronError::Internal(format!(
            "the release signing key built into this binary is {} bytes, an Ed25519 key is 32",
            bytes.len()
        ))
    })?;
    Ok(VerifyingMaterial::Ed25519(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_built_in_key_is_a_usable_ed25519_key() {
        assert_eq!(built_in_key_hex().len(), 64);
        match built_in().expect("decodes") {
            VerifyingMaterial::Ed25519(key) => {
                ed25519_dalek::VerifyingKey::from_bytes(&key).expect("a point on the curve");
            }
            other => panic!("expected Ed25519, got {other:?}"),
        }
    }
}
