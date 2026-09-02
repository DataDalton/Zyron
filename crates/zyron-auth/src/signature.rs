//! Signature verification dispatch and principal key rotation.
//!
//! Verification never assumes a scheme. Every artifact says which scheme
//! signed it, the registry says whether that scheme is still accepted for
//! that artifact kind at this instant, and the verifier for it runs. An
//! unknown scheme, a scheme past its overlap, and a retired scheme whose
//! last artifact has expired all fail closed with an error naming what was
//! found rather than falling through to a default.
//!
//! Key rotation is the other half. A principal's signing key can be replaced
//! and its scheme changed at the same time, with an overlap during which the
//! outgoing key still verifies. After the overlap the old key is refused

use std::sync::Arc;

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use zeroize::Zeroizing;
use zyron_common::format::scheme::{
    ArtifactKind, SchemeError, SchemeIdentifierEncoding, SchemeRegistry,
};
use zyron_common::{Result, ZyronError};

/// Default overlap for a rotation that does not name one, twenty-four hours
pub const DEFAULT_ROTATION_OVERLAP_SECS: u64 = 24 * 3_600;

/// The material one scheme needs to verify a signature
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyingMaterial {
    /// A 32-byte Ed25519 public key
    Ed25519([u8; 32]),
    /// A SEC1-encoded P-256 public key
    Es256(Vec<u8>),
    /// A DER-encoded RSA public key
    Rs256(Vec<u8>),
}

impl VerifyingMaterial {
    /// The scheme this material belongs to
    pub const fn scheme_name(&self) -> &'static str {
        match self {
            VerifyingMaterial::Ed25519(_) => "Ed25519",
            VerifyingMaterial::Es256(_) => "ES256",
            VerifyingMaterial::Rs256(_) => "RS256",
        }
    }
}

/// Verifies a signature with one scheme's verifier.
///
/// The scheme has already been resolved against the registry by the caller,
/// so this is the dispatch and nothing else
pub fn verify_with(material: &VerifyingMaterial, message: &[u8], signature: &[u8]) -> Result<bool> {
    match material {
        VerifyingMaterial::Ed25519(public_key) => {
            let key = VerifyingKey::from_bytes(public_key).map_err(|e| {
                ZyronError::SignatureScheme(format!("Ed25519 public key is not on the curve, {e}"))
            })?;
            let bytes: [u8; 64] = match signature.try_into() {
                Ok(bytes) => bytes,
                Err(_) => return Ok(false),
            };
            Ok(key
                .verify(message, &ed25519_dalek::Signature::from_bytes(&bytes))
                .is_ok())
        }
        VerifyingMaterial::Es256(public_key) => {
            use p256::ecdsa::{Signature, VerifyingKey as P256Key, signature::Verifier as _};
            let point = p256::EncodedPoint::from_bytes(public_key).map_err(|e| {
                ZyronError::SignatureScheme(format!("ES256 public key is not a P-256 point, {e}"))
            })?;
            let key = P256Key::from_encoded_point(&point).map_err(|e| {
                ZyronError::SignatureScheme(format!("ES256 public key is not usable, {e}"))
            })?;
            let Ok(parsed) = Signature::from_slice(signature) else {
                return Ok(false);
            };
            Ok(key.verify(message, &parsed).is_ok())
        }
        VerifyingMaterial::Rs256(public_key) => {
            // The key arrives DER encoded, either as a bare PKCS#1
            // RSAPublicKey or wrapped in a SubjectPublicKeyInfo, which is
            // what a JWKS or an IdP metadata document hands over. Both
            // spellings are accepted so a caller does not have to unwrap one
            // into the other before verifying
            let verify = |encoding: &'static ring::signature::RsaParameters, key: &[u8]| {
                ring::signature::UnparsedPublicKey::new(encoding, key)
                    .verify(message, signature)
                    .is_ok()
            };
            if verify(&ring::signature::RSA_PKCS1_2048_8192_SHA256, public_key) {
                return Ok(true);
            }
            match rsa_public_key_from_spki(public_key) {
                Some(inner) => Ok(verify(&ring::signature::RSA_PKCS1_2048_8192_SHA256, inner)),
                // A key that parses as neither spelling is a configuration
                // error rather than a failed signature, so it is reported
                // instead of being reduced to "did not verify"
                None => Err(ZyronError::SignatureScheme(
                    "RS256 public key is neither a PKCS#1 RSAPublicKey nor a \
                     SubjectPublicKeyInfo wrapping one"
                        .to_string(),
                )),
            }
        }
    }
}

/// Unwraps the RSAPublicKey out of a SubjectPublicKeyInfo.
///
/// A minimal DER walk rather than a general parser: SEQUENCE, skip the
/// AlgorithmIdentifier, then take the BIT STRING's contents past its unused
/// bits byte. Returns None for anything that is not that shape
fn rsa_public_key_from_spki(der: &[u8]) -> Option<&[u8]> {
    /// Reads one tag-length header, returning the content and what follows
    fn field(input: &[u8]) -> Option<(u8, &[u8], &[u8])> {
        let (&tag, rest) = input.split_first()?;
        let (&first, rest) = rest.split_first()?;
        let (len, rest) = if first < 0x80 {
            (first as usize, rest)
        } else {
            let count = (first & 0x7f) as usize;
            // A length needs at least one byte and must fit a usize
            if count == 0 || count > std::mem::size_of::<usize>() {
                return None;
            }
            let (bytes, rest) = rest.split_at_checked(count)?;
            (
                bytes.iter().fold(0usize, |acc, b| (acc << 8) | *b as usize),
                rest,
            )
        };
        let (content, after) = rest.split_at_checked(len)?;
        Some((tag, content, after))
    }

    const SEQUENCE: u8 = 0x30;
    const BIT_STRING: u8 = 0x03;

    let (tag, body, _) = field(der)?;
    if tag != SEQUENCE {
        return None;
    }
    // AlgorithmIdentifier, whose contents are not inspected: ring rejects a
    // key whose parameters do not suit the algorithm it was handed
    let (_, _, after_algorithm) = field(body)?;
    let (tag, bits, _) = field(after_algorithm)?;
    if tag != BIT_STRING {
        return None;
    }
    // The leading byte counts unused bits, which is zero for a key
    let (&unused, key) = bits.split_first()?;
    if unused != 0 { None } else { Some(key) }
}

/// Resolves the scheme an artifact declares and verifies it.
///
/// The scheme name comes from the artifact's own identity: a JWT's `alg`
/// header, an X.509 `signatureAlgorithm`, or the leading scheme id byte of a
/// binary artifact. The registry decides whether that scheme is accepted for
/// this artifact kind now, and only then does a verifier run
pub fn verify_artifact(
    registry: &SchemeRegistry,
    kind: ArtifactKind,
    declared_scheme: &str,
    material: &VerifyingMaterial,
    message: &[u8],
    signature: &[u8],
    now_secs: u64,
) -> Result<bool> {
    let scheme = registry.resolve_for_verification(kind, declared_scheme, now_secs)?;
    if !scheme
        .scheme_name
        .eq_ignore_ascii_case(material.scheme_name())
    {
        return Err(ZyronError::SignatureScheme(format!(
            "the artifact declares scheme `{}` but the key material is for `{}`",
            scheme.scheme_name,
            material.scheme_name()
        )));
    }
    verify_with(material, message, signature)
}

/// Splits a binary artifact into its scheme and its signature bytes, then
/// verifies it
pub fn verify_binary_artifact(
    registry: &SchemeRegistry,
    kind: ArtifactKind,
    material: &VerifyingMaterial,
    message: &[u8],
    artifact: &[u8],
    now_secs: u64,
) -> Result<bool> {
    let (scheme, signature) = registry.resolve_artifact_bytes(kind, artifact, now_secs)?;
    if !scheme
        .scheme_name
        .eq_ignore_ascii_case(material.scheme_name())
    {
        return Err(ZyronError::SignatureScheme(format!(
            "the artifact carries scheme id {} for `{}` but the key material is for `{}`",
            scheme.scheme_id,
            scheme.scheme_name,
            material.scheme_name()
        )));
    }
    verify_with(material, message, signature)
}

/// Prefixes a signature with the scheme id a binary artifact carries
pub fn tag_binary_signature(
    registry: &SchemeRegistry,
    scheme_name: &str,
    signature: &[u8],
) -> Result<Vec<u8>> {
    let scheme = registry
        .by_name(scheme_name)
        .ok_or_else(|| SchemeError::UnknownScheme {
            named: scheme_name.to_string(),
        })?;
    let tag = scheme.scheme_id.as_artifact_byte().ok_or_else(|| {
        ZyronError::SignatureScheme(format!(
            "scheme `{}` has id {}, which does not fit the one byte a binary artifact carries",
            scheme.scheme_name, scheme.scheme_id
        ))
    })?;
    let mut out = Vec::with_capacity(1 + signature.len());
    out.push(tag);
    out.extend_from_slice(signature);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Principal keys
// ---------------------------------------------------------------------------

/// One principal's signing key
#[derive(Debug, Clone)]
pub struct PrincipalKey {
    pub principal: String,
    pub scheme_name: String,
    /// The public half, which is what verification uses.
    ///
    /// Length varies by scheme: 32 raw bytes for Ed25519, a SEC1 point for
    /// ES256, a DER SubjectPublicKeyInfo for RS256. An RSA key does not fit a
    /// fixed 32-byte field, which is what kept RS256 from ever signing
    pub public_key: Vec<u8>,
    /// Unix seconds the key was issued at
    pub issued_at_secs: u64,
}

impl PrincipalKey {
    /// The material a verifier needs for this key, resolved from its scheme.
    ///
    /// Callers verify through this rather than assembling a
    /// `VerifyingMaterial` themselves, so a key whose bytes do not match the
    /// shape its scheme requires is refused here instead of failing later as
    /// a signature that did not verify
    pub fn verifying_material(&self) -> Result<VerifyingMaterial> {
        if self.scheme_name.eq_ignore_ascii_case("Ed25519") {
            let bytes: [u8; 32] = self.public_key.as_slice().try_into().map_err(|_| {
                ZyronError::SignatureScheme(format!(
                    "Ed25519 public key for `{}` is {} bytes, not 32",
                    self.principal,
                    self.public_key.len()
                ))
            })?;
            return Ok(VerifyingMaterial::Ed25519(bytes));
        }
        if self.scheme_name.eq_ignore_ascii_case("ES256") {
            return Ok(VerifyingMaterial::Es256(self.public_key.clone()));
        }
        if self.scheme_name.eq_ignore_ascii_case("RS256") {
            return Ok(VerifyingMaterial::Rs256(self.public_key.clone()));
        }
        Err(ZyronError::SignatureScheme(format!(
            "no verifier is registered for scheme `{}`",
            self.scheme_name
        )))
    }
}

/// A principal's outgoing key during a rotation
#[derive(Debug, Clone)]
pub struct RetiringKey {
    pub key: PrincipalKey,
    /// Unix seconds the outgoing key stops being accepted at
    pub overlap_end_secs: u64,
}

/// What a rotation produced
#[derive(Debug, Clone)]
pub struct RotationOutcome {
    pub principal: String,
    pub new_scheme: String,
    pub previous_scheme: Option<String>,
    pub overlap_end_secs: u64,
}

/// Per-principal signing keys with overlap-aware rotation.
///
/// A rotation issues a new key, marks the old one retiring, and keeps
/// accepting the old one until the overlap ends. Both keys verify during the
/// overlap and only the new one signs, which is what lets a caller holding an
/// artifact signed a moment before the rotation still be believed
#[derive(Debug, Default)]
pub struct PrincipalKeyStore {
    keys: parking_lot::RwLock<Vec<PrincipalKey>>,
    retiring: parking_lot::RwLock<Vec<RetiringKey>>,
    /// Secret halves, never handed out. Length varies by scheme: 32 raw bytes
    /// for Ed25519, a DER PKCS#8 PrivateKeyInfo for RS256. Zeroized on drop
    /// and on replacement, so a rotated-out secret does not stay in freed heap
    secrets: parking_lot::RwLock<Vec<(String, Zeroizing<Vec<u8>>)>>,
}

impl PrincipalKeyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Issues a principal's first key
    pub fn issue(&self, principal: &str, scheme_name: &str, now_secs: u64) -> Result<PrincipalKey> {
        let (public_key, secret) = generate_keypair(scheme_name)?;
        let key = PrincipalKey {
            principal: principal.to_string(),
            scheme_name: scheme_name.to_string(),
            public_key,
            issued_at_secs: now_secs,
        };
        self.keys.write().retain(|k| k.principal != principal);
        self.keys.write().push(key.clone());
        self.secrets.write().retain(|(name, _)| name != principal);
        self.secrets.write().push((principal.to_string(), secret));
        Ok(key)
    }

    /// The key a principal signs with now
    pub fn current(&self, principal: &str) -> Option<PrincipalKey> {
        self.keys
            .read()
            .iter()
            .find(|k| k.principal == principal)
            .cloned()
    }

    /// The keys that still verify for a principal at a point in time, the
    /// current one first
    pub fn verifying(&self, principal: &str, now_secs: u64) -> Vec<PrincipalKey> {
        let mut out = Vec::new();
        if let Some(current) = self.current(principal) {
            out.push(current);
        }
        for retiring in self.retiring.read().iter() {
            if retiring.key.principal == principal && now_secs < retiring.overlap_end_secs {
                out.push(retiring.key.clone());
            }
        }
        out
    }

    /// Rotates a principal's key, optionally onto a different scheme.
    ///
    /// The registry is consulted first, so a rotation onto a reserved or
    /// unregistered scheme is refused before any key is generated
    pub fn rotate(
        &self,
        registry: &SchemeRegistry,
        principal: &str,
        new_scheme: Option<&str>,
        overlap_secs: u64,
        now_secs: u64,
    ) -> Result<RotationOutcome> {
        let previous = self.current(principal);
        let scheme_name = match new_scheme {
            Some(named) => {
                let scheme = registry
                    .by_name(named)
                    .ok_or_else(|| SchemeError::UnknownScheme {
                        named: named.to_string(),
                    })?;
                if !scheme.status.can_sign() {
                    return Err(ZyronError::SignatureScheme(format!(
                        "scheme `{}` is {} and cannot sign new artifacts",
                        scheme.scheme_name, scheme.status
                    )));
                }
                scheme.scheme_name.to_string()
            }
            None => match &previous {
                Some(key) => key.scheme_name.clone(),
                None => "Ed25519".to_string(),
            },
        };

        let overlap_end_secs = now_secs.saturating_add(overlap_secs);
        if let Some(previous) = previous.clone() {
            self.retiring.write().push(RetiringKey {
                key: previous,
                overlap_end_secs,
            });
        }
        self.issue(principal, &scheme_name, now_secs)?;
        Ok(RotationOutcome {
            principal: principal.to_string(),
            new_scheme: scheme_name,
            previous_scheme: previous.map(|k| k.scheme_name),
            overlap_end_secs,
        })
    }

    /// Drops retiring keys whose overlap has ended
    pub fn sweep(&self, now_secs: u64) -> usize {
        let mut retiring = self.retiring.write();
        let before = retiring.len();
        retiring.retain(|k| now_secs < k.overlap_end_secs);
        before - retiring.len()
    }

    /// Signs a message with a principal's current key
    pub fn sign(&self, principal: &str, message: &[u8]) -> Result<Vec<u8>> {
        let key = self.current(principal).ok_or_else(|| {
            ZyronError::SignatureScheme(format!("principal `{principal}` has no signing key"))
        })?;
        let secret = self
            .secrets
            .read()
            .iter()
            .find(|(name, _)| name == principal)
            .map(|(_, secret)| secret.clone())
            .ok_or_else(|| {
                ZyronError::SignatureScheme(format!(
                    "principal `{principal}` has no secret half on this node"
                ))
            })?;

        if key.scheme_name.eq_ignore_ascii_case("Ed25519") {
            let bytes: [u8; 32] = secret.as_slice().try_into().map_err(|_| {
                ZyronError::SignatureScheme(format!(
                    "Ed25519 secret for `{principal}` is {} bytes, not 32",
                    secret.len()
                ))
            })?;
            let signing = SigningKey::from_bytes(&bytes);
            return Ok(signing.sign(message).to_bytes().to_vec());
        }

        if key.scheme_name.eq_ignore_ascii_case("RS256") {
            // PKCS#1 v1.5 over SHA-256 through ring, the same implementation
            // the verifier uses, whose modular arithmetic is constant time
            let pair = ring::signature::RsaKeyPair::from_pkcs8(&secret).map_err(|e| {
                ZyronError::SignatureScheme(format!(
                    "RS256 secret for `{principal}` is not a usable PKCS#8 RSA key, {e}"
                ))
            })?;
            let mut signature = vec![0u8; pair.public().modulus_len()];
            pair.sign(
                &ring::signature::RSA_PKCS1_SHA256,
                &ring::rand::SystemRandom::new(),
                message,
                &mut signature,
            )
            .map_err(|e| {
                ZyronError::SignatureScheme(format!("RS256 signing failed for `{principal}`, {e}"))
            })?;
            return Ok(signature);
        }

        // ES256 lands here deliberately, see generate_keypair
        Err(ZyronError::SignatureScheme(format!(
            "scheme `{}` has no signer in this binary, it verifies signatures produced \
             elsewhere and Zyron never holds its private half",
            key.scheme_name
        )))
    }
}

/// Modulus size for a generated RSA key.
///
/// 2048 is the floor ring will sign with and the size every SAML relying
/// party accepts, and NIST keeps it usable through 2030. Generation is the
/// slow step, seconds rather than microseconds, which is why it happens on
/// issue and rotation and never on a signing path
const RSA_MODULUS_BITS: usize = 2048;

/// Generates a keypair for a scheme, returning the public and secret halves.
///
/// The public half is encoded the way that scheme's verifier reads it: raw
/// bytes for Ed25519, a DER SubjectPublicKeyInfo for RS256, which is the
/// spelling a JWKS and an IdP metadata document publish. The secret half is
/// raw bytes for Ed25519 and a DER PKCS#8 PrivateKeyInfo for RS256, which is
/// what ring's signer takes
fn generate_keypair(scheme_name: &str) -> Result<(Vec<u8>, Zeroizing<Vec<u8>>)> {
    if scheme_name.eq_ignore_ascii_case("Ed25519") {
        let mut secret = [0u8; 32];
        fill_random(&mut secret);
        let signing = SigningKey::from_bytes(&secret);
        let public = signing.verifying_key().to_bytes().to_vec();
        return Ok((public, Zeroizing::new(secret.to_vec())));
    }

    if scheme_name.eq_ignore_ascii_case("RS256") {
        use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey};
        // rsa 0.9 carries its own rand_core, a different major than the
        // workspace one, so the generator has to come from the crate's
        // re-export rather than the workspace rand
        let mut rng = rsa::rand_core::OsRng;
        let private = rsa::RsaPrivateKey::new(&mut rng, RSA_MODULUS_BITS).map_err(|e| {
            ZyronError::SignatureScheme(format!("could not generate an RS256 key, {e}"))
        })?;
        let public = private
            .to_public_key()
            .to_public_key_der()
            .map_err(|e| {
                ZyronError::SignatureScheme(format!("could not encode the RS256 public key, {e}"))
            })?
            .as_bytes()
            .to_vec();
        let secret = private.to_pkcs8_der().map_err(|e| {
            ZyronError::SignatureScheme(format!("could not encode the RS256 private key, {e}"))
        })?;
        return Ok((public, Zeroizing::new(secret.as_bytes().to_vec())));
    }

    // ES256 lands here deliberately. It is a verify-only scheme, accepted so
    // Zyron can check signatures WebAuthn authenticators produced, and it has
    // no issuing path because Zyron never holds the private half
    Err(ZyronError::SignatureScheme(format!(
        "this binary generates Ed25519 and RS256 keys, not `{scheme_name}`. Register a \
         generator for it before rotating a principal onto it"
    )))
}

/// Fills a buffer with cryptographically random bytes
fn fill_random(buffer: &mut [u8; 32]) {
    use rand::Rng;
    rand::rng().fill_bytes(buffer);
}

/// The scheme identity a JWS header names, read from an already-decoded
/// header object
pub fn scheme_from_jws_alg(alg: &str) -> &str {
    // The `alg` value is the scheme name for every scheme Zyron registers,
    // so no translation table is needed and adding a scheme adds nothing here
    alg
}

/// Whether an artifact kind carries a signature at all
pub fn is_signed(kind: ArtifactKind) -> bool {
    kind.identifier_encoding() != SchemeIdentifierEncoding::OpaqueBearer
}

/// The process-wide principal key store
static PRINCIPAL_KEYS: std::sync::OnceLock<Arc<PrincipalKeyStore>> = std::sync::OnceLock::new();

/// The principal key store for this node
pub fn principal_keys() -> Arc<PrincipalKeyStore> {
    Arc::clone(PRINCIPAL_KEYS.get_or_init(|| Arc::new(PrincipalKeyStore::new())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::scheme::{
        ArtifactSchemeBinding, SchemeCategory, SchemeId, SchemeStatus, SignatureSchemeRegistration,
        default_artifact_bindings,
    };

    fn scheme(name: &'static str, id: u16, status: SchemeStatus) -> SignatureSchemeRegistration {
        SignatureSchemeRegistration {
            scheme_name: name,
            scheme_id: SchemeId(id),
            category: SchemeCategory::Signature,
            status,
            first_available_version: "0.11.0",
            retirement_date: None,
            notes: "test scheme",
        }
    }

    fn registry() -> SchemeRegistry {
        SchemeRegistry::from_parts(
            vec![
                scheme("Ed25519", 1, SchemeStatus::Active),
                scheme("ES256", 2, SchemeStatus::Active),
                scheme("RS256", 3, SchemeStatus::Active),
                scheme("ML-DSA-65", 16, SchemeStatus::Reserved),
            ],
            default_artifact_bindings(),
        )
    }

    fn ed25519_pair() -> (SigningKey, VerifyingMaterial) {
        let mut secret = [7u8; 32];
        secret[0] = 42;
        let signing = SigningKey::from_bytes(&secret);
        let material = VerifyingMaterial::Ed25519(signing.verifying_key().to_bytes());
        (signing, material)
    }

    #[test]
    fn test_ed25519_round_trips_through_the_dispatch() {
        let registry = registry();
        let (signing, material) = ed25519_pair();
        let message = b"the artifact body";
        let signature = signing.sign(message).to_bytes().to_vec();
        assert!(
            verify_artifact(
                &registry,
                ArtifactKind::Jwt,
                "Ed25519",
                &material,
                message,
                &signature,
                0
            )
            .expect("verifies")
        );
    }

    #[test]
    fn test_a_tampered_message_does_not_verify() {
        let registry = registry();
        let (signing, material) = ed25519_pair();
        let signature = signing.sign(b"original").to_bytes().to_vec();
        assert!(
            !verify_artifact(
                &registry,
                ArtifactKind::Jwt,
                "Ed25519",
                &material,
                b"tampered",
                &signature,
                0
            )
            .expect("runs")
        );
    }

    #[test]
    fn test_an_unknown_scheme_fails_closed() {
        let registry = registry();
        let (_, material) = ed25519_pair();
        let err = verify_artifact(
            &registry,
            ArtifactKind::Jwt,
            "HS256",
            &material,
            b"m",
            b"s",
            0,
        )
        .expect_err("fails closed");
        assert!(err.to_string().contains("HS256"), "{err}");
        assert!(err.to_string().contains("not registered"), "{err}");
    }

    #[test]
    fn test_a_reserved_scheme_fails_closed() {
        let registry = registry();
        let (_, material) = ed25519_pair();
        let err = verify_artifact(
            &registry,
            ArtifactKind::Jwt,
            "ML-DSA-65",
            &material,
            b"m",
            b"s",
            0,
        )
        .expect_err("fails closed");
        assert!(err.to_string().contains("reserved registry slot"), "{err}");
    }

    #[test]
    fn test_declared_scheme_and_key_material_must_agree() {
        // The kind accepts ES256, so the refusal is about the key material
        // rather than about the binding
        let registry = SchemeRegistry::from_parts(
            vec![
                scheme("Ed25519", 1, SchemeStatus::Active),
                scheme("ES256", 2, SchemeStatus::Active),
            ],
            vec![ArtifactSchemeBinding::new(
                ArtifactKind::SessionToken,
                "ES256",
            )],
        );
        let (_, material) = ed25519_pair();
        let err = verify_artifact(
            &registry,
            ArtifactKind::SessionToken,
            "ES256",
            &material,
            b"m",
            b"s",
            0,
        )
        .expect_err("mismatched");
        assert!(err.to_string().contains("key material is for"), "{err}");
    }

    /// A scheme the registry knows but the artifact kind is not bound to is
    /// refused before any verifier runs, which is what stops an artifact
    /// signed for one purpose being accepted for another
    #[test]
    fn test_a_scheme_not_bound_to_the_artifact_kind_fails_closed() {
        let registry = registry();
        let (_, material) = ed25519_pair();
        let err = verify_artifact(
            &registry,
            ArtifactKind::Jwt,
            "ES256",
            &material,
            b"m",
            b"s",
            0,
        )
        .expect_err("fails closed");
        assert!(err.to_string().contains("not accepted for JWT"), "{err}");
    }

    #[test]
    fn test_a_binary_artifact_dispatches_on_its_leading_byte() {
        let registry = registry();
        let (signing, material) = ed25519_pair();
        let message = b"session token body";
        let signature = signing.sign(message).to_bytes().to_vec();
        let artifact = tag_binary_signature(&registry, "Ed25519", &signature).expect("tags");
        assert_eq!(artifact[0], 1);
        assert!(
            verify_binary_artifact(
                &registry,
                ArtifactKind::SessionToken,
                &material,
                message,
                &artifact,
                0
            )
            .expect("verifies")
        );
    }

    #[test]
    fn test_rotation_keeps_the_outgoing_key_verifying_during_the_overlap() {
        let registry = registry();
        let store = PrincipalKeyStore::new();
        let first = store.issue("sp1", "Ed25519", 0).expect("issues");
        let outcome = store
            .rotate(&registry, "sp1", Some("Ed25519"), 1_000, 100)
            .expect("rotates");
        assert_eq!(outcome.overlap_end_secs, 1_100);
        assert_eq!(outcome.previous_scheme.as_deref(), Some("Ed25519"));

        let during = store.verifying("sp1", 500);
        assert_eq!(during.len(), 2, "both keys verify during the overlap");
        assert!(during.iter().any(|k| k.public_key == first.public_key));

        let after = store.verifying("sp1", 2_000);
        assert_eq!(after.len(), 1, "only the new key verifies after it");
        assert!(!after.iter().any(|k| k.public_key == first.public_key));
    }

    #[test]
    fn test_rotation_refuses_a_reserved_scheme_before_generating_a_key() {
        let registry = registry();
        let store = PrincipalKeyStore::new();
        store.issue("sp1", "Ed25519", 0).expect("issues");
        let before = store.current("sp1").expect("has a key");
        let err = store
            .rotate(&registry, "sp1", Some("ML-DSA-65"), 100, 0)
            .expect_err("refuses");
        assert!(err.to_string().contains("cannot sign"), "{err}");
        assert_eq!(
            store.current("sp1").expect("unchanged").public_key,
            before.public_key
        );
    }

    #[test]
    fn test_the_sweep_drops_keys_past_their_overlap() {
        let registry = registry();
        let store = PrincipalKeyStore::new();
        store.issue("sp1", "Ed25519", 0).expect("issues");
        store
            .rotate(&registry, "sp1", None, 100, 0)
            .expect("rotates");
        assert_eq!(store.sweep(50), 0);
        assert_eq!(store.sweep(200), 1);
        assert_eq!(store.verifying("sp1", 200).len(), 1);
    }

    #[test]
    fn test_a_principal_signs_and_verifies_with_its_own_key() {
        let store = PrincipalKeyStore::new();
        let key = store.issue("sp1", "Ed25519", 0).expect("issues");
        let signature = store.sign("sp1", b"payload").expect("signs");
        let material = key.verifying_material().expect("material");
        assert_eq!(
            material,
            VerifyingMaterial::Ed25519(key.public_key.as_slice().try_into().expect("32 bytes"))
        );
        assert!(verify_with(&material, b"payload", &signature).expect("verifies"));
        assert!(!verify_with(&material, b"other", &signature).expect("runs"));
    }

    /// The whole RS256 round trip through the key store: issue generates an
    /// RSA key, sign produces a PKCS#1 v1.5 signature over SHA-256, and the
    /// verifier accepts it against the SubjectPublicKeyInfo the store kept
    #[test]
    fn test_a_principal_signs_and_verifies_with_an_rs256_key() {
        let store = PrincipalKeyStore::new();
        let key = store.issue("sp1", "RS256", 0).expect("issues");
        assert_eq!(key.scheme_name, "RS256");
        assert!(
            key.public_key.len() > 32,
            "an RSA public key does not fit the old fixed 32-byte field, got {}",
            key.public_key.len()
        );

        let signature = store.sign("sp1", b"payload").expect("signs");
        assert_eq!(
            signature.len(),
            RSA_MODULUS_BITS / 8,
            "a PKCS#1 v1.5 signature is one modulus wide"
        );

        let material = key.verifying_material().expect("material");
        assert!(verify_with(&material, b"payload", &signature).expect("verifies"));
        assert!(
            !verify_with(&material, b"other", &signature).expect("runs"),
            "a signature must not verify over a different message"
        );
    }

    /// A rotation onto RS256 has to leave the outgoing Ed25519 key verifying
    /// through the overlap, which is the case that broke when key material
    /// was a fixed 32-byte field and an RSA key could not be stored beside it
    #[test]
    fn test_rotating_from_ed25519_onto_rs256_keeps_both_verifying() {
        let registry = registry();
        let store = PrincipalKeyStore::new();
        let first = store.issue("sp1", "Ed25519", 0).expect("issues");
        let first_signature = store.sign("sp1", b"before").expect("signs");

        let outcome = store
            .rotate(&registry, "sp1", Some("RS256"), 1_000, 100)
            .expect("rotates");
        assert_eq!(outcome.new_scheme, "RS256");
        assert_eq!(outcome.previous_scheme.as_deref(), Some("Ed25519"));

        // The new key signs
        let after_signature = store.sign("sp1", b"after").expect("signs");
        let current = store.current("sp1").expect("has a key");
        assert_eq!(current.scheme_name, "RS256");
        assert!(
            verify_with(
                &current.verifying_material().expect("material"),
                b"after",
                &after_signature
            )
            .expect("verifies")
        );

        // And the retired one still verifies what it signed before the rotation
        let during = store.verifying("sp1", 500);
        assert_eq!(during.len(), 2, "both keys verify during the overlap");
        let retired = during
            .iter()
            .find(|k| k.scheme_name == "Ed25519")
            .expect("the outgoing key is still offered");
        assert_eq!(retired.public_key, first.public_key);
        assert!(
            verify_with(
                &retired.verifying_material().expect("material"),
                b"before",
                &first_signature
            )
            .expect("verifies")
        );
    }

    /// ES256 is verify only by design. Zyron never holds the private half, so
    /// issuing one is refused rather than half-built
    #[test]
    fn test_issuing_an_es256_key_is_refused_because_it_is_verify_only() {
        let store = PrincipalKeyStore::new();
        let err = store.issue("sp1", "ES256", 0).expect_err("refuses");
        assert!(
            err.to_string().contains("ES256"),
            "the refusal has to name the scheme, got: {err}"
        );
    }

    #[test]
    fn test_signing_for_an_unknown_principal_is_an_error() {
        let store = PrincipalKeyStore::new();
        let err = store.sign("nobody", b"payload").expect_err("no key");
        assert!(err.to_string().contains("no signing key"), "{err}");
    }

    #[test]
    fn test_an_overlap_binding_accepts_both_schemes() {
        let registry = SchemeRegistry::from_parts(
            vec![
                scheme("Ed25519", 1, SchemeStatus::Active),
                scheme("ES256", 2, SchemeStatus::Active),
            ],
            vec![ArtifactSchemeBinding {
                artifact_kind: ArtifactKind::SessionToken,
                current_scheme: "Ed25519".to_string(),
                deprecating_scheme: Some("ES256".to_string()),
                overlap_end_secs: Some(1_000),
            }],
        );
        assert!(
            registry
                .resolve_for_verification(ArtifactKind::SessionToken, "ES256", 500)
                .is_ok()
        );
        assert!(
            registry
                .resolve_for_verification(ArtifactKind::SessionToken, "ES256", 1_500)
                .is_err()
        );
    }

    /// RS256 is what external identity providers sign with, so a real
    /// PKCS#1 v1.5 signature over SHA-256 has to verify, a tampered one has
    /// to fail, and both DER spellings of the public key have to be
    /// accepted because a JWKS and an IdP metadata document disagree on
    /// which one they hand over
    #[test]
    fn test_rs256_verifies_a_real_signature_in_both_key_encodings() {
        use rsa::pkcs1::EncodeRsaPublicKey;
        use rsa::pkcs1v15::SigningKey;
        use rsa::pkcs8::EncodePublicKey;
        use rsa::signature::{SignatureEncoding, Signer};
        use rsa::{RsaPrivateKey, RsaPublicKey};

        // The rsa crate carries its own rand_core major, so the generator
        // comes from there rather than from the workspace rand
        let mut rng = rsa::rand_core::OsRng;
        let private = RsaPrivateKey::new(&mut rng, 2048).expect("generates a key");
        let public = RsaPublicKey::from(&private);
        let signing: SigningKey<sha2::Sha256> = SigningKey::new(private);

        let message = b"an assertion the identity provider signed";
        let signature = signing.sign(message).to_bytes().to_vec();

        // PKCS#1 RSAPublicKey, the bare spelling
        let pkcs1 = public.to_pkcs1_der().expect("pkcs1").as_bytes().to_vec();
        assert!(
            verify_with(
                &VerifyingMaterial::Rs256(pkcs1.clone()),
                message,
                &signature
            )
            .expect("verifies"),
            "a valid signature must verify against a PKCS#1 key"
        );

        // SubjectPublicKeyInfo, the spelling a JWKS or metadata document uses
        let spki = public
            .to_public_key_der()
            .expect("spki")
            .as_bytes()
            .to_vec();
        assert!(
            verify_with(&VerifyingMaterial::Rs256(spki.clone()), message, &signature)
                .expect("verifies"),
            "the same signature must verify against the wrapped key"
        );

        // A changed message is a failed verification, not an error
        assert!(
            !verify_with(
                &VerifyingMaterial::Rs256(spki.clone()),
                b"other",
                &signature
            )
            .expect("returns a verdict"),
            "a signature over different bytes must not verify"
        );

        // A tampered signature likewise
        let mut broken = signature.clone();
        broken[0] ^= 0xFF;
        assert!(
            !verify_with(&VerifyingMaterial::Rs256(spki), message, &broken)
                .expect("returns a verdict"),
            "a tampered signature must not verify"
        );

        // Key material that is neither spelling is a configuration error and
        // is reported rather than reduced to "did not verify"
        assert!(
            verify_with(
                &VerifyingMaterial::Rs256(b"not a key".to_vec()),
                message,
                &signature
            )
            .is_err(),
            "unparseable key material must be an error"
        );
    }
}
