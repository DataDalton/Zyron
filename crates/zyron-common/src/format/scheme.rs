//! The signature scheme registry.
//!
//! Every artifact Zyron signs carries the identity of the scheme that signed
//! it, so verification is a lookup and a dispatch rather than an assumption.
//! JWTs carry it in the `alg` header, X.509 certificates in the
//! `signatureAlgorithm` OID, and Zyron's own binary artifacts in a leading
//! scheme id byte ahead of the signature bytes.
//!
//! Scheme definitions are static, submitted by the crypto module that
//! implements them. The per-artifact-kind mapping is live state, because
//! rotation changes it, so it sits behind an atomic swap and is read on the
//! verification path without a lock

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Numeric tag a binary artifact carries ahead of its signature bytes.
///
/// One byte covers the tags in use, the type is `u16` so the registry can
/// outgrow a byte without the catalog changing shape. Encoding to an
/// artifact refuses a tag above 255 rather than truncating it
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemeId(pub u16);

impl SchemeId {
    /// The leading byte a binary artifact carries, or None when the tag does
    /// not fit one
    #[inline]
    pub const fn as_artifact_byte(self) -> Option<u8> {
        if self.0 <= u8::MAX as u16 {
            Some(self.0 as u8)
        } else {
            None
        }
    }
}

impl fmt::Display for SchemeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a scheme is for
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeCategory {
    /// Key encapsulation
    Kem,
    /// Digital signature
    Signature,
    /// Symmetric cipher or MAC
    Symmetric,
    /// Cryptographic hash, which a commit chain names as the algorithm it
    /// links its entries with
    Hash,
}

impl SchemeCategory {
    pub const fn label(self) -> &'static str {
        match self {
            SchemeCategory::Kem => "kem",
            SchemeCategory::Signature => "signature",
            SchemeCategory::Symmetric => "symmetric",
            SchemeCategory::Hash => "hash",
        }
    }
}

impl fmt::Display for SchemeCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where a scheme sits in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeStatus {
    /// Used to sign new artifacts and to verify existing ones
    Active,
    /// No longer signs, still verifies
    Deprecating,
    /// Verifier is scheduled for removal once the last artifact signed with
    /// it has expired
    Retired,
    /// Registered but not yet available to sign with, which is how a slot is
    /// held for a scheme whose verifier has not shipped
    Reserved,
}

impl SchemeStatus {
    pub const fn label(self) -> &'static str {
        match self {
            SchemeStatus::Active => "active",
            SchemeStatus::Deprecating => "deprecating",
            SchemeStatus::Retired => "retired",
            SchemeStatus::Reserved => "reserved",
        }
    }

    /// Whether the scheme may sign new artifacts
    #[inline]
    pub const fn can_sign(self) -> bool {
        matches!(self, SchemeStatus::Active)
    }

    /// Whether the scheme may still verify existing artifacts
    #[inline]
    pub const fn can_verify(self) -> bool {
        matches!(
            self,
            SchemeStatus::Active | SchemeStatus::Deprecating | SchemeStatus::Retired
        )
    }
}

impl fmt::Display for SchemeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One registered cryptographic scheme
#[derive(Debug, Clone, Copy)]
pub struct SignatureSchemeRegistration {
    /// The name the DDL and the `alg` header use
    pub scheme_name: &'static str,
    pub scheme_id: SchemeId,
    pub category: SchemeCategory,
    pub status: SchemeStatus,
    /// The Zyron version that introduced the scheme
    pub first_available_version: &'static str,
    /// ISO 8601 date the verifier is removed on, when one is scheduled
    pub retirement_date: Option<&'static str>,
    /// One line naming what the scheme is for
    pub notes: &'static str,
}

inventory::collect!(SignatureSchemeRegistration);

/// A kind of artifact Zyron signs or verifies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArtifactKind {
    Jwt,
    ClusterIdentity,
    FederationInviteToken,
    AppImage,
    SamlAssertion,
    MtlsCertificate,
    ScimToken,
    SessionToken,
    PersonalAccessToken,
}

/// Every artifact kind, in catalog order
pub const ALL_ARTIFACT_KINDS: &[ArtifactKind] = &[
    ArtifactKind::Jwt,
    ArtifactKind::ClusterIdentity,
    ArtifactKind::FederationInviteToken,
    ArtifactKind::AppImage,
    ArtifactKind::SamlAssertion,
    ArtifactKind::MtlsCertificate,
    ArtifactKind::ScimToken,
    ArtifactKind::SessionToken,
    ArtifactKind::PersonalAccessToken,
];

impl ArtifactKind {
    pub const fn catalog_name(self) -> &'static str {
        match self {
            ArtifactKind::Jwt => "JWT",
            ArtifactKind::ClusterIdentity => "ClusterIdentity",
            ArtifactKind::FederationInviteToken => "FederationInviteToken",
            ArtifactKind::AppImage => "AppImage",
            ArtifactKind::SamlAssertion => "SamlAssertion",
            ArtifactKind::MtlsCertificate => "mTLSCertificate",
            ArtifactKind::ScimToken => "ScimToken",
            ArtifactKind::SessionToken => "SessionToken",
            ArtifactKind::PersonalAccessToken => "PersonalAccessToken",
        }
    }

    pub fn parse(name: &str) -> Option<ArtifactKind> {
        ALL_ARTIFACT_KINDS
            .iter()
            .copied()
            .find(|kind| kind.catalog_name().eq_ignore_ascii_case(name))
    }

    /// How the artifact carries its scheme identity
    pub const fn identifier_encoding(self) -> SchemeIdentifierEncoding {
        match self {
            ArtifactKind::Jwt => SchemeIdentifierEncoding::JwsAlgHeader,
            ArtifactKind::SamlAssertion => SchemeIdentifierEncoding::XmlSignatureMethod,
            ArtifactKind::ClusterIdentity | ArtifactKind::MtlsCertificate => {
                SchemeIdentifierEncoding::X509SignatureAlgorithmOid
            }
            ArtifactKind::ScimToken => SchemeIdentifierEncoding::OpaqueBearer,
            ArtifactKind::FederationInviteToken
            | ArtifactKind::AppImage
            | ArtifactKind::SessionToken
            | ArtifactKind::PersonalAccessToken => SchemeIdentifierEncoding::LeadingSchemeIdByte,
        }
    }

    /// Whether Zyron signs artifacts of this kind at all. An opaque bearer
    /// token has no signature to dispatch on
    #[inline]
    pub const fn is_signed(self) -> bool {
        !matches!(
            self.identifier_encoding(),
            SchemeIdentifierEncoding::OpaqueBearer
        )
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.catalog_name())
    }
}

/// How an artifact kind carries the identity of the scheme that signed it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemeIdentifierEncoding {
    /// RFC 7515 `alg` header
    JwsAlgHeader,
    /// X.509 `signatureAlgorithm` OID
    X509SignatureAlgorithmOid,
    /// XML signature `SignatureMethod` algorithm URI
    XmlSignatureMethod,
    /// Leading scheme id byte ahead of the signature bytes
    LeadingSchemeIdByte,
    /// Not signed, so no identity to read
    OpaqueBearer,
}

impl SchemeIdentifierEncoding {
    pub const fn label(self) -> &'static str {
        match self {
            SchemeIdentifierEncoding::JwsAlgHeader => "jws_alg_header",
            SchemeIdentifierEncoding::X509SignatureAlgorithmOid => "x509_signature_algorithm_oid",
            SchemeIdentifierEncoding::XmlSignatureMethod => "xml_signature_method",
            SchemeIdentifierEncoding::LeadingSchemeIdByte => "leading_scheme_id_byte",
            SchemeIdentifierEncoding::OpaqueBearer => "opaque_bearer",
        }
    }
}

/// The scheme one artifact kind signs with, and the one still accepted while
/// a rotation overlaps
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactSchemeBinding {
    pub artifact_kind: ArtifactKind,
    pub current_scheme: String,
    /// The outgoing scheme during a rotation
    pub deprecating_scheme: Option<String>,
    /// Unix seconds the outgoing scheme stops being accepted at
    pub overlap_end_secs: Option<u64>,
}

impl ArtifactSchemeBinding {
    pub fn new(artifact_kind: ArtifactKind, current_scheme: impl Into<String>) -> Self {
        Self {
            artifact_kind,
            current_scheme: current_scheme.into(),
            deprecating_scheme: None,
            overlap_end_secs: None,
        }
    }

    /// Whether a scheme is accepted for this artifact kind at a point in
    /// time
    pub fn accepts(&self, scheme_name: &str, now_secs: u64) -> bool {
        if self.current_scheme.eq_ignore_ascii_case(scheme_name) {
            return true;
        }
        match (&self.deprecating_scheme, self.overlap_end_secs) {
            (Some(outgoing), Some(end)) => {
                outgoing.eq_ignore_ascii_case(scheme_name) && now_secs < end
            }
            (Some(outgoing), None) => outgoing.eq_ignore_ascii_case(scheme_name),
            _ => false,
        }
    }

    /// True while a rotation is still inside its overlap window
    pub fn rotation_in_progress(&self, now_secs: u64) -> bool {
        match (&self.deprecating_scheme, self.overlap_end_secs) {
            (Some(_), Some(end)) => now_secs < end,
            (Some(_), None) => true,
            _ => false,
        }
    }
}

/// Default scheme per artifact kind, applied the first time a cluster starts.
///
/// Zyron signs with Ed25519 wherever the choice is Zyron's. SAML assertions
/// go out RS256 because that is what SAML relying parties accept, and
/// inbound assertions are verified against whatever scheme they declare
pub fn default_artifact_bindings() -> Vec<ArtifactSchemeBinding> {
    ALL_ARTIFACT_KINDS
        .iter()
        .filter(|kind| kind.is_signed())
        .map(|kind| {
            let scheme = match kind {
                ArtifactKind::SamlAssertion => "RS256",
                _ => "Ed25519",
            };
            ArtifactSchemeBinding::new(*kind, scheme)
        })
        .collect()
}

/// The loaded scheme registry, with the live per-artifact mapping beside it
#[derive(Debug)]
pub struct SchemeRegistry {
    schemes: Vec<SignatureSchemeRegistration>,
    bindings: std::sync::Mutex<Vec<ArtifactSchemeBinding>>,
    /// Per-scheme latest expiry among artifacts still signed with it, unix
    /// seconds. The retention sweep reads this to decide when a retired
    /// scheme's verifier can go
    last_valid_expiry: Vec<(String, Arc<AtomicU64>)>,
}

/// Why a scheme lookup or a verification dispatch failed
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemeError {
    /// The artifact names a scheme the registry does not hold
    UnknownScheme { named: String },
    /// The artifact carries a scheme id byte the registry does not hold
    UnknownSchemeId { scheme_id: u16 },
    /// The scheme exists but is not accepted for this artifact kind now
    NotAcceptedForArtifact {
        artifact_kind: ArtifactKind,
        scheme_name: String,
        current_scheme: String,
        overlap_end_secs: Option<u64>,
    },
    /// The scheme is retired and the artifact outlives the last artifact the
    /// scheme was allowed to cover
    RetiredScheme {
        scheme_name: String,
        last_valid_artifact_expiry: u64,
    },
    /// The scheme is registered but its verifier has not shipped
    ReservedScheme { scheme_name: String },
    /// The artifact kind carries no signature
    Unsigned { artifact_kind: ArtifactKind },
    /// The artifact is too short to carry a scheme identifier
    MissingIdentifier { artifact_kind: ArtifactKind },
}

impl fmt::Display for SchemeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemeError::UnknownScheme { named } => write!(
                f,
                "signature scheme `{named}` is not registered, so the artifact cannot be \
                 verified. Register the scheme or reissue the artifact under a registered one"
            ),
            SchemeError::UnknownSchemeId { scheme_id } => write!(
                f,
                "signature scheme id {scheme_id} is not registered, so the artifact cannot \
                 be verified"
            ),
            SchemeError::NotAcceptedForArtifact {
                artifact_kind,
                scheme_name,
                current_scheme,
                overlap_end_secs,
            } => match overlap_end_secs {
                Some(end) => write!(
                    f,
                    "scheme `{scheme_name}` is no longer accepted for {artifact_kind}, its \
                     overlap ended at unix second {end}. The current scheme is `{current_scheme}`"
                ),
                None => write!(
                    f,
                    "scheme `{scheme_name}` is not accepted for {artifact_kind}, whose \
                     current scheme is `{current_scheme}`"
                ),
            },
            SchemeError::RetiredScheme {
                scheme_name,
                last_valid_artifact_expiry,
            } => write!(
                f,
                "scheme `{scheme_name}` is retired and every artifact it covered expired at \
                 unix second {last_valid_artifact_expiry}"
            ),
            SchemeError::ReservedScheme { scheme_name } => write!(
                f,
                "scheme `{scheme_name}` holds a reserved registry slot and has no verifier \
                 in this binary"
            ),
            SchemeError::Unsigned { artifact_kind } => {
                write!(f, "{artifact_kind} artifacts carry no signature to verify")
            }
            SchemeError::MissingIdentifier { artifact_kind } => write!(
                f,
                "the {artifact_kind} artifact is too short to carry a scheme identifier"
            ),
        }
    }
}

impl std::error::Error for SchemeError {}

impl SchemeRegistry {
    /// Collects every submitted scheme and seeds the default per-artifact
    /// mapping
    pub fn load() -> SchemeRegistry {
        let schemes: Vec<SignatureSchemeRegistration> =
            inventory::iter::<SignatureSchemeRegistration>
                .into_iter()
                .copied()
                .collect();
        SchemeRegistry::from_parts(schemes, default_artifact_bindings())
    }

    pub fn from_parts(
        mut schemes: Vec<SignatureSchemeRegistration>,
        bindings: Vec<ArtifactSchemeBinding>,
    ) -> SchemeRegistry {
        schemes.sort_by_key(|s| s.scheme_id);
        let last_valid_expiry = schemes
            .iter()
            .map(|s| (s.scheme_name.to_string(), Arc::new(AtomicU64::new(0))))
            .collect();
        SchemeRegistry {
            schemes,
            bindings: std::sync::Mutex::new(bindings),
            last_valid_expiry,
        }
    }

    /// Every registered scheme, ordered by id
    pub fn schemes(&self) -> &[SignatureSchemeRegistration] {
        &self.schemes
    }

    /// A scheme by name, case insensitively
    pub fn by_name(&self, name: &str) -> Option<&SignatureSchemeRegistration> {
        self.schemes
            .iter()
            .find(|s| s.scheme_name.eq_ignore_ascii_case(name))
    }

    /// A scheme by its binary tag
    pub fn by_id(&self, id: SchemeId) -> Option<&SignatureSchemeRegistration> {
        self.schemes.iter().find(|s| s.scheme_id == id)
    }

    /// The current binding for one artifact kind
    pub fn binding(&self, kind: ArtifactKind) -> Option<ArtifactSchemeBinding> {
        self.bindings
            .lock()
            .ok()?
            .iter()
            .find(|b| b.artifact_kind == kind)
            .cloned()
    }

    /// Every binding, in catalog order
    pub fn bindings(&self) -> Vec<ArtifactSchemeBinding> {
        let mut all = self.bindings.lock().map(|b| b.clone()).unwrap_or_default();
        all.sort_by_key(|b| b.artifact_kind);
        all
    }

    /// Points an artifact kind at a scheme with no overlap, which is what
    /// `SET SIGNATURE SCHEME` does
    pub fn set_scheme(&self, kind: ArtifactKind, scheme_name: &str) -> Result<(), SchemeError> {
        let scheme = self
            .by_name(scheme_name)
            .ok_or_else(|| SchemeError::UnknownScheme {
                named: scheme_name.to_string(),
            })?;
        if scheme.status == SchemeStatus::Reserved {
            return Err(SchemeError::ReservedScheme {
                scheme_name: scheme.scheme_name.to_string(),
            });
        }
        let canonical = scheme.scheme_name.to_string();
        let Ok(mut bindings) = self.bindings.lock() else {
            return Err(SchemeError::UnknownScheme {
                named: scheme_name.to_string(),
            });
        };
        match bindings.iter_mut().find(|b| b.artifact_kind == kind) {
            Some(binding) => {
                binding.current_scheme = canonical;
                binding.deprecating_scheme = None;
                binding.overlap_end_secs = None;
            }
            None => bindings.push(ArtifactSchemeBinding::new(kind, canonical)),
        }
        Ok(())
    }

    /// Rotates an artifact kind onto a new scheme, keeping the outgoing one
    /// acceptable until `overlap_end_secs`
    pub fn rotate_scheme(
        &self,
        kind: ArtifactKind,
        new_scheme: &str,
        overlap_end_secs: u64,
    ) -> Result<ArtifactSchemeBinding, SchemeError> {
        let scheme = self
            .by_name(new_scheme)
            .ok_or_else(|| SchemeError::UnknownScheme {
                named: new_scheme.to_string(),
            })?;
        if scheme.status == SchemeStatus::Reserved {
            return Err(SchemeError::ReservedScheme {
                scheme_name: scheme.scheme_name.to_string(),
            });
        }
        let canonical = scheme.scheme_name.to_string();
        let Ok(mut bindings) = self.bindings.lock() else {
            return Err(SchemeError::UnknownScheme {
                named: new_scheme.to_string(),
            });
        };
        let binding = match bindings.iter_mut().find(|b| b.artifact_kind == kind) {
            Some(binding) => binding,
            None => {
                bindings.push(ArtifactSchemeBinding::new(kind, canonical.clone()));
                bindings
                    .last_mut()
                    .ok_or_else(|| SchemeError::UnknownScheme {
                        named: new_scheme.to_string(),
                    })?
            }
        };
        let outgoing = std::mem::replace(&mut binding.current_scheme, canonical);
        binding.deprecating_scheme = Some(outgoing);
        binding.overlap_end_secs = Some(overlap_end_secs);
        Ok(binding.clone())
    }

    /// Whether a named scheme may verify an artifact of this kind now.
    ///
    /// This is the dispatch gate. It resolves the scheme, checks it is
    /// accepted for the artifact kind at this instant, and refuses anything
    /// unknown or past its overlap rather than falling through to a default
    pub fn resolve_for_verification(
        &self,
        kind: ArtifactKind,
        scheme_name: &str,
        now_secs: u64,
    ) -> Result<&SignatureSchemeRegistration, SchemeError> {
        if !kind.is_signed() {
            return Err(SchemeError::Unsigned {
                artifact_kind: kind,
            });
        }
        let scheme = self
            .by_name(scheme_name)
            .ok_or_else(|| SchemeError::UnknownScheme {
                named: scheme_name.to_string(),
            })?;
        if scheme.status == SchemeStatus::Reserved {
            return Err(SchemeError::ReservedScheme {
                scheme_name: scheme.scheme_name.to_string(),
            });
        }
        if scheme.status == SchemeStatus::Retired {
            let expiry = self.last_valid_artifact_expiry(scheme.scheme_name);
            if now_secs >= expiry {
                return Err(SchemeError::RetiredScheme {
                    scheme_name: scheme.scheme_name.to_string(),
                    last_valid_artifact_expiry: expiry,
                });
            }
        }
        let binding = self.binding(kind);
        match binding {
            Some(binding) if binding.accepts(scheme.scheme_name, now_secs) => Ok(scheme),
            Some(binding) => Err(SchemeError::NotAcceptedForArtifact {
                artifact_kind: kind,
                scheme_name: scheme.scheme_name.to_string(),
                current_scheme: binding.current_scheme,
                overlap_end_secs: binding.overlap_end_secs,
            }),
            None => Err(SchemeError::NotAcceptedForArtifact {
                artifact_kind: kind,
                scheme_name: scheme.scheme_name.to_string(),
                current_scheme: String::new(),
                overlap_end_secs: None,
            }),
        }
    }

    /// The scheme a binary artifact's leading id byte names
    pub fn resolve_artifact_bytes<'a>(
        &'a self,
        kind: ArtifactKind,
        artifact: &'a [u8],
        now_secs: u64,
    ) -> Result<(&'a SignatureSchemeRegistration, &'a [u8]), SchemeError> {
        if kind.identifier_encoding() != SchemeIdentifierEncoding::LeadingSchemeIdByte {
            return Err(SchemeError::Unsigned {
                artifact_kind: kind,
            });
        }
        let Some((tag, rest)) = artifact.split_first() else {
            return Err(SchemeError::MissingIdentifier {
                artifact_kind: kind,
            });
        };
        let scheme = self
            .by_id(SchemeId(*tag as u16))
            .ok_or(SchemeError::UnknownSchemeId {
                scheme_id: *tag as u16,
            })?;
        let resolved = self.resolve_for_verification(kind, scheme.scheme_name, now_secs)?;
        Ok((resolved, rest))
    }

    /// One artifact kind's binding rendered as the value it persists and
    /// replicates as.
    ///
    /// `Ed25519` on its own while nothing is rotating, and
    /// `Ed25519|RS256|1757203200` during a rotation, which reads as the
    /// incoming scheme, the outgoing one, and the unix second the outgoing
    /// one stops being accepted
    pub fn binding_setting(&self, kind: ArtifactKind) -> Option<String> {
        let binding = self.binding(kind)?;
        Some(
            match (&binding.deprecating_scheme, binding.overlap_end_secs) {
                (Some(outgoing), Some(end)) => {
                    format!("{}|{outgoing}|{end}", binding.current_scheme)
                }
                (Some(outgoing), None) => format!("{}|{outgoing}|", binding.current_scheme),
                _ => binding.current_scheme.clone(),
            },
        )
    }

    /// Applies a binding in the form `binding_setting` renders, and answers
    /// with the value as it was stored.
    ///
    /// The persisted form and the replicated form are the same string read by
    /// the same function, so a binding seeded at boot and one applied from the
    /// log cannot end up meaning different things
    pub fn apply_binding_setting(
        &self,
        kind: ArtifactKind,
        value: &str,
    ) -> Result<String, SchemeError> {
        let mut parts = value.split('|');
        let incoming = parts.next().unwrap_or_default().trim();
        if incoming.is_empty() {
            return Err(SchemeError::UnknownScheme {
                named: value.to_string(),
            });
        }
        let outgoing = parts.next().map(str::trim).filter(|s| !s.is_empty());
        let overlap_end = match parts.next().map(str::trim).filter(|s| !s.is_empty()) {
            Some(text) => Some(
                text.parse::<u64>()
                    .map_err(|_| SchemeError::UnknownScheme {
                        named: value.to_string(),
                    })?,
            ),
            None => None,
        };

        // Both halves resolve before anything is written, so a binding naming
        // a scheme this binary does not carry is refused whole rather than
        // leaving the kind pointing at half a rotation
        let current = self.canonical_signing_scheme(incoming)?.to_string();
        let deprecating = match outgoing {
            Some(name) => Some(self.canonical_signing_scheme(name)?.to_string()),
            None => None,
        };

        let Ok(mut bindings) = self.bindings.lock() else {
            return Err(SchemeError::UnknownScheme {
                named: value.to_string(),
            });
        };
        match bindings.iter_mut().find(|b| b.artifact_kind == kind) {
            Some(binding) => {
                binding.current_scheme = current;
                binding.deprecating_scheme = deprecating;
                binding.overlap_end_secs = overlap_end;
            }
            None => bindings.push(ArtifactSchemeBinding {
                artifact_kind: kind,
                current_scheme: current,
                deprecating_scheme: deprecating,
                overlap_end_secs: overlap_end,
            }),
        }
        drop(bindings);
        self.binding_setting(kind)
            .ok_or_else(|| SchemeError::UnknownScheme {
                named: value.to_string(),
            })
    }

    /// The registry's own spelling of a scheme that is allowed to sign, so a
    /// binding stores one spelling however it was typed
    fn canonical_signing_scheme(&self, named: &str) -> Result<&'static str, SchemeError> {
        let scheme = self
            .by_name(named)
            .ok_or_else(|| SchemeError::UnknownScheme {
                named: named.to_string(),
            })?;
        if scheme.status == SchemeStatus::Reserved {
            return Err(SchemeError::ReservedScheme {
                scheme_name: scheme.scheme_name.to_string(),
            });
        }
        Ok(scheme.scheme_name)
    }

    /// Records the expiry of an artifact signed with a scheme, so the
    /// retention sweep knows when the verifier can go
    pub fn note_artifact_expiry(&self, scheme_name: &str, expiry_secs: u64) {
        if let Some((_, cell)) = self
            .last_valid_expiry
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(scheme_name))
        {
            cell.fetch_max(expiry_secs, Ordering::Relaxed);
        }
    }

    /// The latest expiry seen among artifacts signed with a scheme
    pub fn last_valid_artifact_expiry(&self, scheme_name: &str) -> u64 {
        self.last_valid_expiry
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(scheme_name))
            .map(|(_, cell)| cell.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Clears the recorded expiry for a scheme, which the sweep does once it
    /// has confirmed no artifact signed with it survives
    pub fn clear_artifact_expiry(&self, scheme_name: &str) {
        if let Some((_, cell)) = self
            .last_valid_expiry
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(scheme_name))
        {
            cell.store(0, Ordering::Relaxed);
        }
    }

    /// Retired schemes whose last artifact has expired, which is the set the
    /// sweep reports as ready for verifier removal
    pub fn verifiers_ready_for_removal(&self, now_secs: u64) -> Vec<&'static str> {
        self.schemes
            .iter()
            .filter(|s| s.status == SchemeStatus::Retired)
            .filter(|s| {
                let expiry = self.last_valid_artifact_expiry(s.scheme_name);
                expiry == 0 || now_secs >= expiry
            })
            .map(|s| s.scheme_name)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_defaults_cover_every_signed_artifact_kind() {
        let registry = registry();
        for kind in ALL_ARTIFACT_KINDS {
            if kind.is_signed() {
                assert!(registry.binding(*kind).is_some(), "{kind} has no binding");
            } else {
                assert!(
                    registry.binding(*kind).is_none(),
                    "{kind} should be unsigned"
                );
            }
        }
        assert_eq!(
            registry
                .binding(ArtifactKind::SamlAssertion)
                .expect("bound")
                .current_scheme,
            "RS256"
        );
        assert_eq!(
            registry
                .binding(ArtifactKind::Jwt)
                .expect("bound")
                .current_scheme,
            "Ed25519"
        );
    }

    #[test]
    fn test_verification_dispatches_by_name() {
        let registry = registry();
        assert!(
            registry
                .resolve_for_verification(ArtifactKind::Jwt, "Ed25519", 0)
                .is_ok()
        );
        let err = registry
            .resolve_for_verification(ArtifactKind::Jwt, "HS256", 0)
            .expect_err("unknown fails closed");
        assert!(matches!(err, SchemeError::UnknownScheme { .. }));
    }

    #[test]
    fn test_reserved_scheme_cannot_be_set_or_verified() {
        let registry = registry();
        assert!(matches!(
            registry.set_scheme(ArtifactKind::Jwt, "ML-DSA-65"),
            Err(SchemeError::ReservedScheme { .. })
        ));
        assert!(matches!(
            registry.resolve_for_verification(ArtifactKind::Jwt, "ML-DSA-65", 0),
            Err(SchemeError::ReservedScheme { .. })
        ));
    }

    #[test]
    fn test_rotation_accepts_both_schemes_during_overlap() {
        let registry = SchemeRegistry::from_parts(
            vec![
                scheme("Ed25519", 1, SchemeStatus::Active),
                scheme("ML-DSA-65", 16, SchemeStatus::Active),
            ],
            default_artifact_bindings(),
        );
        let binding = registry
            .rotate_scheme(ArtifactKind::Jwt, "ML-DSA-65", 1_000)
            .expect("rotates");
        assert_eq!(binding.current_scheme, "ML-DSA-65");
        assert_eq!(binding.deprecating_scheme.as_deref(), Some("Ed25519"));
        assert!(binding.rotation_in_progress(500));

        assert!(
            registry
                .resolve_for_verification(ArtifactKind::Jwt, "Ed25519", 500)
                .is_ok()
        );
        assert!(
            registry
                .resolve_for_verification(ArtifactKind::Jwt, "ML-DSA-65", 500)
                .is_ok()
        );

        let err = registry
            .resolve_for_verification(ArtifactKind::Jwt, "Ed25519", 1_001)
            .expect_err("overlap ended");
        assert!(matches!(err, SchemeError::NotAcceptedForArtifact { .. }));
    }

    #[test]
    fn test_set_scheme_clears_a_rotation() {
        let registry = registry();
        registry
            .rotate_scheme(ArtifactKind::Jwt, "ES256", 1_000)
            .expect("rotates");
        registry
            .set_scheme(ArtifactKind::Jwt, "Ed25519")
            .expect("sets");
        let binding = registry.binding(ArtifactKind::Jwt).expect("bound");
        assert_eq!(binding.current_scheme, "Ed25519");
        assert!(binding.deprecating_scheme.is_none());
    }

    #[test]
    fn test_binary_artifact_dispatches_on_its_leading_byte() {
        let registry = registry();
        let mut artifact = vec![1u8];
        artifact.extend_from_slice(b"signature bytes");
        let (scheme, rest) = registry
            .resolve_artifact_bytes(ArtifactKind::SessionToken, &artifact, 0)
            .expect("dispatches");
        assert_eq!(scheme.scheme_name, "Ed25519");
        assert_eq!(rest, b"signature bytes");

        let unknown = vec![200u8, 0, 0];
        assert!(matches!(
            registry.resolve_artifact_bytes(ArtifactKind::SessionToken, &unknown, 0),
            Err(SchemeError::UnknownSchemeId { scheme_id: 200 })
        ));
        assert!(matches!(
            registry.resolve_artifact_bytes(ArtifactKind::SessionToken, &[], 0),
            Err(SchemeError::MissingIdentifier { .. })
        ));
    }

    #[test]
    fn test_opaque_bearer_has_no_dispatch() {
        let registry = registry();
        assert!(matches!(
            registry.resolve_for_verification(ArtifactKind::ScimToken, "Ed25519", 0),
            Err(SchemeError::Unsigned { .. })
        ));
    }

    #[test]
    fn test_retired_scheme_verifies_until_its_last_artifact_expires() {
        let registry = SchemeRegistry::from_parts(
            vec![
                scheme("Ed25519", 1, SchemeStatus::Active),
                scheme("ES256", 2, SchemeStatus::Retired),
            ],
            default_artifact_bindings(),
        );
        // ES256 signed the JWTs already out there, then the kind rotated to
        // Ed25519 with an overlap that has not ended
        registry
            .set_scheme(ArtifactKind::Jwt, "ES256")
            .expect("sets");
        registry
            .rotate_scheme(ArtifactKind::Jwt, "Ed25519", u64::MAX)
            .expect("rotates");
        registry.note_artifact_expiry("ES256", 5_000);

        assert!(
            registry
                .resolve_for_verification(ArtifactKind::Jwt, "ES256", 4_000)
                .is_ok(),
            "a retired scheme still verifies while one of its artifacts lives"
        );
        assert!(registry.verifiers_ready_for_removal(4_000).is_empty());
        assert_eq!(
            registry.verifiers_ready_for_removal(5_000),
            vec!["ES256"],
            "the verifier is removable once the last artifact expired"
        );
        let err = registry
            .resolve_for_verification(ArtifactKind::Jwt, "ES256", 6_000)
            .expect_err("past expiry");
        assert!(matches!(err, SchemeError::RetiredScheme { .. }));
    }

    #[test]
    fn test_artifact_kinds_round_trip_their_names() {
        for kind in ALL_ARTIFACT_KINDS {
            assert_eq!(ArtifactKind::parse(kind.catalog_name()), Some(*kind));
            assert_eq!(
                ArtifactKind::parse(&kind.catalog_name().to_lowercase()),
                Some(*kind)
            );
        }
        assert_eq!(ArtifactKind::parse("nothing"), None);
    }

    #[test]
    fn test_scheme_id_fits_one_artifact_byte() {
        assert_eq!(SchemeId(1).as_artifact_byte(), Some(1));
        assert_eq!(SchemeId(255).as_artifact_byte(), Some(255));
        assert_eq!(SchemeId(256).as_artifact_byte(), None);
    }
}
