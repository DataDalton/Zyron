# Signature Agility, Developer Notes

Implementation reference for Zyron's cryptographic scheme registry and rotation infrastructure. Customer-facing documentation lives in [`business/security/signature-agility.md`](../../business/security/signature-agility.md).

## Scheme identifier encoding

Every Zyron-signed artifact carries the identity of the scheme that signed it, in one of these forms, chosen per artifact kind by `ArtifactKind::identifier_encoding`:

| Encoding | Artifact kinds | Read by the substrate |
| -------- | -------------- | --------------------- |
| `JwsAlgHeader` | JWT | Yes |
| `LeadingSchemeIdByte` | FederationInviteToken, AppImage, SessionToken, PersonalAccessToken | Yes |
| `X509SignatureAlgorithmOid` | ClusterIdentity, mTLSCertificate | Declared, no reader yet |
| `XmlSignatureMethod` | SamlAssertion | Declared, no reader yet |
| `OpaqueBearer` | ScimToken | Not signed, nothing to read |

For the leading-byte form, `tag_binary_signature` prepends a one-byte `scheme_id` ahead of the signature bytes and `resolve_artifact_bytes` splits it back off on verification.

## Registry types

`crates/zyron-common/src/format/scheme.rs`:

```rust
pub struct SignatureSchemeRegistration {
    pub scheme_name: &'static str,
    pub scheme_id: SchemeId,          // u16
    pub category: SchemeCategory,     // Kem, Signature, Symmetric
    pub status: SchemeStatus,         // Active, Deprecating, Retired, Reserved
    pub first_available_version: &'static str,
    pub retirement_date: Option<&'static str>,
    pub notes: &'static str,
}
```

The registration carries no verifier or signer function pointer. Dispatch is a hand-written match on the key material in `crates/zyron-auth/src/signature.rs`, not a compile-time-inlined function from the registration. Schemes register with `inventory::submit!` from `crates/zyron-auth/src/format.rs`.

`SchemeStatus::can_sign` is true only for `Active`. `can_verify` is true for `Active`, `Deprecating` and `Retired`. A `Reserved` scheme can neither sign nor verify, it holds a name and an id against a future release.

## Registered schemes

| Scheme | id | Status |
| ------ | -- | ------ |
| Ed25519 | 1 | Active, the default signing scheme for Zyron's own artifacts |
| ES256 | 2 | Active, verify only |
| RS256 | 3 | Active |
| ML-DSA-65 | 16 | Reserved |
| SLH-DSA-SHA2-128s | 17 | Reserved |
| Ed25519+ML-DSA-65 | 18 | Reserved |

The three reserved slots hold ids and names for post-quantum schemes so a later release adds one as an ordinary registration. They are visible in the registry view with status `Reserved`.

## Signing and verification

Verification in `verify_with` accepts all three active schemes: Ed25519 through `ed25519-dalek`, ES256 through `p256`, RS256 through `ring`. RS256 accepts a public key in either the bare PKCS#1 or the SPKI-wrapped DER form, and reports an unparseable key as an error rather than a silent verification failure.

Signing in `PrincipalKeyStore::sign` covers Ed25519 and RS256 only. RS256 keys are generated through the `rsa` crate and signed through `ring` with PKCS#1 and SHA-256. ES256 has no signer: Zyron verifies ES256 for WebAuthn interoperability but never holds an ES256 private key, so `sign` and `generate_keypair` refuse it. Verify-only is enforced in the key store, not in the registry, where ES256's status is `Active`.

Secrets held in the key store are wrapped in `Zeroizing` so they are cleared on drop.

## Verification dispatch

`resolve_for_verification` reads the scheme identifier for the artifact kind, looks the scheme up in the registry, and refuses closed in these cases, each a distinct `SchemeError`:

- `UnknownScheme` or `UnknownSchemeId`, the identifier names no registered scheme.
- `ReservedScheme`, the scheme is a reserved future slot.
- `NotAcceptedForArtifact`, the scheme is registered but not the current or in-overlap scheme bound to this artifact kind.
- `Unsigned`, the artifact kind is an opaque bearer token with no signature to dispatch on.
- `MissingIdentifier`, the artifact is too short to carry its identifier.

A retired scheme is refused once the wall clock passes its recorded last-valid-artifact expiry.

## Registry views

`zyron_sys.crypto.scheme_registry`, one row per registered scheme: `scheme_name`, `scheme_id`, `category`, `status`, `first_available_version`, `retirement_date`, `last_valid_artifact_expiry`, `can_sign`, `can_verify`, `notes`.

`zyron_sys.crypto.artifact_scheme_map`, one row per signed artifact kind: `artifact_kind`, `identifier_encoding`, `current_scheme`, `deprecating_scheme`, `overlap_end_secs`, `rotation_in_progress`.

## Default bindings

`default_artifact_bindings` binds every signed artifact kind to Ed25519, with SamlAssertion bound to RS256 because that is what SAML relying parties accept. ScimToken is an opaque bearer token and gets no binding. Inbound SAML assertions and WebAuthn assertions are verified against whatever scheme they declare, which for WebAuthn is handled directly in the WebAuthn path rather than through this registry.

## Rotation with overlap

`ROTATE SIGNATURE SCHEME <artifact_kind> TO <new_scheme> [OVERLAP <interval>]`:

1. `rotate_scheme` moves the current scheme into `deprecating_scheme`, sets `current_scheme` to the new one, and sets `overlap_end_secs` to now plus the overlap. The default overlap when the clause is omitted is 24 hours.
2. New signatures use the current scheme.
3. Verification accepts both the current and the deprecating scheme until the overlap ends, after which only the current scheme is accepted.

A reserved scheme cannot be a rotation target.

## Long-lived artifact retention

Some signed artifacts outlive an overlap window, such as federation invite tokens and app image signatures. The registry tracks a per-scheme last-valid-artifact expiry through `note_artifact_expiry`, and `verifiers_ready_for_removal` reports a scheme whose last artifact has expired. `zyron-ctl release verify` reads that to report a verifier that can be deleted, so the retirement decision is a date and an expiry rather than a judgment call.

## Adding a new scheme

There is no per-scheme file. Registering a scheme touches these sites:

1. Submit a `SignatureSchemeRegistration` from `crates/zyron-auth/src/format.rs` with the name, id, category, status, first available version, and notes.
2. Add a `VerifyingMaterial` variant for the scheme and its arm in `verify_with`.
3. Add its arms in `verifying_material`, and in `PrincipalKeyStore::sign` and `generate_keypair` if Zyron holds its private key.

On restart the scheme appears in `zyron_sys.crypto.scheme_registry`, and `SET SIGNATURE SCHEME '<scheme>' FOR ARTIFACT KIND '<kind>'` starts binding new artifacts to it.

## Validation

The release check verifies the scheme registry for a duplicate id, a duplicate name, an id that does not fit one byte, and a scheme past its retirement date still carrying a verifier. Startup loads the substrate and reports the scheme count, and refuses to start if the substrate fails to load.

## Related

- [../storage/format-agility.md](../storage/format-agility.md), the parallel infrastructure for on-disk file format versioning.
- [../operations/auto-upgrade.md](../operations/auto-upgrade.md), release manifest signing and verification.
- Business-facing counterpart: [`business/security/signature-agility.md`](../../business/security/signature-agility.md).
