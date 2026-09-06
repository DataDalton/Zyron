# Signature Agility

Zyron can rotate the cryptographic scheme used for any signed artifact without breaking anything already signed. Every signed artifact carries a scheme identifier so the verifier knows which algorithm to apply, and rotation to a new scheme happens transparently.

## What this means for you

- You can move to a new signature scheme without reissuing existing artifacts.
- Existing artifacts signed with an older scheme continue to verify until their normal expiration.
- Rotation is an operator-driven decision, not a forced upgrade behavior.
- Unknown or retired schemes fail closed with a clear error.

## Currently registered schemes

- **Ed25519**, Zyron's default signing scheme.
- **ES256**, accepted for WebAuthn interoperability.
- **RS256**, accepted for external identity provider interoperability.

Zyron's own artifacts (JWTs, cluster identity certificates, federation invite tokens, App image signatures, session tokens, PATs) are signed with the current scheme configured for that artifact kind. External artifacts (WebAuthn assertions from hardware keys, JWTs from external identity providers) are verified against whatever scheme the external system uses.

The registry also carries reserved slots for post-quantum schemes, which appear with status `Reserved` until a release enables one. A reserved scheme cannot yet sign or verify.

## Rotation with overlap

Rotation always uses an overlap window. During the overlap, both the outgoing scheme and the new scheme are accepted. After the overlap ends, the outgoing scheme is rejected.

Choose an overlap duration longer than the longest-lived artifact signed with the outgoing scheme, so callers have time to migrate.

## DDL

```sql
-- Set the current scheme for a specific artifact kind
SET SIGNATURE SCHEME 'Ed25519' FOR ARTIFACT KIND 'JWT';

-- Rotate to a new scheme with an overlap window
ROTATE SIGNATURE SCHEME JWT TO '<new_scheme>' OVERLAP '<interval>';

-- Rotate a Service Principal's key, optionally changing its scheme
ROTATE SERVICE PRINCIPAL KEY <sp> SCHEME '<new_scheme>' OVERLAP '<interval>';

-- List registered schemes
LIST SIGNATURE SCHEMES;

-- Show per-artifact-kind scheme mapping
LIST ARTIFACT SCHEMES;
```

## Long-lived artifact retention

Some signed artifacts live past the overlap window. Federation invite tokens, App image signatures, and audit chain entries can outlive normal rotation cycles. For these, Zyron keeps the verifier for the retired scheme active until the last artifact signed with it expires. You cannot accidentally strand old artifacts.

## Retirement lifecycle

A scheme moves through three states over its lifetime:

- **Active**, the scheme is currently used to sign new artifacts and to verify existing ones.
- **Deprecating**, the scheme is no longer used to sign new artifacts, but existing artifacts still verify against it.
- **Retired**, the scheme's verifier is scheduled for removal once the last artifact signed with it has expired.

Query current state via `zyron_sys.crypto.scheme_registry`.

## Registry

Query the registry to see scheme state and per-artifact-kind mappings.

- `zyron_sys.crypto.scheme_registry`, one row per registered scheme with status and retirement timing.
- `zyron_sys.crypto.artifact_scheme_map`, per-artifact-kind current scheme, and the previous scheme when a rotation is in progress.

## Related

- [../storage/format-agility.md](../storage/format-agility.md), the parallel guarantee for on-disk file format upgrades.
- [../operations/auto-upgrade.md](../operations/auto-upgrade.md), how release manifests are signed and verified during auto-upgrade.
