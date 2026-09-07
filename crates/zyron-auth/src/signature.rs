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

/// Ed25519 keys this process has already turned from bytes into a point.
///
/// The 32 bytes a key is stored and configured as are a compressed curve
/// point, and using it means decompressing it, which costs a field square
/// root. That work is identical every time the same key verifies, and a
/// key verifies many signatures: the vendor's release key checks every
/// artifact, and an identity provider's key checks every token. A process
/// holds a handful of distinct keys, so a short list scanned in order beats
/// hashing the key to find it
static ED25519_KEYS: parking_lot::RwLock<Vec<([u8; 32], VerifyingKey)>> =
    parking_lot::RwLock::new(Vec::new());

/// How many parsed keys are kept. Past this the oldest goes, which for a
/// list this size only happens when keys are being rotated through faster
/// than they are used
const ED25519_KEY_CACHE: usize = 16;

/// The parsed form of a stored Ed25519 public key.
///
/// A key that does not decompress is not cached, so a bad key costs the
/// same error every time rather than being remembered as usable
fn ed25519_key(bytes: &[u8; 32]) -> Result<VerifyingKey> {
    if let Some((_, key)) = ED25519_KEYS.read().iter().find(|(seen, _)| seen == bytes) {
        return Ok(*key);
    }
    let key = VerifyingKey::from_bytes(bytes).map_err(|e| {
        ZyronError::SignatureScheme(format!("Ed25519 public key is not on the curve, {e}"))
    })?;
    let mut cache = ED25519_KEYS.write();
    // Another caller may have parsed the same key while this one worked
    if !cache.iter().any(|(seen, _)| seen == bytes) {
        if cache.len() >= ED25519_KEY_CACHE {
            cache.remove(0);
        }
        cache.push((*bytes, key));
    }
    Ok(key)
}

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
            let key = ed25519_key(public_key)?;
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

    /// SHA-256 over the public half, in hex.
    ///
    /// Two members hold different keys for the same principal, because each
    /// draws its own, so this is what an operator compares to tell which node
    /// signed something. Taken over the public half, which is not a secret
    pub fn fingerprint(&self) -> String {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(&self.public_key);
        let mut out = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

/// A principal's outgoing key during a rotation
#[derive(Debug, Clone)]
pub struct RetiringKey {
    pub key: PrincipalKey,
    /// Unix seconds the outgoing key stops being accepted at
    pub overlap_end_secs: u64,
}

/// One key this node holds, as it is reported to an operator.
///
/// The public half and its metadata. There is no method here or on the store
/// that returns a secret half, which is what makes this safe to read out of a
/// view and to compare between members
#[derive(Debug, Clone)]
pub struct PublishedKey {
    pub key: PrincipalKey,
    /// None for the key a principal signs with now, Some for one that is
    /// still accepted until the moment it names
    pub overlap_end_secs: Option<u64>,
}

/// What a rotation produced
#[derive(Debug, Clone)]
pub struct RotationOutcome {
    pub principal: String,
    pub new_scheme: String,
    pub previous_scheme: Option<String>,
    pub overlap_end_secs: u64,
}

/// Bound into the principal key file's authenticated data, so a file lifted
/// from one store cannot be opened as another kind of sealed blob
const PRINCIPAL_KEY_FILE_AAD: &[u8] = b"principal-signing-keys";

fn put_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(buf, bytes.len() as u32);
    buf.extend_from_slice(bytes);
}

/// Walks the decrypted key file, refusing a short read rather than reading
/// past the end of one field into the next
struct KeyFileReader<'a> {
    data: &'a [u8],
    at: usize,
}

impl KeyFileReader<'_> {
    fn short() -> ZyronError {
        ZyronError::Internal("principal key file is truncated".to_string())
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self
            .data
            .get(self.at..self.at + 4)
            .ok_or_else(Self::short)?
            .try_into()
            .map_err(|_| Self::short())?;
        self.at += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes: [u8; 8] = self
            .data
            .get(self.at..self.at + 8)
            .ok_or_else(Self::short)?
            .try_into()
            .map_err(|_| Self::short())?;
        self.at += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    fn bytes(&mut self) -> Result<&[u8]> {
        let len = self.u32()? as usize;
        let slice = self
            .data
            .get(self.at..self.at + len)
            .ok_or_else(Self::short)?;
        self.at += len;
        Ok(slice)
    }

    fn text(&mut self) -> Result<String> {
        let slice = self.bytes()?;
        String::from_utf8(slice.to_vec()).map_err(|_| {
            ZyronError::Internal("principal key file holds a name that is not UTF-8".to_string())
        })
    }
}

/// Writes through a sibling file and a rename, so a crash part way leaves
/// either the keys the last write sealed or the ones before them, never a
/// file the next boot refuses and no key at all
fn write_replacing(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let directory = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(directory).map_err(|e| {
        ZyronError::Internal(format!("{} is not writable, {e}", directory.display()))
    })?;
    let temporary = path.with_extension("tmp");
    {
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary).map_err(|e| {
            ZyronError::Internal(format!("{} is not writable, {e}", temporary.display()))
        })?;
        file.write_all(bytes).map_err(|e| {
            ZyronError::Internal(format!("{} was not written, {e}", temporary.display()))
        })?;
        file.sync_all().map_err(|e| {
            ZyronError::Internal(format!("{} was not flushed, {e}", temporary.display()))
        })?;
    }
    std::fs::rename(&temporary, path).map_err(|e| {
        let _ = std::fs::remove_file(&temporary);
        ZyronError::Internal(format!("{} was not replaced, {e}", path.display()))
    })?;
    #[cfg(unix)]
    if let Ok(dir) = std::fs::File::open(directory) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Per-principal signing keys with overlap-aware rotation.
///
/// A rotation issues a new key, marks the old one retiring, and keeps
/// accepting the old one until the overlap ends. Both keys verify during the
/// overlap and only the new one signs, which is what lets a caller holding an
/// artifact signed a moment before the rotation still be believed.
///
/// # Nothing holds a key here yet
///
/// Service principals do not exist. There is no principal kind one could be,
/// every privilege is granted to a role id, and the only JWT this engine
/// verifies is signed with a shared secret, so a key pair has nothing to be
/// presented to. `ROTATE SERVICE PRINCIPAL KEY` is refused for that reason
/// and this store has no caller outside its own tests.
///
/// It is kept rather than removed because the rotation it implements is the
/// part that is hard to get right, and it is finished: overlap-aware
/// rotation across three schemes, sealed at rest, swept when an overlap
/// lapses. `published` and `zyron_sys.security.principal_keys` report what a
/// node holds.
///
/// # Two questions to settle before anything uses it
///
/// A store belongs to the node that owns the data directory it is sealed in,
/// because the file is wrapped by that node's own master key, which is
/// derived per data directory. So two members that both drew a key for one
/// principal hold different keys and neither can sign as the other. Whether
/// that is right depends on what the key is for, and that is not decided:
///
/// - A principal that authenticates by signing an assertion holds its own
///   secret, and this engine would keep only the public half to verify
///   against. This store draws and seals a secret and never hands it out, so
///   it does not serve that shape as written.
/// - A principal whose tokens this engine issues needs the secret here, and
///   then the public halves are published for others to verify against, one
///   entry per key rather than one per principal.
///
/// A secret must not be carried between members either way: it would travel
/// the consensus log and then sit in every member's write-ahead log, its
/// snapshots, and every backup taken afterwards, permanently
#[derive(Debug, Default)]
pub struct PrincipalKeyStore {
    keys: parking_lot::RwLock<Vec<PrincipalKey>>,
    retiring: parking_lot::RwLock<Vec<RetiringKey>>,
    /// Secret halves, never handed out. Length varies by scheme: 32 raw bytes
    /// for Ed25519, a DER PKCS#8 PrivateKeyInfo for RS256. Zeroized on drop
    /// and on replacement, so a rotated-out secret does not stay in freed heap
    secrets: parking_lot::RwLock<Vec<(String, Zeroizing<Vec<u8>>)>>,
    /// Where the keys are written down, absent for a store that lives only as
    /// long as the process
    storage: Option<PrincipalKeyStorage>,
}

/// Where a key store writes itself down, and what wraps the secret halves on
/// the way
struct PrincipalKeyStorage {
    path: std::path::PathBuf,
    key_store: Arc<dyn crate::encryption::KeyStore>,
    /// The wrapping key's handle, allocated on the first write and read back
    /// out of the file after that, so a restart opens what the last process
    /// sealed
    wrapping_key_id: parking_lot::Mutex<Option<u32>>,
    /// Held across the read, change, and rename of one write, because an
    /// issue and a rotation on two connections would otherwise each write a
    /// file describing only what it knew
    writing: std::sync::Mutex<()>,
}

impl std::fmt::Debug for PrincipalKeyStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrincipalKeyStorage")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl PrincipalKeyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens the store at `path`, putting back whatever the last process
    /// sealed there.
    ///
    /// A missing file is a store that has issued nothing yet. A file that
    /// cannot be opened is an error rather than an empty store, because
    /// starting empty would silently issue a second key for a principal whose
    /// artifacts are already signed with the first
    pub fn open(
        path: std::path::PathBuf,
        key_store: Arc<dyn crate::encryption::KeyStore>,
    ) -> Result<Self> {
        let store = Self {
            keys: parking_lot::RwLock::new(Vec::new()),
            retiring: parking_lot::RwLock::new(Vec::new()),
            secrets: parking_lot::RwLock::new(Vec::new()),
            storage: Some(PrincipalKeyStorage {
                path,
                key_store,
                wrapping_key_id: parking_lot::Mutex::new(None),
                writing: std::sync::Mutex::new(()),
            }),
        };
        store.load()?;
        Ok(store)
    }

    /// How many principals hold a signing key here
    pub fn len(&self) -> usize {
        self.keys.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn load(&self) -> Result<()> {
        let Some(storage) = self.storage.as_ref() else {
            return Ok(());
        };
        let bytes = match std::fs::read(&storage.path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(ZyronError::Internal(format!(
                    "principal key file {} is not readable, {e}",
                    storage.path.display()
                )));
            }
        };
        let parsed = zyron_common::format::envelope::decode_as(
            &bytes,
            zyron_common::format::FormatKind::PrincipalKeyStorePersistence,
        )
        .map_err(|e| {
            ZyronError::Internal(format!(
                "principal key file {}, {e}",
                storage.path.display()
            ))
        })?;
        if parsed.header.version != crate::format::PRINCIPAL_KEY_STORE_FORMAT_VERSION {
            return Err(ZyronError::Internal(format!(
                "principal key file is at format version {}, this binary writes and reads {}",
                parsed.header.version,
                crate::format::PRINCIPAL_KEY_STORE_FORMAT_VERSION
            )));
        }
        let body = parsed.body;
        let truncated = || ZyronError::Internal("principal key file is truncated".to_string());
        let id_bytes: [u8; 4] = body
            .get(0..4)
            .ok_or_else(truncated)?
            .try_into()
            .unwrap_or([0; 4]);
        let wrapping_key_id = u32::from_le_bytes(id_bytes);
        let key_material = storage.key_store.get_key(wrapping_key_id)?;
        let plaintext = Zeroizing::new(crate::encryption::decrypt_value(
            body.get(4..).ok_or_else(truncated)?,
            &key_material,
            crate::encryption::EncryptionAlgorithm::Aes256Gcm,
            PRINCIPAL_KEY_FILE_AAD,
        )?);

        let mut reader = KeyFileReader {
            data: &plaintext,
            at: 0,
        };
        let current_count = reader.u32()? as usize;
        let mut keys = Vec::with_capacity(current_count);
        let mut secrets = Vec::with_capacity(current_count);
        for _ in 0..current_count {
            let principal = reader.text()?;
            let scheme_name = reader.text()?;
            let public_key = reader.bytes()?.to_vec();
            let issued_at_secs = reader.u64()?;
            let secret = Zeroizing::new(reader.bytes()?.to_vec());
            secrets.push((principal.clone(), secret));
            keys.push(PrincipalKey {
                principal,
                scheme_name,
                public_key,
                issued_at_secs,
            });
        }
        let retiring_count = reader.u32()? as usize;
        let mut retiring = Vec::with_capacity(retiring_count);
        for _ in 0..retiring_count {
            let principal = reader.text()?;
            let scheme_name = reader.text()?;
            let public_key = reader.bytes()?.to_vec();
            let issued_at_secs = reader.u64()?;
            let overlap_end_secs = reader.u64()?;
            retiring.push(RetiringKey {
                key: PrincipalKey {
                    principal,
                    scheme_name,
                    public_key,
                    issued_at_secs,
                },
                overlap_end_secs,
            });
        }

        *storage.wrapping_key_id.lock() = Some(wrapping_key_id);
        *self.keys.write() = keys;
        *self.secrets.write() = secrets;
        *self.retiring.write() = retiring;
        Ok(())
    }

    /// Writes the whole store down, sealed under the node's key store.
    ///
    /// Called after anything that changes a key. The file is rewritten whole
    /// rather than appended to, because the set is one principal per row and
    /// small, and a whole rewrite is what lets a rotated-out secret leave the
    /// file instead of staying in a history nothing reads
    fn persist(&self) -> Result<()> {
        let Some(storage) = self.storage.as_ref() else {
            return Ok(());
        };
        let _writing = storage
            .writing
            .lock()
            .map_err(|_| ZyronError::Internal("the principal key file lock is poisoned".into()))?;

        let mut plaintext = Zeroizing::new(Vec::new());
        {
            let keys = self.keys.read();
            let secrets = self.secrets.read();
            put_u32(&mut plaintext, keys.len() as u32);
            for key in keys.iter() {
                put_bytes(&mut plaintext, key.principal.as_bytes());
                put_bytes(&mut plaintext, key.scheme_name.as_bytes());
                put_bytes(&mut plaintext, &key.public_key);
                put_u64(&mut plaintext, key.issued_at_secs);
                let secret = secrets
                    .iter()
                    .find(|(name, _)| name == &key.principal)
                    .map(|(_, secret)| secret.as_slice())
                    .unwrap_or_default();
                put_bytes(&mut plaintext, secret);
            }
            let retiring = self.retiring.read();
            put_u32(&mut plaintext, retiring.len() as u32);
            for entry in retiring.iter() {
                put_bytes(&mut plaintext, entry.key.principal.as_bytes());
                put_bytes(&mut plaintext, entry.key.scheme_name.as_bytes());
                put_bytes(&mut plaintext, &entry.key.public_key);
                put_u64(&mut plaintext, entry.key.issued_at_secs);
                put_u64(&mut plaintext, entry.overlap_end_secs);
            }
        }

        // One wrapping key for the life of the file, allocated on the first
        // write. Allocating a new one each time would leave every previous
        // key in the store with nothing referring to it
        let key_id = {
            let mut held = storage.wrapping_key_id.lock();
            match *held {
                Some(id) => id,
                None => {
                    let id = storage
                        .key_store
                        .create_key(crate::encryption::EncryptionAlgorithm::Aes256Gcm)?;
                    *held = Some(id);
                    id
                }
            }
        };
        let key_material = storage.key_store.get_key(key_id)?;
        let ciphertext = crate::encryption::encrypt_value(
            &plaintext,
            &key_material,
            crate::encryption::EncryptionAlgorithm::Aes256Gcm,
            PRINCIPAL_KEY_FILE_AAD,
        )?;

        let mut body = Vec::with_capacity(4 + ciphertext.len());
        body.extend_from_slice(&key_id.to_le_bytes());
        body.extend_from_slice(&ciphertext);
        let bytes = zyron_common::format::envelope::encode(
            zyron_common::format::FormatKind::PrincipalKeyStorePersistence,
            crate::format::PRINCIPAL_KEY_STORE_FORMAT_VERSION,
            &body,
        );
        write_replacing(&storage.path, &bytes)
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
        // Written down before the caller is told it has a key. A key reported
        // as issued and not persisted would be gone at the next restart while
        // the artifacts it signed were still in circulation
        self.persist()?;
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

    /// Every key this node holds at a point in time, sorted by principal and
    /// with each principal's signing key before its retiring ones.
    ///
    /// Public halves only. A retiring key whose overlap has passed is left
    /// out, because it no longer verifies anything and reporting it would
    /// read as though it still did. Sweeping is what removes it for good, and
    /// that happens on the next rotation
    pub fn published(&self, now_secs: u64) -> Vec<PublishedKey> {
        let mut out: Vec<PublishedKey> = self
            .keys
            .read()
            .iter()
            .map(|key| PublishedKey {
                key: key.clone(),
                overlap_end_secs: None,
            })
            .collect();
        for retiring in self.retiring.read().iter() {
            if now_secs < retiring.overlap_end_secs {
                out.push(PublishedKey {
                    key: retiring.key.clone(),
                    overlap_end_secs: Some(retiring.overlap_end_secs),
                });
            }
        }
        out.sort_by(|a, b| {
            a.key
                .principal
                .cmp(&b.key.principal)
                .then(a.overlap_end_secs.cmp(&b.overlap_end_secs))
        });
        out
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

    /// Drops retiring keys whose overlap has ended.
    ///
    /// A sweep that could not be written down still took effect here, and the
    /// next issue or rotation writes the file again, so the swept keys are
    /// reported and the store carries on rather than holding them for a
    /// failure that has nothing to do with them
    pub fn sweep(&self, now_secs: u64) -> usize {
        let swept = {
            let mut retiring = self.retiring.write();
            let before = retiring.len();
            retiring.retain(|k| now_secs < k.overlap_end_secs);
            before - retiring.len()
        };
        if swept > 0
            && let Err(e) = self.persist()
        {
            tracing::warn!(
                "{swept} retired principal key(s) were dropped here and the key file was not \
                 rewritten, {e}. They are dropped again at the next restart"
            );
        }
        swept
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

/// The principal key store for this node.
///
/// Falls back to a store that lives only as long as the process, which is
/// what a test or a tool that never opened a data directory gets
pub fn principal_keys() -> Arc<PrincipalKeyStore> {
    Arc::clone(PRINCIPAL_KEYS.get_or_init(|| Arc::new(PrincipalKeyStore::new())))
}

/// Opens the node's key file and makes it the process's key store.
///
/// Called once during startup, before anything signs or verifies. Asking for
/// the store first would install the process-lifetime one and this would have
/// nothing to attach to, so that is reported rather than passed over: a node
/// that silently ran on keys it never wrote down would issue a second key for
/// a principal after every restart
pub fn open_principal_keys(
    path: std::path::PathBuf,
    key_store: Arc<dyn crate::encryption::KeyStore>,
) -> Result<Arc<PrincipalKeyStore>> {
    let opened = Arc::new(PrincipalKeyStore::open(path, key_store)?);
    PRINCIPAL_KEYS.set(Arc::clone(&opened)).map_err(|_| {
        ZyronError::Internal(
            "the principal key store was already in use before the key file was opened, so keys \
             issued in this process would not survive a restart"
                .to_string(),
        )
    })?;
    Ok(opened)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::scheme::{
        ArtifactSchemeBinding, SchemeCategory, SchemeId, SchemeStatus, SignatureSchemeRegistration,
        default_artifact_bindings,
    };

    /// A key issued before a restart is the key the node signs with after
    /// one, and the artifacts it already signed still verify.
    ///
    /// Held in memory only, every restart issued a principal a fresh key and
    /// every token the process signed before it became unverifiable
    #[test]
    fn a_key_issued_before_a_restart_is_the_key_after_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("principal_keys.zypk");
        let key_store: Arc<dyn crate::encryption::KeyStore> = Arc::new(
            crate::encryption::FileKeyStore::open([7u8; 32], dir.path().join("wrapping.zykeys"))
                .expect("key store opens"),
        );

        let first = PrincipalKeyStore::open(path.clone(), Arc::clone(&key_store)).expect("opens");
        let issued = first.issue("sp1", "Ed25519", 100).expect("issues");
        first
            .rotate(&SchemeRegistry::load(), "sp1", Some("Ed25519"), 500, 200)
            .expect("rotates");
        let after_rotation = first.current("sp1").expect("has a key");
        drop(first);

        // A second process opening the same file, which is what a restart is
        let reopened = PrincipalKeyStore::open(path, key_store).expect("reopens");
        let current = reopened
            .current("sp1")
            .expect("has a key after the restart");
        assert_eq!(
            current.public_key, after_rotation.public_key,
            "the restart did not come back on the key the rotation left"
        );
        assert!(
            reopened.sign("sp1", b"payload").is_ok(),
            "the secret half did not survive the restart"
        );

        // The outgoing key is still inside its overlap, so an artifact signed
        // before the rotation is still believed after the restart
        let verifying = reopened.verifying("sp1", 300);
        assert_eq!(verifying.len(), 2, "{verifying:?}");
        assert!(
            verifying.iter().any(|k| k.public_key == issued.public_key),
            "the key that signed before the rotation was lost across the restart"
        );

        // Past the overlap the retired key goes, and stays gone
        assert_eq!(reopened.sweep(1_000), 1);
        assert_eq!(reopened.verifying("sp1", 1_000).len(), 1);
    }

    /// A store with no file behind it works exactly as it did, which is what
    /// a tool that never opened a data directory gets
    #[test]
    fn a_store_without_storage_issues_and_signs() {
        let store = PrincipalKeyStore::new();
        store.issue("sp1", "Ed25519", 0).expect("issues");
        assert!(store.sign("sp1", b"payload").is_ok());
        assert_eq!(store.len(), 1);
    }

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

    /// What an operator reads out of `zyron_sys.security.principal_keys`.
    ///
    /// The signing key comes first with no end, a key inside its overlap
    /// comes after it with the moment it stops, and one whose overlap has
    /// passed is gone. Reporting a lapsed key would read as though it still
    /// verified something
    #[test]
    fn test_published_reports_the_signing_key_then_the_ones_still_accepted() {
        let registry = registry();
        let store = PrincipalKeyStore::new();
        store.issue("sp_b", "Ed25519", 10).expect("issues");
        store.issue("sp_a", "Ed25519", 20).expect("issues");
        let outgoing = store.current("sp_a").expect("has a key");
        store
            .rotate(&registry, "sp_a", Some("Ed25519"), 1_000, 100)
            .expect("rotates");

        let during = store.published(500);
        assert_eq!(during.len(), 3, "two principals, one mid rotation");
        assert_eq!(during[0].key.principal, "sp_a", "sorted by principal");
        assert_eq!(
            during[0].overlap_end_secs, None,
            "the signing key comes before the retiring one"
        );
        assert_eq!(during[1].key.principal, "sp_a");
        assert_eq!(during[1].overlap_end_secs, Some(1_100));
        assert_eq!(during[1].key.public_key, outgoing.public_key);
        assert_eq!(during[2].key.principal, "sp_b");

        let after = store.published(2_000);
        assert_eq!(after.len(), 2, "the lapsed key is not reported");
        assert!(after.iter().all(|k| k.overlap_end_secs.is_none()));
    }

    /// The fingerprint is what tells two members apart.
    ///
    /// Each node draws its own key for a principal, so the same principal
    /// name on two members is two different keys, and the fingerprint is how
    /// an operator sees that rather than assuming they match
    #[test]
    fn test_a_fingerprint_is_taken_over_the_public_half_and_differs_per_key() {
        let one = PrincipalKeyStore::new();
        let other = PrincipalKeyStore::new();
        let here = one.issue("sp1", "Ed25519", 0).expect("issues");
        let there = other.issue("sp1", "Ed25519", 0).expect("issues");

        assert_eq!(here.fingerprint().len(), 64, "SHA-256 as hex");
        assert_eq!(
            here.fingerprint(),
            one.current("sp1").expect("has a key").fingerprint(),
            "the same key fingerprints the same way twice"
        );
        assert_ne!(
            here.fingerprint(),
            there.fingerprint(),
            "two nodes drew the same fingerprint for keys that are not the same"
        );
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
