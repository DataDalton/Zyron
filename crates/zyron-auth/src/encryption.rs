//! Column-level AES-GCM encryption with hardware acceleration.
//!
//! Provides authenticated encryption using AES-128-GCM or AES-256-GCM with
//! hardware AES-NI/VAES (x86_64) and AESE/AESD (aarch64) instructions.
//! Each encrypted value gets a unique 12-byte nonce. The output format is:
//! nonce (12 bytes) || ciphertext || auth_tag (16 bytes).
//!
//! The KeyStore trait abstracts key management. LocalKeyStore wraps data
//! encryption keys with a master key derived from Balloon KDF.

use crate::rcu::RcuMap;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use zyron_common::{Result, ZyronError};

/// Supported encryption algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum EncryptionAlgorithm {
    Aes128Gcm = 0,
    Aes256Gcm = 1,
}

impl EncryptionAlgorithm {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(EncryptionAlgorithm::Aes128Gcm),
            1 => Ok(EncryptionAlgorithm::Aes256Gcm),
            _ => Err(ZyronError::DecodingFailed(format!(
                "Unknown EncryptionAlgorithm value: {}",
                v
            ))),
        }
    }

    /// Returns the key length in bytes for this algorithm.
    pub fn key_len(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 16,
            EncryptionAlgorithm::Aes256Gcm => 32,
        }
    }

    /// Returns the number of AES rounds for this algorithm.
    fn rounds(&self) -> usize {
        match self {
            EncryptionAlgorithm::Aes128Gcm => 10,
            EncryptionAlgorithm::Aes256Gcm => 14,
        }
    }
}

/// Column encryption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnEncryption {
    pub table_id: u32,
    pub column_id: u16,
    pub algorithm: EncryptionAlgorithm,
    pub key_id: u32,
}

impl ColumnEncryption {
    /// Serializes to bytes.
    /// Layout: table_id(4) + column_id(2) + algorithm(1) + key_id(4)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(11);
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.extend_from_slice(&self.column_id.to_le_bytes());
        buf.push(self.algorithm as u8);
        buf.extend_from_slice(&self.key_id.to_le_bytes());
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 11 {
            return Err(ZyronError::DecodingFailed(
                "ColumnEncryption data too short".to_string(),
            ));
        }
        let table_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let column_id = u16::from_le_bytes([data[4], data[5]]);
        let algorithm = EncryptionAlgorithm::from_u8(data[6])?;
        let key_id = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);
        Ok(Self {
            table_id,
            column_id,
            algorithm,
            key_id,
        })
    }
}

/// Stores column encryption configurations indexed by (table_id, column_id).
pub struct EncryptionStore {
    configs: RcuMap<(u32, u16), ColumnEncryption>,
}

impl EncryptionStore {
    pub fn new() -> Self {
        Self {
            configs: RcuMap::empty_map(),
        }
    }

    /// Sets encryption config for a column.
    pub fn set_config(&self, config: ColumnEncryption) {
        let key = (config.table_id, config.column_id);
        self.configs.insert(key, config);
    }

    /// Gets encryption config for a column.
    pub fn get_config(&self, table_id: u32, column_id: u16) -> Option<ColumnEncryption> {
        self.configs.get(&(table_id, column_id))
    }

    /// Removes encryption config for a column.
    pub fn remove_config(&self, table_id: u32, column_id: u16) -> bool {
        self.configs.remove(&(table_id, column_id))
    }

    /// Bulk-loads configs from storage.
    pub fn load(&self, configs: Vec<ColumnEncryption>) {
        for config in configs {
            self.set_config(config);
        }
    }
}

/// Trait for key management. Implementations handle key storage and retrieval.
pub trait KeyStore: Send + Sync {
    fn get_key(&self, key_id: u32) -> Result<Vec<u8>>;
    fn create_key(&self, algorithm: EncryptionAlgorithm) -> Result<u32>;
    fn delete_key(&self, key_id: u32) -> Result<()>;
    fn rotate_key(&self, key_id: u32) -> Result<u32>;
}

/// An encryption key wrapped (encrypted) with the master key.
#[derive(Debug, Clone)]
struct EncryptedKey {
    #[allow(dead_code)]
    key_id: u32,
    algorithm: EncryptionAlgorithm,
    /// The data encryption key, encrypted with the master key via AES-GCM.
    encrypted_material: Vec<u8>,
}

/// Local key store that keeps data encryption keys encrypted in memory
/// with a master key derived from a passphrase via Balloon KDF.
pub struct LocalKeyStore {
    master_key: [u8; 32],
    keys: RcuMap<u32, EncryptedKey>,
    next_id: AtomicU32,
}

impl LocalKeyStore {
    /// Creates a new LocalKeyStore with the given master key (32 bytes).
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            master_key,
            keys: RcuMap::empty_map(),
            next_id: AtomicU32::new(1),
        }
    }

    /// Wraps (encrypts) a data encryption key with the master key.
    fn wrap_key(&self, key_material: &[u8]) -> Result<Vec<u8>> {
        encrypt_value(
            key_material,
            &self.master_key,
            EncryptionAlgorithm::Aes256Gcm,
            &[],
        )
    }

    /// Unwraps (decrypts) a data encryption key using the master key.
    fn unwrap_key(&self, wrapped: &[u8]) -> Result<Vec<u8>> {
        decrypt_value(
            wrapped,
            &self.master_key,
            EncryptionAlgorithm::Aes256Gcm,
            &[],
        )
    }
}

impl KeyStore for LocalKeyStore {
    fn get_key(&self, key_id: u32) -> Result<Vec<u8>> {
        let encrypted = self
            .keys
            .get(&key_id)
            .ok_or_else(|| ZyronError::EncryptionKeyNotFound(format!("key id {}", key_id)))?;
        self.unwrap_key(&encrypted.encrypted_material)
    }

    fn create_key(&self, algorithm: EncryptionAlgorithm) -> Result<u32> {
        let key_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        use rand::Rng;
        let key_len = algorithm.key_len();
        let mut key_material = vec![0u8; key_len];
        rand::rng().fill_bytes(&mut key_material);
        let encrypted_material = self.wrap_key(&key_material)?;
        let entry = EncryptedKey {
            key_id,
            algorithm,
            encrypted_material,
        };
        self.keys.insert(key_id, entry);
        Ok(key_id)
    }

    fn delete_key(&self, key_id: u32) -> Result<()> {
        if !self.keys.remove(&key_id) {
            return Err(ZyronError::EncryptionKeyNotFound(format!(
                "key id {}",
                key_id
            )));
        }
        Ok(())
    }

    fn rotate_key(&self, key_id: u32) -> Result<u32> {
        let old = self
            .keys
            .get(&key_id)
            .ok_or_else(|| ZyronError::EncryptionKeyNotFound(format!("key id {}", key_id)))?;
        let new_id = self.create_key(old.algorithm)?;
        // Remove the old key. Data encrypted with it must be re-encrypted
        // with the new key before this point.
        self.keys.remove(&key_id);
        Ok(new_id)
    }
}

// ---------------------------------------------------------------------------
// AES-GCM implementation
// ---------------------------------------------------------------------------

const GCM_NONCE_LEN: usize = 12;
const GCM_TAG_LEN: usize = 16;
const AES_BLOCK_LEN: usize = 16;

// ---------------------------------------------------------------------------
// AES-based fast hash for data masking
// ---------------------------------------------------------------------------

/// Fixed key for AES-based masking hash. Not secret (the hash is public),
/// but provides the AES key schedule for Davies-Meyer compression.
const MASK_HASH_KEY: [u8; 32] = [
    0x6a, 0x09, 0xe6, 0x67, 0xbb, 0x67, 0xae, 0x85, 0x3c, 0x6e, 0xf3, 0x72, 0xa5, 0x4f, 0xf5, 0x3a,
    0x51, 0x0e, 0x52, 0x7f, 0x9b, 0x05, 0x68, 0x8c, 0x1f, 0x83, 0xd9, 0xab, 0x5b, 0xe0, 0xcd, 0x19,
];

/// Pre-expanded AES-256 round keys broadcast to 256-bit YMM layout.
/// 32-byte aligned so VAES can use _mm256_loadu_si256 (aligned, faster).
/// Computed once at first use, then every subsequent hash call is a
/// direct pointer dereference with zero conversion overhead.
#[cfg(target_arch = "x86_64")]
#[repr(align(32))]
struct VaesRoundKeys {
    keys: [[u8; 32]; 15],
}

// SAFETY: VaesRoundKeys is plain aligned byte data, immutable after init.
// __m256i lacks Send/Sync due to a Rust type system limitation for SIMD
// types, not because the data is unsafe to share. This is the same pattern
// used by RustCrypto crates (aes, polyval, ghash).
#[cfg(target_arch = "x86_64")]
unsafe impl Send for VaesRoundKeys {}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for VaesRoundKeys {}

#[cfg(target_arch = "x86_64")]
static MASK_HASH_VAES_RK: OnceLock<VaesRoundKeys> = OnceLock::new();

/// Pre-expanded 128-bit round keys for non-x86 platforms.
#[cfg(not(target_arch = "x86_64"))]
static MASK_HASH_RK: OnceLock<[u8; 240]> = OnceLock::new();

/// AES-based 256-bit hash for data masking. Uses Davies-Meyer compression
/// with AES-256 in two passes to produce 256 bits of one-way output.
///
/// x86_64: VAES processes both blocks in parallel via 256-bit YMM registers.
/// Round keys are pre-broadcast to 256-bit and 32-byte aligned, loaded
/// directly as __m256i with zero per-call conversion.
pub fn aes_hash_256(data: &[u8]) -> [u8; 32] {
    // Pad input to 16 bytes. XOR-fold longer inputs.
    // Length is encoded as 2 bytes (LE) in positions 14-15 to avoid collisions
    // for inputs differing only in length.
    let mut padded = [0u8; 16];
    let len_bytes = (data.len() as u16).to_le_bytes();
    if data.len() <= 14 {
        padded[..data.len()].copy_from_slice(data);
        padded[14] = len_bytes[0];
        padded[15] = len_bytes[1];
    } else {
        for chunk in data.chunks(14) {
            for (i, &b) in chunk.iter().enumerate() {
                padded[i] ^= b;
            }
        }
        padded[14] ^= len_bytes[0];
        padded[15] ^= len_bytes[1];
    }

    #[cfg(target_arch = "x86_64")]
    {
        let rk = MASK_HASH_VAES_RK.get_or_init(|| {
            let rk_bytes = aes_key_expansion(&MASK_HASH_KEY, EncryptionAlgorithm::Aes256Gcm)
                .expect("Fixed key expansion");
            let mut vaes_rk = VaesRoundKeys {
                keys: [[0u8; 32]; 15],
            };
            // Copy round keys as raw bytes (no AVX2 instructions needed at init)
            for i in 0..15 {
                vaes_rk.keys[i][..16].copy_from_slice(&rk_bytes[i * 16..i * 16 + 16]);
                // Duplicate the 128-bit key into the upper half for VAES broadcast
                vaes_rk.keys[i][16..32].copy_from_slice(&rk_bytes[i * 16..i * 16 + 16]);
            }
            vaes_rk
        });
        return unsafe { aes_ni::vaes_hash_256_fast(&rk.keys, &padded) };
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let rk_bytes = MASK_HASH_RK.get_or_init(|| {
            aes_key_expansion(&MASK_HASH_KEY, EncryptionAlgorithm::Aes256Gcm)
                .expect("Fixed key expansion")
                .try_into()
                .expect("AES-256 produces 240 bytes")
        });
        let block1 = aes_encrypt_block(&padded, rk_bytes, 14);
        let mut h1 = [0u8; 16];
        for i in 0..16 {
            h1[i] = block1[i] ^ padded[i];
        }

        let mut input2 = h1;
        for b in input2.iter_mut() {
            *b ^= 0xFF;
        }
        let block2 = aes_encrypt_block(&input2, rk_bytes, 14);
        let mut h2 = [0u8; 16];
        for i in 0..16 {
            h2[i] = block2[i] ^ input2[i];
        }

        let mut result = [0u8; 32];
        result[..16].copy_from_slice(&h1);
        result[16..].copy_from_slice(&h2);
        result
    }
}

/// Encrypts plaintext using AES-GCM. Returns nonce || ciphertext || tag.
/// The `aad` (Additional Authenticated Data) is authenticated but not encrypted.
/// For column encryption, pass table_id and column_id as AAD to prevent
/// ciphertext from one column being swapped to another undetected.
/// Pass &[] if no AAD is needed.
/// Uses hardware AES-NI + PCLMULQDQ on x86_64 when available, falling back
/// to the software implementation on other architectures.
pub fn encrypt_value(
    plaintext: &[u8],
    key: &[u8],
    algorithm: EncryptionAlgorithm,
    aad: &[u8],
) -> Result<Vec<u8>> {
    let expected_key_len = algorithm.key_len();
    if key.len() != expected_key_len {
        return Err(ZyronError::EncryptionFailed(format!(
            "Key length {} does not match algorithm requirement {}",
            key.len(),
            expected_key_len
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if hw_aes_available() {
        return unsafe { aes_ni::hw_encrypt_direct(plaintext, key, algorithm, aad) };
    }

    let ctx = AesGcmContext::new(key, algorithm)?;
    ctx.encrypt(plaintext, aad)
}

/// Decrypts AES-GCM ciphertext. Input format: nonce || ciphertext || tag.
/// The `aad` must match what was passed during encryption, or the
/// authentication tag verification will fail. Pass &[] if no AAD was used.
/// Uses hardware AES-NI + PCLMULQDQ on x86_64 when available, falling back
/// to the software implementation on other architectures.
pub fn decrypt_value(
    data: &[u8],
    key: &[u8],
    algorithm: EncryptionAlgorithm,
    aad: &[u8],
) -> Result<Vec<u8>> {
    let expected_key_len = algorithm.key_len();
    if key.len() != expected_key_len {
        return Err(ZyronError::EncryptionFailed(format!(
            "Key length {} does not match algorithm requirement {}",
            key.len(),
            expected_key_len
        )));
    }

    #[cfg(target_arch = "x86_64")]
    if hw_aes_available() {
        let min_len = GCM_NONCE_LEN + GCM_TAG_LEN;
        if data.len() < min_len {
            return Err(ZyronError::DecryptionFailed(
                "Ciphertext too short for AES-GCM".to_string(),
            ));
        }
        let nonce = &data[..GCM_NONCE_LEN];
        let ciphertext = &data[GCM_NONCE_LEN..data.len() - GCM_TAG_LEN];
        let tag = &data[data.len() - GCM_TAG_LEN..];
        return unsafe { aes_ni::hw_decrypt_direct(ciphertext, aad, nonce, tag, key, algorithm) };
    }

    let ctx = AesGcmContext::new(key, algorithm)?;
    ctx.decrypt(data, aad)
}

/// Increments the 32-bit big-endian counter in the last 4 bytes of the block.
fn increment_counter(block: &mut [u8; AES_BLOCK_LEN]) {
    let ctr = u32::from_be_bytes([block[12], block[13], block[14], block[15]]);
    let new_ctr = ctr.wrapping_add(1);
    block[12..16].copy_from_slice(&new_ctr.to_be_bytes());
}

/// Constant-time comparison of two byte slices.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// AES key expansion (software implementation)
// ---------------------------------------------------------------------------

/// AES round constants.
const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// AES S-box.
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn sub_word(word: u32) -> u32 {
    let b0 = SBOX[(word >> 24) as usize];
    let b1 = SBOX[((word >> 16) & 0xff) as usize];
    let b2 = SBOX[((word >> 8) & 0xff) as usize];
    let b3 = SBOX[(word & 0xff) as usize];
    (b0 as u32) << 24 | (b1 as u32) << 16 | (b2 as u32) << 8 | b3 as u32
}

fn rot_word(word: u32) -> u32 {
    word.rotate_left(8)
}

/// Expands an AES key into round keys. Returns a flat Vec of round key bytes.
/// AES-128: 11 round keys (176 bytes). AES-256: 15 round keys (240 bytes).
fn aes_key_expansion(key: &[u8], algorithm: EncryptionAlgorithm) -> Result<Vec<u8>> {
    let nk = key.len() / 4; // Number of 32-bit words in key
    let nr = algorithm.rounds();
    let total_words = 4 * (nr + 1);

    let mut w = vec![0u32; total_words];

    // Copy key into first nk words
    for i in 0..nk {
        w[i] = u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
    }

    for i in nk..total_words {
        let mut temp = w[i - 1];
        if i % nk == 0 {
            temp = sub_word(rot_word(temp)) ^ ((RCON[i / nk - 1] as u32) << 24);
        } else if nk > 6 && i % nk == 4 {
            temp = sub_word(temp);
        }
        w[i] = w[i - nk] ^ temp;
    }

    // Convert to bytes
    let mut round_keys = vec![0u8; total_words * 4];
    for (i, word) in w.iter().enumerate() {
        round_keys[4 * i..4 * i + 4].copy_from_slice(&word.to_be_bytes());
    }
    Ok(round_keys)
}

/// Single AES block encryption using the expanded round keys.
fn aes_encrypt_block(input: &[u8; 16], round_keys: &[u8], rounds: usize) -> [u8; 16] {
    let mut state = [0u8; 16];
    state.copy_from_slice(input);

    // Initial round key addition
    xor_block(&mut state, &round_keys[0..16]);

    // Main rounds
    for round in 1..rounds {
        sub_bytes(&mut state);
        shift_rows(&mut state);
        mix_columns(&mut state);
        xor_block(&mut state, &round_keys[round * 16..(round + 1) * 16]);
    }

    // Final round (no MixColumns)
    sub_bytes(&mut state);
    shift_rows(&mut state);
    xor_block(&mut state, &round_keys[rounds * 16..(rounds + 1) * 16]);

    state
}

fn xor_block(state: &mut [u8; 16], key: &[u8]) {
    for i in 0..16 {
        state[i] ^= key[i];
    }
}

fn sub_bytes(state: &mut [u8; 16]) {
    for byte in state.iter_mut() {
        *byte = SBOX[*byte as usize];
    }
}

fn shift_rows(state: &mut [u8; 16]) {
    // Row 1: shift left by 1
    let tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    // Row 2: shift left by 2
    let tmp0 = state[2];
    let tmp1 = state[6];
    state[2] = state[10];
    state[6] = state[14];
    state[10] = tmp0;
    state[14] = tmp1;

    // Row 3: shift left by 3
    let tmp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = tmp;
}

fn mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let s0 = state[c * 4];
        let s1 = state[c * 4 + 1];
        let s2 = state[c * 4 + 2];
        let s3 = state[c * 4 + 3];

        state[c * 4] = gf_mul(2, s0) ^ gf_mul(3, s1) ^ s2 ^ s3;
        state[c * 4 + 1] = s0 ^ gf_mul(2, s1) ^ gf_mul(3, s2) ^ s3;
        state[c * 4 + 2] = s0 ^ s1 ^ gf_mul(2, s2) ^ gf_mul(3, s3);
        state[c * 4 + 3] = gf_mul(3, s0) ^ s1 ^ s2 ^ gf_mul(2, s3);
    }
}

/// GF(2^8) multiplication.
fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            result ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b; // AES irreducible polynomial
        }
        b >>= 1;
    }
    result
}

// ---------------------------------------------------------------------------
// GHASH computation for GCM authentication
// ---------------------------------------------------------------------------

/// Precomputed table for 4-bit GHASH multiplication.
/// table[i] = H * i for i in 0..16, where multiplication is in GF(2^128).
/// Using a 4-bit table gives ~8x speedup over bit-by-bit (32 table lookups
/// instead of 128 conditional XORs per multiplication).
struct GHashTable {
    table: [[u8; 16]; 16],
}

impl GHashTable {
    /// Builds the 4-bit multiplication table from the hash key H.
    fn new(h: &[u8; 16]) -> Self {
        let mut table = [[0u8; 16]; 16];
        // table[0] = 0 (already zero)
        // table[1] = H
        table[1] = *h;
        // table[i] = table[i-1] * x (doubling in GF(2^128))
        for i in 2..16 {
            if i % 2 == 0 {
                // Even: double the half entry
                table[i] = gf128_double(&table[i / 2]);
            } else {
                // Odd: XOR the even entry below with H
                let mut val = table[i - 1];
                for j in 0..16 {
                    val[j] ^= h[j];
                }
                table[i] = val;
            }
        }
        Self { table }
    }

    /// Multiplies x by H using the precomputed 4-bit table.
    /// Processes 4 bits at a time (32 iterations instead of 128).
    fn multiply(&self, x: &[u8; 16]) -> [u8; 16] {
        let mut z = [0u8; 16];
        for i in 0..32 {
            // Shift z right by 4 bits in GF(2^128)
            if i > 0 {
                let reduction = z[15] & 0x0f;
                for j in (1..16).rev() {
                    z[j] = (z[j] >> 4) | (z[j - 1] << 4);
                }
                z[0] >>= 4;
                // Apply reduction polynomial for each of the 4 shifted bits
                z[0] ^= GHASH_REDUCTION[reduction as usize];
            }
            // Extract 4-bit nibble from x (high nibble first)
            let byte_idx = i / 2;
            let nibble = if i % 2 == 0 {
                (x[byte_idx] >> 4) & 0x0f
            } else {
                x[byte_idx] & 0x0f
            };
            // XOR in the table entry
            let entry = &self.table[nibble as usize];
            for j in 0..16 {
                z[j] ^= entry[j];
            }
        }
        z
    }
}

/// Reduction constants for 4-bit GHASH shift. When shifting right by 4,
/// the low 4 bits determine which reduction polynomial terms to XOR in.
const GHASH_REDUCTION: [u8; 16] = [
    0x00, 0x1c, 0x38, 0x24, 0x70, 0x6c, 0x48, 0x54, 0xe1, 0xfd, 0xd9, 0xc5, 0x91, 0x8d, 0xa9, 0xb5,
];

/// Doubles a value in GF(2^128) (shift right by 1, apply reduction).
fn gf128_double(a: &[u8; 16]) -> [u8; 16] {
    let mut result = [0u8; 16];
    let carry = a[15] & 1;
    for j in (1..16).rev() {
        result[j] = (a[j] >> 1) | (a[j - 1] << 7);
    }
    result[0] = a[0] >> 1;
    if carry == 1 {
        result[0] ^= 0xe1;
    }
    result
}

/// Computes GHASH over AAD and ciphertext using a 4-bit precomputed table,
/// then XORs with encrypted J0 to produce the authentication tag.
fn ghash_compute(h: &[u8; 16], aad: &[u8], ciphertext: &[u8], encrypted_j0: &[u8; 16]) -> [u8; 16] {
    let table = GHashTable::new(h);
    let mut y = [0u8; 16];

    // Process AAD blocks
    for chunk in aad.chunks(AES_BLOCK_LEN) {
        let mut block = [0u8; 16];
        block[..chunk.len()].copy_from_slice(chunk);
        for i in 0..16 {
            y[i] ^= block[i];
        }
        y = table.multiply(&y);
    }

    // Process ciphertext blocks
    for chunk in ciphertext.chunks(AES_BLOCK_LEN) {
        let mut block = [0u8; 16];
        block[..chunk.len()].copy_from_slice(chunk);
        for i in 0..16 {
            y[i] ^= block[i];
        }
        y = table.multiply(&y);
    }

    // Length block: [len(A) in bits as u64 BE || len(C) in bits as u64 BE]
    let mut len_block = [0u8; 16];
    let aad_bits = (aad.len() as u64) * 8;
    let ct_bits = (ciphertext.len() as u64) * 8;
    len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
    len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
    for i in 0..16 {
        y[i] ^= len_block[i];
    }
    y = table.multiply(&y);

    // XOR with encrypted J0 to get the tag
    for i in 0..16 {
        y[i] ^= encrypted_j0[i];
    }
    y
}

// ---------------------------------------------------------------------------
// Hardware acceleration detection
// ---------------------------------------------------------------------------

/// Returns true if hardware AES-NI and PCLMULQDQ are available (x86_64)
/// or if hardware AES instructions are available (aarch64).
/// Result is cached in a static OnceLock to avoid repeated CPUID checks.
fn hw_aes_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("aes") && is_x86_feature_detected!("pclmulqdq")
        }
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("aes")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    })
}

// ---------------------------------------------------------------------------
// Pre-computed AES-GCM context for repeated encryption with the same key
// ---------------------------------------------------------------------------

/// Cached AES-GCM context that pre-computes expanded round keys and the
/// GHASH hash subkey H. Reusing this context across multiple encrypt/decrypt
/// calls with the same key avoids redundant key expansion and GHASH table
/// construction.
pub struct AesGcmContext {
    round_keys: Vec<u8>,
    rounds: usize,
    h: [u8; 16],
    #[allow(dead_code)]
    algorithm: EncryptionAlgorithm,
    #[cfg(target_arch = "x86_64")]
    hw_round_keys_128: Option<[u128; 15]>,
    #[cfg(target_arch = "x86_64")]
    hw_h_be: Option<u128>,
    /// Precomputed H^2, H^3, H^4 in GHASH big-endian byte order for 4-way
    /// parallel GHASH reduction. Each multiplication is independent, hiding
    /// PCLMULQDQ latency behind parallel execution.
    #[cfg(target_arch = "x86_64")]
    hw_h2_be: Option<u128>,
    #[cfg(target_arch = "x86_64")]
    hw_h3_be: Option<u128>,
    #[cfg(target_arch = "x86_64")]
    hw_h4_be: Option<u128>,
}

impl AesGcmContext {
    /// Creates a new AES-GCM context with pre-expanded round keys and H.
    /// On x86_64 with AES-NI, uses hardware key expansion directly.
    pub fn new(key: &[u8], algorithm: EncryptionAlgorithm) -> Result<Self> {
        let expected_key_len = algorithm.key_len();
        if key.len() != expected_key_len {
            return Err(ZyronError::EncryptionFailed(format!(
                "Key length {} does not match algorithm requirement {}",
                key.len(),
                expected_key_len
            )));
        }

        let rounds = algorithm.rounds();

        #[cfg(target_arch = "x86_64")]
        if hw_aes_available() {
            let ctx = unsafe { aes_ni::hw_key_expand(key, algorithm) }?;
            return Ok(Self {
                round_keys: Vec::new(),
                rounds,
                h: ctx.h,
                algorithm,
                hw_round_keys_128: Some(ctx.rk),
                hw_h_be: Some(ctx.h_be),
                hw_h2_be: Some(ctx.h2_be),
                hw_h3_be: Some(ctx.h3_be),
                hw_h4_be: Some(ctx.h4_be),
            });
        }

        let round_keys = aes_key_expansion(key, algorithm)?;
        let zero_block = [0u8; AES_BLOCK_LEN];
        let h = aes_encrypt_block(&zero_block, &round_keys, rounds);

        Ok(Self {
            round_keys,
            rounds,
            h,
            algorithm,
            #[cfg(target_arch = "x86_64")]
            hw_round_keys_128: None,
            #[cfg(target_arch = "x86_64")]
            hw_h_be: None,
            #[cfg(target_arch = "x86_64")]
            hw_h2_be: None,
            #[cfg(target_arch = "x86_64")]
            hw_h3_be: None,
            #[cfg(target_arch = "x86_64")]
            hw_h4_be: None,
        })
    }

    /// Encrypts plaintext using this pre-computed context. Returns nonce || ciphertext || tag.
    pub fn encrypt(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        use rand::Rng;
        let mut nonce = [0u8; GCM_NONCE_LEN];
        rand::rng().fill_bytes(&mut nonce);

        #[cfg(target_arch = "x86_64")]
        if let (Some(hw_rk), Some(hw_h_be), Some(h2), Some(h3), Some(h4)) = (
            &self.hw_round_keys_128,
            &self.hw_h_be,
            &self.hw_h2_be,
            &self.hw_h3_be,
            &self.hw_h4_be,
        ) {
            return unsafe {
                aes_ni::hw_aes_gcm_encrypt(
                    plaintext,
                    aad,
                    &nonce,
                    hw_rk,
                    *hw_h_be,
                    *h2,
                    *h3,
                    *h4,
                    self.rounds,
                )
            };
        }

        sw_aes_gcm_encrypt(
            plaintext,
            aad,
            &nonce,
            &self.round_keys,
            self.rounds,
            &self.h,
        )
    }

    /// Decrypts AES-GCM ciphertext using this pre-computed context.
    pub fn decrypt(&self, data: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        let min_len = GCM_NONCE_LEN + GCM_TAG_LEN;
        if data.len() < min_len {
            return Err(ZyronError::DecryptionFailed(
                "Ciphertext too short for AES-GCM".to_string(),
            ));
        }

        let nonce = &data[..GCM_NONCE_LEN];
        let ciphertext = &data[GCM_NONCE_LEN..data.len() - GCM_TAG_LEN];
        let tag = &data[data.len() - GCM_TAG_LEN..];

        #[cfg(target_arch = "x86_64")]
        if let (Some(hw_rk), Some(hw_h_be), Some(h2), Some(h3), Some(h4)) = (
            &self.hw_round_keys_128,
            &self.hw_h_be,
            &self.hw_h2_be,
            &self.hw_h3_be,
            &self.hw_h4_be,
        ) {
            return unsafe {
                aes_ni::hw_aes_gcm_decrypt(
                    ciphertext,
                    aad,
                    nonce,
                    tag,
                    hw_rk,
                    *hw_h_be,
                    *h2,
                    *h3,
                    *h4,
                    self.rounds,
                )
            };
        }

        sw_aes_gcm_decrypt(
            ciphertext,
            aad,
            nonce,
            tag,
            &self.round_keys,
            self.rounds,
            &self.h,
        )
    }
}

/// Software AES-GCM encrypt (extracted from encrypt_value for reuse).
fn sw_aes_gcm_encrypt(
    plaintext: &[u8],
    aad: &[u8],
    nonce: &[u8; GCM_NONCE_LEN],
    round_keys: &[u8],
    rounds: usize,
    h: &[u8; 16],
) -> Result<Vec<u8>> {
    let mut j0 = [0u8; AES_BLOCK_LEN];
    j0[..GCM_NONCE_LEN].copy_from_slice(nonce);
    j0[15] = 1;

    let encrypted_j0 = aes_encrypt_block(&j0, round_keys, rounds);

    // Write directly into output buffer: nonce || ciphertext || tag
    let ct_len = plaintext.len();
    let mut output = vec![0u8; GCM_NONCE_LEN + ct_len + GCM_TAG_LEN];
    output[..GCM_NONCE_LEN].copy_from_slice(nonce);

    let ct_slice = &mut output[GCM_NONCE_LEN..GCM_NONCE_LEN + ct_len];
    let mut counter_block = j0;

    for (chunk_idx, chunk) in plaintext.chunks(AES_BLOCK_LEN).enumerate() {
        increment_counter(&mut counter_block);
        let keystream = aes_encrypt_block(&counter_block, round_keys, rounds);
        let offset = chunk_idx * AES_BLOCK_LEN;
        for (i, &byte) in chunk.iter().enumerate() {
            ct_slice[offset + i] = byte ^ keystream[i];
        }
    }

    let tag = ghash_compute(h, aad, ct_slice, &encrypted_j0);
    output[GCM_NONCE_LEN + ct_len..].copy_from_slice(&tag);

    Ok(output)
}

/// Software AES-GCM decrypt (extracted from decrypt_value for reuse).
fn sw_aes_gcm_decrypt(
    ciphertext: &[u8],
    aad: &[u8],
    nonce: &[u8],
    tag: &[u8],
    round_keys: &[u8],
    rounds: usize,
    h: &[u8; 16],
) -> Result<Vec<u8>> {
    let mut j0 = [0u8; AES_BLOCK_LEN];
    j0[..GCM_NONCE_LEN].copy_from_slice(nonce);
    j0[15] = 1;

    let encrypted_j0 = aes_encrypt_block(&j0, round_keys, rounds);

    let computed_tag = ghash_compute(h, aad, ciphertext, &encrypted_j0);
    if !constant_time_eq(tag, &computed_tag) {
        return Err(ZyronError::DecryptionFailed(
            "Authentication tag mismatch, data may be tampered".to_string(),
        ));
    }

    let mut counter_block = j0;
    let mut plaintext = vec![0u8; ciphertext.len()];

    for (chunk_idx, chunk) in ciphertext.chunks(AES_BLOCK_LEN).enumerate() {
        increment_counter(&mut counter_block);
        let keystream = aes_encrypt_block(&counter_block, round_keys, rounds);
        let offset = chunk_idx * AES_BLOCK_LEN;
        for (i, &byte) in chunk.iter().enumerate() {
            plaintext[offset + i] = byte ^ keystream[i];
        }
    }

    Ok(plaintext)
}

// ---------------------------------------------------------------------------
// x86_64 AES-NI + PCLMULQDQ hardware acceleration
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod aes_ni {
    use super::*;
    use core::arch::x86_64::*;

    /// Byte-reverse shuffle mask for converting between GHASH big-endian
    /// and native little-endian byte order.
    const BSWAP_MASK: [u8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

    /// Result of hardware key expansion with precomputed GHASH powers.
    pub(super) struct HwKeyContext {
        pub rk: [u128; 15],
        pub h: [u8; 16],
        pub h_be: u128,
        pub h2_be: u128,
        pub h3_be: u128,
        pub h4_be: u128,
    }

    /// Stores a __m128i as u128.
    #[inline(always)]
    unsafe fn m128_to_u128(v: __m128i) -> u128 {
        let mut buf = [0u8; 16];
        unsafe {
            _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, v);
        }
        u128::from_ne_bytes(buf)
    }

    /// Hardware AES key expansion + H/H^2/H^3/H^4 computation using AES-NI.
    /// Precomputes GHASH powers for 4-way parallel reduction.
    #[target_feature(enable = "aes", enable = "ssse3", enable = "pclmulqdq")]
    pub(super) unsafe fn hw_key_expand(
        key: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Result<HwKeyContext> {
        unsafe {
            let rounds = algorithm.rounds();
            let mut rk_m128 = [_mm_setzero_si128(); 15];

            match algorithm {
                EncryptionAlgorithm::Aes128Gcm => {
                    rk_m128[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                    hw_expand_128(&mut rk_m128);
                }
                EncryptionAlgorithm::Aes256Gcm => {
                    rk_m128[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                    rk_m128[1] = _mm_loadu_si128(key[16..].as_ptr() as *const __m128i);
                    hw_expand_256(&mut rk_m128);
                }
            }

            let mut rk_u128 = [0u128; 15];
            for i in 0..=rounds {
                rk_u128[i] = m128_to_u128(rk_m128[i]);
            }

            // Compute H = AES_K(0^128)
            let zero = _mm_setzero_si128();
            let h_native = aes_ni_encrypt_block(zero, &rk_u128, rounds);
            let mut h_bytes = [0u8; 16];
            _mm_storeu_si128(h_bytes.as_mut_ptr() as *mut __m128i, h_native);

            // H in GHASH big-endian byte order
            let bswap = _mm_loadu_si128(BSWAP_MASK.as_ptr() as *const __m128i);
            let h_be = _mm_shuffle_epi8(h_native, bswap);

            // Precompute H^2, H^3, H^4 for 4-way parallel GHASH
            let h2_be = ghash_mul(h_be, h_be);
            let h3_be = ghash_mul(h2_be, h_be);
            let h4_be = ghash_mul(h3_be, h_be);

            Ok(HwKeyContext {
                rk: rk_u128,
                h: h_bytes,
                h_be: m128_to_u128(h_be),
                h2_be: m128_to_u128(h2_be),
                h3_be: m128_to_u128(h3_be),
                h4_be: m128_to_u128(h4_be),
            })
        }
    }

    /// AES-128 key expansion using _mm_aeskeygenassist_si128.
    #[target_feature(enable = "aes")]
    unsafe fn hw_expand_128(rk: &mut [__m128i; 15]) {
        unsafe {
            macro_rules! expand_round {
                ($i:expr, $rcon:expr) => {{
                    let assist = _mm_aeskeygenassist_si128(rk[$i - 1], $rcon);
                    let assist = _mm_shuffle_epi32(assist, 0xFF);
                    let mut tmp = rk[$i - 1];
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    rk[$i] = _mm_xor_si128(tmp, assist);
                }};
            }
            expand_round!(1, 0x01);
            expand_round!(2, 0x02);
            expand_round!(3, 0x04);
            expand_round!(4, 0x08);
            expand_round!(5, 0x10);
            expand_round!(6, 0x20);
            expand_round!(7, 0x40);
            expand_round!(8, 0x80);
            expand_round!(9, 0x1B);
            expand_round!(10, 0x36);
        }
    }

    /// AES-256 key expansion using _mm_aeskeygenassist_si128.
    #[target_feature(enable = "aes")]
    unsafe fn hw_expand_256(rk: &mut [__m128i; 15]) {
        unsafe {
            macro_rules! expand_even {
                ($i:expr, $rcon:expr) => {{
                    let assist = _mm_aeskeygenassist_si128(rk[$i - 1], $rcon);
                    let assist = _mm_shuffle_epi32(assist, 0xFF);
                    let mut tmp = rk[$i - 2];
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    rk[$i] = _mm_xor_si128(tmp, assist);
                }};
            }
            macro_rules! expand_odd {
                ($i:expr) => {{
                    let assist = _mm_aeskeygenassist_si128(rk[$i - 1], 0x00);
                    let assist = _mm_shuffle_epi32(assist, 0xAA);
                    let mut tmp = rk[$i - 2];
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
                    rk[$i] = _mm_xor_si128(tmp, assist);
                }};
            }
            expand_even!(2, 0x01);
            expand_odd!(3);
            expand_even!(4, 0x02);
            expand_odd!(5);
            expand_even!(6, 0x04);
            expand_odd!(7);
            expand_even!(8, 0x08);
            expand_odd!(9);
            expand_even!(10, 0x10);
            expand_odd!(11);
            expand_even!(12, 0x20);
            expand_odd!(13);
            expand_even!(14, 0x40);
        }
    }

    /// Loads a round key from the u128 array as a __m128i register.
    #[inline(always)]
    unsafe fn ld(rk: &[u128; 15], i: usize) -> __m128i {
        unsafe { _mm_loadu_si128(rk.as_ptr().add(i) as *const __m128i) }
    }

    /// Loads a round key from a __m128i array directly.
    #[inline(always)]
    unsafe fn ldm(rk: &[__m128i; 15], i: usize) -> __m128i {
        unsafe { *rk.get_unchecked(i) }
    }

    /// AES block encryption dispatching to fully unrolled variants.
    /// Avoids dynamic loop so the compiler can pipeline all rounds.
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn aes_ni_encrypt_block(block: __m128i, rk: &[u128; 15], rounds: usize) -> __m128i {
        unsafe {
            if rounds == 10 {
                aes_ni_encrypt_block_128(block, rk)
            } else {
                aes_ni_encrypt_block_256(block, rk)
            }
        }
    }

    /// AES-128 block encryption using direct __m128i round keys.
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn aes128_block(block: __m128i, rk: &[__m128i; 15]) -> __m128i {
        unsafe {
            let mut s = _mm_xor_si128(block, ldm(rk, 0));
            s = _mm_aesenc_si128(s, ldm(rk, 1));
            s = _mm_aesenc_si128(s, ldm(rk, 2));
            s = _mm_aesenc_si128(s, ldm(rk, 3));
            s = _mm_aesenc_si128(s, ldm(rk, 4));
            s = _mm_aesenc_si128(s, ldm(rk, 5));
            s = _mm_aesenc_si128(s, ldm(rk, 6));
            s = _mm_aesenc_si128(s, ldm(rk, 7));
            s = _mm_aesenc_si128(s, ldm(rk, 8));
            s = _mm_aesenc_si128(s, ldm(rk, 9));
            _mm_aesenclast_si128(s, ldm(rk, 10))
        }
    }

    /// AES-256 block encryption using direct __m128i round keys.
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn aes256_block(block: __m128i, rk: &[__m128i; 15]) -> __m128i {
        unsafe {
            let mut s = _mm_xor_si128(block, ldm(rk, 0));
            s = _mm_aesenc_si128(s, ldm(rk, 1));
            s = _mm_aesenc_si128(s, ldm(rk, 2));
            s = _mm_aesenc_si128(s, ldm(rk, 3));
            s = _mm_aesenc_si128(s, ldm(rk, 4));
            s = _mm_aesenc_si128(s, ldm(rk, 5));
            s = _mm_aesenc_si128(s, ldm(rk, 6));
            s = _mm_aesenc_si128(s, ldm(rk, 7));
            s = _mm_aesenc_si128(s, ldm(rk, 8));
            s = _mm_aesenc_si128(s, ldm(rk, 9));
            s = _mm_aesenc_si128(s, ldm(rk, 10));
            s = _mm_aesenc_si128(s, ldm(rk, 11));
            s = _mm_aesenc_si128(s, ldm(rk, 12));
            s = _mm_aesenc_si128(s, ldm(rk, 13));
            _mm_aesenclast_si128(s, ldm(rk, 14))
        }
    }

    /// Fully unrolled AES-128 block encryption (10 rounds).
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn aes_ni_encrypt_block_128(block: __m128i, rk: &[u128; 15]) -> __m128i {
        unsafe {
            let mut s = _mm_xor_si128(block, ld(rk, 0));
            s = _mm_aesenc_si128(s, ld(rk, 1));
            s = _mm_aesenc_si128(s, ld(rk, 2));
            s = _mm_aesenc_si128(s, ld(rk, 3));
            s = _mm_aesenc_si128(s, ld(rk, 4));
            s = _mm_aesenc_si128(s, ld(rk, 5));
            s = _mm_aesenc_si128(s, ld(rk, 6));
            s = _mm_aesenc_si128(s, ld(rk, 7));
            s = _mm_aesenc_si128(s, ld(rk, 8));
            s = _mm_aesenc_si128(s, ld(rk, 9));
            _mm_aesenclast_si128(s, ld(rk, 10))
        }
    }

    /// Fully unrolled AES-256 block encryption (14 rounds).
    #[target_feature(enable = "aes")]
    #[inline]
    unsafe fn aes_ni_encrypt_block_256(block: __m128i, rk: &[u128; 15]) -> __m128i {
        unsafe {
            let mut s = _mm_xor_si128(block, ld(rk, 0));
            s = _mm_aesenc_si128(s, ld(rk, 1));
            s = _mm_aesenc_si128(s, ld(rk, 2));
            s = _mm_aesenc_si128(s, ld(rk, 3));
            s = _mm_aesenc_si128(s, ld(rk, 4));
            s = _mm_aesenc_si128(s, ld(rk, 5));
            s = _mm_aesenc_si128(s, ld(rk, 6));
            s = _mm_aesenc_si128(s, ld(rk, 7));
            s = _mm_aesenc_si128(s, ld(rk, 8));
            s = _mm_aesenc_si128(s, ld(rk, 9));
            s = _mm_aesenc_si128(s, ld(rk, 10));
            s = _mm_aesenc_si128(s, ld(rk, 11));
            s = _mm_aesenc_si128(s, ld(rk, 12));
            s = _mm_aesenc_si128(s, ld(rk, 13));
            _mm_aesenclast_si128(s, ld(rk, 14))
        }
    }

    /// GHASH multiplication using PCLMULQDQ with Karatsuba decomposition.
    /// Inputs and output are in GHASH big-endian (reflected) bit order.
    #[target_feature(enable = "pclmulqdq")]
    #[inline]
    unsafe fn ghash_mul(a: __m128i, h: __m128i) -> __m128i {
        unsafe {
            let lo = _mm_clmulepi64_si128(a, h, 0x00);
            let hi = _mm_clmulepi64_si128(a, h, 0x11);
            let mid_a = _mm_xor_si128(a, _mm_shuffle_epi32(a, 0x4E));
            let mid_h = _mm_xor_si128(h, _mm_shuffle_epi32(h, 0x4E));
            let mid = _mm_xor_si128(
                _mm_clmulepi64_si128(mid_a, mid_h, 0x00),
                _mm_xor_si128(lo, hi),
            );
            let prod_lo = _mm_xor_si128(lo, _mm_slli_si128(mid, 8));
            let prod_hi = _mm_xor_si128(hi, _mm_srli_si128(mid, 8));

            // Two-phase Barrett reduction mod x^128 + x^7 + x^2 + x + 1
            let poly = _mm_set_epi64x(0, 0x00000000_00000087);
            let t1 = _mm_clmulepi64_si128(prod_lo, poly, 0x00);
            let reduced = _mm_xor_si128(prod_lo, _mm_shuffle_epi32(t1, 0x4E));
            let t2 = _mm_clmulepi64_si128(reduced, poly, 0x00);
            _mm_xor_si128(prod_hi, _mm_xor_si128(reduced, _mm_shuffle_epi32(t2, 0x4E)))
        }
    }

    /// 4-way aggregated GHASH: computes (a0*b0) XOR (a1*b1) XOR (a2*b2) XOR (a3*b3)
    /// using 12 Karatsuba clmul + 1 combined Barrett reduction (2 clmul) = 14 total.
    /// The non-aggregated approach uses 4 * 5 = 20 clmul. All 12 partial products
    /// are independent, allowing the CPU to pipeline them fully.
    #[target_feature(enable = "pclmulqdq")]
    #[inline]
    unsafe fn ghash_mul4(
        a0: __m128i,
        b0: __m128i,
        a1: __m128i,
        b1: __m128i,
        a2: __m128i,
        b2: __m128i,
        a3: __m128i,
        b3: __m128i,
    ) -> __m128i {
        unsafe {
            // 12 independent clmul partial products
            let lo0 = _mm_clmulepi64_si128(a0, b0, 0x00);
            let lo1 = _mm_clmulepi64_si128(a1, b1, 0x00);
            let lo2 = _mm_clmulepi64_si128(a2, b2, 0x00);
            let lo3 = _mm_clmulepi64_si128(a3, b3, 0x00);

            let hi0 = _mm_clmulepi64_si128(a0, b0, 0x11);
            let hi1 = _mm_clmulepi64_si128(a1, b1, 0x11);
            let hi2 = _mm_clmulepi64_si128(a2, b2, 0x11);
            let hi3 = _mm_clmulepi64_si128(a3, b3, 0x11);

            let mid0 = _mm_clmulepi64_si128(
                _mm_xor_si128(a0, _mm_shuffle_epi32(a0, 0x4E)),
                _mm_xor_si128(b0, _mm_shuffle_epi32(b0, 0x4E)),
                0x00,
            );
            let mid1 = _mm_clmulepi64_si128(
                _mm_xor_si128(a1, _mm_shuffle_epi32(a1, 0x4E)),
                _mm_xor_si128(b1, _mm_shuffle_epi32(b1, 0x4E)),
                0x00,
            );
            let mid2 = _mm_clmulepi64_si128(
                _mm_xor_si128(a2, _mm_shuffle_epi32(a2, 0x4E)),
                _mm_xor_si128(b2, _mm_shuffle_epi32(b2, 0x4E)),
                0x00,
            );
            let mid3 = _mm_clmulepi64_si128(
                _mm_xor_si128(a3, _mm_shuffle_epi32(a3, 0x4E)),
                _mm_xor_si128(b3, _mm_shuffle_epi32(b3, 0x4E)),
                0x00,
            );

            // Accumulate all products
            let lo_acc = _mm_xor_si128(_mm_xor_si128(lo0, lo1), _mm_xor_si128(lo2, lo3));
            let hi_acc = _mm_xor_si128(_mm_xor_si128(hi0, hi1), _mm_xor_si128(hi2, hi3));
            let mid_raw = _mm_xor_si128(_mm_xor_si128(mid0, mid1), _mm_xor_si128(mid2, mid3));
            let mid_acc = _mm_xor_si128(mid_raw, _mm_xor_si128(lo_acc, hi_acc));

            let prod_lo = _mm_xor_si128(lo_acc, _mm_slli_si128(mid_acc, 8));
            let prod_hi = _mm_xor_si128(hi_acc, _mm_srli_si128(mid_acc, 8));

            // Single Barrett reduction (2 clmul instead of 4 * 2 = 8)
            let poly = _mm_set_epi64x(0, 0x00000000_00000087);
            let t1 = _mm_clmulepi64_si128(prod_lo, poly, 0x00);
            let reduced = _mm_xor_si128(prod_lo, _mm_shuffle_epi32(t1, 0x4E));
            let t2 = _mm_clmulepi64_si128(reduced, poly, 0x00);
            _mm_xor_si128(prod_hi, _mm_xor_si128(reduced, _mm_shuffle_epi32(t2, 0x4E)))
        }
    }

    /// Processes a GHASH block: XOR input into state, multiply by H.
    /// The input block is loaded from memory and byte-swapped to big-endian.
    #[target_feature(enable = "pclmulqdq", enable = "ssse3")]
    #[inline]
    unsafe fn ghash_update(
        state: __m128i,
        block_ptr: *const u8,
        block_len: usize,
        bswap: __m128i,
        h_be: __m128i,
    ) -> __m128i {
        unsafe {
            let mut block = _mm_setzero_si128();
            if block_len == 16 {
                block = _mm_loadu_si128(block_ptr as *const __m128i);
            } else {
                let mut buf = [0u8; 16];
                core::ptr::copy_nonoverlapping(block_ptr, buf.as_mut_ptr(), block_len);
                block = _mm_loadu_si128(buf.as_ptr() as *const __m128i);
            }
            let swapped = _mm_shuffle_epi8(block, bswap);
            ghash_mul(_mm_xor_si128(state, swapped), h_be)
        }
    }

    /// CTR mode: build 4 counter blocks using _mm_insert_epi32 (SSE4.1).
    /// One instruction per counter instead of three (set + shuffle + or).
    /// Counter value is byte-swapped to big-endian before insertion into
    /// the last 32-bit lane (position 3 = bytes 12-15).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    unsafe fn build_4_counters(
        nonce_base: __m128i,
        counter: u32,
    ) -> (__m128i, __m128i, __m128i, __m128i) {
        unsafe {
            let c0 = _mm_insert_epi32(nonce_base, counter.swap_bytes() as i32, 3);
            let c1 = _mm_insert_epi32(nonce_base, (counter + 1).swap_bytes() as i32, 3);
            let c2 = _mm_insert_epi32(nonce_base, (counter + 2).swap_bytes() as i32, 3);
            let c3 = _mm_insert_epi32(nonce_base, (counter + 3).swap_bytes() as i32, 3);
            (c0, c1, c2, c3)
        }
    }

    /// Hardware AES-GCM encrypt with fused CTR+GHASH, register-based counters,
    /// and pre-swapped H. Returns nonce || ciphertext || tag.
    #[target_feature(
        enable = "aes",
        enable = "pclmulqdq",
        enable = "ssse3",
        enable = "sse4.1"
    )]
    pub(super) unsafe fn hw_aes_gcm_encrypt(
        plaintext: &[u8],
        aad: &[u8],
        nonce: &[u8; GCM_NONCE_LEN],
        rk: &[u128; 15],
        h_be_u128: u128,
        h2_be_u128: u128,
        h3_be_u128: u128,
        h4_be_u128: u128,
        rounds: usize,
    ) -> Result<Vec<u8>> {
        unsafe {
            let ct_len = plaintext.len();
            let mut output = vec![0u8; GCM_NONCE_LEN + ct_len + GCM_TAG_LEN];
            output[..GCM_NONCE_LEN].copy_from_slice(nonce);

            let bswap = _mm_loadu_si128(BSWAP_MASK.as_ptr() as *const __m128i);
            let h_be = _mm_loadu_si128(&h_be_u128 as *const u128 as *const __m128i);
            let h2_be = _mm_loadu_si128(&h2_be_u128 as *const u128 as *const __m128i);
            let h3_be = _mm_loadu_si128(&h3_be_u128 as *const u128 as *const __m128i);
            let h4_be = _mm_loadu_si128(&h4_be_u128 as *const u128 as *const __m128i);

            let mut j0_bytes = [0u8; 16];
            j0_bytes[..12].copy_from_slice(nonce);
            let nonce_base = _mm_loadu_si128(j0_bytes.as_ptr() as *const __m128i);

            j0_bytes[15] = 1;
            let j0 = _mm_loadu_si128(j0_bytes.as_ptr() as *const __m128i);
            let encrypted_j0 = aes_ni_encrypt_block(j0, rk, rounds);

            let ct_out = &mut output[GCM_NONCE_LEN..GCM_NONCE_LEN + ct_len];
            let mut counter = 2u32;
            let mut offset = 0;

            let mut ghash_state = _mm_setzero_si128();
            for chunk in aad.chunks(16) {
                ghash_state = ghash_update(ghash_state, chunk.as_ptr(), chunk.len(), bswap, h_be);
            }

            // 4-block parallel: encrypt CTR then 4-way parallel GHASH reduction.
            // state_new = ((state XOR s0) * H^4) XOR (s1 * H^3) XOR (s2 * H^2) XOR (s3 * H)
            // The 4 multiplications are independent (no data dependency between them),
            // allowing the CPU to execute them in parallel on different execution ports.
            let full_quads = ct_len / 64;
            for _ in 0..full_quads {
                let (c0, c1, c2, c3) = build_4_counters(nonce_base, counter);
                let k0 = aes_ni_encrypt_block(c0, rk, rounds);
                let k1 = aes_ni_encrypt_block(c1, rk, rounds);
                let k2 = aes_ni_encrypt_block(c2, rk, rounds);
                let k3 = aes_ni_encrypt_block(c3, rk, rounds);

                let p0 = _mm_loadu_si128(plaintext[offset..].as_ptr() as *const __m128i);
                let p1 = _mm_loadu_si128(plaintext[offset + 16..].as_ptr() as *const __m128i);
                let p2 = _mm_loadu_si128(plaintext[offset + 32..].as_ptr() as *const __m128i);
                let p3 = _mm_loadu_si128(plaintext[offset + 48..].as_ptr() as *const __m128i);

                let ct0 = _mm_xor_si128(p0, k0);
                let ct1 = _mm_xor_si128(p1, k1);
                let ct2 = _mm_xor_si128(p2, k2);
                let ct3 = _mm_xor_si128(p3, k3);

                _mm_storeu_si128(ct_out[offset..].as_mut_ptr() as *mut __m128i, ct0);
                _mm_storeu_si128(ct_out[offset + 16..].as_mut_ptr() as *mut __m128i, ct1);
                _mm_storeu_si128(ct_out[offset + 32..].as_mut_ptr() as *mut __m128i, ct2);
                _mm_storeu_si128(ct_out[offset + 48..].as_mut_ptr() as *mut __m128i, ct3);

                // 4-way parallel GHASH: each multiply is independent
                let s0 = _mm_shuffle_epi8(ct0, bswap);
                let s1 = _mm_shuffle_epi8(ct1, bswap);
                let s2 = _mm_shuffle_epi8(ct2, bswap);
                let s3 = _mm_shuffle_epi8(ct3, bswap);

                ghash_state = ghash_mul4(
                    _mm_xor_si128(ghash_state, s0),
                    h4_be,
                    s1,
                    h3_be,
                    s2,
                    h2_be,
                    s3,
                    h_be,
                );

                counter += 4;
                offset += 64;
            }

            // Remaining full blocks
            while offset + 16 <= ct_len {
                let mut cb = j0_bytes;
                cb[12..16].copy_from_slice(&counter.to_be_bytes());
                let ks = aes_ni_encrypt_block(
                    _mm_loadu_si128(cb.as_ptr() as *const __m128i),
                    rk,
                    rounds,
                );
                let pt = _mm_loadu_si128(plaintext[offset..].as_ptr() as *const __m128i);
                let ct_block = _mm_xor_si128(pt, ks);
                _mm_storeu_si128(ct_out[offset..].as_mut_ptr() as *mut __m128i, ct_block);
                ghash_state = ghash_mul(
                    _mm_xor_si128(ghash_state, _mm_shuffle_epi8(ct_block, bswap)),
                    h_be,
                );
                counter += 1;
                offset += 16;
            }

            // Partial last block
            if offset < ct_len {
                let mut cb = j0_bytes;
                cb[12..16].copy_from_slice(&counter.to_be_bytes());
                let ks = aes_ni_encrypt_block(
                    _mm_loadu_si128(cb.as_ptr() as *const __m128i),
                    rk,
                    rounds,
                );
                let mut ks_bytes = [0u8; 16];
                _mm_storeu_si128(ks_bytes.as_mut_ptr() as *mut __m128i, ks);
                let remain = ct_len - offset;
                for i in 0..remain {
                    ct_out[offset + i] = plaintext[offset + i] ^ ks_bytes[i];
                }
                ghash_state =
                    ghash_update(ghash_state, ct_out[offset..].as_ptr(), remain, bswap, h_be);
            }

            // Length block
            let aad_bits = (aad.len() as u64) * 8;
            let ct_bits = (ct_len as u64) * 8;
            let mut len_block = [0u8; 16];
            len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
            len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
            let lb = _mm_shuffle_epi8(_mm_loadu_si128(len_block.as_ptr() as *const __m128i), bswap);
            ghash_state = ghash_mul(_mm_xor_si128(ghash_state, lb), h_be);

            // Tag = bswap(ghash_state) XOR encrypted_j0
            let tag = _mm_xor_si128(_mm_shuffle_epi8(ghash_state, bswap), encrypted_j0);
            _mm_storeu_si128(
                output[GCM_NONCE_LEN + ct_len..].as_mut_ptr() as *mut __m128i,
                tag,
            );

            Ok(output)
        }
    }

    /// Hardware AES-GCM decrypt with tag verification then CTR decryption.
    #[target_feature(
        enable = "aes",
        enable = "pclmulqdq",
        enable = "ssse3",
        enable = "sse4.1"
    )]
    pub(super) unsafe fn hw_aes_gcm_decrypt(
        ciphertext: &[u8],
        aad: &[u8],
        nonce: &[u8],
        tag: &[u8],
        rk: &[u128; 15],
        h_be_u128: u128,
        h2_be_u128: u128,
        h3_be_u128: u128,
        h4_be_u128: u128,
        rounds: usize,
    ) -> Result<Vec<u8>> {
        unsafe {
            let ct_len = ciphertext.len();
            let bswap = _mm_loadu_si128(BSWAP_MASK.as_ptr() as *const __m128i);
            let h_be = _mm_loadu_si128(&h_be_u128 as *const u128 as *const __m128i);
            let h2_be = _mm_loadu_si128(&h2_be_u128 as *const u128 as *const __m128i);
            let h3_be = _mm_loadu_si128(&h3_be_u128 as *const u128 as *const __m128i);
            let h4_be = _mm_loadu_si128(&h4_be_u128 as *const u128 as *const __m128i);

            let mut j0_bytes = [0u8; 16];
            j0_bytes[..12].copy_from_slice(nonce);
            j0_bytes[15] = 1;
            let j0 = _mm_loadu_si128(j0_bytes.as_ptr() as *const __m128i);
            let encrypted_j0 = aes_ni_encrypt_block(j0, rk, rounds);
            let nonce_base = {
                let mut nb = [0u8; 16];
                nb[..12].copy_from_slice(nonce);
                _mm_loadu_si128(nb.as_ptr() as *const __m128i)
            };

            // GHASH over AAD + ciphertext for tag verification with 4-way parallelism
            let mut ghash_state = _mm_setzero_si128();
            for chunk in aad.chunks(16) {
                ghash_state = ghash_update(ghash_state, chunk.as_ptr(), chunk.len(), bswap, h_be);
            }

            // 4-way parallel GHASH on ciphertext blocks
            let mut ct_offset = 0;
            let ct_full_quads = ct_len / 64;
            for _ in 0..ct_full_quads {
                let s0 = _mm_shuffle_epi8(
                    _mm_loadu_si128(ciphertext[ct_offset..].as_ptr() as *const __m128i),
                    bswap,
                );
                let s1 = _mm_shuffle_epi8(
                    _mm_loadu_si128(ciphertext[ct_offset + 16..].as_ptr() as *const __m128i),
                    bswap,
                );
                let s2 = _mm_shuffle_epi8(
                    _mm_loadu_si128(ciphertext[ct_offset + 32..].as_ptr() as *const __m128i),
                    bswap,
                );
                let s3 = _mm_shuffle_epi8(
                    _mm_loadu_si128(ciphertext[ct_offset + 48..].as_ptr() as *const __m128i),
                    bswap,
                );

                ghash_state = ghash_mul4(
                    _mm_xor_si128(ghash_state, s0),
                    h4_be,
                    s1,
                    h3_be,
                    s2,
                    h2_be,
                    s3,
                    h_be,
                );

                ct_offset += 64;
            }
            // Remaining ciphertext blocks
            while ct_offset < ct_len {
                let remain = core::cmp::min(16, ct_len - ct_offset);
                ghash_state = ghash_update(
                    ghash_state,
                    ciphertext[ct_offset..].as_ptr(),
                    remain,
                    bswap,
                    h_be,
                );
                ct_offset += 16;
            }
            let aad_bits = (aad.len() as u64) * 8;
            let ct_bits = (ct_len as u64) * 8;
            let mut len_block = [0u8; 16];
            len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
            len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
            let lb = _mm_shuffle_epi8(_mm_loadu_si128(len_block.as_ptr() as *const __m128i), bswap);
            ghash_state = ghash_mul(_mm_xor_si128(ghash_state, lb), h_be);

            let computed_tag = _mm_xor_si128(_mm_shuffle_epi8(ghash_state, bswap), encrypted_j0);
            let mut computed_tag_bytes = [0u8; 16];
            _mm_storeu_si128(
                computed_tag_bytes.as_mut_ptr() as *mut __m128i,
                computed_tag,
            );

            if !constant_time_eq(tag, &computed_tag_bytes) {
                return Err(ZyronError::DecryptionFailed(
                    "Authentication tag mismatch, data may be tampered".to_string(),
                ));
            }

            // CTR mode decryption with 4-block parallelism
            let mut plaintext = vec![0u8; ct_len];
            let mut counter = 2u32;
            let mut offset = 0;

            let full_quads = ct_len / 64;
            for _ in 0..full_quads {
                let (c0, c1, c2, c3) = build_4_counters(nonce_base, counter);
                let k0 = aes_ni_encrypt_block(c0, rk, rounds);
                let k1 = aes_ni_encrypt_block(c1, rk, rounds);
                let k2 = aes_ni_encrypt_block(c2, rk, rounds);
                let k3 = aes_ni_encrypt_block(c3, rk, rounds);

                let ct0 = _mm_loadu_si128(ciphertext[offset..].as_ptr() as *const __m128i);
                let ct1 = _mm_loadu_si128(ciphertext[offset + 16..].as_ptr() as *const __m128i);
                let ct2 = _mm_loadu_si128(ciphertext[offset + 32..].as_ptr() as *const __m128i);
                let ct3 = _mm_loadu_si128(ciphertext[offset + 48..].as_ptr() as *const __m128i);

                _mm_storeu_si128(
                    plaintext[offset..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct0, k0),
                );
                _mm_storeu_si128(
                    plaintext[offset + 16..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct1, k1),
                );
                _mm_storeu_si128(
                    plaintext[offset + 32..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct2, k2),
                );
                _mm_storeu_si128(
                    plaintext[offset + 48..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct3, k3),
                );

                counter += 4;
                offset += 64;
            }

            while offset + 16 <= ct_len {
                let mut cb = j0_bytes;
                cb[12..16].copy_from_slice(&counter.to_be_bytes());
                let ks = aes_ni_encrypt_block(
                    _mm_loadu_si128(cb.as_ptr() as *const __m128i),
                    rk,
                    rounds,
                );
                let ct = _mm_loadu_si128(ciphertext[offset..].as_ptr() as *const __m128i);
                _mm_storeu_si128(
                    plaintext[offset..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct, ks),
                );
                counter += 1;
                offset += 16;
            }

            if offset < ct_len {
                let mut cb = j0_bytes;
                cb[12..16].copy_from_slice(&counter.to_be_bytes());
                let ks = aes_ni_encrypt_block(
                    _mm_loadu_si128(cb.as_ptr() as *const __m128i),
                    rk,
                    rounds,
                );
                let mut ks_bytes = [0u8; 16];
                _mm_storeu_si128(ks_bytes.as_mut_ptr() as *mut __m128i, ks);
                for i in 0..(ct_len - offset) {
                    plaintext[offset + i] = ciphertext[offset + i] ^ ks_bytes[i];
                }
            }

            Ok(plaintext)
        }
    }
    /// Macro that generates the CTR encrypt + GHASH body for a specific
    /// AES block function. Eliminates function pointer indirection so the
    /// compiler can inline the block encrypt into the loop body.
    macro_rules! gcm_encrypt_body {
        ($aes_block:ident, $rk:expr, $plaintext:expr, $aad:expr, $nonce:expr,
         $bswap:expr, $h_be:expr, $h2_be:expr, $h3_be:expr, $h4_be:expr) => {{
            let ct_len = $plaintext.len();
            let mut output = vec![0u8; GCM_NONCE_LEN + ct_len + GCM_TAG_LEN];
            output[..GCM_NONCE_LEN].copy_from_slice(&$nonce);

            let mut j0_bytes = [0u8; 16];
            j0_bytes[..12].copy_from_slice(&$nonce);
            let nonce_base = _mm_loadu_si128(j0_bytes.as_ptr() as *const __m128i);

            j0_bytes[15] = 1;
            let j0 = _mm_loadu_si128(j0_bytes.as_ptr() as *const __m128i);
            let encrypted_j0 = $aes_block(j0, $rk);

            let ct_out = &mut output[GCM_NONCE_LEN..GCM_NONCE_LEN + ct_len];
            let mut counter = 2u32;
            let mut offset = 0;

            let mut ghash_state = _mm_setzero_si128();
            for chunk in $aad.chunks(16) {
                ghash_state = ghash_update(ghash_state, chunk.as_ptr(), chunk.len(), $bswap, $h_be);
            }

            let full_quads = ct_len / 64;
            for _ in 0..full_quads {
                let (c0, c1, c2, c3) = build_4_counters(nonce_base, counter);
                let k0 = $aes_block(c0, $rk);
                let k1 = $aes_block(c1, $rk);
                let k2 = $aes_block(c2, $rk);
                let k3 = $aes_block(c3, $rk);

                let p0 = _mm_loadu_si128($plaintext[offset..].as_ptr() as *const __m128i);
                let p1 = _mm_loadu_si128($plaintext[offset + 16..].as_ptr() as *const __m128i);
                let p2 = _mm_loadu_si128($plaintext[offset + 32..].as_ptr() as *const __m128i);
                let p3 = _mm_loadu_si128($plaintext[offset + 48..].as_ptr() as *const __m128i);

                let ct0 = _mm_xor_si128(p0, k0);
                let ct1 = _mm_xor_si128(p1, k1);
                let ct2 = _mm_xor_si128(p2, k2);
                let ct3 = _mm_xor_si128(p3, k3);

                _mm_storeu_si128(ct_out[offset..].as_mut_ptr() as *mut __m128i, ct0);
                _mm_storeu_si128(ct_out[offset + 16..].as_mut_ptr() as *mut __m128i, ct1);
                _mm_storeu_si128(ct_out[offset + 32..].as_mut_ptr() as *mut __m128i, ct2);
                _mm_storeu_si128(ct_out[offset + 48..].as_mut_ptr() as *mut __m128i, ct3);

                let s0 = _mm_shuffle_epi8(ct0, $bswap);
                let s1 = _mm_shuffle_epi8(ct1, $bswap);
                let s2 = _mm_shuffle_epi8(ct2, $bswap);
                let s3 = _mm_shuffle_epi8(ct3, $bswap);

                ghash_state = ghash_mul4(
                    _mm_xor_si128(ghash_state, s0),
                    $h4_be,
                    s1,
                    $h3_be,
                    s2,
                    $h2_be,
                    s3,
                    $h_be,
                );

                counter += 4;
                offset += 64;
            }

            while offset + 16 <= ct_len {
                let c = _mm_insert_epi32(nonce_base, counter.swap_bytes() as i32, 3);
                let ks = $aes_block(c, $rk);
                let pt = _mm_loadu_si128($plaintext[offset..].as_ptr() as *const __m128i);
                let ct_block = _mm_xor_si128(pt, ks);
                _mm_storeu_si128(ct_out[offset..].as_mut_ptr() as *mut __m128i, ct_block);
                ghash_state = ghash_mul(
                    _mm_xor_si128(ghash_state, _mm_shuffle_epi8(ct_block, $bswap)),
                    $h_be,
                );
                counter += 1;
                offset += 16;
            }

            if offset < ct_len {
                let c = _mm_insert_epi32(nonce_base, counter.swap_bytes() as i32, 3);
                let ks = $aes_block(c, $rk);
                let mut ks_bytes = [0u8; 16];
                _mm_storeu_si128(ks_bytes.as_mut_ptr() as *mut __m128i, ks);
                let remain = ct_len - offset;
                for i in 0..remain {
                    ct_out[offset + i] = $plaintext[offset + i] ^ ks_bytes[i];
                }
                ghash_state = ghash_update(
                    ghash_state,
                    ct_out[offset..].as_ptr(),
                    remain,
                    $bswap,
                    $h_be,
                );
            }

            let aad_bits = ($aad.len() as u64) * 8;
            let ct_bits = (ct_len as u64) * 8;
            let mut len_block = [0u8; 16];
            len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
            len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
            let lb = _mm_shuffle_epi8(
                _mm_loadu_si128(len_block.as_ptr() as *const __m128i),
                $bswap,
            );
            ghash_state = ghash_mul(_mm_xor_si128(ghash_state, lb), $h_be);

            let tag = _mm_xor_si128(_mm_shuffle_epi8(ghash_state, $bswap), encrypted_j0);
            _mm_storeu_si128(
                output[GCM_NONCE_LEN + ct_len..].as_mut_ptr() as *mut __m128i,
                tag,
            );

            Ok(output)
        }};
    }

    /// Macro for CTR decrypt body (GHASH verification already done).
    macro_rules! gcm_decrypt_ctr {
        ($aes_block:ident, $rk:expr, $ciphertext:expr, $nonce_base:expr) => {{
            let ct_len = $ciphertext.len();
            let mut plaintext_out = vec![0u8; ct_len];
            let mut counter = 2u32;
            let mut offset = 0;

            let full_quads = ct_len / 64;
            for _ in 0..full_quads {
                let (c0, c1, c2, c3) = build_4_counters($nonce_base, counter);
                let k0 = $aes_block(c0, $rk);
                let k1 = $aes_block(c1, $rk);
                let k2 = $aes_block(c2, $rk);
                let k3 = $aes_block(c3, $rk);

                let ct0 = _mm_loadu_si128($ciphertext[offset..].as_ptr() as *const __m128i);
                let ct1 = _mm_loadu_si128($ciphertext[offset + 16..].as_ptr() as *const __m128i);
                let ct2 = _mm_loadu_si128($ciphertext[offset + 32..].as_ptr() as *const __m128i);
                let ct3 = _mm_loadu_si128($ciphertext[offset + 48..].as_ptr() as *const __m128i);

                _mm_storeu_si128(
                    plaintext_out[offset..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct0, k0),
                );
                _mm_storeu_si128(
                    plaintext_out[offset + 16..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct1, k1),
                );
                _mm_storeu_si128(
                    plaintext_out[offset + 32..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct2, k2),
                );
                _mm_storeu_si128(
                    plaintext_out[offset + 48..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct3, k3),
                );

                counter += 4;
                offset += 64;
            }

            while offset + 16 <= ct_len {
                let c = _mm_insert_epi32($nonce_base, counter.swap_bytes() as i32, 3);
                let ks = $aes_block(c, $rk);
                let ct = _mm_loadu_si128($ciphertext[offset..].as_ptr() as *const __m128i);
                _mm_storeu_si128(
                    plaintext_out[offset..].as_mut_ptr() as *mut __m128i,
                    _mm_xor_si128(ct, ks),
                );
                counter += 1;
                offset += 16;
            }

            if offset < ct_len {
                let c = _mm_insert_epi32($nonce_base, counter.swap_bytes() as i32, 3);
                let ks = $aes_block(c, $rk);
                let mut ks_bytes = [0u8; 16];
                _mm_storeu_si128(ks_bytes.as_mut_ptr() as *mut __m128i, ks);
                for i in 0..(ct_len - offset) {
                    plaintext_out[offset + i] = $ciphertext[offset + i] ^ ks_bytes[i];
                }
            }

            Ok(plaintext_out)
        }};
    }

    /// Direct encrypt path: key expansion + CTR + GHASH in one function.
    /// Round keys stay as __m128i (no u128 roundtrip). H powers computed
    /// inline. Uses macro-expanded block function to avoid function pointer
    /// indirection, enabling the compiler to inline AES rounds into the loop.
    #[target_feature(
        enable = "aes",
        enable = "pclmulqdq",
        enable = "ssse3",
        enable = "sse4.1"
    )]
    pub(super) unsafe fn hw_encrypt_direct(
        plaintext: &[u8],
        key: &[u8],
        algorithm: EncryptionAlgorithm,
        aad: &[u8],
    ) -> Result<Vec<u8>> {
        unsafe {
            let mut rk = [_mm_setzero_si128(); 15];
            match algorithm {
                EncryptionAlgorithm::Aes128Gcm => {
                    rk[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                    hw_expand_128(&mut rk);
                }
                EncryptionAlgorithm::Aes256Gcm => {
                    rk[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                    rk[1] = _mm_loadu_si128(key[16..].as_ptr() as *const __m128i);
                    hw_expand_256(&mut rk);
                }
            }

            let bswap = _mm_loadu_si128(BSWAP_MASK.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();

            use rand::Rng;
            let mut nonce = [0u8; GCM_NONCE_LEN];
            rand::rng().fill_bytes(&mut nonce);

            match algorithm {
                EncryptionAlgorithm::Aes128Gcm => {
                    let h_be = _mm_shuffle_epi8(aes128_block(zero, &rk), bswap);
                    let h2_be = ghash_mul(h_be, h_be);
                    let h3_be = ghash_mul(h2_be, h_be);
                    let h4_be = ghash_mul(h3_be, h_be);
                    gcm_encrypt_body!(
                        aes128_block,
                        &rk,
                        plaintext,
                        aad,
                        nonce,
                        bswap,
                        h_be,
                        h2_be,
                        h3_be,
                        h4_be
                    )
                }
                EncryptionAlgorithm::Aes256Gcm => {
                    let h_be = _mm_shuffle_epi8(aes256_block(zero, &rk), bswap);
                    let h2_be = ghash_mul(h_be, h_be);
                    let h3_be = ghash_mul(h2_be, h_be);
                    let h4_be = ghash_mul(h3_be, h_be);
                    gcm_encrypt_body!(
                        aes256_block,
                        &rk,
                        plaintext,
                        aad,
                        nonce,
                        bswap,
                        h_be,
                        h2_be,
                        h3_be,
                        h4_be
                    )
                }
            }
        }
    }

    /// Macro for GHASH verification over AAD + ciphertext, returns error on mismatch.
    macro_rules! gcm_ghash_verify {
        ($aes_block:ident, $rk:expr, $ciphertext:expr, $aad:expr, $nonce:expr, $tag:expr,
         $bswap:expr, $h_be:expr, $h2_be:expr, $h3_be:expr, $h4_be:expr) => {{
            let ct_len = $ciphertext.len();
            let mut j0_bytes = [0u8; 16];
            j0_bytes[..12].copy_from_slice($nonce);
            j0_bytes[15] = 1;
            let j0 = _mm_loadu_si128(j0_bytes.as_ptr() as *const __m128i);
            let encrypted_j0 = $aes_block(j0, $rk);

            let mut ghash_state = _mm_setzero_si128();
            for chunk in $aad.chunks(16) {
                ghash_state = ghash_update(ghash_state, chunk.as_ptr(), chunk.len(), $bswap, $h_be);
            }

            let mut ct_offset = 0;
            let ct_full_quads = ct_len / 64;
            for _ in 0..ct_full_quads {
                let s0 = _mm_shuffle_epi8(
                    _mm_loadu_si128($ciphertext[ct_offset..].as_ptr() as *const __m128i),
                    $bswap,
                );
                let s1 = _mm_shuffle_epi8(
                    _mm_loadu_si128($ciphertext[ct_offset + 16..].as_ptr() as *const __m128i),
                    $bswap,
                );
                let s2 = _mm_shuffle_epi8(
                    _mm_loadu_si128($ciphertext[ct_offset + 32..].as_ptr() as *const __m128i),
                    $bswap,
                );
                let s3 = _mm_shuffle_epi8(
                    _mm_loadu_si128($ciphertext[ct_offset + 48..].as_ptr() as *const __m128i),
                    $bswap,
                );

                ghash_state = ghash_mul4(
                    _mm_xor_si128(ghash_state, s0),
                    $h4_be,
                    s1,
                    $h3_be,
                    s2,
                    $h2_be,
                    s3,
                    $h_be,
                );
                ct_offset += 64;
            }
            while ct_offset < ct_len {
                let remain = core::cmp::min(16, ct_len - ct_offset);
                ghash_state = ghash_update(
                    ghash_state,
                    $ciphertext[ct_offset..].as_ptr(),
                    remain,
                    $bswap,
                    $h_be,
                );
                ct_offset += 16;
            }
            let aad_bits = ($aad.len() as u64) * 8;
            let ct_bits = (ct_len as u64) * 8;
            let mut len_block = [0u8; 16];
            len_block[..8].copy_from_slice(&aad_bits.to_be_bytes());
            len_block[8..16].copy_from_slice(&ct_bits.to_be_bytes());
            let lb = _mm_shuffle_epi8(
                _mm_loadu_si128(len_block.as_ptr() as *const __m128i),
                $bswap,
            );
            ghash_state = ghash_mul(_mm_xor_si128(ghash_state, lb), $h_be);

            let computed_tag = _mm_xor_si128(_mm_shuffle_epi8(ghash_state, $bswap), encrypted_j0);
            let mut computed_tag_bytes = [0u8; 16];
            _mm_storeu_si128(
                computed_tag_bytes.as_mut_ptr() as *mut __m128i,
                computed_tag,
            );

            if !constant_time_eq($tag, &computed_tag_bytes) {
                return Err(ZyronError::DecryptionFailed(
                    "Authentication tag mismatch, data may be tampered".to_string(),
                ));
            }
        }};
    }

    /// Direct decrypt path: key expansion + GHASH + CTR in one function.
    /// Round keys stay as __m128i (no u128 roundtrip). Uses macros for
    /// the hot loops to avoid function pointer indirection.
    #[target_feature(
        enable = "aes",
        enable = "pclmulqdq",
        enable = "ssse3",
        enable = "sse4.1"
    )]
    pub(super) unsafe fn hw_decrypt_direct(
        ciphertext: &[u8],
        aad: &[u8],
        nonce: &[u8],
        tag: &[u8],
        key: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> Result<Vec<u8>> {
        unsafe {
            let mut rk = [_mm_setzero_si128(); 15];
            match algorithm {
                EncryptionAlgorithm::Aes128Gcm => {
                    rk[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                    hw_expand_128(&mut rk);
                }
                EncryptionAlgorithm::Aes256Gcm => {
                    rk[0] = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                    rk[1] = _mm_loadu_si128(key[16..].as_ptr() as *const __m128i);
                    hw_expand_256(&mut rk);
                }
            }

            let bswap = _mm_loadu_si128(BSWAP_MASK.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();
            let nonce_base = {
                let mut nb = [0u8; 16];
                nb[..12].copy_from_slice(nonce);
                _mm_loadu_si128(nb.as_ptr() as *const __m128i)
            };

            match algorithm {
                EncryptionAlgorithm::Aes128Gcm => {
                    let h_be = _mm_shuffle_epi8(aes128_block(zero, &rk), bswap);
                    let h2_be = ghash_mul(h_be, h_be);
                    let h3_be = ghash_mul(h2_be, h_be);
                    let h4_be = ghash_mul(h3_be, h_be);
                    gcm_ghash_verify!(
                        aes128_block,
                        &rk,
                        ciphertext,
                        aad,
                        nonce,
                        tag,
                        bswap,
                        h_be,
                        h2_be,
                        h3_be,
                        h4_be
                    );
                    gcm_decrypt_ctr!(aes128_block, &rk, ciphertext, nonce_base)
                }
                EncryptionAlgorithm::Aes256Gcm => {
                    let h_be = _mm_shuffle_epi8(aes256_block(zero, &rk), bswap);
                    let h2_be = ghash_mul(h_be, h_be);
                    let h3_be = ghash_mul(h2_be, h_be);
                    let h4_be = ghash_mul(h3_be, h_be);
                    gcm_ghash_verify!(
                        aes256_block,
                        &rk,
                        ciphertext,
                        aad,
                        nonce,
                        tag,
                        bswap,
                        h_be,
                        h2_be,
                        h3_be,
                        h4_be
                    );
                    gcm_decrypt_ctr!(aes256_block, &rk, ciphertext, nonce_base)
                }
            }
        }
    }

    /// VAES-accelerated 256-bit hash using Davies-Meyer compression.
    /// Round keys are pre-broadcast to 256-bit and 32-byte aligned.
    /// Direct _mm256_loadu_si256 from the static, zero per-call conversion.
    /// Both 128-bit blocks processed in parallel through 14 AES rounds.
    #[target_feature(enable = "aes", enable = "avx2", enable = "vaes")]
    pub(super) unsafe fn vaes_hash_256_fast(rk: &[[u8; 32]; 15], padded: &[u8; 16]) -> [u8; 32] {
        unsafe {
            use core::arch::x86_64::*;

            // Block 0 = padded input (for h1)
            let b0 = _mm_loadu_si128(padded.as_ptr() as *const __m128i);

            // Block 1 = padded XOR 0xFF (domain separation for h2)
            let b1 = _mm_xor_si128(b0, _mm_set1_epi8(-1));

            // Pack both blocks into one YMM register: [b0 | b1]
            let inputs = _mm256_inserti128_si256(_mm256_castsi128_si256(b0), b1, 1);

            // Initial round key addition (aligned load from pre-broadcast keys)
            let mut state =
                _mm256_xor_si256(inputs, _mm256_loadu_si256(rk[0].as_ptr() as *const __m256i));

            // 13 main rounds + 1 final round (AES-256)
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[1].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[2].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[3].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[4].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[5].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[6].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[7].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[8].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[9].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[10].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[11].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[12].as_ptr() as *const __m256i));
            state =
                _mm256_aesenc_epi128(state, _mm256_loadu_si256(rk[13].as_ptr() as *const __m256i));
            state = _mm256_aesenclast_epi128(
                state,
                _mm256_loadu_si256(rk[14].as_ptr() as *const __m256i),
            );

            // Davies-Meyer: XOR ciphertext with plaintext for one-wayness
            state = _mm256_xor_si256(state, inputs);

            let mut result = [0u8; 32];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, state);
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ColumnEncryption serialization --

    #[test]
    fn test_column_encryption_roundtrip() {
        let ce = ColumnEncryption {
            table_id: 42,
            column_id: 5,
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_id: 100,
        };
        let bytes = ce.to_bytes();
        let restored = ColumnEncryption::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.table_id, 42);
        assert_eq!(restored.column_id, 5);
        assert_eq!(restored.algorithm, EncryptionAlgorithm::Aes256Gcm);
        assert_eq!(restored.key_id, 100);
    }

    #[test]
    fn test_column_encryption_from_bytes_too_short() {
        assert!(ColumnEncryption::from_bytes(&[0u8; 5]).is_err());
    }

    // -- EncryptionStore --

    #[test]
    fn test_encryption_store_set_get() {
        let store = EncryptionStore::new();
        store.set_config(ColumnEncryption {
            table_id: 1,
            column_id: 2,
            algorithm: EncryptionAlgorithm::Aes128Gcm,
            key_id: 10,
        });
        let config = store.get_config(1, 2).expect("should exist");
        assert_eq!(config.key_id, 10);
    }

    #[test]
    fn test_encryption_store_remove() {
        let store = EncryptionStore::new();
        store.set_config(ColumnEncryption {
            table_id: 1,
            column_id: 2,
            algorithm: EncryptionAlgorithm::Aes128Gcm,
            key_id: 10,
        });
        assert!(store.remove_config(1, 2));
        assert!(store.get_config(1, 2).is_none());
    }

    // -- AES-GCM encrypt/decrypt --

    #[test]
    fn test_aes128_gcm_roundtrip() {
        let key = [0x42u8; 16];
        let plaintext = b"Hello, AES-GCM encryption test!";
        let encrypted =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
        assert!(encrypted.len() > plaintext.len());
        let decrypted =
            decrypt_value(&encrypted, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("decrypt");
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_aes256_gcm_roundtrip() {
        let key = [0xAB; 32];
        let plaintext = b"256-bit key test data";
        let encrypted =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("encrypt");
        let decrypted =
            decrypt_value(&encrypted, &key, EncryptionAlgorithm::Aes256Gcm, &[]).expect("decrypt");
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_aes_gcm_wrong_key_fails() {
        let key1 = [0x42u8; 16];
        let key2 = [0x43u8; 16];
        let plaintext = b"secret data";
        let encrypted =
            encrypt_value(plaintext, &key1, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
        assert!(decrypt_value(&encrypted, &key2, EncryptionAlgorithm::Aes128Gcm, &[]).is_err());
    }

    #[test]
    fn test_aes_gcm_tamper_detection() {
        let key = [0x42u8; 16];
        let plaintext = b"tamper test";
        let mut encrypted =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
        // Flip a bit in the ciphertext
        if encrypted.len() > GCM_NONCE_LEN + 1 {
            encrypted[GCM_NONCE_LEN] ^= 0x01;
        }
        assert!(decrypt_value(&encrypted, &key, EncryptionAlgorithm::Aes128Gcm, &[]).is_err());
    }

    #[test]
    fn test_aes_gcm_empty_plaintext() {
        let key = [0x42u8; 16];
        let plaintext = b"";
        let encrypted =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
        assert_eq!(encrypted.len(), GCM_NONCE_LEN + GCM_TAG_LEN);
        let decrypted =
            decrypt_value(&encrypted, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("decrypt");
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_aes_gcm_unique_nonces() {
        let key = [0x42u8; 16];
        let plaintext = b"same data";
        let e1 =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
        let e2 =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, &[]).expect("encrypt");
        // Different nonces produce different ciphertexts
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_aes_gcm_invalid_key_length() {
        let key = [0x42u8; 15]; // Wrong length for AES-128
        assert!(encrypt_value(b"data", &key, EncryptionAlgorithm::Aes128Gcm, &[]).is_err());
    }

    #[test]
    fn test_aes_gcm_too_short_ciphertext() {
        let key = [0x42u8; 16];
        assert!(decrypt_value(&[0u8; 10], &key, EncryptionAlgorithm::Aes128Gcm, &[]).is_err());
    }

    // -- AAD tests --

    #[test]
    fn test_aes_gcm_aad_roundtrip() {
        let key = [0x42u8; 16];
        let plaintext = b"column data";
        let aad = b"table_5_col_3";
        let encrypted =
            encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, aad).expect("encrypt");
        let decrypted =
            decrypt_value(&encrypted, &key, EncryptionAlgorithm::Aes128Gcm, aad).expect("decrypt");
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_aes_gcm_aad_mismatch_fails() {
        let key = [0x42u8; 16];
        let plaintext = b"column data";
        let encrypted = encrypt_value(
            plaintext,
            &key,
            EncryptionAlgorithm::Aes128Gcm,
            b"table_5_col_3",
        )
        .expect("encrypt");
        // Decrypt with different AAD should fail (tag mismatch)
        assert!(
            decrypt_value(
                &encrypted,
                &key,
                EncryptionAlgorithm::Aes128Gcm,
                b"table_5_col_7"
            )
            .is_err()
        );
    }

    #[test]
    fn test_aes_gcm_aad_vs_no_aad_fails() {
        let key = [0x42u8; 16];
        let plaintext = b"column data";
        let encrypted = encrypt_value(plaintext, &key, EncryptionAlgorithm::Aes128Gcm, b"some_aad")
            .expect("encrypt");
        // Decrypt with empty AAD should fail
        assert!(decrypt_value(&encrypted, &key, EncryptionAlgorithm::Aes128Gcm, &[]).is_err());
    }

    // -- LocalKeyStore --

    #[test]
    fn test_local_key_store_create_and_get() {
        let master = [0xAB; 32];
        let store = LocalKeyStore::new(master);
        let key_id = store
            .create_key(EncryptionAlgorithm::Aes256Gcm)
            .expect("create");
        let key = store.get_key(key_id).expect("get");
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_local_key_store_delete() {
        let master = [0xAB; 32];
        let store = LocalKeyStore::new(master);
        let key_id = store
            .create_key(EncryptionAlgorithm::Aes128Gcm)
            .expect("create");
        store.delete_key(key_id).expect("delete");
        assert!(store.get_key(key_id).is_err());
    }

    #[test]
    fn test_local_key_store_rotate() {
        let master = [0xAB; 32];
        let store = LocalKeyStore::new(master);
        let old_id = store
            .create_key(EncryptionAlgorithm::Aes256Gcm)
            .expect("create");
        let new_id = store.rotate_key(old_id).expect("rotate");
        assert_ne!(old_id, new_id);
        // Old key is deleted after rotation
        assert!(store.get_key(old_id).is_err());
        // New key exists
        let new_key = store.get_key(new_id).expect("new exists");
        assert_eq!(new_key.len(), 32);
    }

    #[test]
    fn test_local_key_store_get_nonexistent() {
        let master = [0xAB; 32];
        let store = LocalKeyStore::new(master);
        assert!(store.get_key(999).is_err());
    }

    // -- AES internals --

    #[test]
    fn test_gf_mul_identity() {
        assert_eq!(gf_mul(1, 0x53), 0x53);
    }

    #[test]
    fn test_gf_mul_known() {
        // GF(2^8) multiplication: 0x57 * 0x83 = 0xc1 (standard AES test)
        assert_eq!(gf_mul(0x57, 0x83), 0xc1);
    }

    #[test]
    fn test_aes_key_expansion_128() {
        let key = [0u8; 16];
        let round_keys = aes_key_expansion(&key, EncryptionAlgorithm::Aes128Gcm).expect("expand");
        assert_eq!(round_keys.len(), 176); // 11 round keys * 16 bytes
    }

    #[test]
    fn test_aes_key_expansion_256() {
        let key = [0u8; 32];
        let round_keys = aes_key_expansion(&key, EncryptionAlgorithm::Aes256Gcm).expect("expand");
        assert_eq!(round_keys.len(), 240); // 15 round keys * 16 bytes
    }

    #[test]
    fn test_constant_time_eq_equal() {
        assert!(constant_time_eq(&[1, 2, 3], &[1, 2, 3]));
    }

    #[test]
    fn test_constant_time_eq_not_equal() {
        assert!(!constant_time_eq(&[1, 2, 3], &[1, 2, 4]));
    }

    #[test]
    fn test_constant_time_eq_different_length() {
        assert!(!constant_time_eq(&[1, 2], &[1, 2, 3]));
    }
}
