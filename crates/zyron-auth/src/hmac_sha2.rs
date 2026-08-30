//! HMAC over the SHA-2 family, built directly on sha2's compression
//! functions
//!
//! A key holds its padded inner and outer blocks, so producing a MAC is two
//! hash passes. Both passes drive `sha2::compress256`/`compress512`, the
//! crate's single non-generic compression symbols with runtime hardware
//! dispatch, and the Merkle-Damgard padding around them is implemented here
//! per FIPS 180-4. The generic `digest` wrapper is deliberately absent from
//! this path: its per-crate LTO instantiations proved pathological, costing
//! microseconds per call where the compression itself costs nanoseconds,
//! and which copy a call site received varied build to build. SHA-2 itself
//! is not reimplemented here, only the padding and the HMAC construction.
//!
//! Outputs are checked against RFC 4231 vectors and differentially against
//! the hmac crate in the tests below, across every padding boundary.

use sha2::digest::generic_array::GenericArray;

/// FIPS 180-4 initial hash values
const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const SHA384_IV: [u64; 8] = [
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4,
];
const SHA512_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// Runs one 64 byte block through the SHA-256 compression
///
/// The `inline(never)` is load-bearing. When LLVM inlines sha2's compress
/// dispatch chain into a larger caller under fat LTO, the resulting code
/// runs two orders of magnitude slower than the same chain compiled in a
/// small function of its own, measured at 12.7us versus 98ns for a six
/// block HMAC. Keeping the per-block call out of line pins the sane
/// compilation on every build instead of rolling inlining dice
#[inline(never)]
fn compress_block_256(state: &mut [u32; 8], block: &[u8]) {
    sha2::compress256(
        state,
        core::slice::from_ref(GenericArray::from_slice(block)),
    );
}

/// Runs one 128 byte block through the SHA-512 compression. The
/// `inline(never)` is load-bearing, see `compress_block_256`
#[inline(never)]
fn compress_block_512(state: &mut [u64; 8], block: &[u8]) {
    sha2::compress512(
        state,
        core::slice::from_ref(GenericArray::from_slice(block)),
    );
}

/// One SHA-256 pass over `pad || message`, or over `message` alone when no
/// pad block is given. The pad is exactly one block, so the stream is the
/// pad block, the message's whole blocks, then the final padded block or
/// two carrying the 0x80 terminator and the 64 bit big-endian bit length
fn sha256_pass(pad: Option<&[u8; 64]>, message: &[u8]) -> [u8; 32] {
    let mut state = SHA256_IV;
    let mut total = message.len();
    if let Some(pad) = pad {
        compress_block_256(&mut state, pad);
        total += 64;
    }
    let mut chunks = message.chunks_exact(64);
    for block in chunks.by_ref() {
        compress_block_256(&mut state, block);
    }
    let rem = chunks.remainder();

    let mut tail = [0u8; 128];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    // The length field needs 8 bytes after the terminator; when the
    // remainder leaves no room the padding spills into a second block
    let end = if rem.len() + 1 + 8 <= 64 { 64 } else { 128 };
    let bits = (total as u64) * 8;
    tail[end - 8..end].copy_from_slice(&bits.to_be_bytes());
    for block in tail[..end].chunks_exact(64) {
        compress_block_256(&mut state, block);
    }

    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// One SHA-384/SHA-512 pass over `pad || message`, parameterized by the
/// initial hash values and the output length, which is how the two family
/// members differ. The length field is 128 bits in this family
fn sha512_pass<const OUT: usize>(
    iv: [u64; 8],
    pad: Option<&[u8; 128]>,
    message: &[u8],
) -> [u8; OUT] {
    let mut state = iv;
    let mut total = message.len();
    if let Some(pad) = pad {
        compress_block_512(&mut state, pad);
        total += 128;
    }
    let mut chunks = message.chunks_exact(128);
    for block in chunks.by_ref() {
        compress_block_512(&mut state, block);
    }
    let rem = chunks.remainder();

    let mut tail = [0u8; 256];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    let end = if rem.len() + 1 + 16 <= 128 { 128 } else { 256 };
    let bits = (total as u128) * 8;
    tail[end - 16..end].copy_from_slice(&bits.to_be_bytes());
    for block in tail[..end].chunks_exact(128) {
        compress_block_512(&mut state, block);
    }

    let mut out = [0u8; OUT];
    for i in 0..OUT / 8 {
        out[i * 8..i * 8 + 8].copy_from_slice(&state[i].to_be_bytes());
    }
    out
}

fn hmac_pass_256(pad: &[u8; 64], message: &[u8]) -> [u8; 32] {
    sha256_pass(Some(pad), message)
}
fn sha256_plain(message: &[u8]) -> [u8; 32] {
    sha256_pass(None, message)
}
fn hmac_pass_384(pad: &[u8; 128], message: &[u8]) -> [u8; 48] {
    sha512_pass::<48>(SHA384_IV, Some(pad), message)
}
fn sha384_plain(message: &[u8]) -> [u8; 48] {
    sha512_pass::<48>(SHA384_IV, None, message)
}
fn hmac_pass_512(pad: &[u8; 128], message: &[u8]) -> [u8; 64] {
    sha512_pass::<64>(SHA512_IV, Some(pad), message)
}
fn sha512_plain(message: &[u8]) -> [u8; 64] {
    sha512_pass::<64>(SHA512_IV, None, message)
}

macro_rules! define_hmac_key {
    ($name:ident, $block:expr, $out:expr, $pass:path, $plain:path, $doc:expr) => {
        #[doc = $doc]
        #[derive(Clone)]
        pub struct $name {
            ipad: [u8; $block],
            opad: [u8; $block],
        }

        impl $name {
            /// Derives the padded key blocks. Per RFC 2104 a key longer than
            /// the hash block size is replaced by its own digest, and a
            /// shorter key is zero padded, which the pad fill already covers
            pub fn new(key: &[u8]) -> Self {
                let mut ipad = [0x36u8; $block];
                let mut opad = [0x5cu8; $block];
                if key.len() > $block {
                    let digest = $plain(key);
                    for (i, byte) in digest.iter().enumerate() {
                        ipad[i] ^= byte;
                        opad[i] ^= byte;
                    }
                } else {
                    for (i, byte) in key.iter().enumerate() {
                        ipad[i] ^= byte;
                        opad[i] ^= byte;
                    }
                }
                Self { ipad, opad }
            }

            /// Computes the MAC over `message`: the inner pass hashes the
            /// inner pad block then the message, the outer pass hashes the
            /// outer pad block then the inner digest
            pub fn sign(&self, message: &[u8]) -> [u8; $out] {
                let inner = $pass(&self.ipad, message);
                $pass(&self.opad, &inner)
            }
        }
    };
}

define_hmac_key!(
    HmacSha256Key,
    64,
    32,
    hmac_pass_256,
    sha256_plain,
    "HMAC-SHA-256 key holding its precomputed pad blocks"
);
define_hmac_key!(
    HmacSha384Key,
    128,
    48,
    hmac_pass_384,
    sha384_plain,
    "HMAC-SHA-384 key holding its precomputed pad blocks"
);
define_hmac_key!(
    HmacSha512Key,
    128,
    64,
    hmac_pass_512,
    sha512_plain,
    "HMAC-SHA-512 key holding its precomputed pad blocks"
);

/// HMAC-SHA-256 for callers that do not retain a key between calls
#[inline(always)]
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    HmacSha256Key::new(key).sign(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::Mac;
    use sha2::Digest;

    fn to_hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }

    // RFC 4231 section 4.2, 20 byte key
    #[test]
    fn rfc4231_case1() {
        let key = [0x0bu8; 20];
        let data = b"Hi There";
        assert_eq!(
            to_hex(&HmacSha256Key::new(&key).sign(data)),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            to_hex(&HmacSha384Key::new(&key).sign(data)),
            "afd03944d84895626b0825f4ab46907f15f9dadbe4101ec682aa034c7cebc59c\
             faea9ea9076ede7f4af152e8b2fa9cb6"
        );
        assert_eq!(
            to_hex(&HmacSha512Key::new(&key).sign(data)),
            "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde\
             daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854"
        );
    }

    // RFC 4231 section 4.3, key shorter than the block size
    #[test]
    fn rfc4231_case2() {
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        assert_eq!(
            to_hex(&HmacSha256Key::new(key).sign(data)),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            to_hex(&HmacSha384Key::new(key).sign(data)),
            "af45d2e376484031617f78d2b58a6b1b9c7ef464f5a01b47e42ec3736322445e\
             8e2240ca5e69e2c78b3239ecfab21649"
        );
        assert_eq!(
            to_hex(&HmacSha512Key::new(key).sign(data)),
            "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554\
             9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737"
        );
    }

    // RFC 4231 section 4.8, 131 byte key exercises the long key branch where
    // the key is hashed before padding
    #[test]
    fn rfc4231_case7_long_key() {
        let key = [0xaau8; 131];
        let data = b"This is a test using a larger than block-size key and a \
larger than block-size data. The key needs to be hashed before being used by \
the HMAC algorithm.";
        assert_eq!(
            to_hex(&HmacSha256Key::new(&key).sign(data)),
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
        );
        assert_eq!(
            to_hex(&HmacSha512Key::new(&key).sign(data)),
            "e37b6a775dc87dbaa4dfa9f96e5e3ffddebd71f8867289865df5a32d20cdc944\
             b6022cac3c4982b10d5eeb55c3e4de15134676fb6de0446065c97440fa8c6a58"
        );
    }

    // The hand-rolled padding against sha2's own one-shot digests, at every
    // length that crosses a final-block boundary in either family
    #[test]
    fn plain_digests_match_sha2() {
        for len in [
            0usize, 1, 54, 55, 56, 57, 63, 64, 65, 110, 111, 112, 113, 119, 120, 127, 128, 129,
            255, 256, 257, 1000, 4096,
        ] {
            let data: Vec<u8> = (0..len).map(|i| (i * 13 + 7) as u8).collect();
            assert_eq!(
                sha256_plain(&data)[..],
                sha2::Sha256::digest(&data)[..],
                "SHA-256 mismatch at len {len}"
            );
            assert_eq!(
                sha384_plain(&data)[..],
                sha2::Sha384::digest(&data)[..],
                "SHA-384 mismatch at len {len}"
            );
            assert_eq!(
                sha512_plain(&data)[..],
                sha2::Sha512::digest(&data)[..],
                "SHA-512 mismatch at len {len}"
            );
        }
    }

    // Every key length that crosses a padding or hashing boundary, compared
    // against the hmac crate so any divergence fails the build. The message
    // lengths cover both families' final-block boundaries, 55/56 for the 64
    // byte block with its 8 byte length field and 111/112 for the 128 byte
    // block with its 16 byte length field, plus multi-block sizes
    #[test]
    fn matches_hmac_crate_across_key_and_message_lengths() {
        for key_len in [0usize, 1, 20, 31, 32, 63, 64, 65, 127, 128, 129, 200] {
            let key: Vec<u8> = (0..key_len).map(|i| (i * 7 + 3) as u8).collect();
            for msg_len in [
                0usize, 1, 55, 56, 63, 64, 65, 111, 112, 113, 119, 120, 127, 128, 129, 133, 895,
                896, 897, 959, 960, 961, 1000, 4096,
            ] {
                let msg: Vec<u8> = (0..msg_len).map(|i| (i * 11 + 5) as u8).collect();

                let mut reference =
                    <hmac::Hmac<sha2::Sha256> as Mac>::new_from_slice(&key).expect("any key len");
                reference.update(&msg);
                assert_eq!(
                    HmacSha256Key::new(&key).sign(&msg)[..],
                    reference.finalize().into_bytes()[..],
                    "HMAC-SHA-256 mismatch at key_len {key_len} msg_len {msg_len}"
                );

                let mut reference =
                    <hmac::Hmac<sha2::Sha384> as Mac>::new_from_slice(&key).expect("any key len");
                reference.update(&msg);
                assert_eq!(
                    HmacSha384Key::new(&key).sign(&msg)[..],
                    reference.finalize().into_bytes()[..],
                    "HMAC-SHA-384 mismatch at key_len {key_len} msg_len {msg_len}"
                );

                let mut reference =
                    <hmac::Hmac<sha2::Sha512> as Mac>::new_from_slice(&key).expect("any key len");
                reference.update(&msg);
                assert_eq!(
                    HmacSha512Key::new(&key).sign(&msg)[..],
                    reference.finalize().into_bytes()[..],
                    "HMAC-SHA-512 mismatch at key_len {key_len} msg_len {msg_len}"
                );
            }
        }
    }

    #[test]
    fn free_function_matches_key_type() {
        let key = b"a shared secret value";
        let msg = b"payload";
        assert_eq!(hmac_sha256(key, msg), HmacSha256Key::new(key).sign(msg));
    }
}
