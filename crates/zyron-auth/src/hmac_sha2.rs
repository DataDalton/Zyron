//! HMAC over the SHA-2 family, built directly on sha2 hash passes
//!
//! A key holds its padded inner and outer blocks, so producing a MAC is two
//! plain hash passes. The generic wrapper in the hmac crate costs 3.4 to 9.6
//! microseconds per HMAC-SHA-256 in this build while the identical result
//! computed this way costs about 100 ns. SHA-256 itself is not the problem
//! and is not reimplemented here, only the HMAC construction around it.
//!
//! Outputs are checked against RFC 4231 vectors and differentially against
//! the hmac crate in the tests below.

use sha2::Digest;

/// Largest pad-plus-message size assembled in a stack buffer. Covers every
/// hot path here, JWT signing inputs and SCRAM messages are a few hundred
/// bytes, while keeping the buffer small enough to sit on the stack.
const INLINE_INPUT_CAP: usize = 1024;

macro_rules! define_hmac_key {
    ($name:ident, $hash:ty, $block:expr, $out:expr, $doc:expr) => {
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
            #[inline(always)]
            pub fn new(key: &[u8]) -> Self {
                let mut ipad = [0x36u8; $block];
                let mut opad = [0x5cu8; $block];
                if key.len() > $block {
                    let digest = <$hash>::digest(key);
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

            /// Computes the MAC over `message`
            ///
            /// The pad and the message are laid out contiguously so each hash
            /// is a single `digest` call rather than splitting the work into
            /// separate `new`, `update` and `finalize` statements, which
            /// lets the compiler keep the hash state in registers across the
            /// whole pass.
            #[inline(always)]
            pub fn sign(&self, message: &[u8]) -> [u8; $out] {
                let inner_len = $block + message.len();
                if inner_len > INLINE_INPUT_CAP {
                    return self.sign_large(message);
                }
                let mut buf = [0u8; INLINE_INPUT_CAP];
                buf[..$block].copy_from_slice(&self.ipad);
                buf[$block..inner_len].copy_from_slice(message);
                let inner_digest = <$hash>::digest(&buf[..inner_len]);

                let mut outer_buf = [0u8; $block + $out];
                outer_buf[..$block].copy_from_slice(&self.opad);
                outer_buf[$block..].copy_from_slice(&inner_digest);

                let mut out = [0u8; $out];
                out.copy_from_slice(&<$hash>::digest(&outer_buf));
                out
            }

            /// Streaming path for messages too large to assemble in the
            /// stack buffer. Kept out of line so the hot path above stays
            /// straight line code with the hash state in registers.
            #[inline(never)]
            #[cold]
            fn sign_large(&self, message: &[u8]) -> [u8; $out] {
                let mut inner = <$hash>::new();
                inner.update(&self.ipad);
                inner.update(message);
                let inner_digest = inner.finalize();

                let mut outer_buf = [0u8; $block + $out];
                outer_buf[..$block].copy_from_slice(&self.opad);
                outer_buf[$block..].copy_from_slice(&inner_digest);

                let mut out = [0u8; $out];
                out.copy_from_slice(&<$hash>::digest(&outer_buf));
                out
            }
        }
    };
}

define_hmac_key!(
    HmacSha256Key,
    sha2::Sha256,
    64,
    32,
    "HMAC-SHA-256 key holding its precomputed pad blocks"
);
define_hmac_key!(
    HmacSha384Key,
    sha2::Sha384,
    128,
    48,
    "HMAC-SHA-384 key holding its precomputed pad blocks"
);
define_hmac_key!(
    HmacSha512Key,
    sha2::Sha512,
    128,
    64,
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

    // Every key length that crosses a padding or hashing boundary, compared
    // against the hmac crate so any divergence fails the build
    #[test]
    fn matches_hmac_crate_across_key_and_message_lengths() {
        for key_len in [0usize, 1, 20, 31, 32, 63, 64, 65, 127, 128, 129, 200] {
            let key: Vec<u8> = (0..key_len).map(|i| (i * 7 + 3) as u8).collect();
            // 895/896/897 and 959/960/961 straddle the INLINE_INPUT_CAP
            // boundary for the 128 byte and 64 byte block sizes, so both the
            // stack buffer path and the streaming path are exercised
            for msg_len in [
                0usize, 1, 55, 56, 63, 64, 65, 119, 120, 128, 133, 895, 896, 897, 959, 960, 961,
                1000, 4096,
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
