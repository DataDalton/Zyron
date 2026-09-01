//! Media descriptor with a versioned binary codec
//!
//! A descriptor names a media value without committing to where the payload
//! lives. Inline mode carries the payload bytes directly after the header,
//! Toast and External modes carry only the content hash, ExternalUri carries
//! a uri and no local payload

use crate::error::{MediaError, MediaResult};

const MAGIC: &[u8; 4] = b"ZYMD";
const CODEC_VERSION: u16 = 1;

/// Broad classification of the media payload
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
    Document,
    ExternalRef,
}

impl MediaKind {
    fn to_code(self) -> u8 {
        match self {
            MediaKind::Image => 0,
            MediaKind::Video => 1,
            MediaKind::Audio => 2,
            MediaKind::Document => 3,
            MediaKind::ExternalRef => 4,
        }
    }

    fn from_code(code: u8) -> MediaResult<Self> {
        match code {
            0 => Ok(MediaKind::Image),
            1 => Ok(MediaKind::Video),
            2 => Ok(MediaKind::Audio),
            3 => Ok(MediaKind::Document),
            4 => Ok(MediaKind::ExternalRef),
            other => Err(MediaError::CorruptDescriptor(format!(
                "unknown media kind code {other}"
            ))),
        }
    }
}

/// Where the payload bytes for a descriptor live
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageMode {
    Inline,
    Toast,
    External,
    ExternalUri,
}

impl StorageMode {
    fn to_code(self) -> u8 {
        match self {
            StorageMode::Inline => 0,
            StorageMode::Toast => 1,
            StorageMode::External => 2,
            StorageMode::ExternalUri => 3,
        }
    }

    fn from_code(code: u8) -> MediaResult<Self> {
        match code {
            0 => Ok(StorageMode::Inline),
            1 => Ok(StorageMode::Toast),
            2 => Ok(StorageMode::External),
            3 => Ok(StorageMode::ExternalUri),
            other => Err(MediaError::CorruptDescriptor(format!(
                "unknown storage mode code {other}"
            ))),
        }
    }
}

/// Descriptor for one media value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaDescriptor {
    pub kind: MediaKind,
    pub mode: StorageMode,
    pub sha256: [u8; 32],
    pub byte_len: u64,
    pub format: String,
    pub metadata_json: String,
    pub uri: Option<String>,
}

impl MediaDescriptor {
    /// Encodes the descriptor header, appending the payload for Inline mode
    ///
    /// Inline mode requires a payload whose length matches byte_len, every
    /// other mode requires no payload
    pub fn to_bytes(&self, inline_payload: Option<&[u8]>) -> MediaResult<Vec<u8>> {
        match (self.mode, inline_payload) {
            (StorageMode::Inline, None) => {
                return Err(MediaError::InvalidArgument(
                    "inline descriptor requires the payload bytes".to_string(),
                ));
            }
            (StorageMode::Inline, Some(payload)) if payload.len() as u64 != self.byte_len => {
                return Err(MediaError::InvalidArgument(format!(
                    "inline payload is {} bytes but descriptor byte_len is {}",
                    payload.len(),
                    self.byte_len
                )));
            }
            (mode, Some(_)) if mode != StorageMode::Inline => {
                return Err(MediaError::InvalidArgument(
                    "only inline descriptors carry a payload".to_string(),
                ));
            }
            _ => {}
        }

        let mut out = Vec::with_capacity(
            64 + self.format.len()
                + self.metadata_json.len()
                + self.uri.as_ref().map_or(0, |u| u.len())
                + inline_payload.map_or(0, |p| p.len()),
        );
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&CODEC_VERSION.to_le_bytes());
        out.push(self.kind.to_code());
        out.push(self.mode.to_code());
        out.extend_from_slice(&self.sha256);
        out.extend_from_slice(&self.byte_len.to_le_bytes());
        write_string(&mut out, &self.format);
        write_string(&mut out, &self.metadata_json);
        match &self.uri {
            Some(uri) => {
                out.push(1);
                write_string(&mut out, uri);
            }
            None => out.push(0),
        }
        if let Some(payload) = inline_payload {
            out.extend_from_slice(payload);
        }
        Ok(out)
    }

    /// Decodes a descriptor, returning the inline payload when present
    pub fn from_bytes(bytes: &[u8]) -> MediaResult<(Self, Option<Vec<u8>>)> {
        let mut cursor = Cursor { bytes, pos: 0 };
        let magic = cursor.take(4)?;
        if magic != MAGIC {
            return Err(MediaError::CorruptDescriptor(
                "magic bytes are not ZYMD".to_string(),
            ));
        }
        let version = u16::from_le_bytes(fixed2(cursor.take(2)?));
        if version != CODEC_VERSION {
            return Err(MediaError::CorruptDescriptor(format!(
                "unknown descriptor version {version}"
            )));
        }
        let kind = MediaKind::from_code(cursor.take(1)?[0])?;
        let mode = StorageMode::from_code(cursor.take(1)?[0])?;
        let mut sha256 = [0u8; 32];
        sha256.copy_from_slice(cursor.take(32)?);
        let byte_len = u64::from_le_bytes(fixed8(cursor.take(8)?));
        let format = cursor.take_string()?;
        let metadata_json = cursor.take_string()?;
        let uri = match cursor.take(1)?[0] {
            0 => None,
            1 => Some(cursor.take_string()?),
            other => {
                return Err(MediaError::CorruptDescriptor(format!(
                    "invalid uri presence flag {other}"
                )));
            }
        };

        let descriptor = MediaDescriptor {
            kind,
            mode,
            sha256,
            byte_len,
            format,
            metadata_json,
            uri,
        };

        let rest = &bytes[cursor.pos..];
        match descriptor.mode {
            StorageMode::Inline => {
                if rest.len() as u64 != descriptor.byte_len {
                    return Err(MediaError::CorruptDescriptor(format!(
                        "inline payload is {} bytes but descriptor byte_len is {}",
                        rest.len(),
                        descriptor.byte_len
                    )));
                }
                Ok((descriptor, Some(rest.to_vec())))
            }
            _ => {
                if !rest.is_empty() {
                    return Err(MediaError::CorruptDescriptor(format!(
                        "{} trailing bytes after a non inline descriptor",
                        rest.len()
                    )));
                }
                Ok((descriptor, None))
            }
        }
    }
}

/// Cheap magic sniff for descriptor bytes
pub fn is_descriptor(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == MAGIC
}

/// Descriptor bytes for a bare uri reference, the form a text value written
/// into an EXTERNAL_REF column takes. The payload lives behind the uri, so
/// there is no content hash and no local byte length
pub fn uri_reference_bytes(uri: &str) -> MediaResult<Vec<u8>> {
    MediaDescriptor {
        kind: MediaKind::ExternalRef,
        mode: StorageMode::ExternalUri,
        sha256: [0u8; 32],
        byte_len: 0,
        format: "uri".to_string(),
        metadata_json: "{}".to_string(),
        uri: Some(uri.to_string()),
    }
    .to_bytes(None)
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn fixed2(bytes: &[u8]) -> [u8; 2] {
    let mut a = [0u8; 2];
    a.copy_from_slice(bytes);
    a
}

fn fixed8(bytes: &[u8]) -> [u8; 8] {
    let mut a = [0u8; 8];
    a.copy_from_slice(bytes);
    a
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> MediaResult<&'a [u8]> {
        if self.pos + n > self.bytes.len() {
            return Err(MediaError::CorruptDescriptor(format!(
                "descriptor truncated, needed {} bytes at offset {} of {}",
                n,
                self.pos,
                self.bytes.len()
            )));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn take_string(&mut self) -> MediaResult<String> {
        let len_bytes = self.take(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(len_bytes);
        let len = u32::from_le_bytes(a) as usize;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec())
            .map_err(|_| MediaError::CorruptDescriptor("string field is not utf8".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(mode: StorageMode) -> MediaDescriptor {
        MediaDescriptor {
            kind: MediaKind::Image,
            mode,
            sha256: [7u8; 32],
            byte_len: 5,
            format: "png".to_string(),
            metadata_json: "{\"width\":64}".to_string(),
            uri: None,
        }
    }

    #[test]
    fn inline_round_trip() {
        let desc = sample(StorageMode::Inline);
        let payload = b"hello";
        let bytes = desc.to_bytes(Some(payload)).expect("encode");
        assert!(is_descriptor(&bytes));
        let (decoded, decoded_payload) = MediaDescriptor::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, desc);
        assert_eq!(decoded_payload.expect("payload"), payload.to_vec());
    }

    #[test]
    fn toast_round_trip_with_uri() {
        let mut desc = sample(StorageMode::Toast);
        desc.uri = Some("s3://bucket/key".to_string());
        let bytes = desc.to_bytes(None).expect("encode");
        let (decoded, payload) = MediaDescriptor::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, desc);
        assert!(payload.is_none());
    }

    #[test]
    fn corruption_errors() {
        let desc = sample(StorageMode::Toast);
        let bytes = desc.to_bytes(None).expect("encode");

        assert!(MediaDescriptor::from_bytes(&bytes[..bytes.len() - 3]).is_err());
        assert!(MediaDescriptor::from_bytes(b"ZY").is_err());

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(MediaDescriptor::from_bytes(&bad_magic).is_err());
        assert!(!is_descriptor(&bad_magic));

        let mut bad_version = bytes.clone();
        bad_version[4] = 9;
        assert!(MediaDescriptor::from_bytes(&bad_version).is_err());

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(MediaDescriptor::from_bytes(&trailing).is_err());
    }

    #[test]
    fn inline_payload_length_enforced() {
        let desc = sample(StorageMode::Inline);
        assert!(desc.to_bytes(None).is_err());
        assert!(desc.to_bytes(Some(b"wrong length")).is_err());
        let toast = sample(StorageMode::Toast);
        assert!(toast.to_bytes(Some(b"hello")).is_err());
    }
}
