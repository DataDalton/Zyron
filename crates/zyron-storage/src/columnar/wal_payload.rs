// -----------------------------------------------------------------------------
// Columnar WAL payload codec
// -----------------------------------------------------------------------------
//
// Single source of truth for the byte layouts of the columnar WAL record
// payloads. The DML, savepoint, branch and streaming writers and the crash
// recovery reader all go through these structs, so the layouts cannot drift
// between producers and the replay path.
//
// All fields are little-endian. Layouts:
//   Supersede / SupersedeRevoke  40 bytes      table, branch, file, rowid, xid
//   ValuePatch                   48 + value    table, branch, file, rowid,
//                                              col, xid, len, value bytes
//   ValuePatchRevoke             44 bytes      table, branch, file, rowid,
//                                              col, xid
//   BranchClear                  16 bytes      table, branch

fn rd_u64(b: &[u8], off: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(w)
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(&b[off..off + 4]);
    u32::from_le_bytes(w)
}

/// Payload of a ColumnarSupersede or ColumnarSupersedeRevoke record. The two
/// record kinds share one layout, the WAL record type distinguishes them
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnarSupersedePayload {
    pub table_id: u64,
    pub branch: u64,
    pub file_id: u64,
    pub sys_rowid: u64,
    pub xid: u64,
}

impl ColumnarSupersedePayload {
    pub const LEN: usize = 40;

    pub fn encode(&self) -> Vec<u8> {
        let mut pl = Vec::with_capacity(Self::LEN);
        pl.extend_from_slice(&self.table_id.to_le_bytes());
        pl.extend_from_slice(&self.branch.to_le_bytes());
        pl.extend_from_slice(&self.file_id.to_le_bytes());
        pl.extend_from_slice(&self.sys_rowid.to_le_bytes());
        pl.extend_from_slice(&self.xid.to_le_bytes());
        pl
    }

    pub fn decode(p: &[u8]) -> Option<Self> {
        if p.len() < Self::LEN {
            return None;
        }
        Some(Self {
            table_id: rd_u64(p, 0),
            branch: rd_u64(p, 8),
            file_id: rd_u64(p, 16),
            sys_rowid: rd_u64(p, 24),
            xid: rd_u64(p, 32),
        })
    }
}

/// Header of a ColumnarPatch record. The value bytes follow the 48-byte
/// header, their length recorded in the len field at offset 44
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnarValuePatchPayload {
    pub table_id: u64,
    pub branch: u64,
    pub file_id: u64,
    pub sys_rowid: u64,
    pub column_id: u32,
    pub xid: u64,
}

impl ColumnarValuePatchPayload {
    pub const HEADER_LEN: usize = 48;

    pub fn encode_with_value(&self, value: &[u8]) -> Vec<u8> {
        let mut pl = Vec::with_capacity(Self::HEADER_LEN + value.len());
        pl.extend_from_slice(&self.table_id.to_le_bytes());
        pl.extend_from_slice(&self.branch.to_le_bytes());
        pl.extend_from_slice(&self.file_id.to_le_bytes());
        pl.extend_from_slice(&self.sys_rowid.to_le_bytes());
        pl.extend_from_slice(&self.column_id.to_le_bytes());
        pl.extend_from_slice(&self.xid.to_le_bytes());
        pl.extend_from_slice(&(value.len() as u32).to_le_bytes());
        pl.extend_from_slice(value);
        pl
    }

    /// Returns the header and the borrowed value bytes
    pub fn decode(p: &[u8]) -> Option<(Self, &[u8])> {
        if p.len() < Self::HEADER_LEN {
            return None;
        }
        let vlen = rd_u32(p, 44) as usize;
        if Self::HEADER_LEN + vlen > p.len() {
            return None;
        }
        Some((
            Self {
                table_id: rd_u64(p, 0),
                branch: rd_u64(p, 8),
                file_id: rd_u64(p, 16),
                sys_rowid: rd_u64(p, 24),
                column_id: rd_u32(p, 32),
                xid: rd_u64(p, 36),
            },
            &p[Self::HEADER_LEN..Self::HEADER_LEN + vlen],
        ))
    }
}

/// Payload of a ColumnarPatchRevoke record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnarPatchRevokePayload {
    pub table_id: u64,
    pub branch: u64,
    pub file_id: u64,
    pub sys_rowid: u64,
    pub column_id: u32,
    pub xid: u64,
}

impl ColumnarPatchRevokePayload {
    pub const LEN: usize = 44;

    pub fn encode(&self) -> Vec<u8> {
        let mut pl = Vec::with_capacity(Self::LEN);
        pl.extend_from_slice(&self.table_id.to_le_bytes());
        pl.extend_from_slice(&self.branch.to_le_bytes());
        pl.extend_from_slice(&self.file_id.to_le_bytes());
        pl.extend_from_slice(&self.sys_rowid.to_le_bytes());
        pl.extend_from_slice(&self.column_id.to_le_bytes());
        pl.extend_from_slice(&self.xid.to_le_bytes());
        pl
    }

    pub fn decode(p: &[u8]) -> Option<Self> {
        if p.len() < Self::LEN {
            return None;
        }
        Some(Self {
            table_id: rd_u64(p, 0),
            branch: rd_u64(p, 8),
            file_id: rd_u64(p, 16),
            sys_rowid: rd_u64(p, 24),
            column_id: rd_u32(p, 32),
            xid: rd_u64(p, 36),
        })
    }
}

/// Payload of a ColumnarBranchClear record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnarBranchClearPayload {
    pub table_id: u64,
    pub branch: u64,
}

impl ColumnarBranchClearPayload {
    pub const LEN: usize = 16;

    pub fn encode(&self) -> Vec<u8> {
        let mut pl = Vec::with_capacity(Self::LEN);
        pl.extend_from_slice(&self.table_id.to_le_bytes());
        pl.extend_from_slice(&self.branch.to_le_bytes());
        pl
    }

    pub fn decode(p: &[u8]) -> Option<Self> {
        if p.len() < Self::LEN {
            return None;
        }
        Some(Self {
            table_id: rd_u64(p, 0),
            branch: rd_u64(p, 8),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supersede_payload_roundtrip() {
        let p = ColumnarSupersedePayload {
            table_id: 7,
            branch: 3,
            file_id: 11,
            sys_rowid: 42,
            xid: 900,
        };
        let bytes = p.encode();
        assert_eq!(bytes.len(), ColumnarSupersedePayload::LEN);
        assert_eq!(ColumnarSupersedePayload::decode(&bytes), Some(p));
        assert_eq!(ColumnarSupersedePayload::decode(&bytes[..39]), None);
    }

    #[test]
    fn test_value_patch_payload_roundtrip() {
        let h = ColumnarValuePatchPayload {
            table_id: 7,
            branch: 0,
            file_id: 11,
            sys_rowid: 42,
            column_id: 5,
            xid: 900,
        };
        let bytes = h.encode_with_value(b"hello");
        assert_eq!(bytes.len(), ColumnarValuePatchPayload::HEADER_LEN + 5);
        let (dh, val) = ColumnarValuePatchPayload::decode(&bytes).expect("decode");
        assert_eq!(dh, h);
        assert_eq!(val, b"hello");
        // truncated value region rejects
        assert!(ColumnarValuePatchPayload::decode(&bytes[..bytes.len() - 1]).is_none());
    }

    #[test]
    fn test_patch_revoke_payload_roundtrip() {
        let p = ColumnarPatchRevokePayload {
            table_id: 7,
            branch: 1,
            file_id: 11,
            sys_rowid: 42,
            column_id: 5,
            xid: 900,
        };
        let bytes = p.encode();
        assert_eq!(bytes.len(), ColumnarPatchRevokePayload::LEN);
        assert_eq!(ColumnarPatchRevokePayload::decode(&bytes), Some(p));
    }

    #[test]
    fn test_branch_clear_payload_roundtrip() {
        let p = ColumnarBranchClearPayload {
            table_id: 7,
            branch: 9,
        };
        let bytes = p.encode();
        assert_eq!(bytes.len(), ColumnarBranchClearPayload::LEN);
        assert_eq!(ColumnarBranchClearPayload::decode(&bytes), Some(p));
    }
}
