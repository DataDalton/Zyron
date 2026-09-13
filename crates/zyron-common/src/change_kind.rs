//! The kinds of change a table records, and the codes they travel as.
//!
//! One vocabulary shared by the change feed that writes them, the planner
//! that prunes on them and the executor that renders them. The code is what
//! goes on disk and in a summary's kind mask, and the name is what
//! `_change_type` shows and what a predicate compares against

/// A row that did not exist before this commit
pub const CHANGE_TYPE_INSERT: u8 = 0;
/// A changed row as it stood before this commit
pub const CHANGE_TYPE_UPDATE_PREIMAGE: u8 = 1;
/// A changed row as it stands after this commit
pub const CHANGE_TYPE_UPDATE_POSTIMAGE: u8 = 2;
/// A row that existed before this commit and does not after it
pub const CHANGE_TYPE_DELETE: u8 = 3;
/// A change to the table's shape rather than to its rows
pub const CHANGE_TYPE_SCHEMA_CHANGE: u8 = 4;
/// Every row removed at once
pub const CHANGE_TYPE_TRUNCATE: u8 = 5;

/// Every code, in order, so a caller can walk them
pub const CHANGE_TYPE_CODES: &[u8] = &[
    CHANGE_TYPE_INSERT,
    CHANGE_TYPE_UPDATE_PREIMAGE,
    CHANGE_TYPE_UPDATE_POSTIMAGE,
    CHANGE_TYPE_DELETE,
    CHANGE_TYPE_SCHEMA_CHANGE,
    CHANGE_TYPE_TRUNCATE,
];

/// The name a change kind shows under
pub fn change_type_label(code: u8) -> Option<&'static str> {
    match code {
        CHANGE_TYPE_INSERT => Some("insert"),
        CHANGE_TYPE_UPDATE_PREIMAGE => Some("update_preimage"),
        CHANGE_TYPE_UPDATE_POSTIMAGE => Some("update_postimage"),
        CHANGE_TYPE_DELETE => Some("delete"),
        CHANGE_TYPE_SCHEMA_CHANGE => Some("schema_change"),
        CHANGE_TYPE_TRUNCATE => Some("truncate"),
        _ => None,
    }
}

/// The code a name addresses, for a predicate that writes one
pub fn change_type_code(label: &str) -> Option<u8> {
    match label {
        "insert" => Some(CHANGE_TYPE_INSERT),
        "update_preimage" => Some(CHANGE_TYPE_UPDATE_PREIMAGE),
        "update_postimage" => Some(CHANGE_TYPE_UPDATE_POSTIMAGE),
        "delete" => Some(CHANGE_TYPE_DELETE),
        "schema_change" => Some(CHANGE_TYPE_SCHEMA_CHANGE),
        "truncate" => Some(CHANGE_TYPE_TRUNCATE),
        _ => None,
    }
}

/// The bit a kind occupies in a change window's kind mask
#[inline]
pub const fn change_type_bit(code: u8) -> u8 {
    1u8 << code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_code_names_itself_back() {
        for code in CHANGE_TYPE_CODES {
            let label = change_type_label(*code).expect("every code has a name");
            assert_eq!(change_type_code(label), Some(*code));
        }
        assert_eq!(change_type_label(99), None);
        assert_eq!(change_type_code("nothing"), None);
    }

    #[test]
    fn test_every_kind_takes_its_own_bit() {
        let mut seen = 0u8;
        for code in CHANGE_TYPE_CODES {
            let bit = change_type_bit(*code);
            assert_eq!(seen & bit, 0, "code {code} shares a bit");
            seen |= bit;
        }
    }
}
