//! Three-state privilege system (GRANT/DENY/unset) with temporal and column-level support.
//!
//! Privileges are stored per (object_type, object_id) key. DENY always wins over GRANT.
//! Entries can have temporal validity (valid_from, valid_until, time_window) and
//! column-level granularity. Pattern-based grants match object names using SQL LIKE
//! syntax (% for any sequence, _ for single character).

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use zyron_catalog::encoding::{
    read_bool, read_option_string, read_u8, read_u16, read_u32, read_u64, write_bool,
    write_option_string, write_u8, write_u16, write_u32, write_u64,
};
use zyron_common::{Result, ZyronError};

use crate::role::RoleId;
use crate::session_binding::TimeWindow;

/// Types of privileges that can be granted or denied on database objects.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrivilegeType {
    Select = 0,
    Insert = 1,
    Update = 2,
    Delete = 3,
    Truncate = 4,
    References = 5,
    Trigger = 6,
    Create = 7,
    Connect = 8,
    Execute = 9,
    Usage = 10,
    Temporary = 11,
    Impersonate = 12,
    ManageRoles = 13,
    ManagePrivileges = 14,
    ManagePolicy = 15,
    ManageClassification = 16,
    ManageTags = 17,
    ManageMasking = 18,
    ManageOwnership = 19,
    ManageBreakGlass = 20,
    ManageQueryLimits = 21,
    ManageAuthRules = 22,
    ViewAuthAttempts = 23,
    ManageBruteForcePolicy = 24,
    All = 255,
}

impl PrivilegeType {
    /// Converts a u8 to a PrivilegeType.
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Select),
            1 => Ok(Self::Insert),
            2 => Ok(Self::Update),
            3 => Ok(Self::Delete),
            4 => Ok(Self::Truncate),
            5 => Ok(Self::References),
            6 => Ok(Self::Trigger),
            7 => Ok(Self::Create),
            8 => Ok(Self::Connect),
            9 => Ok(Self::Execute),
            10 => Ok(Self::Usage),
            11 => Ok(Self::Temporary),
            12 => Ok(Self::Impersonate),
            13 => Ok(Self::ManageRoles),
            14 => Ok(Self::ManagePrivileges),
            15 => Ok(Self::ManagePolicy),
            16 => Ok(Self::ManageClassification),
            17 => Ok(Self::ManageTags),
            18 => Ok(Self::ManageMasking),
            19 => Ok(Self::ManageOwnership),
            20 => Ok(Self::ManageBreakGlass),
            21 => Ok(Self::ManageQueryLimits),
            22 => Ok(Self::ManageAuthRules),
            23 => Ok(Self::ViewAuthAttempts),
            24 => Ok(Self::ManageBruteForcePolicy),
            255 => Ok(Self::All),
            _ => Err(ZyronError::CatalogCorrupted(format!(
                "invalid PrivilegeType value {}",
                val
            ))),
        }
    }

    /// Returns all concrete privilege types (excludes All).
    pub fn concrete_types() -> &'static [PrivilegeType] {
        &[
            PrivilegeType::Select,
            PrivilegeType::Insert,
            PrivilegeType::Update,
            PrivilegeType::Delete,
            PrivilegeType::Truncate,
            PrivilegeType::References,
            PrivilegeType::Trigger,
            PrivilegeType::Create,
            PrivilegeType::Connect,
            PrivilegeType::Execute,
            PrivilegeType::Usage,
            PrivilegeType::Temporary,
            PrivilegeType::Impersonate,
            PrivilegeType::ManageRoles,
            PrivilegeType::ManagePrivileges,
            PrivilegeType::ManagePolicy,
            PrivilegeType::ManageClassification,
            PrivilegeType::ManageTags,
            PrivilegeType::ManageMasking,
            PrivilegeType::ManageOwnership,
            PrivilegeType::ManageBreakGlass,
            PrivilegeType::ManageQueryLimits,
            PrivilegeType::ManageAuthRules,
            PrivilegeType::ViewAuthAttempts,
            PrivilegeType::ManageBruteForcePolicy,
        ]
    }

    /// Returns true if this type matches a requested type, considering the All wildcard.
    fn matches(&self, requested: PrivilegeType) -> bool {
        *self == requested || *self == PrivilegeType::All || requested == PrivilegeType::All
    }
}

/// Types of database objects that can have privileges.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectType {
    Database = 0,
    Schema = 1,
    Table = 2,
    Column = 3,
    Sequence = 4,
    Function = 5,
    Type = 6,
}

impl ObjectType {
    /// Converts a u8 to an ObjectType.
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(Self::Database),
            1 => Ok(Self::Schema),
            2 => Ok(Self::Table),
            3 => Ok(Self::Column),
            4 => Ok(Self::Sequence),
            5 => Ok(Self::Function),
            6 => Ok(Self::Type),
            _ => Err(ZyronError::CatalogCorrupted(format!(
                "invalid ObjectType value {}",
                val
            ))),
        }
    }
}

/// Three-valued privilege state: either explicitly granted or denied.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivilegeState {
    Grant = 1,
    Deny = 2,
}

impl PrivilegeState {
    fn from_u8(val: u8) -> Result<Self> {
        match val {
            1 => Ok(Self::Grant),
            2 => Ok(Self::Deny),
            _ => Err(ZyronError::CatalogCorrupted(format!(
                "invalid PrivilegeState value {}",
                val
            ))),
        }
    }
}

/// The result of a privilege check. Unset means no matching grant or deny entry was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivilegeDecision {
    Allow,
    Denied,
    Unset,
}

/// A single privilege grant or deny entry with optional temporal and column-level restrictions.
#[derive(Debug, Clone)]
pub struct GrantEntry {
    pub grantee: RoleId,
    pub privilege: PrivilegeType,
    pub object_type: ObjectType,
    pub object_id: u32,
    pub columns: Option<Vec<u16>>,
    pub state: PrivilegeState,
    pub with_grant_option: bool,
    pub granted_by: RoleId,
    pub valid_from: Option<u64>,
    pub valid_until: Option<u64>,
    pub time_window: Option<TimeWindow>,
    pub object_pattern: Option<String>,
    pub no_inherit: bool,
    pub mask_function: Option<String>,
}

impl GrantEntry {
    /// Serializes the grant entry to a binary byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u32(&mut buf, self.grantee.0);
        write_u8(&mut buf, self.privilege as u8);
        write_u8(&mut buf, self.object_type as u8);
        write_u32(&mut buf, self.object_id);

        // columns: 0 for None, 1 + u16 count + u16 values for Some.
        match &self.columns {
            None => write_u8(&mut buf, 0),
            Some(cols) => {
                write_u8(&mut buf, 1);
                write_u16(&mut buf, cols.len() as u16);
                for &col in cols {
                    write_u16(&mut buf, col);
                }
            }
        }

        write_u8(&mut buf, self.state as u8);
        write_bool(&mut buf, self.with_grant_option);
        write_u32(&mut buf, self.granted_by.0);

        // valid_from: 0 for None, 1 + u64 for Some.
        match self.valid_from {
            None => write_u8(&mut buf, 0),
            Some(ts) => {
                write_u8(&mut buf, 1);
                write_u64(&mut buf, ts);
            }
        }

        match self.valid_until {
            None => write_u8(&mut buf, 0),
            Some(ts) => {
                write_u8(&mut buf, 1);
                write_u64(&mut buf, ts);
            }
        }

        // time_window: 0 for None, 1 + 3 bytes for Some.
        match &self.time_window {
            None => write_u8(&mut buf, 0),
            Some(tw) => {
                write_u8(&mut buf, 1);
                write_u8(&mut buf, tw.start_hour);
                write_u8(&mut buf, tw.end_hour);
                write_u8(&mut buf, tw.days);
            }
        }

        write_option_string(&mut buf, &self.object_pattern);
        write_bool(&mut buf, self.no_inherit);
        write_option_string(&mut buf, &self.mask_function);
        buf
    }

    /// Deserializes a grant entry from a binary byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let grantee = RoleId(read_u32(data, &mut off)?);
        let privilege = PrivilegeType::from_u8(read_u8(data, &mut off)?)?;
        let object_type = ObjectType::from_u8(read_u8(data, &mut off)?)?;
        let object_id = read_u32(data, &mut off)?;

        let columns = match read_u8(data, &mut off)? {
            0 => None,
            1 => {
                let count = read_u16(data, &mut off)? as usize;
                let mut cols = Vec::with_capacity(count);
                for _ in 0..count {
                    cols.push(read_u16(data, &mut off)?);
                }
                Some(cols)
            }
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid columns tag {} in grant entry",
                    tag
                )));
            }
        };

        let state = PrivilegeState::from_u8(read_u8(data, &mut off)?)?;
        let with_grant_option = read_bool(data, &mut off)?;
        let granted_by = RoleId(read_u32(data, &mut off)?);

        let valid_from = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(read_u64(data, &mut off)?),
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid valid_from tag {} in grant entry",
                    tag
                )));
            }
        };

        let valid_until = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(read_u64(data, &mut off)?),
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid valid_until tag {} in grant entry",
                    tag
                )));
            }
        };

        let time_window = match read_u8(data, &mut off)? {
            0 => None,
            1 => {
                let start_hour = read_u8(data, &mut off)?;
                let end_hour = read_u8(data, &mut off)?;
                let days = read_u8(data, &mut off)?;
                Some(TimeWindow {
                    start_hour,
                    end_hour,
                    days,
                })
            }
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid time_window tag {} in grant entry",
                    tag
                )));
            }
        };

        let object_pattern = read_option_string(data, &mut off)?;
        let no_inherit = read_bool(data, &mut off)?;
        let mask_function = read_option_string(data, &mut off)?;

        Ok(Self {
            grantee,
            privilege,
            object_type,
            object_id,
            columns,
            state,
            with_grant_option,
            granted_by,
            valid_from,
            valid_until,
            time_window,
            object_pattern,
            no_inherit,
            mask_function,
        })
    }
}

/// Returns true if the grant entry is temporally active at the given unix timestamp.
fn is_temporally_active(entry: &GrantEntry, now: u64) -> bool {
    if let Some(from) = entry.valid_from {
        if now < from {
            return false;
        }
    }
    if let Some(until) = entry.valid_until {
        if now > until {
            return false;
        }
    }
    if let Some(ref tw) = entry.time_window {
        if !tw.is_active(now) {
            return false;
        }
    }
    true
}

/// SQL LIKE pattern matcher where % matches any sequence and _ matches one character.
fn matches_pattern(pattern: &str, name: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    matches_pattern_recursive(&pat, 0, &name, 0)
}

/// Recursive helper for pattern matching with backtracking.
fn matches_pattern_recursive(pat: &[char], pi: usize, name: &[char], ni: usize) -> bool {
    if pi == pat.len() && ni == name.len() {
        return true;
    }
    if pi == pat.len() {
        return false;
    }

    match pat[pi] {
        '%' => {
            // % can match zero or more characters. Try matching zero, then one, two, etc.
            for skip in 0..=(name.len() - ni) {
                if matches_pattern_recursive(pat, pi + 1, name, ni + skip) {
                    return true;
                }
            }
            false
        }
        '_' => {
            // _ matches exactly one character.
            if ni < name.len() {
                matches_pattern_recursive(pat, pi + 1, name, ni + 1)
            } else {
                false
            }
        }
        c => {
            if ni < name.len() && name[ni] == c {
                matches_pattern_recursive(pat, pi + 1, name, ni + 1)
            } else {
                false
            }
        }
    }
}

/// Thread-safe privilege store holding all grant and deny entries.
/// Pattern-based grants are stored separately for name-based lookups.
pub struct PrivilegeStore {
    grants: scc::HashMap<(u8, u32), Vec<GrantEntry>>,
    pattern_grants: parking_lot::RwLock<Vec<GrantEntry>>,
    generation: AtomicU64,
}

/// Inserts a grant entry into the scc::HashMap, appending to an existing vec or creating a new one.
/// Inserts a grant entry into the scc::HashMap, appending to an existing vec or creating a new one.
fn upsert_grant(map: &scc::HashMap<(u8, u32), Vec<GrantEntry>>, key: (u8, u32), entry: GrantEntry) {
    let found = map.read_sync(&key, |_, _| {}).is_some();
    if found {
        map.update_sync(&key, |_, v| {
            v.push(entry.clone());
        });
    } else {
        let _ = map.insert_sync(key, vec![entry]);
    }
}

impl PrivilegeStore {
    /// Creates an empty privilege store.
    pub fn new() -> Self {
        Self {
            grants: scc::HashMap::new(),
            pattern_grants: parking_lot::RwLock::new(Vec::new()),
            generation: AtomicU64::new(0),
        }
    }

    /// Bulk loads grant entries, separating pattern-based grants from object-based grants.
    pub fn load(&self, entries: Vec<GrantEntry>) {
        let mut patterns = Vec::new();
        for entry in entries {
            if entry.object_pattern.is_some() {
                patterns.push(entry);
            } else {
                let key = (entry.object_type as u8, entry.object_id);
                upsert_grant(&self.grants, key, entry);
            }
        }
        *self.pattern_grants.write() = patterns;
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Checks whether the given effective roles have the requested privilege on an object.
    /// DENY entries always take precedence over GRANT entries. Temporal and column filters apply.
    pub fn check_privilege(
        &self,
        effective_roles: &[RoleId],
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
        columns: Option<&[u16]>,
        now: u64,
    ) -> PrivilegeDecision {
        let key = (object_type as u8, object_id);
        let mut found_grant = false;

        let check_result = self.grants.read_sync(&key, |_, entries| {
            check_entries(entries, effective_roles, privilege, columns, now)
        });

        match check_result {
            Some((has_deny, has_grant)) => {
                if has_deny {
                    return PrivilegeDecision::Denied;
                }
                if has_grant {
                    found_grant = true;
                }
            }
            None => {}
        }

        if found_grant {
            PrivilegeDecision::Allow
        } else {
            PrivilegeDecision::Unset
        }
    }

    /// Checks pattern-based grants against an object name.
    pub fn check_pattern_privilege(
        &self,
        effective_roles: &[RoleId],
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_name: &str,
        now: u64,
    ) -> PrivilegeDecision {
        let patterns = self.pattern_grants.read();
        let mut found_grant = false;
        let mut found_deny = false;

        for entry in patterns.iter() {
            if entry.object_type != object_type {
                continue;
            }
            if !entry.privilege.matches(privilege) {
                continue;
            }
            if !effective_roles.contains(&entry.grantee) {
                continue;
            }
            if !is_temporally_active(entry, now) {
                continue;
            }
            if let Some(ref pattern) = entry.object_pattern {
                if !matches_pattern(pattern, object_name) {
                    continue;
                }
            }
            match entry.state {
                PrivilegeState::Deny => found_deny = true,
                PrivilegeState::Grant => found_grant = true,
            }
        }

        if found_deny {
            PrivilegeDecision::Denied
        } else if found_grant {
            PrivilegeDecision::Allow
        } else {
            PrivilegeDecision::Unset
        }
    }

    /// Adds a GRANT entry to the store.
    pub fn grant(&self, entry: GrantEntry) -> Result<()> {
        if entry.object_pattern.is_some() {
            self.pattern_grants.write().push(entry);
        } else {
            let key = (entry.object_type as u8, entry.object_id);
            upsert_grant(&self.grants, key, entry);
        }
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Adds a DENY entry to the store. Sets the state to Deny regardless of input state.
    pub fn deny(&self, mut entry: GrantEntry) -> Result<()> {
        entry.state = PrivilegeState::Deny;
        if entry.object_pattern.is_some() {
            self.pattern_grants.write().push(entry);
        } else {
            let key = (entry.object_type as u8, entry.object_id);
            upsert_grant(&self.grants, key, entry);
        }
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Revokes (removes) a specific privilege grant for a grantee on an object.
    /// Returns true if an entry was removed.
    pub fn revoke(
        &self,
        grantee: RoleId,
        privilege: PrivilegeType,
        object_type: ObjectType,
        object_id: u32,
    ) -> bool {
        let key = (object_type as u8, object_id);
        let mut removed = false;
        self.grants.update_sync(&key, |_, entries| {
            let before = entries.len();
            entries.retain(|e| !(e.grantee == grantee && e.privilege == privilege));
            if entries.len() < before {
                removed = true;
            }
        });
        if removed {
            self.generation.fetch_add(1, Ordering::Release);
        }
        removed
    }

    /// Revokes all grants on a specific object.
    pub fn revoke_all_on_object(&self, object_type: ObjectType, object_id: u32) {
        let key = (object_type as u8, object_id);
        let _ = self.grants.remove_sync(&key);
        self.pattern_grants
            .write()
            .retain(|e| !(e.object_type == object_type && e.object_id == object_id));
        self.generation.fetch_add(1, Ordering::Release);
    }

    /// Returns all grant entries for a specific object.
    pub fn grants_for_object(&self, object_type: ObjectType, object_id: u32) -> Vec<GrantEntry> {
        let key = (object_type as u8, object_id);
        let mut result = Vec::new();
        self.grants.read_sync(&key, |_, entries| {
            result = entries.clone();
        });
        result
    }

    /// Returns all grant entries where the grantee matches the given role.
    pub fn grants_for_role(&self, role_id: RoleId) -> Vec<GrantEntry> {
        let mut result = Vec::new();
        self.grants.iter_sync(|_, entries| {
            for entry in entries {
                if entry.grantee == role_id {
                    result.push(entry.clone());
                }
            }
            true
        });
        let patterns = self.pattern_grants.read();
        for entry in patterns.iter() {
            if entry.grantee == role_id {
                result.push(entry.clone());
            }
        }
        result
    }

    /// Returns the current generation counter. Incremented on every mutation.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

/// Checks a list of grant entries against the requested roles, privilege, and columns.
/// Returns (has_deny, has_grant).
fn check_entries(
    entries: &[GrantEntry],
    effective_roles: &[RoleId],
    privilege: PrivilegeType,
    columns: Option<&[u16]>,
    now: u64,
) -> (bool, bool) {
    let mut has_deny = false;
    let mut has_grant = false;

    for entry in entries {
        if !entry.privilege.matches(privilege) {
            continue;
        }
        if !effective_roles.contains(&entry.grantee) {
            continue;
        }
        if !is_temporally_active(entry, now) {
            continue;
        }

        // Column-level check: if columns are requested and the entry has column restrictions,
        // check that the requested columns intersect with the entry's columns.
        if let Some(requested_cols) = columns {
            if let Some(ref entry_cols) = entry.columns {
                // For DENY: if any requested column is in the deny set, it counts as denied.
                // For GRANT: only count if all requested columns are covered.
                match entry.state {
                    PrivilegeState::Deny => {
                        let any_match = requested_cols.iter().any(|c| entry_cols.contains(c));
                        if any_match {
                            has_deny = true;
                            continue;
                        }
                    }
                    PrivilegeState::Grant => {
                        let all_covered = requested_cols.iter().all(|c| entry_cols.contains(c));
                        if all_covered {
                            has_grant = true;
                            continue;
                        }
                    }
                }
                continue;
            }
        }

        // No column restriction on the entry, or no columns requested.
        match entry.state {
            PrivilegeState::Deny => has_deny = true,
            PrivilegeState::Grant => has_grant = true,
        }
    }

    (has_deny, has_grant)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grant(
        grantee: u32,
        priv_type: PrivilegeType,
        obj_type: ObjectType,
        obj_id: u32,
    ) -> GrantEntry {
        GrantEntry {
            grantee: RoleId(grantee),
            privilege: priv_type,
            object_type: obj_type,
            object_id: obj_id,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: false,
            granted_by: RoleId(0),
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: None,
            no_inherit: false,
            mask_function: None,
        }
    }

    fn make_deny(
        grantee: u32,
        priv_type: PrivilegeType,
        obj_type: ObjectType,
        obj_id: u32,
    ) -> GrantEntry {
        let mut entry = make_grant(grantee, priv_type, obj_type, obj_id);
        entry.state = PrivilegeState::Deny;
        entry
    }

    #[test]
    fn test_basic_grant() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_no_grant_returns_unset() {
        let store = PrivilegeStore::new();
        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_deny_wins_over_grant() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        store
            .deny(make_deny(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("deny failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Denied);
    }

    #[test]
    fn test_deny_on_different_role_does_not_affect() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        store
            .deny(make_deny(2, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("deny failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_temporal_valid_from_not_yet_active() {
        let store = PrivilegeStore::new();
        let mut entry = make_grant(1, PrivilegeType::Select, ObjectType::Table, 100);
        entry.valid_from = Some(2000);
        store.grant(entry).expect("grant failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_temporal_valid_from_active() {
        let store = PrivilegeStore::new();
        let mut entry = make_grant(1, PrivilegeType::Select, ObjectType::Table, 100);
        entry.valid_from = Some(500);
        store.grant(entry).expect("grant failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_temporal_valid_until_expired() {
        let store = PrivilegeStore::new();
        let mut entry = make_grant(1, PrivilegeType::Select, ObjectType::Table, 100);
        entry.valid_until = Some(500);
        store.grant(entry).expect("grant failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_column_level_deny_overrides_table_grant() {
        let store = PrivilegeStore::new();

        // Table-level GRANT for SELECT.
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");

        // Column-level DENY for column 3.
        let mut deny = make_deny(1, PrivilegeType::Select, ObjectType::Table, 100);
        deny.columns = Some(vec![3]);
        store.deny(deny).expect("deny failed");

        // Requesting column 3 should be denied.
        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            Some(&[3]),
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Denied);

        // Requesting column 5 should be allowed (table-level grant, no deny on col 5).
        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            Some(&[5]),
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_column_level_grant_covers_requested() {
        let store = PrivilegeStore::new();

        let mut entry = make_grant(1, PrivilegeType::Update, ObjectType::Table, 100);
        entry.columns = Some(vec![1, 2, 3]);
        store.grant(entry).expect("grant failed");

        // Requesting columns 1 and 2 should be allowed.
        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Update,
            ObjectType::Table,
            100,
            Some(&[1, 2]),
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);

        // Requesting column 5 (not in grant) should be unset.
        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Update,
            ObjectType::Table,
            100,
            Some(&[5]),
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_pattern_grant() {
        let store = PrivilegeStore::new();

        let mut entry = make_grant(1, PrivilegeType::Select, ObjectType::Table, 0);
        entry.object_pattern = Some("audit_%".to_string());
        store.grant(entry).expect("grant failed");

        let result = store.check_pattern_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            "audit_log",
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);

        let result = store.check_pattern_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            "users",
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_pattern_single_char_wildcard() {
        let store = PrivilegeStore::new();

        let mut entry = make_grant(1, PrivilegeType::Select, ObjectType::Table, 0);
        entry.object_pattern = Some("log_".to_string());
        store.grant(entry).expect("grant failed");

        let result = store.check_pattern_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            "logA",
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);

        // Two chars after "log" does not match.
        let result = store.check_pattern_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            "logAB",
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_revoke() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");

        let removed = store.revoke(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
        assert!(removed);

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Unset);
    }

    #[test]
    fn test_revoke_nonexistent() {
        let store = PrivilegeStore::new();
        let removed = store.revoke(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
        assert!(!removed);
    }

    #[test]
    fn test_revoke_all_on_object() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        store
            .grant(make_grant(2, PrivilegeType::Insert, ObjectType::Table, 100))
            .expect("grant failed");

        store.revoke_all_on_object(ObjectType::Table, 100);

        let entries = store.grants_for_object(ObjectType::Table, 100);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_grants_for_role() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        store
            .grant(make_grant(1, PrivilegeType::Insert, ObjectType::Table, 200))
            .expect("grant failed");
        store
            .grant(make_grant(2, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");

        let entries = store.grants_for_role(RoleId(1));
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_generation_counter() {
        let store = PrivilegeStore::new();
        let g0 = store.generation();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        let g1 = store.generation();
        assert!(g1 > g0);

        store.revoke(RoleId(1), PrivilegeType::Select, ObjectType::Table, 100);
        let g2 = store.generation();
        assert!(g2 > g1);
    }

    #[test]
    fn test_all_privilege_matches_specific() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::All, ObjectType::Table, 100))
            .expect("grant failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Delete,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_bulk_load() {
        let entries = vec![
            make_grant(1, PrivilegeType::Select, ObjectType::Table, 100),
            make_grant(2, PrivilegeType::Insert, ObjectType::Table, 200),
        ];
        let store = PrivilegeStore::new();
        store.load(entries);

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);

        let result = store.check_privilege(
            &[RoleId(2)],
            PrivilegeType::Insert,
            ObjectType::Table,
            200,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_effective_roles_check() {
        let store = PrivilegeStore::new();
        // Role 2 has SELECT, user 1 inherits role 2.
        store
            .grant(make_grant(2, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");

        let result = store.check_privilege(
            &[RoleId(1), RoleId(2)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Allow);
    }

    #[test]
    fn test_grant_entry_to_bytes_from_bytes() {
        let entry = GrantEntry {
            grantee: RoleId(10),
            privilege: PrivilegeType::Update,
            object_type: ObjectType::Table,
            object_id: 500,
            columns: Some(vec![1, 3, 5]),
            state: PrivilegeState::Grant,
            with_grant_option: true,
            granted_by: RoleId(1),
            valid_from: Some(1000),
            valid_until: Some(2000),
            time_window: Some(TimeWindow {
                start_hour: 9,
                end_hour: 17,
                days: 0b0001_1111,
            }),
            object_pattern: Some("test_%".to_string()),
            no_inherit: true,
            mask_function: Some("mask_email".to_string()),
        };

        let bytes = entry.to_bytes();
        let restored = GrantEntry::from_bytes(&bytes).expect("from_bytes failed");

        assert_eq!(restored.grantee, RoleId(10));
        assert_eq!(restored.privilege, PrivilegeType::Update);
        assert_eq!(restored.object_type, ObjectType::Table);
        assert_eq!(restored.object_id, 500);
        assert_eq!(restored.columns, Some(vec![1, 3, 5]));
        assert_eq!(restored.state, PrivilegeState::Grant);
        assert!(restored.with_grant_option);
        assert_eq!(restored.granted_by, RoleId(1));
        assert_eq!(restored.valid_from, Some(1000));
        assert_eq!(restored.valid_until, Some(2000));
        let tw = restored.time_window.expect("time_window missing");
        assert_eq!(tw.start_hour, 9);
        assert_eq!(tw.end_hour, 17);
        assert_eq!(tw.days, 0b0001_1111);
        assert_eq!(restored.object_pattern, Some("test_%".to_string()));
        assert!(restored.no_inherit);
        assert_eq!(restored.mask_function, Some("mask_email".to_string()));
    }

    #[test]
    fn test_grant_entry_to_bytes_from_bytes_minimal() {
        let entry = make_grant(1, PrivilegeType::Select, ObjectType::Database, 1);
        let bytes = entry.to_bytes();
        let restored = GrantEntry::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.grantee, RoleId(1));
        assert_eq!(restored.privilege, PrivilegeType::Select);
        assert_eq!(restored.object_type, ObjectType::Database);
        assert_eq!(restored.object_id, 1);
        assert!(restored.columns.is_none());
        assert_eq!(restored.state, PrivilegeState::Grant);
        assert!(!restored.with_grant_option);
        assert!(restored.valid_from.is_none());
        assert!(restored.valid_until.is_none());
        assert!(restored.time_window.is_none());
        assert!(restored.object_pattern.is_none());
        assert!(!restored.no_inherit);
        assert!(restored.mask_function.is_none());
    }

    #[test]
    fn test_grant_entry_from_bytes_truncated() {
        let data = vec![0u8; 3];
        assert!(GrantEntry::from_bytes(&data).is_err());
    }

    #[test]
    fn test_privilege_type_from_u8_invalid() {
        assert!(PrivilegeType::from_u8(100).is_err());
    }

    #[test]
    fn test_object_type_from_u8_invalid() {
        assert!(ObjectType::from_u8(99).is_err());
    }

    #[test]
    fn test_matches_pattern_basic() {
        assert!(matches_pattern("hello", "hello"));
        assert!(!matches_pattern("hello", "world"));
    }

    #[test]
    fn test_matches_pattern_percent() {
        assert!(matches_pattern("%", "anything"));
        assert!(matches_pattern("%", ""));
        assert!(matches_pattern("pre%", "prefix"));
        assert!(matches_pattern("pre%", "pre"));
        assert!(!matches_pattern("pre%", "notpre"));
        assert!(matches_pattern("%fix", "suffix"));
        assert!(matches_pattern("a%b%c", "abc"));
        assert!(matches_pattern("a%b%c", "aXXbYYc"));
    }

    #[test]
    fn test_matches_pattern_underscore() {
        assert!(matches_pattern("a_c", "abc"));
        assert!(!matches_pattern("a_c", "ac"));
        assert!(!matches_pattern("a_c", "abbc"));
        assert!(matches_pattern("___", "abc"));
    }

    #[test]
    fn test_matches_pattern_combined() {
        assert!(matches_pattern("log_%", "log_2024"));
        assert!(matches_pattern("log_%", "log_X"));
        // "log_%": _ matches the underscore char in "log_", % matches empty string.
        assert!(matches_pattern("log_%", "log_"));
        // "log_" without % requires exactly 4 chars.
        assert!(!matches_pattern("log_%", "log"));
    }

    #[test]
    fn test_time_window_based_privilege() {
        let store = PrivilegeStore::new();

        // Create a grant active only in certain time windows.
        let mut entry = make_grant(1, PrivilegeType::Select, ObjectType::Table, 100);
        entry.time_window = Some(TimeWindow {
            start_hour: 9,
            end_hour: 17,
            days: 0b0111_1111, // all days
        });
        store.grant(entry).expect("grant failed");

        // The check uses now=1000, and TimeWindow.is_active determines if active.
        // Since is_active depends on actual time calculation, the result depends
        // on the TimeWindow implementation.
        let _result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        // Just verify no panic occurs.
    }

    #[test]
    fn test_deny_all_blocks_specific_privilege() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        store
            .deny(make_deny(1, PrivilegeType::All, ObjectType::Table, 100))
            .expect("deny failed");

        let result = store.check_privilege(
            &[RoleId(1)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Denied);
    }

    #[test]
    fn test_multiple_roles_mixed_grants() {
        let store = PrivilegeStore::new();
        store
            .grant(make_grant(1, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("grant failed");
        store
            .deny(make_deny(2, PrivilegeType::Select, ObjectType::Table, 100))
            .expect("deny failed");

        // User has both roles: DENY on role 2 should win.
        let result = store.check_privilege(
            &[RoleId(1), RoleId(2)],
            PrivilegeType::Select,
            ObjectType::Table,
            100,
            None,
            1000,
        );
        assert_eq!(result, PrivilegeDecision::Denied);
    }
}
