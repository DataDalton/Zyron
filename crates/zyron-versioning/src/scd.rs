//! Slowly Changing Dimensions (SCD) handlers.
//!
//! SCD types control how dimension table rows are managed when source
//! data changes. The ScdHandler generates DML actions based on the
//! configured type, which the executor applies through standard
//! insert/update paths.

use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::error::{Result, ZyronError};
use zyron_storage::TupleId;

// ---------------------------------------------------------------------------
// SCD type definitions
// ---------------------------------------------------------------------------

/// Slowly Changing Dimension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScdType {
    /// Type 1: Overwrite existing record, no history.
    Type1 = 1,
    /// Type 2: Add new row with validity period. Full history.
    Type2 = 2,
    /// Type 3: Store current and previous values in the same row.
    Type3 = 3,
    /// Type 4: Separate history table, main table holds current only.
    Type4 = 4,
    /// Type 6: Hybrid of Type 1 + 2 + 3.
    Type6 = 6,
}

impl ScdType {
    /// Parses an SCD type from a numeric value.
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            1 => Ok(Self::Type1),
            2 => Ok(Self::Type2),
            3 => Ok(Self::Type3),
            4 => Ok(Self::Type4),
            6 => Ok(Self::Type6),
            _ => Err(ZyronError::ScdConfigError(format!(
                "unknown scd_type {val}"
            ))),
        }
    }
}

/// Configuration for SCD behavior on a table.
#[derive(Debug, Clone)]
pub struct ScdConfig {
    pub scd_type: ScdType,
    /// Columns that form the natural (business) key.
    pub natural_key_columns: Vec<String>,
    /// Columns tracked for changes. None means all non-key columns.
    pub tracked_columns: Option<Vec<String>>,
    /// Name of the valid_from column (default: "valid_from").
    pub valid_from_column: String,
    /// Name of the valid_to column (default: "valid_to").
    pub valid_to_column: String,
    /// Name of the is_current column (default: "is_current").
    pub is_current_column: String,
    /// For Type 4: the history table's table_id.
    pub history_table_id: Option<u32>,
    /// For Type 3/6: prefix for previous-value columns (default: "prev_").
    pub previous_prefix: String,
}

impl Default for ScdConfig {
    fn default() -> Self {
        Self {
            scd_type: ScdType::Type2,
            natural_key_columns: Vec::new(),
            tracked_columns: None,
            valid_from_column: "valid_from".to_string(),
            valid_to_column: "valid_to".to_string(),
            is_current_column: "is_current".to_string(),
            history_table_id: None,
            previous_prefix: "prev_".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// SCD actions (fixed-size per type, no unnecessary allocations)
// ---------------------------------------------------------------------------

/// Type 1: simple overwrite of existing row.
#[derive(Debug)]
pub struct ScdType1Actions {
    pub update_tuple_id: TupleId,
    pub new_data: Vec<u8>,
}

/// Type 2: expire old row + insert new row.
#[derive(Debug)]
pub struct ScdType2Actions {
    /// Tuple to expire (set valid_to, is_current=false).
    pub expire_tuple_id: TupleId,
    /// Timestamp to set as valid_to on the old row.
    pub expire_valid_to: i64,
    /// valid_from for the new row.
    pub new_row_valid_from: i64,
    /// Full row data for the new current row.
    pub new_row_data: Vec<u8>,
    /// Auto-generated time-ordered surrogate key.
    pub surrogate_key: u64,
}

/// Type 3: update row with current + previous values.
#[derive(Debug)]
pub struct ScdType3Actions {
    pub update_tuple_id: TupleId,
    /// Row data with current values and prev_ columns filled.
    pub new_data: Vec<u8>,
}

/// Type 4: insert into history table + update main table.
#[derive(Debug)]
pub struct ScdType4Actions {
    pub history_table_id: u32,
    /// Full row to insert into the history table.
    pub history_row_data: Vec<u8>,
    /// Tuple to update on the main table.
    pub update_tuple_id: TupleId,
    /// New data for the main table row.
    pub new_data: Vec<u8>,
}

/// Type 6: hybrid (expire + prev columns + insert new).
#[derive(Debug)]
pub struct ScdType6Actions {
    /// Tuple to expire on the old row.
    pub expire_tuple_id: TupleId,
    /// Updated data for the old row (valid_to + prev_ columns).
    pub expire_updates: Vec<u8>,
    /// Full row data for the new current row.
    pub new_row_data: Vec<u8>,
    /// Auto-generated time-ordered surrogate key.
    pub surrogate_key: u64,
}

/// Discriminated union of SCD actions by type.
#[derive(Debug)]
pub enum ScdActions {
    Type1(ScdType1Actions),
    Type2(ScdType2Actions),
    Type3(ScdType3Actions),
    Type4(ScdType4Actions),
    Type6(ScdType6Actions),
    /// For Type2/Type6 deletes: expire the row (set valid_to, is_current=false).
    DeleteExpire {
        tuple_id: TupleId,
        valid_to: i64,
    },
}

// ---------------------------------------------------------------------------
// Surrogate key generator
// ---------------------------------------------------------------------------

/// Generates time-ordered unique surrogate keys.
///
/// Key format: (timestamp_millis << 20) | (counter & 0xFFFFF).
/// 20 bits for counter gives 1M keys per millisecond.
/// Counter resets when the millisecond changes.
pub struct SurrogateKeyGenerator {
    last_millis: AtomicU64,
    counter: AtomicU64,
}

impl SurrogateKeyGenerator {
    /// Creates a new surrogate key generator.
    pub fn new() -> Self {
        Self {
            last_millis: AtomicU64::new(0),
            counter: AtomicU64::new(0),
        }
    }

    /// Generates the next time-ordered unique key.
    pub fn next_key(&self, now_millis: u64) -> u64 {
        let last = self.last_millis.load(Ordering::Relaxed);
        if now_millis > last {
            // New millisecond: try to reset counter.
            // CAS failure means another thread already advanced, which is fine.
            if self
                .last_millis
                .compare_exchange(last, now_millis, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.counter.store(0, Ordering::Relaxed);
            }
        }
        let seq = self.counter.fetch_add(1, Ordering::Relaxed) & 0xFFFFF;
        (now_millis << 20) | seq
    }
}

impl Default for SurrogateKeyGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SCD handler
// ---------------------------------------------------------------------------

/// Handles SCD logic for a specific table configuration.
pub struct ScdHandler {
    config: ScdConfig,
    surrogate_gen: SurrogateKeyGenerator,
}

impl ScdHandler {
    /// Creates a new SCD handler with the given configuration.
    pub fn new(config: ScdConfig) -> Self {
        Self {
            config,
            surrogate_gen: SurrogateKeyGenerator::new(),
        }
    }

    /// Returns the SCD configuration.
    pub fn config(&self) -> &ScdConfig {
        &self.config
    }

    /// Parses SCD configuration from table WITH options.
    ///
    /// Recognized keys: "scd_type", "natural_key", "tracked_columns",
    /// "valid_from_column", "valid_to_column", "is_current_column",
    /// "previous_prefix".
    pub fn from_table_options(options: &[(String, String)]) -> Result<Option<ScdConfig>> {
        let mut scd_type: Option<ScdType> = None;
        let mut config = ScdConfig::default();

        for (key, value) in options {
            match key.as_str() {
                "scd_type" => {
                    let val: u8 = value.parse().map_err(|_| {
                        ZyronError::ScdConfigError(format!("invalid scd_type value: {value}"))
                    })?;
                    scd_type = Some(ScdType::from_u8(val)?);
                }
                "natural_key" => {
                    // Parse comma-separated or bracket-enclosed list
                    let cleaned = value.trim_matches(|c| c == '[' || c == ']' || c == '\'');
                    config.natural_key_columns = cleaned
                        .split(',')
                        .map(|s| s.trim().trim_matches('\'').to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
                "tracked_columns" => {
                    let cleaned = value.trim_matches(|c| c == '[' || c == ']' || c == '\'');
                    config.tracked_columns = Some(
                        cleaned
                            .split(',')
                            .map(|s| s.trim().trim_matches('\'').to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                }
                "valid_from_column" => config.valid_from_column = value.clone(),
                "valid_to_column" => config.valid_to_column = value.clone(),
                "is_current_column" => config.is_current_column = value.clone(),
                "previous_prefix" => config.previous_prefix = value.clone(),
                _ => {} // ignore unknown options
            }
        }

        match scd_type {
            Some(t) => {
                config.scd_type = t;
                if config.natural_key_columns.is_empty() {
                    return Err(ZyronError::ScdConfigError(
                        "natural_key is required for SCD tables".to_string(),
                    ));
                }
                Ok(Some(config))
            }
            None => Ok(None),
        }
    }

    /// Generates a Type 2 update: expire old row, insert new row.
    pub fn generate_type2_update(
        &self,
        old_tuple_id: TupleId,
        new_row_data: Vec<u8>,
        now_micros: i64,
        now_millis: u64,
    ) -> ScdType2Actions {
        ScdType2Actions {
            expire_tuple_id: old_tuple_id,
            expire_valid_to: now_micros,
            new_row_valid_from: now_micros,
            new_row_data,
            surrogate_key: self.surrogate_gen.next_key(now_millis),
        }
    }

    /// Generates a Type 1 update: simple overwrite.
    pub fn generate_type1_update(
        &self,
        old_tuple_id: TupleId,
        new_data: Vec<u8>,
    ) -> ScdType1Actions {
        ScdType1Actions {
            update_tuple_id: old_tuple_id,
            new_data,
        }
    }

    /// Generates a Type 3 update: store previous values alongside current.
    pub fn generate_type3_update(
        &self,
        old_tuple_id: TupleId,
        new_data_with_prev: Vec<u8>,
    ) -> ScdType3Actions {
        ScdType3Actions {
            update_tuple_id: old_tuple_id,
            new_data: new_data_with_prev,
        }
    }

    /// Generates a Type 4 update: copy to history table, update main.
    pub fn generate_type4_update(
        &self,
        old_tuple_id: TupleId,
        history_row_data: Vec<u8>,
        new_data: Vec<u8>,
    ) -> Result<ScdType4Actions> {
        let history_table_id = self.config.history_table_id.ok_or_else(|| {
            ZyronError::ScdConfigError("history_table_id not set for SCD Type 4".to_string())
        })?;
        Ok(ScdType4Actions {
            history_table_id,
            history_row_data,
            update_tuple_id: old_tuple_id,
            new_data,
        })
    }

    /// Generates a Type 6 update: hybrid of Type 1+2+3.
    pub fn generate_type6_update(
        &self,
        old_tuple_id: TupleId,
        expire_updates: Vec<u8>,
        new_row_data: Vec<u8>,
        now_millis: u64,
    ) -> ScdType6Actions {
        ScdType6Actions {
            expire_tuple_id: old_tuple_id,
            expire_updates,
            new_row_data,
            surrogate_key: self.surrogate_gen.next_key(now_millis),
        }
    }

    /// Generates SCD actions based on the configured type.
    pub fn generate_update(
        &self,
        old_tuple_id: TupleId,
        new_row_data: Vec<u8>,
        now_micros: i64,
        now_millis: u64,
    ) -> Result<ScdActions> {
        match self.config.scd_type {
            ScdType::Type1 => Ok(ScdActions::Type1(
                self.generate_type1_update(old_tuple_id, new_row_data),
            )),
            ScdType::Type2 => Ok(ScdActions::Type2(self.generate_type2_update(
                old_tuple_id,
                new_row_data,
                now_micros,
                now_millis,
            ))),
            ScdType::Type3 => Ok(ScdActions::Type3(
                self.generate_type3_update(old_tuple_id, new_row_data),
            )),
            ScdType::Type4 => {
                // For Type4, new_row_data is used for both history and main table.
                // The executor must split this appropriately.
                let history_data = new_row_data.clone();
                Ok(ScdActions::Type4(self.generate_type4_update(
                    old_tuple_id,
                    history_data,
                    new_row_data,
                )?))
            }
            ScdType::Type6 => {
                let expire_updates = Vec::new(); // filled by executor with prev_ columns
                Ok(ScdActions::Type6(self.generate_type6_update(
                    old_tuple_id,
                    expire_updates,
                    new_row_data,
                    now_millis,
                )))
            }
        }
    }

    /// Generates a delete action (for Type 2/Type 6: expire the row).
    pub fn generate_delete(&self, tuple_id: TupleId, now_micros: i64) -> ScdActions {
        match self.config.scd_type {
            ScdType::Type2 | ScdType::Type6 => ScdActions::DeleteExpire {
                tuple_id,
                valid_to: now_micros,
            },
            // Type 1/3/4: standard delete (no special handling)
            _ => ScdActions::DeleteExpire {
                tuple_id,
                valid_to: now_micros,
            },
        }
    }

    /// Returns the columns required by this SCD type that must exist on the table.
    pub fn required_columns(&self) -> Vec<(String, &'static str)> {
        match self.config.scd_type {
            ScdType::Type1 => Vec::new(),
            ScdType::Type2 => vec![
                (self.config.valid_from_column.clone(), "TIMESTAMPTZ"),
                (self.config.valid_to_column.clone(), "TIMESTAMPTZ"),
                (self.config.is_current_column.clone(), "BOOLEAN"),
            ],
            ScdType::Type3 => {
                // prev_{column} for each tracked column
                match &self.config.tracked_columns {
                    Some(cols) => cols
                        .iter()
                        .map(|c| {
                            (
                                format!("{}{}", self.config.previous_prefix, c),
                                "TEXT", // actual type matches the column
                            )
                        })
                        .collect(),
                    None => Vec::new(), // determined at table creation time
                }
            }
            ScdType::Type4 => Vec::new(), // no extra columns on main table
            ScdType::Type6 => {
                let mut cols = vec![
                    (self.config.valid_from_column.clone(), "TIMESTAMPTZ"),
                    (self.config.valid_to_column.clone(), "TIMESTAMPTZ"),
                    (self.config.is_current_column.clone(), "BOOLEAN"),
                ];
                if let Some(tracked) = &self.config.tracked_columns {
                    for c in tracked {
                        cols.push((format!("{}{}", self.config.previous_prefix, c), "TEXT"));
                    }
                }
                cols
            }
        }
    }

    /// Returns the name for a Type 4 history table.
    pub fn history_table_name(base_table: &str) -> String {
        format!("{base_table}_history")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::page::PageId;

    fn dummy_tuple_id() -> TupleId {
        TupleId::new(PageId::new(1, 0), 0)
    }

    #[test]
    fn test_scd_type_from_u8() {
        assert_eq!(ScdType::from_u8(1).unwrap(), ScdType::Type1);
        assert_eq!(ScdType::from_u8(2).unwrap(), ScdType::Type2);
        assert_eq!(ScdType::from_u8(3).unwrap(), ScdType::Type3);
        assert_eq!(ScdType::from_u8(4).unwrap(), ScdType::Type4);
        assert_eq!(ScdType::from_u8(6).unwrap(), ScdType::Type6);
        assert!(ScdType::from_u8(5).is_err());
        assert!(ScdType::from_u8(0).is_err());
    }

    #[test]
    fn test_from_table_options() {
        let options = vec![
            ("scd_type".to_string(), "2".to_string()),
            ("natural_key".to_string(), "customer_id".to_string()),
        ];
        let config = ScdHandler::from_table_options(&options)
            .expect("parse")
            .expect("some");
        assert_eq!(config.scd_type, ScdType::Type2);
        assert_eq!(config.natural_key_columns, vec!["customer_id"]);
    }

    #[test]
    fn test_from_table_options_missing_natural_key() {
        let options = vec![("scd_type".to_string(), "2".to_string())];
        assert!(ScdHandler::from_table_options(&options).is_err());
    }

    #[test]
    fn test_from_table_options_no_scd() {
        let options = vec![("retention".to_string(), "30d".to_string())];
        let result = ScdHandler::from_table_options(&options).expect("parse");
        assert!(result.is_none());
    }

    #[test]
    fn test_from_table_options_bracket_list() {
        let options = vec![
            ("scd_type".to_string(), "2".to_string()),
            (
                "natural_key".to_string(),
                "['customer_id', 'region']".to_string(),
            ),
        ];
        let config = ScdHandler::from_table_options(&options)
            .expect("parse")
            .expect("some");
        assert_eq!(config.natural_key_columns, vec!["customer_id", "region"]);
    }

    #[test]
    fn test_type2_update_actions() {
        let config = ScdConfig {
            scd_type: ScdType::Type2,
            natural_key_columns: vec!["id".to_string()],
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tid = dummy_tuple_id();

        let actions = handler.generate_type2_update(tid, vec![1, 2, 3], 1000000, 1000);

        assert_eq!(actions.expire_tuple_id, tid);
        assert_eq!(actions.expire_valid_to, 1000000);
        assert_eq!(actions.new_row_valid_from, 1000000);
        assert_eq!(actions.new_row_data, vec![1, 2, 3]);
        assert!(actions.surrogate_key > 0);
    }

    #[test]
    fn test_type1_update_actions() {
        let config = ScdConfig {
            scd_type: ScdType::Type1,
            natural_key_columns: vec!["id".to_string()],
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tid = dummy_tuple_id();

        let actions = handler.generate_type1_update(tid, vec![4, 5, 6]);
        assert_eq!(actions.update_tuple_id, tid);
        assert_eq!(actions.new_data, vec![4, 5, 6]);
    }

    #[test]
    fn test_surrogate_key_uniqueness() {
        let keygen = SurrogateKeyGenerator::new();
        let now = 1700000000000u64;
        let k1 = keygen.next_key(now);
        let k2 = keygen.next_key(now);
        let k3 = keygen.next_key(now);

        assert_ne!(k1, k2);
        assert_ne!(k2, k3);
        assert!(k1 < k2);
        assert!(k2 < k3);
    }

    #[test]
    fn test_surrogate_key_time_ordering() {
        let keygen = SurrogateKeyGenerator::new();
        let k1 = keygen.next_key(1000);
        let k2 = keygen.next_key(2000);
        assert!(k2 > k1);
    }

    #[test]
    fn test_required_columns_type2() {
        let config = ScdConfig {
            scd_type: ScdType::Type2,
            natural_key_columns: vec!["id".to_string()],
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let cols = handler.required_columns();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].0, "valid_from");
        assert_eq!(cols[1].0, "valid_to");
        assert_eq!(cols[2].0, "is_current");
    }

    #[test]
    fn test_required_columns_type3() {
        let config = ScdConfig {
            scd_type: ScdType::Type3,
            natural_key_columns: vec!["id".to_string()],
            tracked_columns: Some(vec!["name".to_string(), "email".to_string()]),
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let cols = handler.required_columns();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].0, "prev_name");
        assert_eq!(cols[1].0, "prev_email");
    }

    #[test]
    fn test_generate_update_dispatches() {
        let config = ScdConfig {
            scd_type: ScdType::Type2,
            natural_key_columns: vec!["id".to_string()],
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tid = dummy_tuple_id();

        let actions = handler
            .generate_update(tid, vec![1, 2, 3], 1000000, 1000)
            .expect("gen");
        assert!(matches!(actions, ScdActions::Type2(_)));
    }

    #[test]
    fn test_generate_delete_type2() {
        let config = ScdConfig {
            scd_type: ScdType::Type2,
            natural_key_columns: vec!["id".to_string()],
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tid = dummy_tuple_id();

        let actions = handler.generate_delete(tid, 5000000);
        match actions {
            ScdActions::DeleteExpire { tuple_id, valid_to } => {
                assert_eq!(tuple_id, tid);
                assert_eq!(valid_to, 5000000);
            }
            _ => panic!("expected DeleteExpire"),
        }
    }

    #[test]
    fn test_history_table_name() {
        assert_eq!(
            ScdHandler::history_table_name("customers"),
            "customers_history"
        );
    }

    #[test]
    fn test_type4_requires_history_table_id() {
        let config = ScdConfig {
            scd_type: ScdType::Type4,
            natural_key_columns: vec!["id".to_string()],
            history_table_id: None,
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tid = dummy_tuple_id();

        // Should fail because history_table_id is not set
        assert!(handler.generate_type4_update(tid, vec![], vec![]).is_err());
    }

    #[test]
    fn test_type4_with_history_table() {
        let config = ScdConfig {
            scd_type: ScdType::Type4,
            natural_key_columns: vec!["id".to_string()],
            history_table_id: Some(500),
            ..Default::default()
        };
        let handler = ScdHandler::new(config);
        let tid = dummy_tuple_id();

        let actions = handler
            .generate_type4_update(tid, vec![1, 2], vec![3, 4])
            .expect("gen");
        assert_eq!(actions.history_table_id, 500);
        assert_eq!(actions.history_row_data, vec![1, 2]);
        assert_eq!(actions.new_data, vec![3, 4]);
    }
}
