//! DML hook bridges connecting the executor to CDC and trigger subsystems.
//!
//! CdcHookBridge implements the CdcHook trait (AFTER triggers + CDC capture).
//! DmlHookBridge implements the DmlHook trait (BEFORE triggers).

use std::sync::Arc;

use zyron_cdc::{CdfRegistry, ChangeRecord, ChangeType};
use zyron_common::Result;
use zyron_executor::context::{CdcHook, DmlHook};
use zyron_pipeline::trigger::{TriggerEvent, TriggerManager, TriggerTiming};

/// Bridges DML AFTER events to CDC change feed capture and AFTER trigger dispatch.
pub struct CdcHookBridge {
    cdc_registry: Arc<CdfRegistry>,
    trigger_manager: Arc<TriggerManager>,
}

impl CdcHookBridge {
    pub fn new(cdc_registry: Arc<CdfRegistry>, trigger_manager: Arc<TriggerManager>) -> Self {
        Self {
            cdc_registry,
            trigger_manager,
        }
    }
}

impl CdcHook for CdcHookBridge {
    fn on_insert(
        &self,
        table_id: u32,
        tuples: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u32,
        is_last_in_txn: bool,
    ) -> Result<()> {
        // Write change records to CDC feed if enabled for this table
        if let Some(feed) = self.cdc_registry.get_feed(table_id) {
            for (i, tuple) in tuples.iter().enumerate() {
                let record = ChangeRecord {
                    change_type: ChangeType::Insert,
                    commit_version: version,
                    commit_timestamp: timestamp,
                    table_id,
                    txn_id,
                    schema_version: 0,
                    row_data: tuple.to_vec(),
                    primary_key_data: Vec::new(),
                    is_last_in_txn: is_last_in_txn && i == tuples.len() - 1,
                };
                feed.append_change(&record)?;
            }
        }

        // AFTER INSERT triggers: retrieve matching triggers. Trigger function
        // bodies require UDF registry to resolve the function name to code.
        let _triggers = self.trigger_manager.getMatchingTriggers(
            table_id,
            TriggerTiming::After,
            TriggerEvent::Insert,
        );

        Ok(())
    }

    fn on_delete(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u32,
        is_last_in_txn: bool,
    ) -> Result<()> {
        if let Some(feed) = self.cdc_registry.get_feed(table_id) {
            for (i, old) in old_data.iter().enumerate() {
                let record = ChangeRecord {
                    change_type: ChangeType::Delete,
                    commit_version: version,
                    commit_timestamp: timestamp,
                    table_id,
                    txn_id,
                    schema_version: 0,
                    row_data: old.to_vec(),
                    primary_key_data: Vec::new(),
                    is_last_in_txn: is_last_in_txn && i == old_data.len() - 1,
                };
                feed.append_change(&record)?;
            }
        }

        let _triggers = self.trigger_manager.getMatchingTriggers(
            table_id,
            TriggerTiming::After,
            TriggerEvent::Delete,
        );

        Ok(())
    }

    fn on_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u32,
        is_last_in_txn: bool,
    ) -> Result<()> {
        if let Some(feed) = self.cdc_registry.get_feed(table_id) {
            for (i, (old, new)) in old_data.iter().zip(new_data.iter()).enumerate() {
                let is_last = is_last_in_txn && i == old_data.len() - 1;
                // Record pre-image (old data)
                let pre_record = ChangeRecord {
                    change_type: ChangeType::UpdatePreimage,
                    commit_version: version,
                    commit_timestamp: timestamp,
                    table_id,
                    txn_id,
                    schema_version: 0,
                    row_data: old.to_vec(),
                    primary_key_data: Vec::new(),
                    is_last_in_txn: false,
                };
                feed.append_change(&pre_record)?;
                // Record post-image (new data)
                let post_record = ChangeRecord {
                    change_type: ChangeType::UpdatePostimage,
                    commit_version: version,
                    commit_timestamp: timestamp,
                    table_id,
                    txn_id,
                    schema_version: 0,
                    row_data: new.to_vec(),
                    primary_key_data: Vec::new(),
                    is_last_in_txn: is_last,
                };
                feed.append_change(&post_record)?;
            }
        }

        let _triggers = self.trigger_manager.getMatchingTriggers(
            table_id,
            TriggerTiming::After,
            TriggerEvent::Update,
        );

        Ok(())
    }
}

/// Bridges DML BEFORE events to BEFORE trigger dispatch.
pub struct DmlHookBridge {
    trigger_manager: Arc<TriggerManager>,
}

impl DmlHookBridge {
    pub fn new(trigger_manager: Arc<TriggerManager>) -> Self {
        Self { trigger_manager }
    }
}

impl DmlHook for DmlHookBridge {
    fn before_insert(&self, table_id: u32, _tuples: &[&[u8]], _txn_id: u32) -> Result<bool> {
        // If no BEFORE INSERT triggers, allow the operation
        if !self
            .trigger_manager
            .hasTriggers(table_id, TriggerTiming::Before, TriggerEvent::Insert)
        {
            return Ok(true);
        }

        // Retrieve matching triggers. Trigger function bodies require
        // UDF registry to resolve the function name to callable code.
        let _triggers = self.trigger_manager.getMatchingTriggers(
            table_id,
            TriggerTiming::Before,
            TriggerEvent::Insert,
        );

        // Trigger function bodies are retrieved but execution requires the UDF
        // registry to resolve the function name to callable code. Allow the
        // operation to proceed. When UDF execution is wired, triggers that
        // return NULL or raise an exception will cancel the operation.
        Ok(true)
    }

    fn before_delete(&self, table_id: u32, _old_data: &[&[u8]], _txn_id: u32) -> Result<bool> {
        if !self
            .trigger_manager
            .hasTriggers(table_id, TriggerTiming::Before, TriggerEvent::Delete)
        {
            return Ok(true);
        }

        let _triggers = self.trigger_manager.getMatchingTriggers(
            table_id,
            TriggerTiming::Before,
            TriggerEvent::Delete,
        );

        Ok(true)
    }

    fn before_update(
        &self,
        table_id: u32,
        _old_data: &[&[u8]],
        _new_data: &[&[u8]],
        _txn_id: u32,
    ) -> Result<bool> {
        if !self
            .trigger_manager
            .hasTriggers(table_id, TriggerTiming::Before, TriggerEvent::Update)
        {
            return Ok(true);
        }

        let _triggers = self.trigger_manager.getMatchingTriggers(
            table_id,
            TriggerTiming::Before,
            TriggerEvent::Update,
        );

        Ok(true)
    }
}
