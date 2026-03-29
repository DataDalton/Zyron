//! System table file ID constants and storage layer for pipeline metadata persistence.

// System table file ID assignments for the pipeline crate.
// Each system table has a paired heap file and free space map (FSM) file.
// Auth crate uses IDs 110-159, so pipeline starts at 160.

pub const PIPELINES_HEAP_FILE_ID: u32 = 160;
pub const PIPELINES_FSM_FILE_ID: u32 = 161;
pub const PIPELINE_STAGES_HEAP_FILE_ID: u32 = 162;
pub const PIPELINE_STAGES_FSM_FILE_ID: u32 = 163;
pub const MATERIALIZED_VIEWS_HEAP_FILE_ID: u32 = 164;
pub const MATERIALIZED_VIEWS_FSM_FILE_ID: u32 = 165;
pub const QUALITY_CHECKS_HEAP_FILE_ID: u32 = 166;
pub const QUALITY_CHECKS_FSM_FILE_ID: u32 = 167;
pub const WATERMARKS_HEAP_FILE_ID: u32 = 168;
pub const WATERMARKS_FSM_FILE_ID: u32 = 169;
pub const SCHEDULES_HEAP_FILE_ID: u32 = 170;
pub const SCHEDULES_FSM_FILE_ID: u32 = 171;
pub const TRIGGERS_HEAP_FILE_ID: u32 = 172;
pub const TRIGGERS_FSM_FILE_ID: u32 = 173;
pub const FUNCTIONS_HEAP_FILE_ID: u32 = 174;
pub const FUNCTIONS_FSM_FILE_ID: u32 = 175;
pub const AGGREGATES_HEAP_FILE_ID: u32 = 176;
pub const AGGREGATES_FSM_FILE_ID: u32 = 177;
pub const EVENT_HANDLERS_HEAP_FILE_ID: u32 = 178;
pub const EVENT_HANDLERS_FSM_FILE_ID: u32 = 179;
pub const STORED_PROCEDURES_HEAP_FILE_ID: u32 = 180;
pub const STORED_PROCEDURES_FSM_FILE_ID: u32 = 181;
pub const DEPENDENCIES_HEAP_FILE_ID: u32 = 182;
pub const DEPENDENCIES_FSM_FILE_ID: u32 = 183;
pub const TRIGGER_TRACE_HEAP_FILE_ID: u32 = 184;
pub const TRIGGER_TRACE_FSM_FILE_ID: u32 = 185;
pub const LINEAGE_HEAP_FILE_ID: u32 = 186;
pub const LINEAGE_FSM_FILE_ID: u32 = 187;
pub const QUALITY_HISTORY_HEAP_FILE_ID: u32 = 188;
pub const QUALITY_HISTORY_FSM_FILE_ID: u32 = 189;
pub const PIPELINE_SLA_HEAP_FILE_ID: u32 = 190;
pub const PIPELINE_SLA_FSM_FILE_ID: u32 = 191;
pub const MV_ADVISOR_HEAP_FILE_ID: u32 = 192;
pub const MV_ADVISOR_FSM_FILE_ID: u32 = 193;

/// Descriptor for a system table file pair (heap + FSM).
#[derive(Debug, Clone)]
pub struct SystemTableFiles {
    pub name: &'static str,
    pub heap_file_id: u32,
    pub fsm_file_id: u32,
}

/// All system tables managed by the pipeline crate.
pub const SYSTEM_TABLES: &[SystemTableFiles] = &[
    SystemTableFiles {
        name: "zyron_pipelines",
        heap_file_id: PIPELINES_HEAP_FILE_ID,
        fsm_file_id: PIPELINES_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_pipeline_stages",
        heap_file_id: PIPELINE_STAGES_HEAP_FILE_ID,
        fsm_file_id: PIPELINE_STAGES_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_materialized_views",
        heap_file_id: MATERIALIZED_VIEWS_HEAP_FILE_ID,
        fsm_file_id: MATERIALIZED_VIEWS_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_quality_checks",
        heap_file_id: QUALITY_CHECKS_HEAP_FILE_ID,
        fsm_file_id: QUALITY_CHECKS_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_watermarks",
        heap_file_id: WATERMARKS_HEAP_FILE_ID,
        fsm_file_id: WATERMARKS_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_schedules",
        heap_file_id: SCHEDULES_HEAP_FILE_ID,
        fsm_file_id: SCHEDULES_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_triggers",
        heap_file_id: TRIGGERS_HEAP_FILE_ID,
        fsm_file_id: TRIGGERS_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_functions",
        heap_file_id: FUNCTIONS_HEAP_FILE_ID,
        fsm_file_id: FUNCTIONS_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_aggregates",
        heap_file_id: AGGREGATES_HEAP_FILE_ID,
        fsm_file_id: AGGREGATES_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_event_handlers",
        heap_file_id: EVENT_HANDLERS_HEAP_FILE_ID,
        fsm_file_id: EVENT_HANDLERS_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_stored_procedures",
        heap_file_id: STORED_PROCEDURES_HEAP_FILE_ID,
        fsm_file_id: STORED_PROCEDURES_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_dependencies",
        heap_file_id: DEPENDENCIES_HEAP_FILE_ID,
        fsm_file_id: DEPENDENCIES_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_trigger_trace",
        heap_file_id: TRIGGER_TRACE_HEAP_FILE_ID,
        fsm_file_id: TRIGGER_TRACE_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_lineage",
        heap_file_id: LINEAGE_HEAP_FILE_ID,
        fsm_file_id: LINEAGE_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_quality_history",
        heap_file_id: QUALITY_HISTORY_HEAP_FILE_ID,
        fsm_file_id: QUALITY_HISTORY_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_pipeline_sla",
        heap_file_id: PIPELINE_SLA_HEAP_FILE_ID,
        fsm_file_id: PIPELINE_SLA_FSM_FILE_ID,
    },
    SystemTableFiles {
        name: "zyron_mv_advisor",
        heap_file_id: MV_ADVISOR_HEAP_FILE_ID,
        fsm_file_id: MV_ADVISOR_FSM_FILE_ID,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_id_continuity() {
        // All heap IDs should be even, FSM IDs should be heap+1
        for table in SYSTEM_TABLES {
            assert_eq!(
                table.heap_file_id % 2,
                0,
                "Heap file ID for {} should be even",
                table.name
            );
            assert_eq!(
                table.fsm_file_id,
                table.heap_file_id + 1,
                "FSM file ID for {} should be heap+1",
                table.name
            );
        }
    }

    #[test]
    fn test_no_id_overlap() {
        let mut ids: Vec<u32> = SYSTEM_TABLES
            .iter()
            .flat_map(|t| vec![t.heap_file_id, t.fsm_file_id])
            .collect();
        ids.sort();
        ids.dedup();
        assert_eq!(
            ids.len(),
            SYSTEM_TABLES.len() * 2,
            "File IDs should be unique"
        );
    }

    #[test]
    fn test_system_table_count() {
        assert_eq!(SYSTEM_TABLES.len(), 17);
    }

    #[test]
    fn test_id_range() {
        for table in SYSTEM_TABLES {
            assert!(
                table.heap_file_id >= 160,
                "{} heap ID below 160",
                table.name
            );
            assert!(table.fsm_file_id <= 193, "{} FSM ID above 193", table.name);
        }
    }
}
