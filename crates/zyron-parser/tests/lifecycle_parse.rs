//! Parser coverage for Phase 17 data-lifecycle SQL.

use zyron_parser::ast::*;
use zyron_parser::parse;

fn one(sql: &str) -> Statement {
    let mut s = parse(sql).expect("parse failed");
    assert_eq!(s.len(), 1, "expected exactly one statement: {sql}");
    s.pop().unwrap()
}

#[test]
fn legal_hold_create_drop_release() {
    match one("LEGAL HOLD CREATE h1 ON transactions WHERE customer_id = 5 REASON 'litigation'") {
        Statement::LegalHold(b) => match b.operation {
            LegalHoldOperation::Create {
                name,
                table,
                where_clause,
                reason,
            } => {
                assert_eq!(name, "h1");
                assert_eq!(table, "transactions");
                assert!(where_clause.is_some());
                assert_eq!(reason.as_deref(), Some("litigation"));
            }
            _ => panic!("expected Create"),
        },
        _ => panic!("expected LegalHold"),
    }
    assert!(matches!(
        one("LEGAL HOLD RELEASE h1"),
        Statement::LegalHold(_)
    ));
    assert!(matches!(
        one("LEGAL HOLD DROP IF EXISTS h1"),
        Statement::LegalHold(_)
    ));
}

#[test]
fn forget_and_export_user() {
    match one("FORGET USER 'u-1' CASCADE DRY RUN") {
        Statement::ForgetUser(b) => {
            assert_eq!(b.user_id, "u-1");
            assert!(b.cascade && b.dry_run);
        }
        _ => panic!("expected ForgetUser"),
    }
    match one("EXPORT USER 'u-1' TO 'fs:///tmp/dsar'") {
        Statement::ExportUser(b) => {
            assert_eq!(b.user_id, "u-1");
            assert_eq!(b.destination.as_deref(), Some("fs:///tmp/dsar"));
        }
        _ => panic!("expected ExportUser"),
    }
}

#[test]
fn alter_table_move_and_classification() {
    match one("ALTER TABLE logs MOVE WHERE id < 100 TO TIER 'cold'") {
        Statement::AlterTableMove(b) => {
            assert_eq!(b.table, "logs");
            assert_eq!(b.tier, "cold");
            assert!(matches!(b.target, MoveTarget::Where(_)));
        }
        _ => panic!("expected AlterTableMove"),
    }
    match one("ALTER TABLE logs MOVE PARTITION 'year=2023' TO TIER 'cold'") {
        Statement::AlterTableMove(b) => {
            assert!(matches!(b.target, MoveTarget::Partition(_)));
        }
        _ => panic!("expected AlterTableMove"),
    }
    match one("ALTER TABLE customers ALTER COLUMN email SET CLASSIFICATION 'confidential'") {
        Statement::AlterColumnClassification(b) => {
            assert_eq!(b.table, "customers");
            assert_eq!(b.column, "email");
            assert_eq!(b.level, "confidential");
        }
        _ => panic!("expected AlterColumnClassification"),
    }
}

#[test]
fn ttl_clause_create_and_alter() {
    match one("CREATE TABLE e (id INT, created_at TIMESTAMPTZ) TTL 30 DAYS ON created_at") {
        Statement::CreateTable(b) => {
            let ttl = b.ttl.expect("ttl clause");
            assert_eq!(ttl.column, "created_at");
            assert_eq!(ttl.action, TtlAction::Delete);
        }
        _ => panic!("expected CreateTable"),
    }
    match one("ALTER TABLE e SET TTL 7 DAYS ON created_at ACTION ANONYMIZE") {
        Statement::AlterTableTtl(b) => match b.operation {
            TtlOperation::Set { action, .. } => assert_eq!(action, TtlAction::Anonymize),
            _ => panic!("expected Set"),
        },
        _ => panic!("expected AlterTableTtl"),
    }
}

#[test]
fn soft_delete_modifiers_and_hard_delete() {
    match one("SELECT * FROM t INCLUDING DELETED") {
        Statement::Select(b) => {
            assert_eq!(b.soft_delete_mode, SoftDeleteSelectMode::IncludingDeleted)
        }
        _ => panic!("expected Select"),
    }
    match one("SELECT * FROM t ONLY DELETED") {
        Statement::Select(b) => {
            assert_eq!(b.soft_delete_mode, SoftDeleteSelectMode::OnlyDeleted)
        }
        _ => panic!("expected Select"),
    }
    match one("SELECT * FROM t") {
        Statement::Select(b) => {
            assert_eq!(b.soft_delete_mode, SoftDeleteSelectMode::Default)
        }
        _ => panic!("expected Select"),
    }
    match one("DELETE FROM t WHERE id = 1 HARD") {
        Statement::Delete(b) => assert!(b.hard),
        _ => panic!("expected Delete"),
    }
    match one("DELETE FROM t WHERE id = 1") {
        Statement::Delete(b) => assert!(!b.hard),
        _ => panic!("expected Delete"),
    }
}

#[test]
fn restore_forms_and_run_retention_and_undrop() {
    assert!(matches!(
        one("RESTORE FROM t WHERE id = 1"),
        Statement::RestoreSoftDelete(_)
    ));
    assert!(matches!(
        one("RESTORE TABLE t FROM 'fs:///a/b.parquet' INTO t2"),
        Statement::RestoreTable(_)
    ));
    assert!(matches!(one("RESTORE TABLE t"), Statement::UndropTable(_)));
    assert!(matches!(one("UNDROP TABLE t"), Statement::UndropTable(_)));
    match one("RUN RETENTION JOB ON events DRY RUN") {
        Statement::RunRetentionJob(b) => {
            assert_eq!(b.table.as_deref(), Some("events"));
            assert!(b.dry_run);
        }
        _ => panic!("expected RunRetentionJob"),
    }
    match one("ARCHIVE TABLE t WHERE id > 0 TO 's3://bucket/arch/' DRY RUN") {
        Statement::ArchiveTable(b) => {
            assert_eq!(b.table, "t");
            assert!(b.dry_run);
        }
        _ => panic!("expected ArchiveTable"),
    }
}
