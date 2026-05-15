#![allow(non_snake_case)]
// Correctness tests for CREATE FEATURE GROUP and CREATE MODEL syntax

use zyron_parser::Parser;
use zyron_parser::Statement;

fn parseOne(sql: &str) -> Statement {
    let mut p = Parser::new(sql).expect("lexer init");
    p.parse_statement().expect("parse")
}

#[test]
fn createFeatureGroupBasic() {
    let sql = "CREATE FEATURE GROUP user_features (\
        ENTITY KEY user_id, \
        FEATURES (\
            total_purchases AS (1), \
            avg_order_value AS (2)\
        ), \
        REFRESH EVERY '1 hour'\
    )";
    let stmt = parseOne(sql);
    match stmt {
        Statement::CreateFeatureGroup(s) => {
            assert_eq!(s.name, "user_features");
            assert_eq!(s.entity_key, "user_id");
            assert_eq!(s.features.len(), 2);
            assert_eq!(s.features[0].name, "total_purchases");
            assert_eq!(s.features[1].name, "avg_order_value");
            assert_eq!(s.refresh_interval.as_deref(), Some("1 hour"));
        }
        other => panic!("expected CreateFeatureGroup, got {:?}", other),
    }
}

#[test]
fn createFeatureGroupIfNotExists() {
    let sql = "CREATE FEATURE GROUP IF NOT EXISTS g (\
        ENTITY KEY id, \
        FEATURES (a AS (1))\
    )";
    let stmt = parseOne(sql);
    match stmt {
        Statement::CreateFeatureGroup(s) => {
            assert!(s.if_not_exists);
            assert_eq!(s.name, "g");
        }
        other => panic!("expected CreateFeatureGroup, got {:?}", other),
    }
}

#[test]
fn createFeatureGroupWithOptions() {
    let sql = "CREATE FEATURE GROUP g (\
        ENTITY KEY id, \
        FEATURES (a AS (1)), \
        REFRESH EVERY '30 minutes', \
        WITH (retention_days = 90, max_staleness_seconds = 1800)\
    )";
    let stmt = parseOne(sql);
    match stmt {
        Statement::CreateFeatureGroup(s) => {
            assert_eq!(s.options.len(), 2);
            assert_eq!(s.options[0].key, "retention_days");
        }
        other => panic!("expected CreateFeatureGroup, got {:?}", other),
    }
}

#[test]
fn dropFeatureGroup() {
    let stmt = parseOne("DROP FEATURE GROUP IF EXISTS g");
    match stmt {
        Statement::DropFeatureGroup(s) => {
            assert!(s.if_exists);
            assert_eq!(s.name, "g");
        }
        other => panic!("expected DropFeatureGroup, got {:?}", other),
    }
}

#[test]
fn createModelBasic() {
    let sql = "CREATE MODEL churn TYPE logistic_regression \
        FEATURES (total_purchases, avg_order_value) \
        TARGET is_churned \
        USING (SELECT * FROM training_data)";
    let stmt = parseOne(sql);
    match stmt {
        Statement::CreateModel(s) => {
            assert_eq!(s.name, "churn");
            assert_eq!(s.model_type, "logistic_regression");
            assert_eq!(s.features.len(), 2);
            assert_eq!(s.target.as_deref(), Some("is_churned"));
            assert!(s.training_query.is_some());
        }
        other => panic!("expected CreateModel, got {:?}", other),
    }
}

#[test]
fn createModelWithHyperparams() {
    let sql = "CREATE MODEL m TYPE linear_regression \
        FEATURES (x) \
        USING (SELECT x, y FROM data) \
        WITH (lambda = 1, learning_rate = 1, max_epochs = 100)";
    let stmt = parseOne(sql);
    match stmt {
        Statement::CreateModel(s) => {
            assert_eq!(s.options.len(), 3);
            assert_eq!(s.options[0].key, "lambda");
        }
        other => panic!("expected CreateModel, got {:?}", other),
    }
}

#[test]
fn dropModel() {
    let stmt = parseOne("DROP MODEL IF EXISTS churn");
    match stmt {
        Statement::DropModel(s) => {
            assert!(s.if_exists);
            assert_eq!(s.name, "churn");
        }
        other => panic!("expected DropModel, got {:?}", other),
    }
}

#[test]
fn createFeatureGroupWithSourceQuery() {
    let sql = "CREATE FEATURE GROUP g (\
        ENTITY KEY id, \
        FEATURES (a AS (1)), \
        SOURCE AS (SELECT id, x FROM raw)\
    )";
    let stmt = parseOne(sql);
    match stmt {
        Statement::CreateFeatureGroup(s) => {
            assert!(s.source_query.is_some());
        }
        other => panic!("expected CreateFeatureGroup, got {:?}", other),
    }
}
