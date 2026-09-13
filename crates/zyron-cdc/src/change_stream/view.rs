//! Change streams over a view.
//!
//! A view that resolves to one base table with an optional WHERE and a
//! projection is the filtered-bronze shape, and a stream over it reads that
//! base table's feed and applies the view's predicate and projection. A view
//! carrying a join, an aggregate, a set operation or a subquery is refused at
//! CREATE naming the construct: reproducing one of those from a change set is
//! incremental view maintenance, which nothing here attempts.

use zyron_common::{Result, ZyronError};

/// What a view's query holds, as far as a change stream cares.
///
/// The binder fills this from the parsed definition. Holding the answer as
/// data rather than as a parsed tree keeps the refusal message in one place
/// and keeps this crate off the parser.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ViewShape {
    /// Base tables the query reads
    pub base_tables: Vec<u32>,
    /// True when the query joins
    pub has_join: bool,
    /// True when the query aggregates or groups
    pub has_aggregate: bool,
    /// True when the query is a UNION, INTERSECT or EXCEPT
    pub has_set_operation: bool,
    /// True when the query holds a subquery in any position
    pub has_subquery: bool,
    /// True when the query is DISTINCT, which collapses rows the same way an
    /// aggregate does
    pub has_distinct: bool,
    /// True when the query carries LIMIT or OFFSET, which makes a row's
    /// presence depend on rows a change set does not carry
    pub has_limit: bool,
    /// The WHERE clause as written, empty when the view has none
    pub predicate: Option<String>,
    /// Column ids the view projects, empty when it projects everything
    pub projection: Vec<u16>,
}

/// What a stream over a view reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewStreamPlan {
    pub base_table: u32,
    pub predicate: Option<String>,
    pub projection: Vec<u16>,
}

/// Decides whether a view can carry a change stream.
///
/// Accepts a single base table with an optional predicate and projection.
/// Every other construct is refused naming itself, and the message points at
/// the two things that do serve the case.
pub fn plan_view_stream(view_name: &str, shape: &ViewShape) -> Result<ViewStreamPlan> {
    let construct = if shape.has_join {
        Some("a join")
    } else if shape.has_aggregate {
        Some("an aggregate")
    } else if shape.has_set_operation {
        Some("a set operation")
    } else if shape.has_subquery {
        Some("a subquery")
    } else if shape.has_distinct {
        Some("DISTINCT")
    } else if shape.has_limit {
        Some("LIMIT")
    } else {
        None
    };
    if let Some(construct) = construct {
        return Err(ZyronError::CdcStreamError(format!(
            "view '{view_name}' carries {construct}, which a change stream cannot read. A stream \
             over a view reads one base table's changes and applies the view's predicate and \
             projection. Use CREATE CHANGE STREAM ... ON TABLES for several sources, or a \
             materialized view for a derived result"
        )));
    }
    match shape.base_tables.as_slice() {
        [base_table] => Ok(ViewStreamPlan {
            base_table: *base_table,
            predicate: shape.predicate.clone(),
            projection: shape.projection.clone(),
        }),
        [] => Err(ZyronError::CdcStreamError(format!(
            "view '{view_name}' reads no table, so there is nothing for a change stream to follow"
        ))),
        many => Err(ZyronError::CdcStreamError(format!(
            "view '{view_name}' reads {} tables, which a change stream cannot read. A stream over \
             a view reads one base table's changes and applies the view's predicate and \
             projection. Use CREATE CHANGE STREAM ... ON TABLES for several sources, or a \
             materialized view for a derived result",
            many.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_table() -> ViewShape {
        ViewShape {
            base_tables: vec![7],
            predicate: Some("region = 'eu'".into()),
            projection: vec![0, 2],
            ..ViewShape::default()
        }
    }

    #[test]
    fn test_a_filtered_projection_over_one_table_is_accepted() {
        let plan = plan_view_stream("eu_orders", &single_table()).expect("accepted");
        assert_eq!(plan.base_table, 7);
        assert_eq!(plan.predicate.as_deref(), Some("region = 'eu'"));
        assert_eq!(plan.projection, vec![0, 2]);
    }

    #[test]
    fn test_every_refused_construct_names_itself() {
        let cases: [(&str, fn(&mut ViewShape)); 6] = [
            ("a join", |s| s.has_join = true),
            ("an aggregate", |s| s.has_aggregate = true),
            ("a set operation", |s| s.has_set_operation = true),
            ("a subquery", |s| s.has_subquery = true),
            ("DISTINCT", |s| s.has_distinct = true),
            ("LIMIT", |s| s.has_limit = true),
        ];
        for (construct, set) in cases {
            let mut shape = single_table();
            set(&mut shape);
            let text = plan_view_stream("v", &shape)
                .expect_err("refused")
                .to_string();
            assert!(
                text.contains(construct),
                "{construct} was not named: {text}"
            );
            assert!(text.contains("ON TABLES"), "{text}");
            assert!(text.contains("materialized view"), "{text}");
        }
    }

    #[test]
    fn test_a_view_over_several_tables_is_refused() {
        let shape = ViewShape {
            base_tables: vec![1, 2],
            ..ViewShape::default()
        };
        let text = plan_view_stream("v", &shape)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("reads 2 tables"), "{text}");
    }

    #[test]
    fn test_a_view_over_no_table_is_refused() {
        let text = plan_view_stream("v", &ViewShape::default())
            .expect_err("refused")
            .to_string();
        assert!(text.contains("reads no table"), "{text}");
    }
}
