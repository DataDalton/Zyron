//! Graph schema definitions for ZyronDB's property graph model.
//!
//! Defines node labels, edge labels, and property definitions that
//! describe the structure of a graph stored in relational backing tables.

use serde::{Deserialize, Serialize};
use zyron_common::{Result, TypeId, ZyronError};

/// Unique identifier for a node within a graph.
pub type NodeId = u64;

/// Unique identifier for an edge within a graph.
pub type EdgeId = u64;

/// Identifier for a node or edge label within a graph schema.
pub type LabelId = u16;

/// Definition of a single property on a node or edge label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyDef {
    /// Property name.
    pub name: String,
    /// Data type of this property.
    pub type_id: TypeId,
    /// Whether this property allows null values.
    pub nullable: bool,
}

/// A node label describing a category of nodes and their properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLabel {
    /// Label identifier, unique within the graph schema.
    pub label_id: LabelId,
    /// Human-readable label name.
    pub name: String,
    /// Property definitions for nodes with this label.
    pub properties: Vec<PropertyDef>,
    /// Catalog TableId of the backing node table.
    pub node_table_id: u32,
}

/// An edge label describing a category of edges and their properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeLabel {
    /// Label identifier, unique within the graph schema.
    pub label_id: LabelId,
    /// Human-readable label name.
    pub name: String,
    /// Label ID of the source node type.
    pub from_label_id: LabelId,
    /// Label ID of the target node type.
    pub to_label_id: LabelId,
    /// Property definitions for edges with this label.
    pub properties: Vec<PropertyDef>,
    /// Catalog TableId of the backing edge table.
    pub edge_table_id: u32,
    /// Whether this edge type is directed (true) or undirected (false).
    pub directed: bool,
}

/// Complete schema for a property graph, containing node and edge labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSchema {
    /// Graph schema name.
    pub name: String,
    /// Catalog OID for this graph schema.
    pub schema_oid: u32,
    /// All registered node labels.
    pub node_labels: Vec<NodeLabel>,
    /// All registered edge labels.
    pub edge_labels: Vec<EdgeLabel>,
    /// Next label ID to assign (auto-incrementing counter).
    pub next_label_id: LabelId,
}

impl GraphSchema {
    /// Creates a new empty graph schema.
    pub fn new(name: String, schema_oid: u32) -> Self {
        Self {
            name,
            schema_oid,
            node_labels: Vec::new(),
            edge_labels: Vec::new(),
            next_label_id: 1,
        }
    }

    /// Registers a new node label and returns its assigned LabelId.
    pub fn add_node_label(
        &mut self,
        name: String,
        properties: Vec<PropertyDef>,
        node_table_id: u32,
    ) -> LabelId {
        let labelId = self.next_label_id;
        self.next_label_id += 1;
        self.node_labels.push(NodeLabel {
            label_id: labelId,
            name,
            properties,
            node_table_id,
        });
        labelId
    }

    /// Registers a new edge label and returns its assigned LabelId.
    /// Returns an error if the from_label_id or to_label_id do not reference existing node labels.
    pub fn add_edge_label(
        &mut self,
        name: String,
        from_label_id: LabelId,
        to_label_id: LabelId,
        properties: Vec<PropertyDef>,
        edge_table_id: u32,
        directed: bool,
    ) -> Result<LabelId> {
        if self.get_node_label_by_id(from_label_id).is_none() {
            return Err(ZyronError::GraphQueryError(format!(
                "source node label not found: {from_label_id}"
            )));
        }
        if self.get_node_label_by_id(to_label_id).is_none() {
            return Err(ZyronError::GraphQueryError(format!(
                "target node label not found: {to_label_id}"
            )));
        }
        let labelId = self.next_label_id;
        self.next_label_id += 1;
        self.edge_labels.push(EdgeLabel {
            label_id: labelId,
            name,
            from_label_id,
            to_label_id,
            properties,
            edge_table_id,
            directed,
        });
        Ok(labelId)
    }

    /// Looks up a node label by name.
    pub fn get_node_label(&self, name: &str) -> Option<&NodeLabel> {
        self.node_labels.iter().find(|l| l.name == name)
    }

    /// Looks up an edge label by name.
    pub fn get_edge_label(&self, name: &str) -> Option<&EdgeLabel> {
        self.edge_labels.iter().find(|l| l.name == name)
    }

    /// Looks up a node label by its LabelId.
    pub fn get_node_label_by_id(&self, id: LabelId) -> Option<&NodeLabel> {
        self.node_labels.iter().find(|l| l.label_id == id)
    }

    /// Looks up an edge label by its LabelId.
    pub fn get_edge_label_by_id(&self, id: LabelId) -> Option<&EdgeLabel> {
        self.edge_labels.iter().find(|l| l.label_id == id)
    }

    /// Serializes this graph schema to a binary format for catalog persistence.
    ///
    /// Format: 2-byte name length + name bytes + 4-byte oid +
    ///         2-byte node_label_count + [node_labels...] +
    ///         2-byte edge_label_count + [edge_labels...] +
    ///         2-byte next_label_id
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Schema name
        write_string(&mut buf, &self.name);
        // Schema OID
        buf.extend_from_slice(&self.schema_oid.to_le_bytes());

        // Node labels
        buf.extend_from_slice(&(self.node_labels.len() as u16).to_le_bytes());
        for nl in &self.node_labels {
            write_node_label(&mut buf, nl);
        }

        // Edge labels
        buf.extend_from_slice(&(self.edge_labels.len() as u16).to_le_bytes());
        for el in &self.edge_labels {
            write_edge_label(&mut buf, el);
        }

        // Next label ID
        buf.extend_from_slice(&self.next_label_id.to_le_bytes());

        buf
    }

    /// Deserializes a graph schema from binary data produced by to_bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;

        let name = read_string(data, &mut off)?;
        let schemaOid = read_u32(data, &mut off)?;

        let nodeCount = read_u16(data, &mut off)? as usize;
        let mut nodeLabels = Vec::with_capacity(nodeCount);
        for _ in 0..nodeCount {
            nodeLabels.push(read_node_label(data, &mut off)?);
        }

        let edgeCount = read_u16(data, &mut off)? as usize;
        let mut edgeLabels = Vec::with_capacity(edgeCount);
        for _ in 0..edgeCount {
            edgeLabels.push(read_edge_label(data, &mut off)?);
        }

        let nextLabelId = read_u16(data, &mut off)?;

        Ok(Self {
            name,
            schema_oid: schemaOid,
            node_labels: nodeLabels,
            edge_labels: edgeLabels,
            next_label_id: nextLabelId,
        })
    }
}

// ---------------------------------------------------------------------------
// Binary serialization helpers
// ---------------------------------------------------------------------------

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u16).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_property_def(buf: &mut Vec<u8>, prop: &PropertyDef) {
    write_string(buf, &prop.name);
    buf.push(prop.type_id as u8);
    buf.push(if prop.nullable { 1 } else { 0 });
}

fn write_node_label(buf: &mut Vec<u8>, nl: &NodeLabel) {
    buf.extend_from_slice(&nl.label_id.to_le_bytes());
    write_string(buf, &nl.name);
    buf.extend_from_slice(&nl.node_table_id.to_le_bytes());
    buf.extend_from_slice(&(nl.properties.len() as u16).to_le_bytes());
    for prop in &nl.properties {
        write_property_def(buf, prop);
    }
}

fn write_edge_label(buf: &mut Vec<u8>, el: &EdgeLabel) {
    buf.extend_from_slice(&el.label_id.to_le_bytes());
    write_string(buf, &el.name);
    buf.extend_from_slice(&el.from_label_id.to_le_bytes());
    buf.extend_from_slice(&el.to_label_id.to_le_bytes());
    buf.extend_from_slice(&el.edge_table_id.to_le_bytes());
    buf.push(if el.directed { 1 } else { 0 });
    buf.extend_from_slice(&(el.properties.len() as u16).to_le_bytes());
    for prop in &el.properties {
        write_property_def(buf, prop);
    }
}

fn ensure_remaining(data: &[u8], off: usize, need: usize) -> Result<()> {
    if off + need > data.len() {
        return Err(ZyronError::GraphQueryError(format!(
            "graph schema bytes truncated at offset {off}, need {need} more bytes"
        )));
    }
    Ok(())
}

fn read_u16(data: &[u8], off: &mut usize) -> Result<u16> {
    ensure_remaining(data, *off, 2)?;
    let val = u16::from_le_bytes([data[*off], data[*off + 1]]);
    *off += 2;
    Ok(val)
}

fn read_u32(data: &[u8], off: &mut usize) -> Result<u32> {
    ensure_remaining(data, *off, 4)?;
    let val = u32::from_le_bytes([data[*off], data[*off + 1], data[*off + 2], data[*off + 3]]);
    *off += 4;
    Ok(val)
}

fn read_u8(data: &[u8], off: &mut usize) -> Result<u8> {
    ensure_remaining(data, *off, 1)?;
    let val = data[*off];
    *off += 1;
    Ok(val)
}

fn read_string(data: &[u8], off: &mut usize) -> Result<String> {
    let len = read_u16(data, off)? as usize;
    ensure_remaining(data, *off, len)?;
    let s = std::str::from_utf8(&data[*off..*off + len]).map_err(|e| {
        ZyronError::GraphQueryError(format!(
            "invalid UTF-8 in graph schema at offset {off}: {e}"
        ))
    })?;
    *off += len;
    Ok(s.to_string())
}

fn type_id_from_u8(val: u8) -> Result<TypeId> {
    match val {
        0 => Ok(TypeId::Null),
        1 => Ok(TypeId::Boolean),
        10 => Ok(TypeId::Int8),
        11 => Ok(TypeId::Int16),
        12 => Ok(TypeId::Int32),
        13 => Ok(TypeId::Int64),
        14 => Ok(TypeId::Int128),
        20 => Ok(TypeId::UInt8),
        21 => Ok(TypeId::UInt16),
        22 => Ok(TypeId::UInt32),
        23 => Ok(TypeId::UInt64),
        24 => Ok(TypeId::UInt128),
        30 => Ok(TypeId::Float32),
        31 => Ok(TypeId::Float64),
        40 => Ok(TypeId::Decimal),
        50 => Ok(TypeId::Char),
        51 => Ok(TypeId::Varchar),
        52 => Ok(TypeId::Text),
        60 => Ok(TypeId::Binary),
        61 => Ok(TypeId::Varbinary),
        62 => Ok(TypeId::Bytea),
        70 => Ok(TypeId::Date),
        71 => Ok(TypeId::Time),
        72 => Ok(TypeId::Timestamp),
        73 => Ok(TypeId::TimestampTz),
        74 => Ok(TypeId::Interval),
        80 => Ok(TypeId::Uuid),
        90 => Ok(TypeId::Json),
        91 => Ok(TypeId::Jsonb),
        100 => Ok(TypeId::Array),
        110 => Ok(TypeId::Composite),
        120 => Ok(TypeId::Vector),
        _ => Err(ZyronError::GraphQueryError(format!(
            "unknown TypeId value in graph schema: {val}"
        ))),
    }
}

fn read_property_def(data: &[u8], off: &mut usize) -> Result<PropertyDef> {
    let name = read_string(data, off)?;
    let typeIdByte = read_u8(data, off)?;
    let typeId = type_id_from_u8(typeIdByte)?;
    let nullable = read_u8(data, off)? != 0;
    Ok(PropertyDef {
        name,
        type_id: typeId,
        nullable,
    })
}

fn read_node_label(data: &[u8], off: &mut usize) -> Result<NodeLabel> {
    let labelId = read_u16(data, off)?;
    let name = read_string(data, off)?;
    let nodeTableId = read_u32(data, off)?;
    let propCount = read_u16(data, off)? as usize;
    let mut properties = Vec::with_capacity(propCount);
    for _ in 0..propCount {
        properties.push(read_property_def(data, off)?);
    }
    Ok(NodeLabel {
        label_id: labelId,
        name,
        properties,
        node_table_id: nodeTableId,
    })
}

fn read_edge_label(data: &[u8], off: &mut usize) -> Result<EdgeLabel> {
    let labelId = read_u16(data, off)?;
    let name = read_string(data, off)?;
    let fromLabelId = read_u16(data, off)?;
    let toLabelId = read_u16(data, off)?;
    let edgeTableId = read_u32(data, off)?;
    let directed = read_u8(data, off)? != 0;
    let propCount = read_u16(data, off)? as usize;
    let mut properties = Vec::with_capacity(propCount);
    for _ in 0..propCount {
        properties.push(read_property_def(data, off)?);
    }
    Ok(EdgeLabel {
        label_id: labelId,
        name,
        from_label_id: fromLabelId,
        to_label_id: toLabelId,
        properties,
        edge_table_id: edgeTableId,
        directed,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_properties() -> Vec<PropertyDef> {
        vec![
            PropertyDef {
                name: "id".to_string(),
                type_id: TypeId::Int64,
                nullable: false,
            },
            PropertyDef {
                name: "name".to_string(),
                type_id: TypeId::Varchar,
                nullable: true,
            },
        ]
    }

    #[test]
    fn test_new_graph_schema() {
        let schema = GraphSchema::new("social".to_string(), 500);
        assert_eq!(schema.name, "social");
        assert_eq!(schema.schema_oid, 500);
        assert!(schema.node_labels.is_empty());
        assert!(schema.edge_labels.is_empty());
        assert_eq!(schema.next_label_id, 1);
    }

    #[test]
    fn test_add_node_label() {
        let mut schema = GraphSchema::new("social".to_string(), 500);
        let props = make_test_properties();
        let id1 = schema.add_node_label("Person".to_string(), props.clone(), 200);
        assert_eq!(id1, 1);
        assert_eq!(schema.next_label_id, 2);
        assert_eq!(schema.node_labels.len(), 1);

        let id2 = schema.add_node_label("Company".to_string(), vec![], 202);
        assert_eq!(id2, 2);
        assert_eq!(schema.node_labels.len(), 2);
    }

    #[test]
    fn test_add_edge_label() {
        let mut schema = GraphSchema::new("social".to_string(), 500);
        let personId = schema.add_node_label("Person".to_string(), vec![], 200);
        let companyId = schema.add_node_label("Company".to_string(), vec![], 202);

        let edgeProps = vec![PropertyDef {
            name: "since".to_string(),
            type_id: TypeId::Date,
            nullable: true,
        }];
        let edgeId = schema
            .add_edge_label(
                "WORKS_AT".to_string(),
                personId,
                companyId,
                edgeProps,
                10001,
                true,
            )
            .expect("add_edge_label should succeed");
        assert_eq!(edgeId, 3);
        assert_eq!(schema.edge_labels.len(), 1);
        assert!(schema.edge_labels[0].directed);
    }

    #[test]
    fn test_add_edge_label_missing_from() {
        let mut schema = GraphSchema::new("g".to_string(), 1);
        schema.add_node_label("A".to_string(), vec![], 200);
        let result = schema.add_edge_label("E".to_string(), 99, 1, vec![], 10001, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_edge_label_missing_to() {
        let mut schema = GraphSchema::new("g".to_string(), 1);
        schema.add_node_label("A".to_string(), vec![], 200);
        let result = schema.add_edge_label("E".to_string(), 1, 99, vec![], 10001, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_node_label_by_name() {
        let mut schema = GraphSchema::new("g".to_string(), 1);
        schema.add_node_label("Person".to_string(), vec![], 200);
        schema.add_node_label("Company".to_string(), vec![], 202);

        let found = schema.get_node_label("Person");
        assert!(found.is_some());
        assert_eq!(found.expect("should exist").name, "Person");

        assert!(schema.get_node_label("Missing").is_none());
    }

    #[test]
    fn test_get_edge_label_by_name() {
        let mut schema = GraphSchema::new("g".to_string(), 1);
        let p = schema.add_node_label("Person".to_string(), vec![], 200);
        let c = schema.add_node_label("Company".to_string(), vec![], 202);
        schema
            .add_edge_label("WORKS_AT".to_string(), p, c, vec![], 10001, true)
            .expect("should succeed");

        let found = schema.get_edge_label("WORKS_AT");
        assert!(found.is_some());
        assert_eq!(found.expect("should exist").name, "WORKS_AT");

        assert!(schema.get_edge_label("MISSING").is_none());
    }

    #[test]
    fn test_get_node_label_by_id() {
        let mut schema = GraphSchema::new("g".to_string(), 1);
        let id = schema.add_node_label("Person".to_string(), vec![], 200);

        let found = schema.get_node_label_by_id(id);
        assert!(found.is_some());
        assert_eq!(found.expect("should exist").label_id, id);

        assert!(schema.get_node_label_by_id(999).is_none());
    }

    #[test]
    fn test_get_edge_label_by_id() {
        let mut schema = GraphSchema::new("g".to_string(), 1);
        let p = schema.add_node_label("Person".to_string(), vec![], 200);
        let c = schema.add_node_label("Company".to_string(), vec![], 202);
        let edgeId = schema
            .add_edge_label("WORKS_AT".to_string(), p, c, vec![], 10001, false)
            .expect("should succeed");

        let found = schema.get_edge_label_by_id(edgeId);
        assert!(found.is_some());
        let el = found.expect("should exist");
        assert_eq!(el.label_id, edgeId);
        assert!(!el.directed);

        assert!(schema.get_edge_label_by_id(999).is_none());
    }

    #[test]
    fn test_serialization_roundtrip_empty() {
        let schema = GraphSchema::new("empty_graph".to_string(), 42);
        let bytes = schema.to_bytes();
        let restored = GraphSchema::from_bytes(&bytes).expect("from_bytes should succeed");
        assert_eq!(restored.name, "empty_graph");
        assert_eq!(restored.schema_oid, 42);
        assert!(restored.node_labels.is_empty());
        assert!(restored.edge_labels.is_empty());
        assert_eq!(restored.next_label_id, 1);
    }

    #[test]
    fn test_serialization_roundtrip_with_labels() {
        let mut schema = GraphSchema::new("social".to_string(), 500);

        let personProps = vec![
            PropertyDef {
                name: "age".to_string(),
                type_id: TypeId::Int32,
                nullable: false,
            },
            PropertyDef {
                name: "email".to_string(),
                type_id: TypeId::Varchar,
                nullable: true,
            },
        ];
        let personId = schema.add_node_label("Person".to_string(), personProps, 200);
        let companyId = schema.add_node_label("Company".to_string(), vec![], 202);

        let edgeProps = vec![PropertyDef {
            name: "since".to_string(),
            type_id: TypeId::Date,
            nullable: true,
        }];
        schema
            .add_edge_label(
                "WORKS_AT".to_string(),
                personId,
                companyId,
                edgeProps,
                10001,
                true,
            )
            .expect("should succeed");
        schema
            .add_edge_label(
                "KNOWS".to_string(),
                personId,
                personId,
                vec![],
                10002,
                false,
            )
            .expect("should succeed");

        let bytes = schema.to_bytes();
        let restored = GraphSchema::from_bytes(&bytes).expect("from_bytes should succeed");

        assert_eq!(restored.name, "social");
        assert_eq!(restored.schema_oid, 500);
        assert_eq!(restored.node_labels.len(), 2);
        assert_eq!(restored.edge_labels.len(), 2);
        assert_eq!(restored.next_label_id, schema.next_label_id);

        // Verify node labels
        let person = restored
            .get_node_label("Person")
            .expect("Person label should exist");
        assert_eq!(person.label_id, personId);
        assert_eq!(person.node_table_id, 200);
        assert_eq!(person.properties.len(), 2);
        assert_eq!(person.properties[0].name, "age");
        assert_eq!(person.properties[0].type_id, TypeId::Int32);
        assert!(!person.properties[0].nullable);
        assert_eq!(person.properties[1].name, "email");
        assert_eq!(person.properties[1].type_id, TypeId::Varchar);
        assert!(person.properties[1].nullable);

        let company = restored
            .get_node_label("Company")
            .expect("Company label should exist");
        assert_eq!(company.node_table_id, 202);
        assert!(company.properties.is_empty());

        // Verify edge labels
        let worksAt = restored
            .get_edge_label("WORKS_AT")
            .expect("WORKS_AT label should exist");
        assert_eq!(worksAt.from_label_id, personId);
        assert_eq!(worksAt.to_label_id, companyId);
        assert_eq!(worksAt.edge_table_id, 10001);
        assert!(worksAt.directed);
        assert_eq!(worksAt.properties.len(), 1);
        assert_eq!(worksAt.properties[0].name, "since");

        let knows = restored
            .get_edge_label("KNOWS")
            .expect("KNOWS label should exist");
        assert_eq!(knows.from_label_id, personId);
        assert_eq!(knows.to_label_id, personId);
        assert!(!knows.directed);
        assert!(knows.properties.is_empty());
    }

    #[test]
    fn test_from_bytes_truncated() {
        let result = GraphSchema::from_bytes(&[0x01]);
        assert!(result.is_err());
    }

    #[test]
    fn test_property_def_all_types() {
        // Verify that property defs with various TypeId values round-trip correctly.
        let types = vec![
            TypeId::Boolean,
            TypeId::Int64,
            TypeId::Float64,
            TypeId::Varchar,
            TypeId::Uuid,
            TypeId::Jsonb,
            TypeId::Vector,
        ];
        let mut schema = GraphSchema::new("types_test".to_string(), 1);
        let props: Vec<PropertyDef> = types
            .iter()
            .enumerate()
            .map(|(i, t)| PropertyDef {
                name: format!("col_{i}"),
                type_id: *t,
                nullable: i % 2 == 0,
            })
            .collect();
        schema.add_node_label("TypeNode".to_string(), props.clone(), 300);

        let bytes = schema.to_bytes();
        let restored = GraphSchema::from_bytes(&bytes).expect("from_bytes should succeed");
        let restoredLabel = restored
            .get_node_label("TypeNode")
            .expect("TypeNode should exist");
        assert_eq!(restoredLabel.properties.len(), props.len());
        for (i, prop) in restoredLabel.properties.iter().enumerate() {
            assert_eq!(prop.name, props[i].name);
            assert_eq!(prop.type_id, props[i].type_id);
            assert_eq!(prop.nullable, props[i].nullable);
        }
    }
}
