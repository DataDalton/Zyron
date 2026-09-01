// -----------------------------------------------------------------------------
// Arrow extension type registry
// -----------------------------------------------------------------------------
//
// Canonical zyron.* extension names for Zyron types with no native Arrow
// analogue. Export stamps ARROW:extension:name onto field metadata over a
// declared storage DataType, import restores the TypeId from that name
// before falling back to raw DataType inference. One table drives both
// directions. Inet covers both v4 and v6 in a single TypeId, so it exports
// under the single name zyron.inet, the ip4 and ip6 spellings are accepted
// on import only

use super::ColumnSpec;
use super::schema::{arrow_to_type_id, timestamp_arrow_type};
use arrow::datatypes::{DataType as ArrowDataType, Field};
use zyron_common::{Result, TypeId};

/// Arrow field metadata key carrying the extension name
pub const EXTENSION_NAME_KEY: &str = "ARROW:extension:name";

/// One registry row, the canonical extension name, the TypeId it restores,
/// and the storage DataType the exported column carries
struct ExtensionEntry {
    name: &'static str,
    type_id: TypeId,
    storage: ArrowDataType,
}

// Storage types mirror the physical byte layout each TypeId ships through
// StreamValue so round trips stay byte exact. Fixed widths follow
// TypeId::fixed_size, variable length payloads stay Binary
const ENTRIES: &[ExtensionEntry] = &[
    ExtensionEntry {
        name: "zyron.uuid",
        type_id: TypeId::Uuid,
        storage: ArrowDataType::FixedSizeBinary(16),
    },
    ExtensionEntry {
        name: "zyron.money",
        type_id: TypeId::Money,
        storage: ArrowDataType::FixedSizeBinary(10),
    },
    ExtensionEntry {
        name: "zyron.inet",
        type_id: TypeId::Inet,
        storage: ArrowDataType::Binary,
    },
    ExtensionEntry {
        name: "zyron.cidr",
        type_id: TypeId::Cidr,
        storage: ArrowDataType::Binary,
    },
    ExtensionEntry {
        name: "zyron.macaddr",
        type_id: TypeId::MacAddr,
        storage: ArrowDataType::FixedSizeBinary(6),
    },
    ExtensionEntry {
        name: "zyron.hll",
        type_id: TypeId::HyperLogLog,
        storage: ArrowDataType::Binary,
    },
    ExtensionEntry {
        name: "zyron.tdigest",
        type_id: TypeId::TDigest,
        storage: ArrowDataType::Binary,
    },
    ExtensionEntry {
        name: "zyron.geometry",
        type_id: TypeId::Geometry,
        storage: ArrowDataType::Binary,
    },
    ExtensionEntry {
        name: "zyron.interval",
        type_id: TypeId::Interval,
        storage: ArrowDataType::FixedSizeBinary(16),
    },
    ExtensionEntry {
        name: "zyron.vector",
        type_id: TypeId::Vector,
        storage: ArrowDataType::Binary,
    },
];

// Import only spellings accepted in addition to the canonical names. Inet is
// one TypeId for v4 and v6, so files annotated with the split names restore
// to the same TypeId
const IMPORT_ALIASES: &[(&str, TypeId)] =
    &[("zyron.ip4", TypeId::Inet), ("zyron.ip6", TypeId::Inet)];

/// Returns the canonical extension name for a TypeId, None when the type has
/// no extension mapping
pub fn extension_name_for(type_id: TypeId) -> Option<&'static str> {
    ENTRIES
        .iter()
        .find(|e| e.type_id == type_id)
        .map(|e| e.name)
}

/// Resolves an extension name back to its TypeId, accepting canonical names
/// and import aliases. None for unknown names
pub fn type_id_for_extension(name: &str) -> Option<TypeId> {
    if let Some(e) = ENTRIES.iter().find(|e| e.name == name) {
        return Some(e.type_id);
    }
    IMPORT_ALIASES
        .iter()
        .find(|(alias, _)| *alias == name)
        .map(|(_, t)| *t)
}

/// Returns the declared storage DataType for a TypeId with an extension
/// mapping, None otherwise
pub fn storage_type_for(type_id: TypeId) -> Option<&'static ArrowDataType> {
    ENTRIES
        .iter()
        .find(|e| e.type_id == type_id)
        .map(|e| &e.storage)
}

/// Attaches the extension name to the field metadata when the TypeId has an
/// extension mapping, otherwise returns the field unchanged
pub fn annotate_field(field: Field, type_id: TypeId) -> Field {
    match extension_name_for(type_id) {
        Some(name) => {
            let mut metadata = field.metadata().clone();
            metadata.insert(EXTENSION_NAME_KEY.to_string(), name.to_string());
            field.with_metadata(metadata)
        }
        None => field,
    }
}

/// Reads the extension name from field metadata and resolves it to a TypeId.
/// None when the field carries no metadata or an unknown name
pub fn type_id_from_field(field: &Field) -> Option<TypeId> {
    field
        .metadata()
        .get(EXTENSION_NAME_KEY)
        .and_then(|name| type_id_for_extension(name))
}

/// Builds the export Field for a column, using the registry storage type for
/// extension types and the shared timestamp aware mapping otherwise, then
/// stamps the extension name onto the metadata
pub fn export_field(spec: &ColumnSpec) -> Field {
    let data_type = match storage_type_for(spec.type_id) {
        Some(storage) => storage.clone(),
        None => timestamp_arrow_type(spec.type_id, spec.fractional_digits),
    };
    annotate_field(Field::new(&spec.name, data_type, true), spec.type_id)
}

/// Resolves a field to a TypeId, extension metadata first, then raw DataType
/// inference for plain fields and unknown extension names
pub fn import_type_id(field: &Field) -> Result<TypeId> {
    match type_id_from_field(field) {
        Some(t) => Ok(t),
        None => arrow_to_type_id(field.data_type()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::schema::type_id_to_arrow;
    use super::*;
    use std::collections::HashMap;

    const ROUND_TRIP_TYPES: &[TypeId] = &[
        TypeId::Uuid,
        TypeId::Money,
        TypeId::Inet,
        TypeId::MacAddr,
        TypeId::Geometry,
        TypeId::Vector,
        TypeId::Interval,
        TypeId::HyperLogLog,
        TypeId::TDigest,
        TypeId::Cidr,
    ];

    #[test]
    fn extension_schema_round_trip_restores_type_ids() {
        let specs: Vec<ColumnSpec> = ROUND_TRIP_TYPES
            .iter()
            .enumerate()
            .map(|(i, t)| ColumnSpec::new(format!("c{i}"), *t))
            .collect();
        let fields: Vec<Field> = specs.iter().map(export_field).collect();
        for (field, spec) in fields.iter().zip(specs.iter()) {
            assert_eq!(
                type_id_from_field(field),
                Some(spec.type_id),
                "metadata read back failed for {:?}",
                spec.type_id
            );
            assert_eq!(
                import_type_id(field).unwrap(),
                spec.type_id,
                "import failed for {:?}",
                spec.type_id
            );
        }
    }

    #[test]
    fn table_is_consistent_both_directions() {
        for entry in ENTRIES {
            assert!(entry.name.starts_with("zyron."), "name {}", entry.name);
            assert_eq!(extension_name_for(entry.type_id), Some(entry.name));
            assert_eq!(type_id_for_extension(entry.name), Some(entry.type_id));
            assert_eq!(
                type_id_to_arrow(entry.type_id),
                entry.storage,
                "storage for {:?} must match the shared TypeId mapping",
                entry.type_id
            );
        }
        let mut names: Vec<&str> = ENTRIES.iter().map(|e| e.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ENTRIES.len(), "duplicate extension name");
        let mut ids: Vec<u8> = ENTRIES.iter().map(|e| e.type_id as u8).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), ENTRIES.len(), "duplicate TypeId entry");
    }

    #[test]
    fn inet_exports_one_name_and_imports_aliases() {
        assert_eq!(extension_name_for(TypeId::Inet), Some("zyron.inet"));
        assert_eq!(type_id_for_extension("zyron.inet"), Some(TypeId::Inet));
        assert_eq!(type_id_for_extension("zyron.ip4"), Some(TypeId::Inet));
        assert_eq!(type_id_for_extension("zyron.ip6"), Some(TypeId::Inet));
    }

    #[test]
    fn unknown_extension_name_falls_back_to_storage_inference() {
        let metadata = HashMap::from([(
            EXTENSION_NAME_KEY.to_string(),
            "zyron.does_not_exist".to_string(),
        )]);
        let field = Field::new("x", ArrowDataType::Binary, true).with_metadata(metadata);
        assert_eq!(type_id_from_field(&field), None);
        assert_eq!(import_type_id(&field).unwrap(), TypeId::Binary);
    }

    #[test]
    fn plain_fields_are_unchanged() {
        let int_field = export_field(&ColumnSpec::new("id", TypeId::Int64));
        assert_eq!(int_field.data_type(), &ArrowDataType::Int64);
        assert!(int_field.metadata().is_empty());
        assert_eq!(import_type_id(&int_field).unwrap(), TypeId::Int64);

        let text_field = export_field(&ColumnSpec::new("name", TypeId::Text));
        assert_eq!(text_field.data_type(), &ArrowDataType::Utf8);
        assert!(text_field.metadata().is_empty());
        assert_eq!(import_type_id(&text_field).unwrap(), TypeId::Text);
    }

    #[test]
    fn annotate_field_preserves_existing_metadata() {
        let metadata = HashMap::from([("origin".to_string(), "test".to_string())]);
        let field =
            Field::new("u", ArrowDataType::FixedSizeBinary(16), true).with_metadata(metadata);
        let annotated = annotate_field(field, TypeId::Uuid);
        assert_eq!(
            annotated.metadata().get("origin").map(String::as_str),
            Some("test")
        );
        assert_eq!(
            annotated
                .metadata()
                .get(EXTENSION_NAME_KEY)
                .map(String::as_str),
            Some("zyron.uuid")
        );
    }

    #[test]
    fn export_field_keeps_timestamp_precision_mapping() {
        use arrow::datatypes::TimeUnit;
        let spec = ColumnSpec::with_precision("t9", TypeId::Timestamp, Some(9));
        let field = export_field(&spec);
        assert_eq!(
            field.data_type(),
            &ArrowDataType::Timestamp(TimeUnit::Nanosecond, None)
        );
        assert!(field.metadata().is_empty());
    }
}
