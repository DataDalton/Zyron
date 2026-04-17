//! Unit-aware quantity type.
//!
//! Storage: f64 value + u16 unit_id = 10 bytes.
//! Operations prevent unit mismatch bugs by requiring explicit conversion
//! between different dimensions. Same-dimension arithmetic auto-converts.

use zyron_common::{Result, ZyronError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Length,
    Mass,
    Time,
    Temperature,
    Speed,
    Area,
    Volume,
    Pressure,
    Energy,
    Power,
    Force,
    Frequency,
    DataSize,
}

#[derive(Debug, Clone, Copy)]
pub struct UnitInfo {
    pub id: u16,
    pub name: &'static str,
    pub symbol: &'static str,
    pub dimension: Dimension,
    /// Conversion factor: value_in_si = value * factor + offset
    pub factor: f64,
    pub offset: f64,
}

/// Look up unit info by name (case-insensitive).
pub fn unit_lookup(name: &str) -> Option<UnitInfo> {
    let lower = name.to_lowercase();
    UNITS
        .iter()
        .find(|u| u.name.eq_ignore_ascii_case(&lower) || u.symbol == name)
        .copied()
}

/// Look up unit info by id.
pub fn unit_by_id(id: u16) -> Option<UnitInfo> {
    UNITS.iter().find(|u| u.id == id).copied()
}

/// Creates a quantity from a value and unit name.
pub fn quantity_create(value: f64, unit: &str) -> Result<(f64, u16)> {
    let info = unit_lookup(unit)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown unit: {}", unit)))?;
    Ok((value, info.id))
}

/// Converts a value from one unit to another within the same dimension.
pub fn quantity_convert(value: f64, from_unit: u16, to_unit: u16) -> Result<f64> {
    let from = unit_by_id(from_unit)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown unit id: {}", from_unit)))?;
    let to = unit_by_id(to_unit)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown unit id: {}", to_unit)))?;

    if from.dimension != to.dimension {
        return Err(ZyronError::ExecutionError(format!(
            "Dimension mismatch: {:?} vs {:?}",
            from.dimension, to.dimension
        )));
    }

    // Convert to SI base, then to target
    let si_value = value * from.factor + from.offset;
    let target_value = (si_value - to.offset) / to.factor;
    Ok(target_value)
}

/// Adds two quantities. Both must have the same dimension. Result in first unit.
pub fn quantity_add(a_val: f64, a_unit: u16, b_val: f64, b_unit: u16) -> Result<(f64, u16)> {
    let b_in_a = quantity_convert(b_val, b_unit, a_unit)?;
    Ok((a_val + b_in_a, a_unit))
}

/// Subtracts two quantities. Both must have the same dimension.
pub fn quantity_subtract(a_val: f64, a_unit: u16, b_val: f64, b_unit: u16) -> Result<(f64, u16)> {
    let b_in_a = quantity_convert(b_val, b_unit, a_unit)?;
    Ok((a_val - b_in_a, a_unit))
}

/// Multiplies a quantity by a scalar.
pub fn quantity_scale(val: f64, unit: u16, factor: f64) -> (f64, u16) {
    (val * factor, unit)
}

/// Multiplies two quantities.
/// Length * Length -> Area, Length * Area -> Volume, Mass/Time -> special cases.
/// Otherwise returns an error since derived dimension handling is limited.
pub fn quantity_multiply(a_val: f64, a_unit: u16, b_val: f64, b_unit: u16) -> Result<(f64, u16)> {
    let a = unit_by_id(a_unit)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown unit: {}", a_unit)))?;
    let b = unit_by_id(b_unit)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown unit: {}", b_unit)))?;

    match (a.dimension, b.dimension) {
        (Dimension::Length, Dimension::Length) => {
            // Convert both to meters, multiply, return in m^2
            let a_m = a_val * a.factor;
            let b_m = b_val * b.factor;
            let m2_id = unit_lookup("m2")
                .ok_or_else(|| ZyronError::ExecutionError("m2 unit not found".into()))?;
            Ok((a_m * b_m, m2_id.id))
        }
        (Dimension::Length, Dimension::Area) | (Dimension::Area, Dimension::Length) => {
            let (len_val, len_unit, area_val, area_unit) = if a.dimension == Dimension::Length {
                (a_val, a, b_val, b)
            } else {
                (b_val, b, a_val, a)
            };
            let len_m = len_val * len_unit.factor;
            let area_m2 = area_val * area_unit.factor;
            let l_id = unit_lookup("liter")
                .ok_or_else(|| ZyronError::ExecutionError("liter unit not found".into()))?;
            // m^3 -> liters: 1 m^3 = 1000 L
            Ok((len_m * area_m2 * 1000.0, l_id.id))
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "Cannot multiply quantities of dimensions {:?} and {:?}",
            a.dimension, b.dimension
        ))),
    }
}

/// Formats a quantity value with its unit symbol.
pub fn quantity_format(value: f64, unit: u16) -> String {
    match unit_by_id(unit) {
        Some(info) => format!("{} {}", format_value(value), info.symbol),
        None => format!("{} (unit {})", format_value(value), unit),
    }
}

fn format_value(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

/// Returns the dimension name for a unit.
pub fn quantity_dimension(unit: u16) -> &'static str {
    match unit_by_id(unit).map(|u| u.dimension) {
        Some(Dimension::Length) => "length",
        Some(Dimension::Mass) => "mass",
        Some(Dimension::Time) => "time",
        Some(Dimension::Temperature) => "temperature",
        Some(Dimension::Speed) => "speed",
        Some(Dimension::Area) => "area",
        Some(Dimension::Volume) => "volume",
        Some(Dimension::Pressure) => "pressure",
        Some(Dimension::Energy) => "energy",
        Some(Dimension::Power) => "power",
        Some(Dimension::Force) => "force",
        Some(Dimension::Frequency) => "frequency",
        Some(Dimension::DataSize) => "data_size",
        None => "unknown",
    }
}

/// Returns the full name of a unit.
pub fn quantity_unit_name(unit: u16) -> &'static str {
    unit_by_id(unit).map(|u| u.name).unwrap_or("unknown")
}

const UNITS: &[UnitInfo] = &[
    // Length (SI base: meters)
    UnitInfo {
        id: 1,
        name: "meter",
        symbol: "m",
        dimension: Dimension::Length,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 2,
        name: "kilometer",
        symbol: "km",
        dimension: Dimension::Length,
        factor: 1000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 3,
        name: "centimeter",
        symbol: "cm",
        dimension: Dimension::Length,
        factor: 0.01,
        offset: 0.0,
    },
    UnitInfo {
        id: 4,
        name: "millimeter",
        symbol: "mm",
        dimension: Dimension::Length,
        factor: 0.001,
        offset: 0.0,
    },
    UnitInfo {
        id: 5,
        name: "mile",
        symbol: "mi",
        dimension: Dimension::Length,
        factor: 1609.344,
        offset: 0.0,
    },
    UnitInfo {
        id: 6,
        name: "yard",
        symbol: "yd",
        dimension: Dimension::Length,
        factor: 0.9144,
        offset: 0.0,
    },
    UnitInfo {
        id: 7,
        name: "foot",
        symbol: "ft",
        dimension: Dimension::Length,
        factor: 0.3048,
        offset: 0.0,
    },
    UnitInfo {
        id: 8,
        name: "inch",
        symbol: "in",
        dimension: Dimension::Length,
        factor: 0.0254,
        offset: 0.0,
    },
    // Mass (SI base: kilograms)
    UnitInfo {
        id: 20,
        name: "kilogram",
        symbol: "kg",
        dimension: Dimension::Mass,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 21,
        name: "gram",
        symbol: "g",
        dimension: Dimension::Mass,
        factor: 0.001,
        offset: 0.0,
    },
    UnitInfo {
        id: 22,
        name: "milligram",
        symbol: "mg",
        dimension: Dimension::Mass,
        factor: 0.000001,
        offset: 0.0,
    },
    UnitInfo {
        id: 23,
        name: "pound",
        symbol: "lb",
        dimension: Dimension::Mass,
        factor: 0.45359237,
        offset: 0.0,
    },
    UnitInfo {
        id: 24,
        name: "ounce",
        symbol: "oz",
        dimension: Dimension::Mass,
        factor: 0.028349523125,
        offset: 0.0,
    },
    UnitInfo {
        id: 25,
        name: "tonne",
        symbol: "t",
        dimension: Dimension::Mass,
        factor: 1000.0,
        offset: 0.0,
    },
    // Time (SI base: seconds)
    UnitInfo {
        id: 40,
        name: "second",
        symbol: "s",
        dimension: Dimension::Time,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 41,
        name: "millisecond",
        symbol: "ms",
        dimension: Dimension::Time,
        factor: 0.001,
        offset: 0.0,
    },
    UnitInfo {
        id: 42,
        name: "microsecond",
        symbol: "us",
        dimension: Dimension::Time,
        factor: 0.000001,
        offset: 0.0,
    },
    UnitInfo {
        id: 43,
        name: "minute",
        symbol: "min",
        dimension: Dimension::Time,
        factor: 60.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 44,
        name: "hour",
        symbol: "h",
        dimension: Dimension::Time,
        factor: 3600.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 45,
        name: "day",
        symbol: "d",
        dimension: Dimension::Time,
        factor: 86400.0,
        offset: 0.0,
    },
    // Temperature (SI base: kelvin) - uses offset
    UnitInfo {
        id: 60,
        name: "kelvin",
        symbol: "K",
        dimension: Dimension::Temperature,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 61,
        name: "celsius",
        symbol: "\u{00B0}C",
        dimension: Dimension::Temperature,
        factor: 1.0,
        offset: 273.15,
    },
    UnitInfo {
        id: 62,
        name: "fahrenheit",
        symbol: "\u{00B0}F",
        dimension: Dimension::Temperature,
        factor: 5.0 / 9.0,
        offset: 255.372222222,
    },
    // Speed (SI base: m/s)
    UnitInfo {
        id: 80,
        name: "meter_per_second",
        symbol: "m/s",
        dimension: Dimension::Speed,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 81,
        name: "km_per_hour",
        symbol: "km/h",
        dimension: Dimension::Speed,
        factor: 1.0 / 3.6,
        offset: 0.0,
    },
    UnitInfo {
        id: 82,
        name: "mph",
        symbol: "mph",
        dimension: Dimension::Speed,
        factor: 0.44704,
        offset: 0.0,
    },
    UnitInfo {
        id: 83,
        name: "knot",
        symbol: "kn",
        dimension: Dimension::Speed,
        factor: 0.5144444,
        offset: 0.0,
    },
    // Area (SI base: square meters)
    UnitInfo {
        id: 100,
        name: "m2",
        symbol: "m\u{00B2}",
        dimension: Dimension::Area,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 101,
        name: "hectare",
        symbol: "ha",
        dimension: Dimension::Area,
        factor: 10000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 102,
        name: "acre",
        symbol: "ac",
        dimension: Dimension::Area,
        factor: 4046.8564224,
        offset: 0.0,
    },
    // Volume (SI base: liters)
    UnitInfo {
        id: 120,
        name: "liter",
        symbol: "L",
        dimension: Dimension::Volume,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 121,
        name: "milliliter",
        symbol: "mL",
        dimension: Dimension::Volume,
        factor: 0.001,
        offset: 0.0,
    },
    UnitInfo {
        id: 122,
        name: "gallon",
        symbol: "gal",
        dimension: Dimension::Volume,
        factor: 3.785411784,
        offset: 0.0,
    },
    // Pressure (SI base: pascal)
    UnitInfo {
        id: 140,
        name: "pascal",
        symbol: "Pa",
        dimension: Dimension::Pressure,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 141,
        name: "bar",
        symbol: "bar",
        dimension: Dimension::Pressure,
        factor: 100000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 142,
        name: "psi",
        symbol: "psi",
        dimension: Dimension::Pressure,
        factor: 6894.757,
        offset: 0.0,
    },
    UnitInfo {
        id: 143,
        name: "atm",
        symbol: "atm",
        dimension: Dimension::Pressure,
        factor: 101325.0,
        offset: 0.0,
    },
    // Energy (SI base: joule)
    UnitInfo {
        id: 160,
        name: "joule",
        symbol: "J",
        dimension: Dimension::Energy,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 161,
        name: "calorie",
        symbol: "cal",
        dimension: Dimension::Energy,
        factor: 4.184,
        offset: 0.0,
    },
    UnitInfo {
        id: 162,
        name: "kwh",
        symbol: "kWh",
        dimension: Dimension::Energy,
        factor: 3600000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 163,
        name: "btu",
        symbol: "BTU",
        dimension: Dimension::Energy,
        factor: 1055.06,
        offset: 0.0,
    },
    // Power (SI base: watt)
    UnitInfo {
        id: 180,
        name: "watt",
        symbol: "W",
        dimension: Dimension::Power,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 181,
        name: "horsepower",
        symbol: "hp",
        dimension: Dimension::Power,
        factor: 745.7,
        offset: 0.0,
    },
    UnitInfo {
        id: 182,
        name: "kilowatt",
        symbol: "kW",
        dimension: Dimension::Power,
        factor: 1000.0,
        offset: 0.0,
    },
    // Force (SI base: newton)
    UnitInfo {
        id: 200,
        name: "newton",
        symbol: "N",
        dimension: Dimension::Force,
        factor: 1.0,
        offset: 0.0,
    },
    // Frequency (SI base: hertz)
    UnitInfo {
        id: 220,
        name: "hertz",
        symbol: "Hz",
        dimension: Dimension::Frequency,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 221,
        name: "kilohertz",
        symbol: "kHz",
        dimension: Dimension::Frequency,
        factor: 1000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 222,
        name: "megahertz",
        symbol: "MHz",
        dimension: Dimension::Frequency,
        factor: 1000000.0,
        offset: 0.0,
    },
    // DataSize (SI base: byte)
    UnitInfo {
        id: 240,
        name: "byte",
        symbol: "B",
        dimension: Dimension::DataSize,
        factor: 1.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 241,
        name: "kilobyte",
        symbol: "KB",
        dimension: Dimension::DataSize,
        factor: 1000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 242,
        name: "megabyte",
        symbol: "MB",
        dimension: Dimension::DataSize,
        factor: 1000000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 243,
        name: "gigabyte",
        symbol: "GB",
        dimension: Dimension::DataSize,
        factor: 1000000000.0,
        offset: 0.0,
    },
    UnitInfo {
        id: 244,
        name: "terabyte",
        symbol: "TB",
        dimension: Dimension::DataSize,
        factor: 1000000000000.0,
        offset: 0.0,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_lookup() {
        let m = unit_lookup("meter").unwrap();
        assert_eq!(m.dimension, Dimension::Length);
    }

    #[test]
    fn test_unit_lookup_case() {
        assert!(unit_lookup("Meter").is_some());
        assert!(unit_lookup("METER").is_some());
    }

    #[test]
    fn test_quantity_create() {
        let (v, id) = quantity_create(42.0, "kg").unwrap();
        assert_eq!(v, 42.0);
        assert_eq!(quantity_dimension(id), "mass");
    }

    #[test]
    fn test_convert_length() {
        let km = unit_lookup("kilometer").unwrap().id;
        let mi = unit_lookup("mile").unwrap().id;
        let result = quantity_convert(1.0, km, mi).unwrap();
        assert!((result - 0.621371).abs() < 0.001);
    }

    #[test]
    fn test_convert_mass() {
        let kg = unit_lookup("kilogram").unwrap().id;
        let lb = unit_lookup("pound").unwrap().id;
        let result = quantity_convert(1.0, lb, kg).unwrap();
        assert!((result - 0.45359237).abs() < 0.001);
    }

    #[test]
    fn test_convert_temperature() {
        let c = unit_lookup("celsius").unwrap().id;
        let f = unit_lookup("fahrenheit").unwrap().id;
        let result = quantity_convert(100.0, c, f).unwrap();
        assert!((result - 212.0).abs() < 0.1);
    }

    #[test]
    fn test_convert_dimension_mismatch() {
        let kg = unit_lookup("kilogram").unwrap().id;
        let m = unit_lookup("meter").unwrap().id;
        assert!(quantity_convert(1.0, kg, m).is_err());
    }

    #[test]
    fn test_add_same_unit() {
        let kg = unit_lookup("kilogram").unwrap().id;
        let (v, _) = quantity_add(5.0, kg, 3.0, kg).unwrap();
        assert_eq!(v, 8.0);
    }

    #[test]
    fn test_add_different_units_same_dimension() {
        let kg = unit_lookup("kilogram").unwrap().id;
        let g = unit_lookup("gram").unwrap().id;
        let (v, _) = quantity_add(1.0, kg, 500.0, g).unwrap();
        assert!((v - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        let kg = unit_lookup("kilogram").unwrap().id;
        let m = unit_lookup("meter").unwrap().id;
        assert!(quantity_add(1.0, kg, 1.0, m).is_err());
    }

    #[test]
    fn test_subtract() {
        let m = unit_lookup("meter").unwrap().id;
        let (v, _) = quantity_subtract(10.0, m, 3.0, m).unwrap();
        assert_eq!(v, 7.0);
    }

    #[test]
    fn test_scale() {
        let m = unit_lookup("meter").unwrap().id;
        let (v, _) = quantity_scale(5.0, m, 2.5);
        assert_eq!(v, 12.5);
    }

    #[test]
    fn test_multiply_length_length() {
        let m = unit_lookup("meter").unwrap().id;
        let (v, unit) = quantity_multiply(3.0, m, 4.0, m).unwrap();
        assert_eq!(v, 12.0);
        assert_eq!(quantity_dimension(unit), "area");
    }

    #[test]
    fn test_format() {
        let kg = unit_lookup("kilogram").unwrap().id;
        assert_eq!(quantity_format(42.0, kg), "42 kg");
    }

    #[test]
    fn test_dimension_name() {
        let m = unit_lookup("meter").unwrap().id;
        assert_eq!(quantity_dimension(m), "length");
    }

    #[test]
    fn test_unit_name() {
        let m = unit_lookup("meter").unwrap().id;
        assert_eq!(quantity_unit_name(m), "meter");
    }

    #[test]
    fn test_convert_roundtrip() {
        let km = unit_lookup("kilometer").unwrap().id;
        let mi = unit_lookup("mile").unwrap().id;
        let value = 100.0;
        let converted = quantity_convert(value, km, mi).unwrap();
        let back = quantity_convert(converted, mi, km).unwrap();
        assert!((value - back).abs() < 0.001);
    }

    #[test]
    fn test_convert_zero_celsius_to_kelvin() {
        let c = unit_lookup("celsius").unwrap().id;
        let k = unit_lookup("kelvin").unwrap().id;
        let result = quantity_convert(0.0, c, k).unwrap();
        assert!((result - 273.15).abs() < 0.01);
    }
}
