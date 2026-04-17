//! Geospatial types and operations.
//!
//! Geometry types (Point, LineString, Polygon, etc.) stored as WKB binary.
//! ST_* functions use Haversine for geographic distances (SRID 4326 default).
//! H3 hex grid support via simplified hierarchical indexing.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Geometry types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub x: f64, // longitude
    pub y: f64, // latitude
}

#[derive(Debug, Clone, PartialEq)]
pub struct LineString {
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    pub exterior: Vec<Point>,
    pub holes: Vec<Vec<Point>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiPoint {
    pub points: Vec<Point>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiLineString {
    pub lines: Vec<LineString>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiPolygon {
    pub polygons: Vec<Polygon>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryKind {
    Point(Point),
    LineString(LineString),
    Polygon(Polygon),
    MultiPoint(MultiPoint),
    MultiLineString(MultiLineString),
    MultiPolygon(MultiPolygon),
    GeometryCollection(Vec<Geometry>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub kind: GeometryKind,
    pub srid: u32,
}

impl Geometry {
    pub fn with_srid(kind: GeometryKind, srid: u32) -> Self {
        Self { kind, srid }
    }

    pub fn point(x: f64, y: f64) -> Self {
        Self::with_srid(GeometryKind::Point(Point { x, y }), 4326)
    }
}

// ---------------------------------------------------------------------------
// WKB encoding (simplified format: type tag + SRID + coords)
// ---------------------------------------------------------------------------

const WKB_POINT: u8 = 1;
const WKB_LINESTRING: u8 = 2;
const WKB_POLYGON: u8 = 3;
const WKB_MULTIPOINT: u8 = 4;
const WKB_MULTILINESTRING: u8 = 5;
const WKB_MULTIPOLYGON: u8 = 6;
const WKB_GEOMETRY_COLLECTION: u8 = 7;

/// Encodes a geometry to WKB-like binary format.
pub fn encode_wkb(geom: &Geometry) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&geom.srid.to_le_bytes());
    encode_kind(&geom.kind, &mut out);
    out
}

fn encode_kind(kind: &GeometryKind, out: &mut Vec<u8>) {
    match kind {
        GeometryKind::Point(p) => {
            out.push(WKB_POINT);
            out.extend_from_slice(&p.x.to_le_bytes());
            out.extend_from_slice(&p.y.to_le_bytes());
        }
        GeometryKind::LineString(ls) => {
            out.push(WKB_LINESTRING);
            encode_points(&ls.points, out);
        }
        GeometryKind::Polygon(p) => {
            out.push(WKB_POLYGON);
            out.extend_from_slice(&(p.holes.len() as u32 + 1).to_le_bytes());
            encode_points(&p.exterior, out);
            for hole in &p.holes {
                encode_points(hole, out);
            }
        }
        GeometryKind::MultiPoint(mp) => {
            out.push(WKB_MULTIPOINT);
            encode_points(&mp.points, out);
        }
        GeometryKind::MultiLineString(mls) => {
            out.push(WKB_MULTILINESTRING);
            out.extend_from_slice(&(mls.lines.len() as u32).to_le_bytes());
            for line in &mls.lines {
                encode_points(&line.points, out);
            }
        }
        GeometryKind::MultiPolygon(mp) => {
            out.push(WKB_MULTIPOLYGON);
            out.extend_from_slice(&(mp.polygons.len() as u32).to_le_bytes());
            for poly in &mp.polygons {
                out.extend_from_slice(&(poly.holes.len() as u32 + 1).to_le_bytes());
                encode_points(&poly.exterior, out);
                for hole in &poly.holes {
                    encode_points(hole, out);
                }
            }
        }
        GeometryKind::GeometryCollection(geoms) => {
            out.push(WKB_GEOMETRY_COLLECTION);
            out.extend_from_slice(&(geoms.len() as u32).to_le_bytes());
            for g in geoms {
                out.extend_from_slice(&g.srid.to_le_bytes());
                encode_kind(&g.kind, out);
            }
        }
    }
}

fn encode_points(points: &[Point], out: &mut Vec<u8>) {
    out.extend_from_slice(&(points.len() as u32).to_le_bytes());
    for p in points {
        out.extend_from_slice(&p.x.to_le_bytes());
        out.extend_from_slice(&p.y.to_le_bytes());
    }
}

/// Decodes WKB-like binary format back to a geometry.
pub fn decode_wkb(bytes: &[u8]) -> Result<Geometry> {
    let (geom, _) = decode_wkb_at(bytes, 0)?;
    Ok(geom)
}

fn decode_wkb_at(bytes: &[u8], offset: usize) -> Result<(Geometry, usize)> {
    if bytes.len() < offset + 5 {
        return Err(ZyronError::ExecutionError("WKB too short".into()));
    }
    let srid = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    let mut pos = offset + 4;
    let (kind, next_pos) = decode_kind(bytes, pos)?;
    pos = next_pos;
    Ok((Geometry { kind, srid }, pos))
}

fn decode_kind(bytes: &[u8], mut pos: usize) -> Result<(GeometryKind, usize)> {
    if pos >= bytes.len() {
        return Err(ZyronError::ExecutionError("WKB missing type byte".into()));
    }
    let type_byte = bytes[pos];
    pos += 1;

    match type_byte {
        WKB_POINT => {
            if bytes.len() < pos + 16 {
                return Err(ZyronError::ExecutionError("Point WKB truncated".into()));
            }
            let x = read_f64(bytes, pos);
            let y = read_f64(bytes, pos + 8);
            Ok((GeometryKind::Point(Point { x, y }), pos + 16))
        }
        WKB_LINESTRING => {
            let (points, next) = decode_points(bytes, pos)?;
            Ok((GeometryKind::LineString(LineString { points }), next))
        }
        WKB_POLYGON => {
            if bytes.len() < pos + 4 {
                return Err(ZyronError::ExecutionError("Polygon truncated".into()));
            }
            let ring_count = read_u32(bytes, pos);
            pos += 4;
            let (exterior, next) = decode_points(bytes, pos)?;
            pos = next;
            let mut holes = Vec::new();
            for _ in 1..ring_count {
                let (hole, next) = decode_points(bytes, pos)?;
                holes.push(hole);
                pos = next;
            }
            Ok((GeometryKind::Polygon(Polygon { exterior, holes }), pos))
        }
        WKB_MULTIPOINT => {
            let (points, next) = decode_points(bytes, pos)?;
            Ok((GeometryKind::MultiPoint(MultiPoint { points }), next))
        }
        WKB_MULTILINESTRING => {
            if bytes.len() < pos + 4 {
                return Err(ZyronError::ExecutionError(
                    "MultiLineString truncated".into(),
                ));
            }
            let n = read_u32(bytes, pos);
            pos += 4;
            let mut lines = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let (points, next) = decode_points(bytes, pos)?;
                lines.push(LineString { points });
                pos = next;
            }
            Ok((
                GeometryKind::MultiLineString(MultiLineString { lines }),
                pos,
            ))
        }
        WKB_MULTIPOLYGON => {
            if bytes.len() < pos + 4 {
                return Err(ZyronError::ExecutionError("MultiPolygon truncated".into()));
            }
            let n = read_u32(bytes, pos);
            pos += 4;
            let mut polygons = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let ring_count = read_u32(bytes, pos);
                pos += 4;
                let (exterior, next) = decode_points(bytes, pos)?;
                pos = next;
                let mut holes = Vec::new();
                for _ in 1..ring_count {
                    let (hole, next) = decode_points(bytes, pos)?;
                    holes.push(hole);
                    pos = next;
                }
                polygons.push(Polygon { exterior, holes });
            }
            Ok((GeometryKind::MultiPolygon(MultiPolygon { polygons }), pos))
        }
        WKB_GEOMETRY_COLLECTION => {
            if bytes.len() < pos + 4 {
                return Err(ZyronError::ExecutionError(
                    "GeometryCollection truncated".into(),
                ));
            }
            let n = read_u32(bytes, pos);
            pos += 4;
            let mut geoms = Vec::with_capacity(n as usize);
            for _ in 0..n {
                let (g, next) = decode_wkb_at(bytes, pos)?;
                geoms.push(g);
                pos = next;
            }
            Ok((GeometryKind::GeometryCollection(geoms), pos))
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "Unknown WKB type: {}",
            type_byte
        ))),
    }
}

fn decode_points(bytes: &[u8], mut pos: usize) -> Result<(Vec<Point>, usize)> {
    if bytes.len() < pos + 4 {
        return Err(ZyronError::ExecutionError("Points count truncated".into()));
    }
    let n = read_u32(bytes, pos);
    pos += 4;
    if bytes.len() < pos + (n as usize) * 16 {
        return Err(ZyronError::ExecutionError("Points data truncated".into()));
    }
    let mut points = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let x = read_f64(bytes, pos);
        let y = read_f64(bytes, pos + 8);
        points.push(Point { x, y });
        pos += 16;
    }
    Ok((points, pos))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

pub fn st_make_point(lon: f64, lat: f64) -> Geometry {
    Geometry::point(lon, lat)
}

// ---------------------------------------------------------------------------
// WKT parsing and output
// ---------------------------------------------------------------------------

pub fn st_geom_from_text(wkt: &str) -> Result<Geometry> {
    let trimmed = wkt.trim();
    parse_wkt(trimmed)
}

fn parse_wkt(s: &str) -> Result<Geometry> {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("POINT") {
        let rest = rest
            .trim()
            .strip_prefix('(')
            .and_then(|r| r.strip_suffix(')'))
            .ok_or_else(|| ZyronError::ExecutionError("Invalid POINT syntax".into()))?;
        let parts: Vec<&str> = rest.trim().split_whitespace().collect();
        if parts.len() < 2 {
            return Err(ZyronError::ExecutionError(
                "POINT requires two coordinates".into(),
            ));
        }
        let x: f64 = parts[0].parse().map_err(|_| {
            ZyronError::ExecutionError(format!("Invalid x coordinate: {}", parts[0]))
        })?;
        let y: f64 = parts[1].parse().map_err(|_| {
            ZyronError::ExecutionError(format!("Invalid y coordinate: {}", parts[1]))
        })?;
        Ok(Geometry::point(x, y))
    } else if let Some(rest) = s.strip_prefix("LINESTRING") {
        let rest = rest
            .trim()
            .strip_prefix('(')
            .and_then(|r| r.strip_suffix(')'))
            .ok_or_else(|| ZyronError::ExecutionError("Invalid LINESTRING syntax".into()))?;
        let points = parse_wkt_points(rest)?;
        Ok(Geometry::with_srid(
            GeometryKind::LineString(LineString { points }),
            4326,
        ))
    } else if let Some(rest) = s.strip_prefix("POLYGON") {
        let rest = rest
            .trim()
            .strip_prefix('(')
            .and_then(|r| r.strip_suffix(')'))
            .ok_or_else(|| ZyronError::ExecutionError("Invalid POLYGON syntax".into()))?;
        let rings = parse_wkt_rings(rest)?;
        if rings.is_empty() {
            return Err(ZyronError::ExecutionError(
                "POLYGON needs at least one ring".into(),
            ));
        }
        let mut iter = rings.into_iter();
        let exterior = iter.next().unwrap();
        let holes: Vec<Vec<Point>> = iter.collect();
        Ok(Geometry::with_srid(
            GeometryKind::Polygon(Polygon { exterior, holes }),
            4326,
        ))
    } else {
        Err(ZyronError::ExecutionError(format!(
            "Unsupported WKT type: {}",
            s
        )))
    }
}

fn parse_wkt_points(s: &str) -> Result<Vec<Point>> {
    let mut points = Vec::new();
    for part in s.split(',') {
        let coords: Vec<&str> = part.trim().split_whitespace().collect();
        if coords.len() < 2 {
            return Err(ZyronError::ExecutionError(format!(
                "Invalid coordinates: {}",
                part
            )));
        }
        let x: f64 = coords[0]
            .parse()
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid x: {}", coords[0])))?;
        let y: f64 = coords[1]
            .parse()
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid y: {}", coords[1])))?;
        points.push(Point { x, y });
    }
    Ok(points)
}

fn parse_wkt_rings(s: &str) -> Result<Vec<Vec<Point>>> {
    // Split on "),(" to get individual rings
    let mut rings = Vec::new();
    let mut current = String::new();
    let mut paren_depth = 0;
    for c in s.chars() {
        match c {
            '(' => {
                paren_depth += 1;
                if paren_depth > 1 {
                    current.push(c);
                }
            }
            ')' => {
                paren_depth -= 1;
                if paren_depth == 0 {
                    rings.push(parse_wkt_points(&current)?);
                    current.clear();
                } else {
                    current.push(c);
                }
            }
            ',' if paren_depth == 0 => {}
            c => current.push(c),
        }
    }
    Ok(rings)
}

pub fn st_as_text(geom: &Geometry) -> String {
    match &geom.kind {
        GeometryKind::Point(p) => format!("POINT({} {})", p.x, p.y),
        GeometryKind::LineString(ls) => {
            let pts: Vec<String> = ls
                .points
                .iter()
                .map(|p| format!("{} {}", p.x, p.y))
                .collect();
            format!("LINESTRING({})", pts.join(", "))
        }
        GeometryKind::Polygon(p) => {
            let ext: Vec<String> = p
                .exterior
                .iter()
                .map(|pt| format!("{} {}", pt.x, pt.y))
                .collect();
            let mut rings = vec![format!("({})", ext.join(", "))];
            for hole in &p.holes {
                let h: Vec<String> = hole.iter().map(|pt| format!("{} {}", pt.x, pt.y)).collect();
                rings.push(format!("({})", h.join(", ")));
            }
            format!("POLYGON({})", rings.join(", "))
        }
        GeometryKind::MultiPoint(mp) => {
            let pts: Vec<String> = mp
                .points
                .iter()
                .map(|p| format!("({} {})", p.x, p.y))
                .collect();
            format!("MULTIPOINT({})", pts.join(", "))
        }
        _ => format!("GEOMETRY(srid={})", geom.srid),
    }
}

pub fn st_geom_from_geojson(json: &str) -> Result<Geometry> {
    let val = crate::diff::JsonValue::parse(json)?;
    geojson_to_geometry(&val)
}

fn geojson_to_geometry(val: &crate::diff::JsonValue) -> Result<Geometry> {
    use crate::diff::JsonValue;
    let obj = match val {
        JsonValue::Object(items) => items,
        _ => return Err(ZyronError::ExecutionError("GeoJSON must be object".into())),
    };

    let type_str = obj
        .iter()
        .find(|(k, _)| k == "type")
        .and_then(|(_, v)| match v {
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .ok_or_else(|| ZyronError::ExecutionError("GeoJSON missing 'type'".into()))?;

    let coords = obj.iter().find(|(k, _)| k == "coordinates").map(|(_, v)| v);

    match type_str.as_str() {
        "Point" => {
            let arr = match coords {
                Some(JsonValue::Array(a)) => a,
                _ => {
                    return Err(ZyronError::ExecutionError(
                        "Point needs coordinates array".into(),
                    ));
                }
            };
            if arr.len() < 2 {
                return Err(ZyronError::ExecutionError("Point needs 2 coords".into()));
            }
            let x = json_number(&arr[0])?;
            let y = json_number(&arr[1])?;
            Ok(Geometry::point(x, y))
        }
        "LineString" => {
            let arr = match coords {
                Some(JsonValue::Array(a)) => a,
                _ => return Err(ZyronError::ExecutionError("LineString needs coords".into())),
            };
            let points: Result<Vec<Point>> = arr.iter().map(json_to_point).collect();
            Ok(Geometry::with_srid(
                GeometryKind::LineString(LineString { points: points? }),
                4326,
            ))
        }
        "Polygon" => {
            let arr = match coords {
                Some(JsonValue::Array(a)) => a,
                _ => return Err(ZyronError::ExecutionError("Polygon needs coords".into())),
            };
            if arr.is_empty() {
                return Err(ZyronError::ExecutionError(
                    "Polygon needs at least 1 ring".into(),
                ));
            }
            let rings: Result<Vec<Vec<Point>>> = arr
                .iter()
                .map(|ring| match ring {
                    JsonValue::Array(pts) => pts.iter().map(json_to_point).collect(),
                    _ => Err(ZyronError::ExecutionError("Ring must be array".into())),
                })
                .collect();
            let mut rings = rings?;
            let exterior = rings.remove(0);
            Ok(Geometry::with_srid(
                GeometryKind::Polygon(Polygon {
                    exterior,
                    holes: rings,
                }),
                4326,
            ))
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "Unsupported GeoJSON type: {}",
            type_str
        ))),
    }
}

fn json_number(val: &crate::diff::JsonValue) -> Result<f64> {
    match val {
        crate::diff::JsonValue::Number(n) => Ok(*n),
        _ => Err(ZyronError::ExecutionError("Expected number".into())),
    }
}

fn json_to_point(val: &crate::diff::JsonValue) -> Result<Point> {
    if let crate::diff::JsonValue::Array(arr) = val {
        if arr.len() >= 2 {
            return Ok(Point {
                x: json_number(&arr[0])?,
                y: json_number(&arr[1])?,
            });
        }
    }
    Err(ZyronError::ExecutionError(
        "Invalid point coordinates".into(),
    ))
}

pub fn st_as_geojson(geom: &Geometry) -> String {
    match &geom.kind {
        GeometryKind::Point(p) => {
            format!(r#"{{"type":"Point","coordinates":[{},{}]}}"#, p.x, p.y)
        }
        GeometryKind::LineString(ls) => {
            let pts: Vec<String> = ls
                .points
                .iter()
                .map(|p| format!("[{},{}]", p.x, p.y))
                .collect();
            format!(
                r#"{{"type":"LineString","coordinates":[{}]}}"#,
                pts.join(",")
            )
        }
        GeometryKind::Polygon(p) => {
            let ext_pts: Vec<String> = p
                .exterior
                .iter()
                .map(|pt| format!("[{},{}]", pt.x, pt.y))
                .collect();
            let mut rings = vec![format!("[{}]", ext_pts.join(","))];
            for hole in &p.holes {
                let h: Vec<String> = hole
                    .iter()
                    .map(|pt| format!("[{},{}]", pt.x, pt.y))
                    .collect();
                rings.push(format!("[{}]", h.join(",")));
            }
            format!(
                r#"{{"type":"Polygon","coordinates":[{}]}}"#,
                rings.join(",")
            )
        }
        _ => r#"{"type":"Geometry","coordinates":[]}"#.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Geographic functions (SRID 4326 / WGS84)
// ---------------------------------------------------------------------------

const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Haversine distance between two geographic points in meters.
fn haversine(p1: &Point, p2: &Point) -> f64 {
    let lat1_rad = p1.y.to_radians();
    let lat2_rad = p2.y.to_radians();
    let dlat = (p2.y - p1.y).to_radians();
    let dlon = (p2.x - p1.x).to_radians();

    let a =
        (dlat / 2.0).sin().powi(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    EARTH_RADIUS_M * c
}

/// Distance between two geometries.
pub fn st_distance(a: &Geometry, b: &Geometry) -> Result<f64> {
    match (&a.kind, &b.kind) {
        (GeometryKind::Point(p1), GeometryKind::Point(p2)) => {
            if a.srid == 4326 || b.srid == 4326 {
                Ok(haversine(p1, p2))
            } else {
                // Cartesian distance
                Ok(((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt())
            }
        }
        _ => Err(ZyronError::ExecutionError(
            "st_distance currently supports Point-Point only".into(),
        )),
    }
}

/// Returns true if two geometries are within `radius` meters of each other.
pub fn st_dwithin(geom: &Geometry, center: &Geometry, radius: f64) -> Result<bool> {
    let dist = st_distance(geom, center)?;
    Ok(dist <= radius)
}

/// Point-in-polygon using ray casting.
pub fn st_contains(polygon: &Geometry, point: &Geometry) -> Result<bool> {
    let (poly, pt) = match (&polygon.kind, &point.kind) {
        (GeometryKind::Polygon(p), GeometryKind::Point(pt)) => (p, pt),
        _ => {
            return Err(ZyronError::ExecutionError(
                "st_contains requires Polygon and Point".into(),
            ));
        }
    };

    if !point_in_ring(pt, &poly.exterior) {
        return Ok(false);
    }
    // Must not be in any hole
    for hole in &poly.holes {
        if point_in_ring(pt, hole) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn point_in_ring(pt: &Point, ring: &[Point]) -> bool {
    if ring.len() < 3 {
        return false;
    }
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let xi = ring[i].x;
        let yi = ring[i].y;
        let xj = ring[j].x;
        let yj = ring[j].y;
        if ((yi > pt.y) != (yj > pt.y)) && (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Checks if two polygons share any common area.
/// Simplified: returns true if any point of one polygon is contained in the other.
pub fn st_intersects(a: &Geometry, b: &Geometry) -> Result<bool> {
    match (&a.kind, &b.kind) {
        (GeometryKind::Point(_), GeometryKind::Polygon(_)) => st_contains(b, a),
        (GeometryKind::Polygon(_), GeometryKind::Point(_)) => st_contains(a, b),
        (GeometryKind::Polygon(pa), GeometryKind::Polygon(pb)) => {
            // Check if any vertex of one is inside the other
            for p in &pa.exterior {
                let pt_geom = Geometry::point(p.x, p.y);
                if st_contains(b, &pt_geom)? {
                    return Ok(true);
                }
            }
            for p in &pb.exterior {
                let pt_geom = Geometry::point(p.x, p.y);
                if st_contains(a, &pt_geom)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        (GeometryKind::Point(p1), GeometryKind::Point(p2)) => {
            Ok((p1.x - p2.x).abs() < 1e-9 && (p1.y - p2.y).abs() < 1e-9)
        }
        _ => Err(ZyronError::ExecutionError(
            "st_intersects not supported for these geometry types".into(),
        )),
    }
}

/// Area of a polygon (planar, for geographic coordinates use cautiously).
/// Uses the shoelace formula.
pub fn st_area(polygon: &Geometry) -> Result<f64> {
    let p = match &polygon.kind {
        GeometryKind::Polygon(p) => p,
        _ => {
            return Err(ZyronError::ExecutionError(
                "st_area requires Polygon".into(),
            ));
        }
    };

    let ring_area = |ring: &[Point]| -> f64 {
        if ring.len() < 3 {
            return 0.0;
        }
        let mut sum = 0.0;
        let n = ring.len();
        for i in 0..n {
            let j = (i + 1) % n;
            sum += ring[i].x * ring[j].y;
            sum -= ring[j].x * ring[i].y;
        }
        (sum / 2.0).abs()
    };

    let mut area = ring_area(&p.exterior);
    for hole in &p.holes {
        area -= ring_area(hole);
    }
    Ok(area)
}

/// Creates a circular buffer around a point (returned as polygon approximation).
pub fn st_buffer(geom: &Geometry, distance: f64) -> Result<Geometry> {
    let p = match &geom.kind {
        GeometryKind::Point(p) => p,
        _ => {
            return Err(ZyronError::ExecutionError(
                "st_buffer currently supports Point only".into(),
            ));
        }
    };

    // Generate approximation circle with 32 segments
    let segments = 32;
    let mut points = Vec::with_capacity(segments + 1);
    for i in 0..segments {
        let angle = (i as f64 / segments as f64) * 2.0 * std::f64::consts::PI;
        points.push(Point {
            x: p.x + distance * angle.cos(),
            y: p.y + distance * angle.sin(),
        });
    }
    // Close the ring
    if let Some(first) = points.first().cloned() {
        points.push(first);
    }

    Ok(Geometry::with_srid(
        GeometryKind::Polygon(Polygon {
            exterior: points,
            holes: Vec::new(),
        }),
        geom.srid,
    ))
}

/// Simplified union of two geometries (returns collection).
pub fn st_union(a: &Geometry, b: &Geometry) -> Result<Geometry> {
    Ok(Geometry::with_srid(
        GeometryKind::GeometryCollection(vec![a.clone(), b.clone()]),
        a.srid,
    ))
}

/// Returns the centroid of a geometry.
pub fn st_centroid(geom: &Geometry) -> Result<Geometry> {
    let point = match &geom.kind {
        GeometryKind::Point(p) => p.clone(),
        GeometryKind::LineString(ls) => {
            if ls.points.is_empty() {
                return Err(ZyronError::ExecutionError("Empty LineString".into()));
            }
            let n = ls.points.len() as f64;
            let sum_x: f64 = ls.points.iter().map(|p| p.x).sum();
            let sum_y: f64 = ls.points.iter().map(|p| p.y).sum();
            Point {
                x: sum_x / n,
                y: sum_y / n,
            }
        }
        GeometryKind::Polygon(p) => {
            if p.exterior.len() < 3 {
                return Err(ZyronError::ExecutionError(
                    "Polygon exterior too short".into(),
                ));
            }
            // Centroid of exterior ring via shoelace-weighted average
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut a = 0.0;
            let ring = &p.exterior;
            let n = ring.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let cross = ring[i].x * ring[j].y - ring[j].x * ring[i].y;
                cx += (ring[i].x + ring[j].x) * cross;
                cy += (ring[i].y + ring[j].y) * cross;
                a += cross;
            }
            a /= 2.0;
            if a.abs() < 1e-15 {
                // Degenerate polygon: use simple average
                let sum_x: f64 = ring.iter().map(|p| p.x).sum();
                let sum_y: f64 = ring.iter().map(|p| p.y).sum();
                Point {
                    x: sum_x / n as f64,
                    y: sum_y / n as f64,
                }
            } else {
                Point {
                    x: cx / (6.0 * a),
                    y: cy / (6.0 * a),
                }
            }
        }
        _ => {
            return Err(ZyronError::ExecutionError(
                "st_centroid not supported for this type".into(),
            ));
        }
    };
    Ok(Geometry::with_srid(GeometryKind::Point(point), geom.srid))
}

// ---------------------------------------------------------------------------
// H3 hex grid (simplified)
// ---------------------------------------------------------------------------

/// Converts a point to a simplified H3-like index at the given resolution.
/// Uses a regular hex grid approximation (not the true H3 icosahedral grid).
pub fn h3_from_point(lon: f64, lat: f64, resolution: u8) -> Result<u64> {
    if resolution > 15 {
        return Err(ZyronError::ExecutionError(
            "H3 resolution must be 0-15".into(),
        ));
    }
    // Normalize to [0, 1]
    let nx = (lon + 180.0) / 360.0;
    let ny = (lat + 90.0) / 180.0;
    // Scale by 2^resolution per axis
    let scale = (1u64 << resolution) as f64;
    let cell_x = (nx * scale).floor() as u64;
    let cell_y = (ny * scale).floor() as u64;
    // Pack: [resolution 4bits][x 30bits][y 30bits]
    Ok(((resolution as u64) << 60) | ((cell_x & 0x3FFFFFFF) << 30) | (cell_y & 0x3FFFFFFF))
}

/// Returns the approximate boundary polygon of an H3 cell.
pub fn h3_to_boundary(h3_index: u64) -> Result<Geometry> {
    let resolution = (h3_index >> 60) as u8;
    let cell_x = (h3_index >> 30) & 0x3FFFFFFF;
    let cell_y = h3_index & 0x3FFFFFFF;
    let scale = (1u64 << resolution) as f64;

    let x_size = 360.0 / scale;
    let y_size = 180.0 / scale;
    let x0 = (cell_x as f64 / scale) * 360.0 - 180.0;
    let y0 = (cell_y as f64 / scale) * 180.0 - 90.0;

    let points = vec![
        Point { x: x0, y: y0 },
        Point {
            x: x0 + x_size,
            y: y0,
        },
        Point {
            x: x0 + x_size,
            y: y0 + y_size,
        },
        Point {
            x: x0,
            y: y0 + y_size,
        },
        Point { x: x0, y: y0 },
    ];

    Ok(Geometry::with_srid(
        GeometryKind::Polygon(Polygon {
            exterior: points,
            holes: Vec::new(),
        }),
        4326,
    ))
}

/// Returns the approximate grid distance between two H3 cells (same resolution).
pub fn h3_distance(a: u64, b: u64) -> Result<i32> {
    let res_a = (a >> 60) as u8;
    let res_b = (b >> 60) as u8;
    if res_a != res_b {
        return Err(ZyronError::ExecutionError(
            "H3 distance requires same resolution".into(),
        ));
    }
    let ax = ((a >> 30) & 0x3FFFFFFF) as i64;
    let ay = (a & 0x3FFFFFFF) as i64;
    let bx = ((b >> 30) & 0x3FFFFFFF) as i64;
    let by = (b & 0x3FFFFFFF) as i64;
    // Chebyshev distance
    let dx = (ax - bx).abs();
    let dy = (ay - by).abs();
    Ok(dx.max(dy) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_point() {
        let p = st_make_point(-73.9857, 40.7484);
        if let GeometryKind::Point(pt) = &p.kind {
            assert_eq!(pt.x, -73.9857);
            assert_eq!(pt.y, 40.7484);
        } else {
            panic!("Expected Point");
        }
        assert_eq!(p.srid, 4326);
    }

    #[test]
    fn test_wkb_roundtrip_point() {
        let p = st_make_point(1.0, 2.0);
        let bytes = encode_wkb(&p);
        let decoded = decode_wkb(&bytes).unwrap();
        assert_eq!(p, decoded);
    }

    #[test]
    fn test_wkb_roundtrip_linestring() {
        let ls = Geometry::with_srid(
            GeometryKind::LineString(LineString {
                points: vec![Point { x: 0.0, y: 0.0 }, Point { x: 1.0, y: 1.0 }],
            }),
            4326,
        );
        let bytes = encode_wkb(&ls);
        let decoded = decode_wkb(&bytes).unwrap();
        assert_eq!(ls, decoded);
    }

    #[test]
    fn test_wkb_roundtrip_polygon() {
        let poly = Geometry::with_srid(
            GeometryKind::Polygon(Polygon {
                exterior: vec![
                    Point { x: 0.0, y: 0.0 },
                    Point { x: 10.0, y: 0.0 },
                    Point { x: 10.0, y: 10.0 },
                    Point { x: 0.0, y: 10.0 },
                    Point { x: 0.0, y: 0.0 },
                ],
                holes: vec![],
            }),
            4326,
        );
        let bytes = encode_wkb(&poly);
        let decoded = decode_wkb(&bytes).unwrap();
        assert_eq!(poly, decoded);
    }

    #[test]
    fn test_wkt_parse_point() {
        let g = st_geom_from_text("POINT(1.5 2.5)").unwrap();
        if let GeometryKind::Point(p) = &g.kind {
            assert_eq!(p.x, 1.5);
            assert_eq!(p.y, 2.5);
        }
    }

    #[test]
    fn test_wkt_parse_linestring() {
        let g = st_geom_from_text("LINESTRING(0 0, 1 1, 2 2)").unwrap();
        if let GeometryKind::LineString(ls) = &g.kind {
            assert_eq!(ls.points.len(), 3);
        }
    }

    #[test]
    fn test_as_text_point() {
        let p = st_make_point(1.0, 2.0);
        assert!(st_as_text(&p).contains("POINT"));
    }

    #[test]
    fn test_as_geojson_point() {
        let p = st_make_point(1.5, 2.5);
        let json = st_as_geojson(&p);
        assert!(json.contains("\"Point\""));
        assert!(json.contains("1.5"));
    }

    #[test]
    fn test_geojson_parse_point() {
        let g = st_geom_from_geojson(r#"{"type":"Point","coordinates":[1.5,2.5]}"#).unwrap();
        if let GeometryKind::Point(p) = &g.kind {
            assert_eq!(p.x, 1.5);
            assert_eq!(p.y, 2.5);
        }
    }

    #[test]
    fn test_distance_nyc_london() {
        // Distance from NYC (40.7484, -73.9857) to London (51.5074, -0.1278)
        // is approximately 5570 km
        let nyc = st_make_point(-73.9857, 40.7484);
        let london = st_make_point(-0.1278, 51.5074);
        let dist = st_distance(&nyc, &london).unwrap();
        assert!((dist - 5_570_000.0).abs() < 50_000.0);
    }

    #[test]
    fn test_distance_same_point() {
        let p = st_make_point(1.0, 2.0);
        let dist = st_distance(&p, &p).unwrap();
        assert!(dist < 1.0);
    }

    #[test]
    fn test_dwithin() {
        let a = st_make_point(-73.9857, 40.7484);
        let b = st_make_point(-73.9856, 40.7485);
        // Very close - should be within 1km
        assert!(st_dwithin(&a, &b, 1000.0).unwrap());
        // Not within 1 meter
        assert!(!st_dwithin(&a, &b, 1.0).unwrap());
    }

    #[test]
    fn test_contains_simple() {
        let polygon = st_geom_from_text("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
        let inside = st_make_point(5.0, 5.0);
        let outside = st_make_point(15.0, 5.0);
        assert!(st_contains(&polygon, &inside).unwrap());
        assert!(!st_contains(&polygon, &outside).unwrap());
    }

    #[test]
    fn test_area_square() {
        let poly = st_geom_from_text("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
        let area = st_area(&poly).unwrap();
        assert!((area - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_area_triangle() {
        let poly = st_geom_from_text("POLYGON((0 0, 10 0, 0 10, 0 0))").unwrap();
        let area = st_area(&poly).unwrap();
        assert!((area - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_buffer_creates_polygon() {
        let p = st_make_point(0.0, 0.0);
        let buffered = st_buffer(&p, 1.0).unwrap();
        if let GeometryKind::Polygon(poly) = &buffered.kind {
            assert!(poly.exterior.len() > 30);
        } else {
            panic!("Expected Polygon");
        }
    }

    #[test]
    fn test_union_collection() {
        let a = st_make_point(0.0, 0.0);
        let b = st_make_point(1.0, 1.0);
        let u = st_union(&a, &b).unwrap();
        if let GeometryKind::GeometryCollection(geoms) = &u.kind {
            assert_eq!(geoms.len(), 2);
        }
    }

    #[test]
    fn test_centroid_point() {
        let p = st_make_point(3.0, 4.0);
        let c = st_centroid(&p).unwrap();
        if let GeometryKind::Point(pt) = &c.kind {
            assert_eq!(pt.x, 3.0);
            assert_eq!(pt.y, 4.0);
        }
    }

    #[test]
    fn test_centroid_square() {
        let poly = st_geom_from_text("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
        let c = st_centroid(&poly).unwrap();
        if let GeometryKind::Point(pt) = &c.kind {
            assert!((pt.x - 5.0).abs() < 0.01);
            assert!((pt.y - 5.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_intersects_point_polygon() {
        let poly = st_geom_from_text("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
        let inside = st_make_point(5.0, 5.0);
        assert!(st_intersects(&poly, &inside).unwrap());
    }

    #[test]
    fn test_h3_roundtrip() {
        let idx = h3_from_point(0.0, 0.0, 5).unwrap();
        let boundary = h3_to_boundary(idx).unwrap();
        if let GeometryKind::Polygon(_) = &boundary.kind {
            // Boundary exists
        } else {
            panic!("Expected Polygon");
        }
    }

    #[test]
    fn test_h3_different_points_different_cells() {
        let a = h3_from_point(0.0, 0.0, 5).unwrap();
        let b = h3_from_point(90.0, 45.0, 5).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_h3_distance_same_cell() {
        let a = h3_from_point(0.0, 0.0, 5).unwrap();
        assert_eq!(h3_distance(a, a).unwrap(), 0);
    }

    #[test]
    fn test_h3_distance_different_cells() {
        let a = h3_from_point(0.0, 0.0, 5).unwrap();
        let b = h3_from_point(90.0, 45.0, 5).unwrap();
        let d = h3_distance(a, b).unwrap();
        assert!(d > 0);
    }

    #[test]
    fn test_h3_invalid_resolution() {
        assert!(h3_from_point(0.0, 0.0, 20).is_err());
    }
}
