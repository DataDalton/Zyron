//! Image metadata extraction

use std::io::Cursor;

use serde_json::{Map, Value, json};

use crate::error::{MediaError, MediaResult};

/// Image formats recognized by magic bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormatKind {
    Jpeg,
    Png,
    Webp,
}

impl ImageFormatKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ImageFormatKind::Jpeg => "jpeg",
            ImageFormatKind::Png => "png",
            ImageFormatKind::Webp => "webp",
        }
    }

    pub fn to_image_format(self) -> image::ImageFormat {
        match self {
            ImageFormatKind::Jpeg => image::ImageFormat::Jpeg,
            ImageFormatKind::Png => image::ImageFormat::Png,
            ImageFormatKind::Webp => image::ImageFormat::WebP,
        }
    }
}

/// Detects jpeg, png or webp from magic bytes
pub fn detect_image_format(bytes: &[u8]) -> MediaResult<ImageFormatKind> {
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return Ok(ImageFormatKind::Jpeg);
    }
    if bytes.len() >= 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Ok(ImageFormatKind::Png);
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Ok(ImageFormatKind::Webp);
    }
    Err(MediaError::UnsupportedFormat(
        "unrecognized image bytes, supported formats are jpeg, png and webp".to_string(),
    ))
}

/// Extracts width, height, format, color space, bit depth, dpi and exif tags.
/// Reads only the image header, every field here is header data and the
/// write path runs this on each media insert, so decoding the pixels would
/// charge megapixels of work for a handful of header bytes
pub fn image_metadata(bytes: &[u8]) -> MediaResult<Value> {
    use image::ImageDecoder;

    let kind = detect_image_format(bytes)?;
    let decoder = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| MediaError::UnsupportedFormat(format!("image sniff failed: {e}")))?
        .into_decoder()
        .map_err(|e| MediaError::UnsupportedFormat(format!("image header parse failed: {e}")))?;

    let (width, height) = decoder.dimensions();
    let (color_space, bit_depth) = describe_color(decoder.color_type());
    let exif_map = extract_exif(bytes);
    let dpi = extract_dpi(bytes, kind, &exif_map);

    Ok(json!({
        "width": width,
        "height": height,
        "format": kind.as_str(),
        "color_space": color_space,
        "bit_depth": bit_depth,
        "dpi": dpi,
        "exif": Value::Object(exif_map),
    }))
}

fn describe_color(color: image::ColorType) -> (&'static str, u32) {
    match color {
        image::ColorType::L8 => ("grayscale", 8),
        image::ColorType::La8 => ("grayscale_alpha", 8),
        image::ColorType::Rgb8 => ("rgb", 8),
        image::ColorType::Rgba8 => ("rgba", 8),
        image::ColorType::L16 => ("grayscale", 16),
        image::ColorType::La16 => ("grayscale_alpha", 16),
        image::ColorType::Rgb16 => ("rgb", 16),
        image::ColorType::Rgba16 => ("rgba", 16),
        image::ColorType::Rgb32F => ("rgb", 32),
        image::ColorType::Rgba32F => ("rgba", 32),
        _ => ("unknown", 0),
    }
}

/// Flattened primary ifd exif tags, empty when the image carries none
fn extract_exif(bytes: &[u8]) -> Map<String, Value> {
    let mut map = Map::new();
    let mut cursor = Cursor::new(bytes);
    let reader = exif::Reader::new();
    if let Ok(parsed) = reader.read_from_container(&mut cursor) {
        for field in parsed.fields() {
            if field.ifd_num != exif::In::PRIMARY {
                continue;
            }
            let name = field.tag.to_string();
            let value = field.display_value().to_string();
            map.insert(name, Value::String(value));
        }
    }
    map
}

fn extract_dpi(bytes: &[u8], kind: ImageFormatKind, exif_map: &Map<String, Value>) -> Value {
    if kind == ImageFormatKind::Jpeg
        && let Some(dpi) = jfif_dpi(bytes)
    {
        return json!(dpi);
    }
    if kind == ImageFormatKind::Png
        && let Some(dpi) = png_phys_dpi(bytes)
    {
        return json!(dpi);
    }
    if let Some(dpi) = exif_dpi(bytes, exif_map) {
        return json!(dpi);
    }
    Value::Null
}

/// Reads the density fields from a jpeg JFIF APP0 segment
fn jfif_dpi(bytes: &[u8]) -> Option<f64> {
    if bytes.len() < 18 || bytes[2] != 0xFF || bytes[3] != 0xE0 {
        return None;
    }
    if &bytes[6..11] != b"JFIF\0" {
        return None;
    }
    let units = bytes[13];
    let x_density = u16::from_be_bytes([bytes[14], bytes[15]]) as f64;
    match units {
        1 if x_density > 0.0 => Some(x_density),
        2 if x_density > 0.0 => Some(x_density * 2.54),
        _ => None,
    }
}

/// Reads a png pHYs chunk, converting pixels per meter to dpi
fn png_phys_dpi(bytes: &[u8]) -> Option<f64> {
    let mut pos = 8usize;
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        let chunk_type = &bytes[pos + 4..pos + 8];
        if chunk_type == b"pHYs" && pos + 8 + 9 <= bytes.len() {
            let data = &bytes[pos + 8..pos + 17];
            let ppux = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as f64;
            let unit = data[8];
            if unit == 1 && ppux > 0.0 {
                return Some((ppux * 0.0254 * 100.0).round() / 100.0);
            }
            return None;
        }
        if chunk_type == b"IDAT" || chunk_type == b"IEND" {
            return None;
        }
        pos += 12 + len;
    }
    None
}

/// Falls back to exif XResolution with ResolutionUnit handling
fn exif_dpi(bytes: &[u8], exif_map: &Map<String, Value>) -> Option<f64> {
    if !exif_map.contains_key("XResolution") {
        return None;
    }
    let mut cursor = Cursor::new(bytes);
    let parsed = exif::Reader::new().read_from_container(&mut cursor).ok()?;
    let xres = parsed.get_field(exif::Tag::XResolution, exif::In::PRIMARY)?;
    let value = match &xres.value {
        exif::Value::Rational(r) if !r.is_empty() => r[0].to_f64(),
        _ => return None,
    };
    let unit = parsed
        .get_field(exif::Tag::ResolutionUnit, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .unwrap_or(2);
    match unit {
        2 => Some(value),
        3 => Some(value * 2.54),
        _ => None,
    }
}
