//! Image transformations returning encoded bytes

use std::io::Cursor;

use image::DynamicImage;

use crate::error::{MediaError, MediaResult};
use crate::image_meta::{ImageFormatKind, detect_image_format};

/// How resize maps the source onto the requested dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Fits within the box preserving aspect ratio
    Fit,
    /// Fills the box preserving aspect ratio, cropping overflow
    Cover,
    /// Matches the box exactly, ignoring aspect ratio
    Stretch,
}

/// Resizes and re encodes in the source format
pub fn resize(bytes: &[u8], width: u32, height: u32, mode: ResizeMode) -> MediaResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(MediaError::InvalidArgument(
            "resize dimensions must be greater than zero".to_string(),
        ));
    }
    let (kind, img) = decode(bytes)?;
    let filter = image::imageops::FilterType::Lanczos3;
    let resized = match mode {
        ResizeMode::Fit => img.resize(width, height, filter),
        ResizeMode::Cover => img.resize_to_fill(width, height, filter),
        ResizeMode::Stretch => img.resize_exact(width, height, filter),
    };
    encode(&resized, kind)
}

/// Crops a rectangle and re encodes in the source format
pub fn crop(bytes: &[u8], x: u32, y: u32, width: u32, height: u32) -> MediaResult<Vec<u8>> {
    let (kind, img) = decode(bytes)?;
    if width == 0 || height == 0 {
        return Err(MediaError::InvalidArgument(
            "crop dimensions must be greater than zero".to_string(),
        ));
    }
    let (src_w, src_h) = (img.width(), img.height());
    let x_end = x.checked_add(width);
    let y_end = y.checked_add(height);
    match (x_end, y_end) {
        (Some(xe), Some(ye)) if xe <= src_w && ye <= src_h => {}
        _ => {
            return Err(MediaError::InvalidArgument(format!(
                "crop rectangle {x},{y} {width}x{height} exceeds image bounds {src_w}x{src_h}"
            )));
        }
    }
    let cropped = img.crop_imm(x, y, width, height);
    encode(&cropped, kind)
}

/// Rotates by an exact quarter turn and re encodes in the source format
pub fn rotate(bytes: &[u8], degrees: u32) -> MediaResult<Vec<u8>> {
    let (kind, img) = decode(bytes)?;
    let rotated = match degrees % 360 {
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        other => {
            return Err(MediaError::InvalidArgument(format!(
                "rotation by {other} degrees is not supported, supported angles are 90, 180 and 270"
            )));
        }
    };
    encode(&rotated, kind)
}

/// Re encodes into the named target format
pub fn convert_format(bytes: &[u8], target: &str) -> MediaResult<Vec<u8>> {
    let target_kind = match target.to_ascii_lowercase().as_str() {
        "jpeg" | "jpg" => ImageFormatKind::Jpeg,
        "png" => ImageFormatKind::Png,
        "webp" => ImageFormatKind::Webp,
        other => {
            return Err(MediaError::InvalidArgument(format!(
                "unknown target image format {other}, supported targets are jpeg, png and webp"
            )));
        }
    };
    let (_, img) = decode(bytes)?;
    encode(&img, target_kind)
}

fn decode(bytes: &[u8]) -> MediaResult<(ImageFormatKind, DynamicImage)> {
    let kind = detect_image_format(bytes)?;
    let img = image::load_from_memory_with_format(bytes, kind.to_image_format())
        .map_err(|e| MediaError::UnsupportedFormat(format!("image decode failed: {e}")))?;
    Ok((kind, img))
}

fn encode(img: &DynamicImage, kind: ImageFormatKind) -> MediaResult<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    match kind {
        // jpeg carries no alpha channel, flatten before encoding
        ImageFormatKind::Jpeg => {
            let rgb = DynamicImage::ImageRgb8(img.to_rgb8());
            rgb.write_to(&mut out, image::ImageFormat::Jpeg)
        }
        ImageFormatKind::Png => img.write_to(&mut out, image::ImageFormat::Png),
        ImageFormatKind::Webp => {
            // the pure rust webp encoder accepts rgb8 and rgba8 only
            let rgba = DynamicImage::ImageRgba8(img.to_rgba8());
            rgba.write_to(&mut out, image::ImageFormat::WebP)
        }
    }
    .map_err(|e| MediaError::UnsupportedFormat(format!("image encode failed: {e}")))?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_meta::image_metadata;

    fn sample_png(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("encode png");
        out.into_inner()
    }

    fn sample_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 64])
        });
        let mut out = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Jpeg)
            .expect("encode jpeg");
        out.into_inner()
    }

    fn dims(bytes: &[u8]) -> (u64, u64) {
        let meta = image_metadata(bytes).expect("metadata");
        (
            meta["width"].as_u64().expect("width"),
            meta["height"].as_u64().expect("height"),
        )
    }

    #[test]
    fn metadata_reports_dimensions_and_format() {
        let png = sample_png(64, 48);
        let meta = image_metadata(&png).expect("metadata");
        assert_eq!(meta["width"], 64);
        assert_eq!(meta["height"], 48);
        assert_eq!(meta["format"], "png");
        assert_eq!(meta["color_space"], "rgb");
        assert_eq!(meta["bit_depth"], 8);
        assert!(meta["exif"].as_object().expect("exif").is_empty());

        let jpeg = sample_jpeg(64, 48);
        let meta = image_metadata(&jpeg).expect("metadata");
        assert_eq!(meta["format"], "jpeg");
        assert_eq!(meta["width"], 64);
    }

    #[test]
    fn resize_modes_produce_expected_dimensions() {
        let png = sample_png(64, 48);
        assert_eq!(
            dims(&resize(&png, 32, 32, ResizeMode::Fit).expect("fit")),
            (32, 24)
        );
        assert_eq!(
            dims(&resize(&png, 32, 32, ResizeMode::Cover).expect("cover")),
            (32, 32)
        );
        assert_eq!(
            dims(&resize(&png, 32, 32, ResizeMode::Stretch).expect("stretch")),
            (32, 32)
        );
    }

    #[test]
    fn crop_respects_bounds() {
        let png = sample_png(64, 48);
        assert_eq!(dims(&crop(&png, 10, 10, 20, 20).expect("crop")), (20, 20));
        assert!(crop(&png, 60, 40, 10, 10).is_err());
        assert!(crop(&png, 0, 0, 0, 5).is_err());
    }

    #[test]
    fn rotate_quarter_turns() {
        let png = sample_png(64, 48);
        assert_eq!(dims(&rotate(&png, 90).expect("rotate 90")), (48, 64));
        assert_eq!(dims(&rotate(&png, 180).expect("rotate 180")), (64, 48));
        assert_eq!(dims(&rotate(&png, 270).expect("rotate 270")), (48, 64));
        assert!(rotate(&png, 45).is_err());
    }

    #[test]
    fn convert_chain_png_jpeg_webp() {
        let png = sample_png(64, 48);
        let jpeg = convert_format(&png, "jpeg").expect("to jpeg");
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        let webp = convert_format(&jpeg, "webp").expect("to webp");
        assert_eq!(&webp[..4], b"RIFF");
        assert_eq!(&webp[8..12], b"WEBP");
        let meta = image_metadata(&webp).expect("webp metadata");
        assert_eq!(meta["format"], "webp");
        assert_eq!(meta["width"], 64);
        assert!(convert_format(&png, "tiff").is_err());
    }
}
