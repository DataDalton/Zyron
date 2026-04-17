//! RGBA color type stored as a packed u32 (4 bytes).
//!
//! Layout: bits 24-31 = red, 16-23 = green, 8-15 = blue, 0-7 = alpha.
//! All color space conversions happen at function call time, not at storage time.

use zyron_common::{Result, ZyronError};

/// Packs RGB values into a u32 with alpha = 255 (fully opaque).
pub fn color_from_rgb(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | 0xFF
}

/// Packs RGBA values into a u32.
pub fn color_from_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32)
}

/// Extracts RGBA components from a packed u32.
fn unpack(rgba: u32) -> (u8, u8, u8, u8) {
    let r = ((rgba >> 24) & 0xFF) as u8;
    let g = ((rgba >> 16) & 0xFF) as u8;
    let b = ((rgba >> 8) & 0xFF) as u8;
    let a = (rgba & 0xFF) as u8;
    (r, g, b, a)
}

/// Parses a hex color string into a packed RGBA u32.
/// Accepts: "#RGB", "#RGBA", "#RRGGBB", "#RRGGBBAA" (with or without '#').
pub fn color_from_hex(hex: &str) -> Result<u32> {
    let s = hex.strip_prefix('#').unwrap_or(hex);
    let parse_err = || ZyronError::ExecutionError(format!("Invalid hex color: {}", hex));

    match s.len() {
        3 => {
            // #RGB -> #RRGGBB with alpha FF
            let r = hex_nibble(s.as_bytes()[0]).ok_or_else(parse_err)?;
            let g = hex_nibble(s.as_bytes()[1]).ok_or_else(parse_err)?;
            let b = hex_nibble(s.as_bytes()[2]).ok_or_else(parse_err)?;
            Ok(color_from_rgb(r << 4 | r, g << 4 | g, b << 4 | b))
        }
        4 => {
            // #RGBA
            let r = hex_nibble(s.as_bytes()[0]).ok_or_else(parse_err)?;
            let g = hex_nibble(s.as_bytes()[1]).ok_or_else(parse_err)?;
            let b = hex_nibble(s.as_bytes()[2]).ok_or_else(parse_err)?;
            let a = hex_nibble(s.as_bytes()[3]).ok_or_else(parse_err)?;
            Ok(color_from_rgba(
                r << 4 | r,
                g << 4 | g,
                b << 4 | b,
                a << 4 | a,
            ))
        }
        6 => {
            let r = hex_byte(s.as_bytes(), 0).ok_or_else(parse_err)?;
            let g = hex_byte(s.as_bytes(), 2).ok_or_else(parse_err)?;
            let b = hex_byte(s.as_bytes(), 4).ok_or_else(parse_err)?;
            Ok(color_from_rgb(r, g, b))
        }
        8 => {
            let r = hex_byte(s.as_bytes(), 0).ok_or_else(parse_err)?;
            let g = hex_byte(s.as_bytes(), 2).ok_or_else(parse_err)?;
            let b = hex_byte(s.as_bytes(), 4).ok_or_else(parse_err)?;
            let a = hex_byte(s.as_bytes(), 6).ok_or_else(parse_err)?;
            Ok(color_from_rgba(r, g, b, a))
        }
        _ => Err(parse_err()),
    }
}

fn hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn hex_byte(bytes: &[u8], offset: usize) -> Option<u8> {
    let hi = hex_nibble(bytes[offset])?;
    let lo = hex_nibble(bytes[offset + 1])?;
    Some((hi << 4) | lo)
}

/// Converts HSL (hue 0-360, saturation 0-1, lightness 0-1) to a packed RGBA u32.
/// Alpha is set to 255.
pub fn color_from_hsl(h: f64, s: f64, l: f64) -> u32 {
    let (r, g, b) = hsl_to_rgb(h, s, l);
    color_from_rgb(
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    )
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = ((h % 360.0) + 360.0) % 360.0 / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);
    (r, g, b)
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

/// Formats a packed RGBA u32 as a hex string "#RRGGBB" or "#RRGGBBAA" if alpha != 255.
pub fn color_to_hex(rgba: u32) -> String {
    let (r, g, b, a) = unpack(rgba);
    if a == 0xFF {
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", r, g, b, a)
    }
}

/// Converts a packed RGBA u32 to HSL (hue 0-360, saturation 0-1, lightness 0-1).
pub fn color_to_hsl(rgba: u32) -> (f64, f64, f64) {
    let (r, g, b, _) = unpack(rgba);
    let rf = r as f64 / 255.0;
    let gf = g as f64 / 255.0;
    let bf = b as f64 / 255.0;

    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let l = (max + min) / 2.0;

    if (max - min).abs() < f64::EPSILON {
        return (0.0, 0.0, l);
    }

    let delta = max - min;
    let s = if l > 0.5 {
        delta / (2.0 - max - min)
    } else {
        delta / (max + min)
    };

    let h = if (max - rf).abs() < f64::EPSILON {
        let mut hue = (gf - bf) / delta;
        if gf < bf {
            hue += 6.0;
        }
        hue
    } else if (max - gf).abs() < f64::EPSILON {
        (bf - rf) / delta + 2.0
    } else {
        (rf - gf) / delta + 4.0
    };

    (h * 60.0, s, l)
}

/// Blends two colors by the given ratio (0.0 = color a, 1.0 = color b).
/// Alpha is also blended.
pub fn color_blend(a: u32, b: u32, ratio: f64) -> u32 {
    let ratio = ratio.clamp(0.0, 1.0);
    let inv = 1.0 - ratio;

    let (ar, ag, ab, aa) = unpack(a);
    let (br, bg, bb, ba) = unpack(b);

    let r = (ar as f64 * inv + br as f64 * ratio + 0.5) as u8;
    let g = (ag as f64 * inv + bg as f64 * ratio + 0.5) as u8;
    let b_out = (ab as f64 * inv + bb as f64 * ratio + 0.5) as u8;
    let a_out = (aa as f64 * inv + ba as f64 * ratio + 0.5) as u8;

    color_from_rgba(r, g, b_out, a_out)
}

/// Lightens a color by the given amount (0.0 = no change, 1.0 = white).
/// Operates in HSL space.
pub fn color_lighten(color: u32, amount: f64) -> u32 {
    let (h, s, l) = color_to_hsl(color);
    let new_l = (l + amount).clamp(0.0, 1.0);
    let (_, _, _, a) = unpack(color);
    let base = color_from_hsl(h, s, new_l);
    // Preserve original alpha
    (base & 0xFFFFFF00) | (a as u32)
}

/// Darkens a color by the given amount (0.0 = no change, 1.0 = black).
/// Operates in HSL space.
pub fn color_darken(color: u32, amount: f64) -> u32 {
    let (h, s, l) = color_to_hsl(color);
    let new_l = (l - amount).clamp(0.0, 1.0);
    let (_, _, _, a) = unpack(color);
    let base = color_from_hsl(h, s, new_l);
    (base & 0xFFFFFF00) | (a as u32)
}

/// Computes the WCAG 2.0 contrast ratio between two colors.
/// Returns a value between 1.0 (identical) and 21.0 (black vs white).
pub fn wcag_contrast_ratio(fg: u32, bg: u32) -> f64 {
    let l1 = relative_luminance(fg);
    let l2 = relative_luminance(bg);
    let lighter = l1.max(l2);
    let darker = l1.min(l2);
    (lighter + 0.05) / (darker + 0.05)
}

/// Computes the relative luminance of a color per WCAG 2.0.
fn relative_luminance(rgba: u32) -> f64 {
    let (r, g, b, _) = unpack(rgba);
    let rs = srgb_to_linear(r as f64 / 255.0);
    let gs = srgb_to_linear(g as f64 / 255.0);
    let bs = srgb_to_linear(b as f64 / 255.0);
    0.2126 * rs + 0.7152 * gs + 0.0722 * bs
}

fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Checks if the foreground/background contrast meets the given WCAG level.
/// Level must be "AA" (ratio >= 4.5) or "AAA" (ratio >= 7.0).
pub fn wcag_compliant(fg: u32, bg: u32, level: &str) -> Result<bool> {
    let ratio = wcag_contrast_ratio(fg, bg);
    match level.to_uppercase().as_str() {
        "AA" => Ok(ratio >= 4.5),
        "AAA" => Ok(ratio >= 7.0),
        _ => Err(ZyronError::ExecutionError(format!(
            "Unknown WCAG level '{}', expected 'AA' or 'AAA'",
            level
        ))),
    }
}

/// Generates a color palette from a base color using the given scheme.
/// Supported schemes: "complementary", "analogous", "triadic", "split-complementary".
pub fn color_palette(base: u32, scheme: &str) -> Result<Vec<u32>> {
    let (h, s, l) = color_to_hsl(base);
    let (_, _, _, a) = unpack(base);

    let hues = match scheme.to_lowercase().as_str() {
        "complementary" => vec![h, (h + 180.0) % 360.0],
        "analogous" => vec![(h + 330.0) % 360.0, h, (h + 30.0) % 360.0],
        "triadic" => vec![h, (h + 120.0) % 360.0, (h + 240.0) % 360.0],
        "split-complementary" | "split_complementary" => {
            vec![h, (h + 150.0) % 360.0, (h + 210.0) % 360.0]
        }
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "Unknown palette scheme '{}', expected 'complementary', 'analogous', 'triadic', or 'split-complementary'",
                scheme
            )));
        }
    };

    let palette = hues
        .into_iter()
        .map(|hue| {
            let c = color_from_hsl(hue, s, l);
            (c & 0xFFFFFF00) | (a as u32)
        })
        .collect();
    Ok(palette)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_packing() {
        let c = color_from_rgb(255, 128, 0);
        let (r, g, b, a) = unpack(c);
        assert_eq!((r, g, b, a), (255, 128, 0, 255));
    }

    #[test]
    fn test_rgba_packing() {
        let c = color_from_rgba(10, 20, 30, 40);
        let (r, g, b, a) = unpack(c);
        assert_eq!((r, g, b, a), (10, 20, 30, 40));
    }

    #[test]
    fn test_hex_6_digit() {
        let c = color_from_hex("#FF5733").unwrap();
        let (r, g, b, a) = unpack(c);
        assert_eq!((r, g, b, a), (0xFF, 0x57, 0x33, 0xFF));
    }

    #[test]
    fn test_hex_8_digit() {
        let c = color_from_hex("#FF573380").unwrap();
        let (r, g, b, a) = unpack(c);
        assert_eq!((r, g, b, a), (0xFF, 0x57, 0x33, 0x80));
    }

    #[test]
    fn test_hex_3_digit() {
        let c = color_from_hex("#F00").unwrap();
        let (r, g, b, _) = unpack(c);
        assert_eq!((r, g, b), (0xFF, 0x00, 0x00));
    }

    #[test]
    fn test_hex_no_hash() {
        let c = color_from_hex("FF5733").unwrap();
        let (r, g, b, _) = unpack(c);
        assert_eq!((r, g, b), (0xFF, 0x57, 0x33));
    }

    #[test]
    fn test_hex_invalid() {
        assert!(color_from_hex("#GGG").is_err());
        assert!(color_from_hex("#12345").is_err());
        assert!(color_from_hex("").is_err());
    }

    #[test]
    fn test_to_hex() {
        let c = color_from_rgb(255, 87, 51);
        assert_eq!(color_to_hex(c), "#ff5733");
    }

    #[test]
    fn test_to_hex_with_alpha() {
        let c = color_from_rgba(255, 87, 51, 128);
        assert_eq!(color_to_hex(c), "#ff573380");
    }

    #[test]
    fn test_hsl_pure_red() {
        let c = color_from_hsl(0.0, 1.0, 0.5);
        let (r, g, b, _) = unpack(c);
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_hsl_pure_green() {
        let c = color_from_hsl(120.0, 1.0, 0.5);
        let (r, g, b, _) = unpack(c);
        assert_eq!(r, 0);
        assert_eq!(g, 255);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_hsl_pure_blue() {
        let c = color_from_hsl(240.0, 1.0, 0.5);
        let (r, g, b, _) = unpack(c);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_hsl_white() {
        let c = color_from_hsl(0.0, 0.0, 1.0);
        let (r, g, b, _) = unpack(c);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_hsl_black() {
        let c = color_from_hsl(0.0, 0.0, 0.0);
        let (r, g, b, _) = unpack(c);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_hsl_roundtrip() {
        let original = color_from_rgb(180, 100, 50);
        let (h, s, l) = color_to_hsl(original);
        let recovered = color_from_hsl(h, s, l);
        let (r1, g1, b1, _) = unpack(original);
        let (r2, g2, b2, _) = unpack(recovered);
        assert!((r1 as i32 - r2 as i32).unsigned_abs() <= 1);
        assert!((g1 as i32 - g2 as i32).unsigned_abs() <= 1);
        assert!((b1 as i32 - b2 as i32).unsigned_abs() <= 1);
    }

    #[test]
    fn test_to_hsl_gray() {
        let c = color_from_rgb(128, 128, 128);
        let (h, s, l) = color_to_hsl(c);
        assert_eq!(h, 0.0);
        assert!(s.abs() < 0.001);
        assert!((l - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_blend_midpoint() {
        let black = color_from_rgb(0, 0, 0);
        let white = color_from_rgb(255, 255, 255);
        let mid = color_blend(black, white, 0.5);
        let (r, g, b, _) = unpack(mid);
        assert!((r as i32 - 128).unsigned_abs() <= 1);
        assert!((g as i32 - 128).unsigned_abs() <= 1);
        assert!((b as i32 - 128).unsigned_abs() <= 1);
    }

    #[test]
    fn test_blend_endpoints() {
        let a = color_from_rgb(100, 200, 50);
        let b = color_from_rgb(200, 50, 100);
        let at_a = color_blend(a, b, 0.0);
        let at_b = color_blend(a, b, 1.0);
        assert_eq!(at_a, a);
        assert_eq!(at_b, b);
    }

    #[test]
    fn test_lighten() {
        let c = color_from_rgb(128, 0, 0);
        let lighter = color_lighten(c, 0.2);
        let (_, _, l_orig) = color_to_hsl(c);
        let (_, _, l_new) = color_to_hsl(lighter);
        assert!(l_new > l_orig);
    }

    #[test]
    fn test_darken() {
        let c = color_from_rgb(128, 128, 128);
        let darker = color_darken(c, 0.2);
        let (_, _, l_orig) = color_to_hsl(c);
        let (_, _, l_new) = color_to_hsl(darker);
        assert!(l_new < l_orig);
    }

    #[test]
    fn test_wcag_black_white() {
        let black = color_from_rgb(0, 0, 0);
        let white = color_from_rgb(255, 255, 255);
        let ratio = wcag_contrast_ratio(black, white);
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn test_wcag_same_color() {
        let c = color_from_rgb(128, 128, 128);
        let ratio = wcag_contrast_ratio(c, c);
        assert!((ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_wcag_compliant_aa() {
        let black = color_from_rgb(0, 0, 0);
        let white = color_from_rgb(255, 255, 255);
        assert!(wcag_compliant(black, white, "AA").unwrap());
        assert!(wcag_compliant(black, white, "AAA").unwrap());
    }

    #[test]
    fn test_wcag_compliant_low_contrast() {
        let gray1 = color_from_rgb(128, 128, 128);
        let gray2 = color_from_rgb(150, 150, 150);
        assert!(!wcag_compliant(gray1, gray2, "AA").unwrap());
    }

    #[test]
    fn test_wcag_invalid_level() {
        assert!(wcag_compliant(0, 0, "B").is_err());
    }

    #[test]
    fn test_palette_complementary() {
        let red = color_from_hsl(0.0, 1.0, 0.5);
        let palette = color_palette(red, "complementary").unwrap();
        assert_eq!(palette.len(), 2);
        // Second color should be cyan (180 degrees)
        let (h, _, _) = color_to_hsl(palette[1]);
        assert!((h - 180.0).abs() < 2.0);
    }

    #[test]
    fn test_palette_triadic() {
        let red = color_from_hsl(0.0, 1.0, 0.5);
        let palette = color_palette(red, "triadic").unwrap();
        assert_eq!(palette.len(), 3);
    }

    #[test]
    fn test_palette_analogous() {
        let c = color_from_hsl(90.0, 0.8, 0.5);
        let palette = color_palette(c, "analogous").unwrap();
        assert_eq!(palette.len(), 3);
    }

    #[test]
    fn test_palette_split_complementary() {
        let c = color_from_hsl(0.0, 1.0, 0.5);
        let palette = color_palette(c, "split-complementary").unwrap();
        assert_eq!(palette.len(), 3);
    }

    #[test]
    fn test_palette_invalid_scheme() {
        assert!(color_palette(0, "invalid").is_err());
    }

    #[test]
    fn test_lighten_preserves_alpha() {
        let c = color_from_rgba(128, 0, 0, 100);
        let lighter = color_lighten(c, 0.2);
        let (_, _, _, a) = unpack(lighter);
        assert_eq!(a, 100);
    }

    #[test]
    fn test_darken_preserves_alpha() {
        let c = color_from_rgba(200, 200, 200, 50);
        let darker = color_darken(c, 0.1);
        let (_, _, _, a) = unpack(darker);
        assert_eq!(a, 50);
    }
}
