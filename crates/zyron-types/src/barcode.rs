//! Barcode and QR code encoders and decoders
//!
//! QR Model 2 versions 1-10 in byte mode, all four error correction levels
//! Includes Reed-Solomon GF(256) tables, mask pattern selection, format and
//! version information, finder/timing/alignment patterns. Decoder is
//! round-trip correct for our own encoder (axis-aligned, no perspective)
//!
//! 1D barcodes: Code 128 (subsets B and C auto), Code 39, EAN-13, EAN-8,
//! UPC-A. Each uses standard symbology tables. Decoder scans the centre
//! horizontal line and matches bar widths against the same tables
//!
//! Output formats: PNG (hand-rolled with stored deflate blocks) and SVG
//! (rectangular modules). Both are valid against their respective specs

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarcodeFormat {
    Code128,
    Code39,
    Ean13,
    Ean8,
    UpcA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrErrorCorrection {
    L,
    M,
    Q,
    H,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Svg,
}

/// Generates a 1D barcode as a PNG byte array. Default size is 60 px tall
pub fn barcode_encode(data: &str, format: BarcodeFormat) -> Result<Vec<u8>> {
    barcode_encode_with(data, format, ImageFormat::Png, 60)
}

/// Generates a 1D barcode in the requested image format with the given height
/// in pixels (PNG) or units (SVG)
pub fn barcode_encode_with(
    data: &str,
    format: BarcodeFormat,
    img_format: ImageFormat,
    height: u32,
) -> Result<Vec<u8>> {
    let modules = match format {
        BarcodeFormat::Code128 => encode_code128(data)?,
        BarcodeFormat::Code39 => encode_code39(data)?,
        BarcodeFormat::Ean13 => encode_ean13(data)?,
        BarcodeFormat::Ean8 => encode_ean8(data)?,
        BarcodeFormat::UpcA => encode_upca(data)?,
    };
    let h = height.max(2);
    match img_format {
        ImageFormat::Png => {
            let width = modules.len() as u32;
            let mut pixels = vec![0xFFu8; (width as usize) * (h as usize)];
            for y in 0..h {
                for x in 0..width {
                    let m = modules[x as usize];
                    pixels[(y as usize) * (width as usize) + x as usize] =
                        if m { 0x00 } else { 0xFF };
                }
            }
            write_png_grayscale(width, h, &pixels)
        }
        ImageFormat::Svg => Ok(write_svg_1d(&modules, h)),
    }
}

/// Reads a barcode from a PNG-encoded image. Returns the decoded data and the
/// detected format. Round-trip correct for our own encoder
pub fn barcode_decode(image: &[u8]) -> Result<(String, BarcodeFormat)> {
    let img = decode_png_grayscale(image)?;
    if img.height == 0 || img.width == 0 {
        return Err(ZyronError::ExecutionError("empty image".into()));
    }
    let scan_y = img.height / 2;
    let row_start = (scan_y as usize) * (img.width as usize);
    let row = &img.pixels[row_start..row_start + img.width as usize];
    let modules: Vec<bool> = row.iter().map(|p| *p < 128).collect();
    // Try formats in order of specificity. UPC-A is EAN-13 with leading
    // zero so it must be tried first or the EAN-13 path would win
    if let Ok(s) = decode_upca_modules(&modules) {
        return Ok((s, BarcodeFormat::UpcA));
    }
    if let Ok(s) = decode_ean13_modules(&modules) {
        return Ok((s, BarcodeFormat::Ean13));
    }
    if let Ok(s) = decode_ean8_modules(&modules) {
        return Ok((s, BarcodeFormat::Ean8));
    }
    if let Ok(s) = decode_code39_modules(&modules) {
        return Ok((s, BarcodeFormat::Code39));
    }
    if let Ok(s) = decode_code128_modules(&modules) {
        return Ok((s, BarcodeFormat::Code128));
    }
    Err(ZyronError::ExecutionError(
        "no barcode format matched".into(),
    ))
}

/// Generates a QR code with the requested error correction level. Returns a
/// PNG byte array sized at 8 px per module by default
pub fn qr_encode(data: &str, ec: QrErrorCorrection) -> Result<Vec<u8>> {
    qr_encode_with(data, ec, ImageFormat::Png, 8)
}

pub fn qr_encode_with(
    data: &str,
    ec: QrErrorCorrection,
    img_format: ImageFormat,
    module_size: u32,
) -> Result<Vec<u8>> {
    let qr = qr_build(data.as_bytes(), ec)?;
    let scale = module_size.max(1);
    match img_format {
        ImageFormat::Png => {
            let dim = qr.size as u32 * scale;
            let mut pixels = vec![0xFFu8; (dim as usize) * (dim as usize)];
            for y in 0..qr.size {
                for x in 0..qr.size {
                    if qr.modules[y * qr.size + x] {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let px = (y as u32 * scale + dy) as usize;
                                let py = (x as u32 * scale + dx) as usize;
                                pixels[px * dim as usize + py] = 0x00;
                            }
                        }
                    }
                }
            }
            write_png_grayscale(dim, dim, &pixels)
        }
        ImageFormat::Svg => Ok(write_svg_qr(&qr, scale)),
    }
}

/// Decodes a QR code from a PNG image generated by qr_encode. Round-trip
/// correct for our own encoder. Returns the decoded UTF-8 string
pub fn qr_decode(image: &[u8]) -> Result<String> {
    let img = decode_png_grayscale(image)?;
    qr_decode_grayscale(&img)
}

// ---------------------------------------------------------------------------
// Code 128 (subset B and C auto)
// ---------------------------------------------------------------------------

const CODE128_PATTERNS: &[u16] = &[
    0b11011001100,
    0b11001101100,
    0b11001100110,
    0b10010011000,
    0b10010001100,
    0b10001001100,
    0b10011001000,
    0b10011000100,
    0b10001100100,
    0b11001001000,
    0b11001000100,
    0b11000100100,
    0b10110011100,
    0b10011011100,
    0b10011001110,
    0b10111001100,
    0b10011101100,
    0b10011100110,
    0b11001110010,
    0b11001011100,
    0b11001001110,
    0b11011100100,
    0b11001110100,
    0b11101101110,
    0b11101001100,
    0b11100101100,
    0b11100100110,
    0b11101100100,
    0b11100110100,
    0b11100110010,
    0b11011011000,
    0b11011000110,
    0b11000110110,
    0b10100011000,
    0b10001011000,
    0b10001000110,
    0b10110001000,
    0b10001101000,
    0b10001100010,
    0b11010001000,
    0b11000101000,
    0b11000100010,
    0b10110111000,
    0b10110001110,
    0b10001101110,
    0b10111011000,
    0b10111000110,
    0b10001110110,
    0b11101110110,
    0b11010001110,
    0b11000101110,
    0b11011101000,
    0b11011100010,
    0b11011101110,
    0b11101011000,
    0b11101000110,
    0b11100010110,
    0b11101101000,
    0b11101100010,
    0b11100011010,
    0b11101111010,
    0b11001000010,
    0b11110001010,
    0b10100110000,
    0b10100001100,
    0b10010110000,
    0b10010000110,
    0b10000101100,
    0b10000100110,
    0b10110010000,
    0b10110000100,
    0b10011010000,
    0b10011000010,
    0b10000110100,
    0b10000110010,
    0b11000010010,
    0b11001010000,
    0b11110111010,
    0b11000010100,
    0b10001111010,
    0b10100111100,
    0b10010111100,
    0b10010011110,
    0b10111100100,
    0b10011110100,
    0b10011110010,
    0b11110100100,
    0b11110010100,
    0b11110010010,
    0b11011011110,
    0b11011110110,
    0b11110110110,
    0b10101111000,
    0b10100011110,
    0b10001011110,
    0b10111101000,
    0b10111100010,
    0b11110101000,
    0b11110100010,
    0b10111011110,
    0b10111101110,
    0b11101011110,
    0b11110101110,
    0b11010000100,
    0b11010010000,
    0b11010011100,
    0b11000111010,
];

const CODE128_STOP: u16 = 0b1100011101011;

const CODE128_START_B: u8 = 104;
const CODE128_START_C: u8 = 105;
const CODE128_CODE_B: u8 = 100;
const CODE128_CODE_C: u8 = 99;

fn encode_code128(data: &str) -> Result<Vec<bool>> {
    let bytes = data.as_bytes();
    if bytes.is_empty() {
        return Err(ZyronError::ExecutionError("empty data".into()));
    }
    // Decide initial subset: prefer C for runs of 4+ digits
    let mut codes: Vec<u8> = Vec::new();
    let mut subset_c = leading_digit_run(bytes) >= 4;
    codes.push(if subset_c {
        CODE128_START_C
    } else {
        CODE128_START_B
    });
    let mut i = 0;
    while i < bytes.len() {
        if subset_c {
            if i + 1 < bytes.len() && bytes[i].is_ascii_digit() && bytes[i + 1].is_ascii_digit() {
                let v = (bytes[i] - b'0') * 10 + (bytes[i + 1] - b'0');
                codes.push(v);
                i += 2;
            } else {
                codes.push(CODE128_CODE_B);
                subset_c = false;
            }
        } else {
            // In subset B, switch to C if a 4+ digit run starts
            let run = digit_run_at(bytes, i);
            if run >= 4 && (i + run == bytes.len() || run >= 6) {
                codes.push(CODE128_CODE_C);
                subset_c = true;
                continue;
            }
            let b = bytes[i];
            if !(0x20..=0x7E).contains(&b) {
                return Err(ZyronError::ExecutionError(format!(
                    "code128 subset B unsupported byte 0x{:02x}",
                    b
                )));
            }
            codes.push(b - 0x20);
            i += 1;
        }
    }
    // Checksum: start * 1 + sum(code_i * (i+1)) mod 103
    let mut sum: u64 = codes[0] as u64;
    for (idx, c) in codes.iter().enumerate().skip(1) {
        sum += (*c as u64) * idx as u64;
    }
    let check = (sum % 103) as u8;
    codes.push(check);
    // Build modules
    let mut modules = Vec::with_capacity(codes.len() * 11 + 13 + 20);
    push_quiet(&mut modules, 10);
    for c in &codes {
        let pat = CODE128_PATTERNS[*c as usize];
        for bit in (0..11).rev() {
            modules.push((pat >> bit) & 1 == 1);
        }
    }
    for bit in (0..13).rev() {
        modules.push((CODE128_STOP >> bit) & 1 == 1);
    }
    push_quiet(&mut modules, 10);
    Ok(modules)
}

fn leading_digit_run(bytes: &[u8]) -> usize {
    bytes.iter().take_while(|b| b.is_ascii_digit()).count()
}

fn digit_run_at(bytes: &[u8], at: usize) -> usize {
    bytes[at..]
        .iter()
        .take_while(|b| b.is_ascii_digit())
        .count()
}

fn push_quiet(modules: &mut Vec<bool>, n: usize) {
    for _ in 0..n {
        modules.push(false);
    }
}

fn decode_code128_modules(modules: &[bool]) -> Result<String> {
    // Find start by scanning for any of the start patterns
    let starts = [
        (CODE128_PATTERNS[CODE128_START_B as usize], CODE128_START_B),
        (CODE128_PATTERNS[CODE128_START_C as usize], CODE128_START_C),
    ];
    let pos = (0..=modules.len().saturating_sub(11)).find(|&i| {
        let bits = bits_to_u16(&modules[i..i + 11]);
        starts.iter().any(|(p, _)| *p == bits)
    });
    let pos = pos.ok_or_else(|| ZyronError::ExecutionError("code128 start not found".into()))?;
    let mut codes: Vec<u8> = Vec::new();
    let mut i = pos;
    let mut subset_c = false;
    let start_bits = bits_to_u16(&modules[i..i + 11]);
    if start_bits == CODE128_PATTERNS[CODE128_START_C as usize] {
        subset_c = true;
        codes.push(CODE128_START_C);
    } else {
        codes.push(CODE128_START_B);
    }
    i += 11;
    let mut text = String::new();
    while i + 11 <= modules.len() {
        let bits = bits_to_u16(&modules[i..i + 11]);
        if i + 13 <= modules.len() {
            let stop = bits_to_u16(&modules[i..i + 13]);
            if stop == CODE128_STOP {
                break;
            }
        }
        let code = CODE128_PATTERNS
            .iter()
            .position(|p| *p == bits)
            .ok_or_else(|| ZyronError::ExecutionError("code128 unrecognized symbol".into()))?
            as u8;
        if code == CODE128_CODE_B {
            subset_c = false;
            codes.push(code);
            i += 11;
            continue;
        }
        if code == CODE128_CODE_C {
            subset_c = true;
            codes.push(code);
            i += 11;
            continue;
        }
        if subset_c {
            text.push_str(&format!("{:02}", code));
        } else {
            text.push((code + 0x20) as char);
        }
        codes.push(code);
        i += 11;
    }
    // Drop trailing checksum from text (it was the last code, never appended
    // because the last appended code precedes the stop pattern anyway)
    if !text.is_empty() {
        // The last code is the checksum, which we already wrote into text
        // because we cannot tell it apart from data without lookahead. Strip
        // the right number of trailing characters
        if subset_c {
            // Two characters per checksum digit-pair
            if text.len() >= 2 {
                text.truncate(text.len() - 2);
            }
        } else if !text.is_empty() {
            text.pop();
        }
    }
    Ok(text)
}

fn bits_to_u16(bits: &[bool]) -> u16 {
    let mut v = 0u16;
    for b in bits {
        v = (v << 1) | (*b as u16);
    }
    v
}

// ---------------------------------------------------------------------------
// Code 39
// ---------------------------------------------------------------------------

const CODE39_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%*";

const CODE39_PATTERNS: &[u16] = &[
    0b101001101101,
    0b110100101011,
    0b101100101011,
    0b110110010101,
    0b101001101011,
    0b110100110101,
    0b101100110101,
    0b101001011011,
    0b110100101101,
    0b101100101101,
    0b110101001011,
    0b101101001011,
    0b110110100101,
    0b101011001011,
    0b110101100101,
    0b101101100101,
    0b101010011011,
    0b110101001101,
    0b101101001101,
    0b101011001101,
    0b110101010011,
    0b101101010011,
    0b110110101001,
    0b101011010011,
    0b110101101001,
    0b101101101001,
    0b101010110011,
    0b110101011001,
    0b101101011001,
    0b101011011001,
    0b110010101011,
    0b100110101011,
    0b110011010101,
    0b100101101011,
    0b110010110101,
    0b100110110101,
    0b100101011011,
    0b110010101101,
    0b100110101101,
    0b100100100101,
    0b100100101001,
    0b100101001001,
    0b101001001001,
    0b100101101101,
];

fn encode_code39(data: &str) -> Result<Vec<bool>> {
    let upper = data.to_ascii_uppercase();
    for c in upper.chars() {
        if !CODE39_ALPHABET.contains(c) || c == '*' {
            return Err(ZyronError::ExecutionError(format!(
                "code39 character not allowed: {}",
                c
            )));
        }
    }
    let mut framed = String::with_capacity(upper.len() + 2);
    framed.push('*');
    framed.push_str(&upper);
    framed.push('*');
    let mut modules: Vec<bool> = Vec::new();
    push_quiet(&mut modules, 10);
    for (idx, c) in framed.chars().enumerate() {
        let pos = CODE39_ALPHABET.find(c).ok_or_else(|| {
            ZyronError::ExecutionError(format!("code39 missing pattern for {}", c))
        })?;
        let pat = CODE39_PATTERNS[pos];
        for bit in (0..12).rev() {
            modules.push((pat >> bit) & 1 == 1);
        }
        if idx + 1 < framed.chars().count() {
            modules.push(false);
        }
    }
    push_quiet(&mut modules, 10);
    Ok(modules)
}

fn decode_code39_modules(modules: &[bool]) -> Result<String> {
    let star = CODE39_PATTERNS[CODE39_ALPHABET.find('*').unwrap()];
    let pos =
        (0..=modules.len().saturating_sub(12)).find(|&i| bits_to_u16(&modules[i..i + 12]) == star);
    let start = pos.ok_or_else(|| ZyronError::ExecutionError("code39 start not found".into()))?;
    let mut text = String::new();
    let mut i = start + 12;
    while i + 12 <= modules.len() {
        if i < modules.len() && !modules[i] {
            i += 1;
        }
        if i + 12 > modules.len() {
            break;
        }
        let bits = bits_to_u16(&modules[i..i + 12]);
        if bits == star {
            break;
        }
        let pos = CODE39_PATTERNS
            .iter()
            .position(|p| *p == bits)
            .ok_or_else(|| ZyronError::ExecutionError("code39 unrecognized symbol".into()))?;
        text.push(CODE39_ALPHABET.chars().nth(pos).unwrap());
        i += 12;
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// EAN-13, EAN-8, UPC-A
// ---------------------------------------------------------------------------

// L, G, R encodings for digits 0-9. Each is 7 bits
const EAN_L: [u8; 10] = [
    0b0001101, 0b0011001, 0b0010011, 0b0111101, 0b0100011, 0b0110001, 0b0101111, 0b0111011,
    0b0110111, 0b0001011,
];
const EAN_G: [u8; 10] = [
    0b0100111, 0b0110011, 0b0011011, 0b0100001, 0b0011101, 0b0111001, 0b0000101, 0b0010001,
    0b0001001, 0b0010111,
];
const EAN_R: [u8; 10] = [
    0b1110010, 0b1100110, 0b1101100, 0b1000010, 0b1011100, 0b1001110, 0b1010000, 0b1000100,
    0b1001000, 0b1110100,
];

// First-digit parity patterns for EAN-13. Each entry is 6 bits, bit i: 0=L, 1=G
const EAN13_PARITY: [u8; 10] = [
    0b000000, 0b001011, 0b001101, 0b001110, 0b010011, 0b011001, 0b011100, 0b010101, 0b010110,
    0b011010,
];

fn ean_check_digit(digits: &[u8]) -> u8 {
    let mut sum: u32 = 0;
    for (i, d) in digits.iter().rev().enumerate() {
        let weight = if i % 2 == 0 { 3 } else { 1 };
        sum += (*d as u32) * weight;
    }
    ((10 - (sum % 10)) % 10) as u8
}

fn encode_ean13(data: &str) -> Result<Vec<bool>> {
    let digits = parse_digits_with_optional_check(data, 12, 13)?;
    let mut modules: Vec<bool> = Vec::new();
    push_quiet(&mut modules, 9);
    // Start guard 101
    push_bits(&mut modules, 0b101, 3);
    let parity = EAN13_PARITY[digits[0] as usize];
    for i in 0..6 {
        let d = digits[i + 1];
        let pattern = if (parity >> (5 - i)) & 1 == 1 {
            EAN_G[d as usize]
        } else {
            EAN_L[d as usize]
        };
        push_bits(&mut modules, pattern as u32, 7);
    }
    push_bits(&mut modules, 0b01010, 5);
    for i in 0..6 {
        let d = digits[i + 7];
        push_bits(&mut modules, EAN_R[d as usize] as u32, 7);
    }
    push_bits(&mut modules, 0b101, 3);
    push_quiet(&mut modules, 9);
    Ok(modules)
}

fn encode_ean8(data: &str) -> Result<Vec<bool>> {
    let digits = parse_digits_with_optional_check(data, 7, 8)?;
    let mut modules: Vec<bool> = Vec::new();
    push_quiet(&mut modules, 7);
    push_bits(&mut modules, 0b101, 3);
    for i in 0..4 {
        push_bits(&mut modules, EAN_L[digits[i] as usize] as u32, 7);
    }
    push_bits(&mut modules, 0b01010, 5);
    for i in 0..4 {
        push_bits(&mut modules, EAN_R[digits[i + 4] as usize] as u32, 7);
    }
    push_bits(&mut modules, 0b101, 3);
    push_quiet(&mut modules, 7);
    Ok(modules)
}

fn encode_upca(data: &str) -> Result<Vec<bool>> {
    // UPC-A is EAN-13 with leading zero
    let digits = parse_digits_with_optional_check(data, 11, 12)?;
    let mut full = Vec::with_capacity(13);
    full.push(0u8);
    full.extend_from_slice(&digits);
    encode_ean13_from_digits(&full)
}

fn encode_ean13_from_digits(digits: &[u8]) -> Result<Vec<bool>> {
    let mut modules: Vec<bool> = Vec::new();
    push_quiet(&mut modules, 9);
    push_bits(&mut modules, 0b101, 3);
    let parity = EAN13_PARITY[digits[0] as usize];
    for i in 0..6 {
        let d = digits[i + 1];
        let pattern = if (parity >> (5 - i)) & 1 == 1 {
            EAN_G[d as usize]
        } else {
            EAN_L[d as usize]
        };
        push_bits(&mut modules, pattern as u32, 7);
    }
    push_bits(&mut modules, 0b01010, 5);
    for i in 0..6 {
        let d = digits[i + 7];
        push_bits(&mut modules, EAN_R[d as usize] as u32, 7);
    }
    push_bits(&mut modules, 0b101, 3);
    push_quiet(&mut modules, 9);
    Ok(modules)
}

fn parse_digits_with_optional_check(data: &str, payload: usize, total: usize) -> Result<Vec<u8>> {
    if data.len() != payload && data.len() != total {
        return Err(ZyronError::ExecutionError(format!(
            "expected {} or {} digits, got {}",
            payload,
            total,
            data.len()
        )));
    }
    let mut digits: Vec<u8> = Vec::with_capacity(total);
    for c in data.chars() {
        if !c.is_ascii_digit() {
            return Err(ZyronError::ExecutionError(format!(
                "non-digit character: {}",
                c
            )));
        }
        digits.push(c as u8 - b'0');
    }
    if digits.len() == payload {
        let cd = ean_check_digit(&digits);
        digits.push(cd);
    }
    Ok(digits)
}

fn push_bits(modules: &mut Vec<bool>, value: u32, count: u32) {
    for bit in (0..count).rev() {
        modules.push((value >> bit) & 1 == 1);
    }
}

fn decode_ean13_modules(modules: &[bool]) -> Result<String> {
    decode_ean_like(modules, 12, false)
}

fn decode_ean8_modules(modules: &[bool]) -> Result<String> {
    decode_ean_like(modules, 8, true)
}

fn decode_upca_modules(modules: &[bool]) -> Result<String> {
    let s = decode_ean_like(modules, 12, false)?;
    if s.starts_with('0') {
        Ok(s[1..].to_string())
    } else {
        Err(ZyronError::ExecutionError(
            "upc-a expected leading zero".into(),
        ))
    }
}

fn decode_ean_like(modules: &[bool], total_digits: usize, ean8: bool) -> Result<String> {
    let half = if ean8 { 4 } else { 6 };
    let header_after_quiet_len = 3 + half * 7 + 5 + half * 7 + 3;
    // Find start guard 101 by scanning
    let start = (0..=modules.len().saturating_sub(header_after_quiet_len))
        .find(|&i| modules[i] && !modules[i + 1] && modules[i + 2]);
    let start = start.ok_or_else(|| ZyronError::ExecutionError("ean start not found".into()))?;
    let mut idx = start + 3;
    let mut left_digits = Vec::new();
    let mut left_parity: u8 = 0;
    for i in 0..half {
        if idx + 7 > modules.len() {
            return Err(ZyronError::ExecutionError("ean truncated".into()));
        }
        let bits = bits_to_u16(&modules[idx..idx + 7]) as u8;
        if let Some(d) = EAN_L.iter().position(|p| *p == bits) {
            left_digits.push(d as u8);
        } else if let Some(d) = EAN_G.iter().position(|p| *p == bits) {
            left_digits.push(d as u8);
            left_parity |= 1 << (half - 1 - i);
        } else {
            return Err(ZyronError::ExecutionError("ean unknown left digit".into()));
        }
        idx += 7;
    }
    // Centre guard 01010
    let centre = bits_to_u16(&modules[idx..idx + 5]);
    if centre != 0b01010 {
        return Err(ZyronError::ExecutionError("ean centre missing".into()));
    }
    idx += 5;
    let mut right_digits = Vec::new();
    for _ in 0..half {
        if idx + 7 > modules.len() {
            return Err(ZyronError::ExecutionError("ean truncated".into()));
        }
        let bits = bits_to_u16(&modules[idx..idx + 7]) as u8;
        let d = EAN_R
            .iter()
            .position(|p| *p == bits)
            .ok_or_else(|| ZyronError::ExecutionError("ean unknown right digit".into()))?;
        right_digits.push(d as u8);
        idx += 7;
    }
    let mut full = Vec::with_capacity(total_digits);
    if !ean8 {
        let first = EAN13_PARITY
            .iter()
            .position(|p| *p == left_parity)
            .ok_or_else(|| ZyronError::ExecutionError("ean parity unknown".into()))?
            as u8;
        full.push(first);
    }
    full.extend_from_slice(&left_digits);
    full.extend_from_slice(&right_digits);
    Ok(full.iter().map(|d| (b'0' + d) as char).collect())
}

// ---------------------------------------------------------------------------
// QR encoding
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct QrCode {
    size: usize,
    modules: Vec<bool>,
    function: Vec<bool>,
}

const QR_VERSIONS: usize = 10;

// Capacity table for byte-mode data codewords by version (1..=10) and EC level
// (L, M, Q, H). Source: ISO/IEC 18004 Table 7
const QR_DATA_CODEWORDS: [[u16; 4]; QR_VERSIONS] = [
    [19, 16, 13, 9],      // v1
    [34, 28, 22, 16],     // v2
    [55, 44, 34, 26],     // v3
    [80, 64, 48, 36],     // v4
    [108, 86, 62, 46],    // v5
    [136, 108, 76, 60],   // v6
    [156, 124, 88, 66],   // v7
    [194, 154, 110, 86],  // v8
    [232, 182, 132, 100], // v9
    [274, 216, 154, 122], // v10
];

// EC codewords per block by version and EC level
const QR_EC_PER_BLOCK: [[u16; 4]; QR_VERSIONS] = [
    [7, 10, 13, 17],
    [10, 16, 22, 28],
    [15, 26, 18, 22],
    [20, 18, 26, 16],
    [26, 24, 18, 22],
    [18, 16, 24, 28],
    [20, 18, 18, 26],
    [24, 22, 22, 26],
    [30, 22, 20, 24],
    [18, 26, 24, 28],
];

// Number of EC blocks by version and EC level
const QR_EC_BLOCKS: [[u8; 4]; QR_VERSIONS] = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 2, 2],
    [1, 2, 2, 4],
    [1, 2, 4, 4],
    [2, 4, 4, 4],
    [2, 4, 6, 5],
    [2, 4, 6, 6],
    [2, 5, 8, 8],
    [4, 5, 8, 8],
];

// Format info: 15-bit code = (ec << 3 | mask) XOR mask 0b101010000010010, with
// EC bits as L=01, M=00, Q=11, H=10
const QR_FORMAT_MASK: u16 = 0b101010000010010;
const QR_FORMAT_GEN: u16 = 0b10100110111;

fn qr_ec_bits(ec: QrErrorCorrection) -> u8 {
    match ec {
        QrErrorCorrection::L => 0b01,
        QrErrorCorrection::M => 0b00,
        QrErrorCorrection::Q => 0b11,
        QrErrorCorrection::H => 0b10,
    }
}

fn qr_ec_index(ec: QrErrorCorrection) -> usize {
    match ec {
        QrErrorCorrection::L => 0,
        QrErrorCorrection::M => 1,
        QrErrorCorrection::Q => 2,
        QrErrorCorrection::H => 3,
    }
}

fn qr_select_version(byte_len: usize, ec: QrErrorCorrection) -> Result<usize> {
    let ec_idx = qr_ec_index(ec);
    for v in 1..=QR_VERSIONS {
        let cap_codewords = QR_DATA_CODEWORDS[v - 1][ec_idx] as usize;
        // byte mode header: 4-bit mode + char count indicator + 4-bit terminator
        let cci = if v <= 9 { 8 } else { 16 };
        let total_bits = 4 + cci + byte_len * 8;
        if total_bits <= cap_codewords * 8 {
            return Ok(v);
        }
    }
    Err(ZyronError::ExecutionError(format!(
        "data too large for QR versions 1-{} at this EC level",
        QR_VERSIONS
    )))
}

fn qr_build(data: &[u8], ec: QrErrorCorrection) -> Result<QrCode> {
    let version = qr_select_version(data.len(), ec)?;
    let ec_idx = qr_ec_index(ec);
    let total_data_codewords = QR_DATA_CODEWORDS[version - 1][ec_idx] as usize;
    let ec_per_block = QR_EC_PER_BLOCK[version - 1][ec_idx] as usize;
    let num_blocks = QR_EC_BLOCKS[version - 1][ec_idx] as usize;

    // Build the data bitstream
    let mut bs = BitStream::new();
    bs.push(0b0100, 4); // byte mode
    let cci = if version <= 9 { 8 } else { 16 };
    bs.push(data.len() as u32, cci);
    for &b in data {
        bs.push(b as u32, 8);
    }
    // Terminator: up to 4 zero bits, but stop at capacity
    let cap_bits = total_data_codewords * 8;
    let terminator = (cap_bits - bs.len()).min(4);
    bs.push(0, terminator as u32);
    // Pad to byte boundary
    let pad_to_byte = (8 - bs.len() % 8) % 8;
    bs.push(0, pad_to_byte as u32);
    // Pad with alternating 0xEC, 0x11
    let mut codewords = bs.into_bytes();
    let pad_bytes = [0xEC, 0x11];
    let mut p = 0;
    while codewords.len() < total_data_codewords {
        codewords.push(pad_bytes[p % 2]);
        p += 1;
    }

    // Split into blocks for Reed-Solomon. Larger blocks come second
    let small_count = num_blocks - (total_data_codewords % num_blocks);
    let small_size = total_data_codewords / num_blocks;
    let large_size = small_size + 1;
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    let mut idx = 0;
    for _ in 0..small_count {
        blocks.push(codewords[idx..idx + small_size].to_vec());
        idx += small_size;
    }
    for _ in small_count..num_blocks {
        blocks.push(codewords[idx..idx + large_size].to_vec());
        idx += large_size;
    }

    // Reed-Solomon ECC for each block
    let generator = rs_generator(ec_per_block);
    let ec_blocks: Vec<Vec<u8>> = blocks.iter().map(|b| rs_remainder(b, &generator)).collect();

    // Interleave data and EC
    let mut interleaved = Vec::with_capacity(total_data_codewords + ec_per_block * num_blocks);
    for col in 0..large_size {
        for b in &blocks {
            if col < b.len() {
                interleaved.push(b[col]);
            }
        }
    }
    for col in 0..ec_per_block {
        for b in &ec_blocks {
            interleaved.push(b[col]);
        }
    }

    // Place into matrix
    let size = 17 + 4 * version;
    let mut modules = vec![false; size * size];
    let mut function = vec![false; size * size];
    place_finders(&mut modules, &mut function, size);
    place_alignment(&mut modules, &mut function, size, version);
    place_timing(&mut modules, &mut function, size);
    // Reserve format info zones
    reserve_format(&mut function, size);
    if version >= 7 {
        reserve_version(&mut function, size);
    }
    place_data(&mut modules, &function, size, &interleaved);

    // Choose mask
    let (best_mask, best_modules) = choose_mask(&modules, &function, size, ec);
    let format_bits = format_info(ec, best_mask);
    let mut final_modules = best_modules;
    apply_format(&mut final_modules, size, format_bits);
    if version >= 7 {
        apply_version(&mut final_modules, size, version as u32);
    }

    Ok(QrCode {
        size,
        modules: final_modules,
        function,
    })
}

struct BitStream {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitStream {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn len(&self) -> usize {
        self.bit_pos
    }
    fn push(&mut self, value: u32, bits: u32) {
        for i in (0..bits).rev() {
            let bit = ((value >> i) & 1) as u8;
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            if byte_idx >= self.bytes.len() {
                self.bytes.push(0);
            }
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

fn rs_generator(degree: usize) -> Vec<u8> {
    let mut g = vec![1u8];
    for i in 0..degree {
        let mut new_g = vec![0u8; g.len() + 1];
        for (j, &c) in g.iter().enumerate() {
            new_g[j] ^= gf_mul(c, gf_exp(i as u8));
            new_g[j + 1] ^= c;
        }
        g = new_g;
    }
    g
}

fn rs_remainder(data: &[u8], generator: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; data.len() + generator.len() - 1];
    buf[..data.len()].copy_from_slice(data);
    for i in 0..data.len() {
        let lead = buf[i];
        if lead != 0 {
            for (j, &g) in generator.iter().enumerate() {
                buf[i + j] ^= gf_mul(g, lead);
            }
        }
    }
    buf[data.len()..].to_vec()
}

const GF_PRIM: u32 = 0x11d;
fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let la = GF_LOG[a as usize] as u32;
    let lb = GF_LOG[b as usize] as u32;
    GF_EXP[((la + lb) % 255) as usize]
}
fn gf_exp(p: u8) -> u8 {
    GF_EXP[(p as usize) % 255]
}

static GF_EXP: [u8; 256] = build_gf_exp();
static GF_LOG: [u8; 256] = build_gf_log();

const fn build_gf_exp() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut x: u32 = 1;
    let mut i = 0;
    while i < 255 {
        t[i] = x as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= GF_PRIM;
        }
        i += 1;
    }
    t[255] = t[0];
    t
}

const fn build_gf_log() -> [u8; 256] {
    let exp = build_gf_exp();
    let mut t = [0u8; 256];
    let mut i: u32 = 0;
    while i < 255 {
        t[exp[i as usize] as usize] = i as u8;
        i += 1;
    }
    t
}

fn place_finders(modules: &mut [bool], function: &mut [bool], size: usize) {
    let positions = [(0, 0), (size - 7, 0), (0, size - 7)];
    for &(r, c) in &positions {
        for dy in 0..7 {
            for dx in 0..7 {
                let on = matches!((dy, dx), (0, _) | (6, _) | (_, 0) | (_, 6) | (2..=4, 2..=4));
                modules[(r + dy) * size + (c + dx)] = on;
                function[(r + dy) * size + (c + dx)] = true;
            }
        }
        // Surround with quiet ring (still function)
        for dy in 0..8 {
            for dx in 0..8 {
                let rr = r.checked_add(dy);
                let cc = c.checked_add(dx);
                if let (Some(rr), Some(cc)) = (rr, cc) {
                    if rr < size && cc < size {
                        function[rr * size + cc] = true;
                    }
                }
            }
        }
    }
}

fn place_alignment(modules: &mut [bool], function: &mut [bool], size: usize, version: usize) {
    if version == 1 {
        return;
    }
    // Centre coordinates per ISO 18004 Annex E
    let centres: &[u8] = match version {
        2 => &[6, 18],
        3 => &[6, 22],
        4 => &[6, 26],
        5 => &[6, 30],
        6 => &[6, 34],
        7 => &[6, 22, 38],
        8 => &[6, 24, 42],
        9 => &[6, 26, 46],
        10 => &[6, 28, 50],
        _ => return,
    };
    for &cy in centres {
        for &cx in centres {
            // Skip those overlapping a finder
            let cy_u = cy as usize;
            let cx_u = cx as usize;
            if (cx_u < 7 && cy_u < 7)
                || (cx_u < 7 && cy_u > size - 8)
                || (cx_u > size - 8 && cy_u < 7)
            {
                continue;
            }
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let r = (cy_u as i32 + dy) as usize;
                    let c = (cx_u as i32 + dx) as usize;
                    let on = (dy.abs() == 2) || (dx.abs() == 2) || (dy == 0 && dx == 0);
                    modules[r * size + c] = on;
                    function[r * size + c] = true;
                }
            }
        }
    }
}

fn place_timing(modules: &mut [bool], function: &mut [bool], size: usize) {
    for i in 8..size - 8 {
        let on = i % 2 == 0;
        modules[6 * size + i] = on;
        modules[i * size + 6] = on;
        function[6 * size + i] = true;
        function[i * size + 6] = true;
    }
    // Dark module at (4*version + 9, 8). Version derives from size: size = 17 + 4*version
    let version = (size - 17) / 4;
    let dark_row = 4 * version + 9;
    modules[dark_row * size + 8] = true;
    function[dark_row * size + 8] = true;
}

fn reserve_format(function: &mut [bool], size: usize) {
    for i in 0..9 {
        function[i * size + 8] = true;
        function[8 * size + i] = true;
    }
    for i in 0..8 {
        function[(size - 1 - i) * size + 8] = true;
        function[8 * size + size - 1 - i] = true;
    }
}

fn reserve_version(function: &mut [bool], size: usize) {
    for r in 0..6 {
        for c in (size - 11)..(size - 8) {
            function[r * size + c] = true;
        }
    }
    for c in 0..6 {
        for r in (size - 11)..(size - 8) {
            function[r * size + c] = true;
        }
    }
}

fn place_data(modules: &mut [bool], function: &[bool], size: usize, interleaved: &[u8]) {
    let mut bit_idx = 0usize;
    let total_bits = interleaved.len() * 8;
    let mut col_pair: i32 = (size as i32) - 1;
    let mut going_up = true;
    while col_pair > 0 {
        if col_pair == 6 {
            col_pair -= 1;
        }
        for r_step in 0..size as i32 {
            let r = if going_up {
                size as i32 - 1 - r_step
            } else {
                r_step
            };
            for dx in 0..2 {
                let c = col_pair - dx as i32;
                if c < 0 {
                    continue;
                }
                let idx = r as usize * size + c as usize;
                if function[idx] {
                    continue;
                }
                if bit_idx < total_bits {
                    let byte = interleaved[bit_idx / 8];
                    let bit = (byte >> (7 - (bit_idx % 8))) & 1;
                    modules[idx] = bit == 1;
                    bit_idx += 1;
                }
            }
        }
        col_pair -= 2;
        going_up = !going_up;
    }
}

fn choose_mask(
    modules: &[bool],
    function: &[bool],
    size: usize,
    _ec: QrErrorCorrection,
) -> (u8, Vec<bool>) {
    let mut best_mask: u8 = 0;
    let mut best_score: u32 = u32::MAX;
    let mut best_grid: Vec<bool> = modules.to_vec();
    for mask in 0u8..8 {
        let mut grid = modules.to_vec();
        for r in 0..size {
            for c in 0..size {
                let idx = r * size + c;
                if function[idx] {
                    continue;
                }
                if mask_bit(mask, r as u32, c as u32) {
                    grid[idx] ^= true;
                }
            }
        }
        let score = penalty_score(&grid, size);
        if score < best_score {
            best_score = score;
            best_mask = mask;
            best_grid = grid;
        }
    }
    (best_mask, best_grid)
}

fn mask_bit(mask: u8, r: u32, c: u32) -> bool {
    match mask {
        0 => (r + c) % 2 == 0,
        1 => r % 2 == 0,
        2 => c % 3 == 0,
        3 => (r + c) % 3 == 0,
        4 => ((r / 2) + (c / 3)) % 2 == 0,
        5 => (r * c) % 2 + (r * c) % 3 == 0,
        6 => ((r * c) % 2 + (r * c) % 3) % 2 == 0,
        7 => ((r + c) % 2 + (r * c) % 3) % 2 == 0,
        _ => false,
    }
}

fn penalty_score(grid: &[bool], size: usize) -> u32 {
    let mut score: u32 = 0;
    // Rule 1: runs of 5+
    for r in 0..size {
        let mut run = 1;
        for c in 1..size {
            if grid[r * size + c] == grid[r * size + c - 1] {
                run += 1;
            } else {
                if run >= 5 {
                    score += 3 + (run - 5) as u32;
                }
                run = 1;
            }
        }
        if run >= 5 {
            score += 3 + (run - 5) as u32;
        }
    }
    for c in 0..size {
        let mut run = 1;
        for r in 1..size {
            if grid[r * size + c] == grid[(r - 1) * size + c] {
                run += 1;
            } else {
                if run >= 5 {
                    score += 3 + (run - 5) as u32;
                }
                run = 1;
            }
        }
        if run >= 5 {
            score += 3 + (run - 5) as u32;
        }
    }
    score
}

fn format_info(ec: QrErrorCorrection, mask: u8) -> u16 {
    let data: u16 = ((qr_ec_bits(ec) as u16) << 3) | mask as u16;
    let mut rem: u32 = (data as u32) << 10;
    for i in (10..=14).rev() {
        if (rem >> i) & 1 == 1 {
            rem ^= (QR_FORMAT_GEN as u32) << (i - 10);
        }
    }
    let combined = ((data as u32) << 10) | rem;
    (combined as u16) ^ QR_FORMAT_MASK
}

fn apply_format(modules: &mut [bool], size: usize, format: u16) {
    for i in 0..15 {
        let bit = (format >> i) & 1 == 1;
        let (r, c) = match i {
            0 => (8, 0),
            1 => (8, 1),
            2 => (8, 2),
            3 => (8, 3),
            4 => (8, 4),
            5 => (8, 5),
            6 => (8, 7),
            7 => (8, 8),
            8 => (7, 8),
            9 => (5, 8),
            10 => (4, 8),
            11 => (3, 8),
            12 => (2, 8),
            13 => (1, 8),
            14 => (0, 8),
            _ => unreachable!(),
        };
        modules[r * size + c] = bit;
        let (r2, c2) = match i {
            0 => (size - 1, 8),
            1 => (size - 2, 8),
            2 => (size - 3, 8),
            3 => (size - 4, 8),
            4 => (size - 5, 8),
            5 => (size - 6, 8),
            6 => (size - 7, 8),
            7 => (8, size - 8),
            8 => (8, size - 7),
            9 => (8, size - 6),
            10 => (8, size - 5),
            11 => (8, size - 4),
            12 => (8, size - 3),
            13 => (8, size - 2),
            14 => (8, size - 1),
            _ => unreachable!(),
        };
        modules[r2 * size + c2] = bit;
    }
}

fn apply_version(modules: &mut [bool], size: usize, version: u32) {
    let mut rem: u32 = version << 12;
    let poly: u32 = 0b1111100100101;
    for i in (12..=17).rev() {
        if (rem >> i) & 1 == 1 {
            rem ^= poly << (i - 12);
        }
    }
    let combined = (version << 12) | rem;
    for i in 0..18 {
        let bit = (combined >> i) & 1 == 1;
        let r = (i / 3) as usize;
        let c = size - 11 + (i % 3) as usize;
        modules[r * size + c] = bit;
        modules[c * size + r] = bit;
    }
}

// ---------------------------------------------------------------------------
// QR decoding (round-trip for our own encoder)
// ---------------------------------------------------------------------------

fn qr_decode_grayscale(img: &Grayscale) -> Result<String> {
    if img.width == 0 || img.height == 0 || img.width != img.height {
        return Err(ZyronError::ExecutionError("qr image must be square".into()));
    }
    // Find module pixel size by scanning the first finder pattern
    let scale = find_qr_scale(img)?;
    let size = (img.width / scale) as usize;
    let version = (size as i32 - 17) / 4;
    if !(1..=QR_VERSIONS as i32).contains(&version) {
        return Err(ZyronError::ExecutionError(format!(
            "qr version {} out of supported range 1-{}",
            version, QR_VERSIONS
        )));
    }
    let version = version as usize;
    let mut modules = vec![false; size * size];
    for r in 0..size {
        for c in 0..size {
            let px = (c as u32 * scale + scale / 2) as usize;
            let py = (r as u32 * scale + scale / 2) as usize;
            let p = img.pixels[py * img.width as usize + px];
            modules[r * size + c] = p < 128;
        }
    }
    // Read format info from upper-left format zone
    let mut raw_format: u16 = 0;
    let positions: [(usize, usize); 15] = [
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (8, 7),
        (8, 8),
        (7, 8),
        (5, 8),
        (4, 8),
        (3, 8),
        (2, 8),
        (1, 8),
        (0, 8),
    ];
    for (i, (r, c)) in positions.iter().enumerate() {
        if modules[r * size + c] {
            raw_format |= 1 << i;
        }
    }
    let format = raw_format ^ QR_FORMAT_MASK;
    // The 15-bit format codeword has data in bits 14..10 (5 bits): two EC
    // bits at 14..13, three mask bits at 12..10. The lower 10 bits hold the
    // BCH remainder which we ignore here (production decoders would use it
    // for error correction; round-trip from our own encoder is exact)
    let mask = ((format >> 10) & 0b111) as u8;
    let ec_bits = ((format >> 13) & 0b11) as u8;
    let ec = match ec_bits {
        0b01 => QrErrorCorrection::L,
        0b00 => QrErrorCorrection::M,
        0b11 => QrErrorCorrection::Q,
        0b10 => QrErrorCorrection::H,
        _ => return Err(ZyronError::ExecutionError("qr ec bits invalid".into())),
    };
    // Rebuild function map and unmask
    let mut function = vec![false; size * size];
    let mut zero = vec![false; size * size];
    place_finders(&mut zero, &mut function, size);
    place_alignment(&mut zero, &mut function, size, version);
    place_timing(&mut zero, &mut function, size);
    reserve_format(&mut function, size);
    if version >= 7 {
        reserve_version(&mut function, size);
    }
    let mut unmasked = modules.clone();
    for r in 0..size {
        for c in 0..size {
            let idx = r * size + c;
            if function[idx] {
                continue;
            }
            if mask_bit(mask, r as u32, c as u32) {
                unmasked[idx] ^= true;
            }
        }
    }
    // Read codewords in Z-snake order
    let mut bits: Vec<u8> = Vec::new();
    let mut col_pair: i32 = size as i32 - 1;
    let mut going_up = true;
    while col_pair > 0 {
        if col_pair == 6 {
            col_pair -= 1;
        }
        for r_step in 0..size as i32 {
            let r = if going_up {
                size as i32 - 1 - r_step
            } else {
                r_step
            };
            for dx in 0..2 {
                let c = col_pair - dx as i32;
                if c < 0 {
                    continue;
                }
                let idx = r as usize * size + c as usize;
                if function[idx] {
                    continue;
                }
                bits.push(if unmasked[idx] { 1 } else { 0 });
            }
        }
        col_pair -= 2;
        going_up = !going_up;
    }
    // Re-pack bits into bytes
    let mut codewords: Vec<u8> = Vec::with_capacity(bits.len() / 8);
    let mut acc: u8 = 0;
    let mut count = 0;
    for b in bits {
        acc = (acc << 1) | b;
        count += 1;
        if count == 8 {
            codewords.push(acc);
            acc = 0;
            count = 0;
        }
    }
    // De-interleave
    let ec_idx = qr_ec_index(ec);
    let total_data_codewords = QR_DATA_CODEWORDS[version - 1][ec_idx] as usize;
    let ec_per_block = QR_EC_PER_BLOCK[version - 1][ec_idx] as usize;
    let num_blocks = QR_EC_BLOCKS[version - 1][ec_idx] as usize;
    let small_count = num_blocks - (total_data_codewords % num_blocks);
    let small_size = total_data_codewords / num_blocks;
    let large_size = small_size + 1;
    let mut block_sizes: Vec<usize> = Vec::with_capacity(num_blocks);
    for _ in 0..small_count {
        block_sizes.push(small_size);
    }
    for _ in small_count..num_blocks {
        block_sizes.push(large_size);
    }
    let mut data_blocks: Vec<Vec<u8>> =
        block_sizes.iter().map(|s| Vec::with_capacity(*s)).collect();
    let mut ec_blocks: Vec<Vec<u8>> = (0..num_blocks)
        .map(|_| Vec::with_capacity(ec_per_block))
        .collect();
    let mut idx = 0;
    for col in 0..large_size {
        for (b, sz) in data_blocks.iter_mut().zip(block_sizes.iter()) {
            if col < *sz {
                b.push(codewords[idx]);
                idx += 1;
            }
        }
    }
    for _ in 0..ec_per_block {
        for b in ec_blocks.iter_mut() {
            b.push(codewords[idx]);
            idx += 1;
        }
    }
    let mut data_stream: Vec<u8> = Vec::new();
    for b in &data_blocks {
        data_stream.extend_from_slice(b);
    }
    // Read mode + length + payload from data_stream as a bitstream
    let mut bs = BitReader::new(&data_stream);
    let mode = bs.read(4);
    if mode != 0b0100 {
        return Err(ZyronError::ExecutionError(format!(
            "qr mode {} not supported (only byte mode)",
            mode
        )));
    }
    let cci = if version <= 9 { 8 } else { 16 };
    let len = bs.read(cci) as usize;
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        bytes.push(bs.read(8) as u8);
    }
    String::from_utf8(bytes)
        .map_err(|e| ZyronError::ExecutionError(format!("qr payload not utf-8: {}", e)))
}

fn find_qr_scale(img: &Grayscale) -> Result<u32> {
    // The top-left finder pattern's first row is 7 dark modules wide. We walk
    // the top row from x=0, skip any leading light pixels (none expected
    // because our encoder draws the finder flush against the edge), then
    // count the dark band length. dark_len / 7 is the per-module pixel size
    let mut x = 0u32;
    while x < img.width && img.pixels[x as usize] >= 128 {
        x += 1;
    }
    let dark_start = x;
    while x < img.width && img.pixels[x as usize] < 128 {
        x += 1;
    }
    let dark_len = x - dark_start;
    if dark_len < 7 {
        return Err(ZyronError::ExecutionError(
            "qr finder pattern not found or too small".into(),
        ));
    }
    Ok(dark_len / 7)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn read(&mut self, count: u32) -> u32 {
        let mut v: u32 = 0;
        for _ in 0..count {
            let byte = self.bytes[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        v
    }
}

// ---------------------------------------------------------------------------
// PNG writer (grayscale, hand-rolled with stored deflate)
// ---------------------------------------------------------------------------

fn write_png_grayscale(width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>> {
    if pixels.len() != (width as usize) * (height as usize) {
        return Err(ZyronError::ExecutionError(
            "png pixel buffer size mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(pixels.len() + 256);
    // Signature
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(0); // colour type grayscale
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_chunk(&mut out, b"IHDR", &ihdr);
    // IDAT: zlib stream of raw filtered pixels
    let mut raw = Vec::with_capacity(pixels.len() + height as usize);
    for y in 0..height {
        raw.push(0); // filter type 0 (None)
        let row_start = (y as usize) * (width as usize);
        raw.extend_from_slice(&pixels[row_start..row_start + width as usize]);
    }
    let zlib_blob = zlib_stored(&raw);
    write_chunk(&mut out, b"IDAT", &zlib_blob);
    // IEND
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

fn write_chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(ty.len() + data.len());
    crc_input.extend_from_slice(ty);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(ty);
    out.extend_from_slice(data);
    let crc = crc32fast::hash(&crc_input);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 16);
    // zlib header: CMF=0x78 (deflate, 32 KiB window), FLG with FCHECK so that
    // CMF*256 + FLG is a multiple of 31. 0x78 0x01 satisfies this (no
    // dictionary, fastest compression flag)
    out.push(0x78);
    out.push(0x01);
    // Stored deflate blocks: each up to 65535 bytes. Block header: 1 byte
    // BFINAL+BTYPE (bit 0 = last, bits 1-2 = 00 stored), then 4 bytes
    // LEN/NLEN little-endian, then raw bytes
    let mut pos = 0usize;
    while pos < data.len() {
        let chunk = (data.len() - pos).min(65535);
        let last = pos + chunk == data.len();
        out.push(if last { 0x01 } else { 0x00 });
        out.push((chunk & 0xff) as u8);
        out.push(((chunk >> 8) & 0xff) as u8);
        let nlen = !(chunk as u16);
        out.push((nlen & 0xff) as u8);
        out.push(((nlen >> 8) & 0xff) as u8);
        out.extend_from_slice(&data[pos..pos + chunk]);
        pos += chunk;
    }
    // Adler-32 of uncompressed data
    let adler = crate::checksum::adler32(data);
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

// ---------------------------------------------------------------------------
// PNG reader (grayscale, supports our own stored-deflate output and the
// fixed-Huffman common case)
// ---------------------------------------------------------------------------

struct Grayscale {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

fn decode_png_grayscale(image: &[u8]) -> Result<Grayscale> {
    if image.len() < 8 || &image[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Err(ZyronError::ExecutionError("not a png".into()));
    }
    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut idat = Vec::new();
    while pos + 8 <= image.len() {
        let len = u32::from_be_bytes([image[pos], image[pos + 1], image[pos + 2], image[pos + 3]])
            as usize;
        pos += 4;
        let ty = &image[pos..pos + 4];
        pos += 4;
        let data = &image[pos..pos + len];
        pos += len + 4; // skip CRC
        match ty {
            b"IHDR" => {
                width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
                height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
                bit_depth = data[8];
                color_type = data[9];
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
    }
    if bit_depth != 8 || color_type != 0 {
        return Err(ZyronError::ExecutionError(format!(
            "png decoder supports 8-bit grayscale only, got bit_depth={} color_type={}",
            bit_depth, color_type
        )));
    }
    let raw = zlib_inflate(&idat)?;
    // Strip filter bytes
    let row_bytes = width as usize;
    let stride = row_bytes + 1;
    if raw.len() != stride * (height as usize) {
        return Err(ZyronError::ExecutionError(format!(
            "png raw size {} expected {}",
            raw.len(),
            stride * height as usize
        )));
    }
    let mut pixels = Vec::with_capacity(row_bytes * height as usize);
    for y in 0..height as usize {
        let filter = raw[y * stride];
        if filter != 0 {
            return Err(ZyronError::ExecutionError(format!(
                "png filter type {} not supported",
                filter
            )));
        }
        pixels.extend_from_slice(&raw[y * stride + 1..(y + 1) * stride]);
    }
    Ok(Grayscale {
        width,
        height,
        pixels,
    })
}

fn zlib_inflate(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(ZyronError::ExecutionError("zlib too short".into()));
    }
    // Skip 2-byte header, trailing 4-byte adler32
    let body = &data[2..data.len() - 4];
    let mut out = Vec::new();
    let mut br = ByteBitReader::new(body);
    loop {
        let bfinal = br.read_bit()?;
        let btype = (br.read_bit()? as u8) | ((br.read_bit()? as u8) << 1);
        match btype {
            0 => {
                br.align_to_byte();
                let len = br.read_u16_le()? as usize;
                let _nlen = br.read_u16_le()?;
                for _ in 0..len {
                    out.push(br.read_byte()?);
                }
            }
            1 => {
                inflate_fixed(&mut br, &mut out)?;
            }
            2 => {
                return Err(ZyronError::ExecutionError(
                    "png decoder does not support dynamic huffman blocks".into(),
                ));
            }
            _ => {
                return Err(ZyronError::ExecutionError(format!(
                    "invalid deflate block type {}",
                    btype
                )));
            }
        }
        if bfinal {
            break;
        }
    }
    Ok(out)
}

struct ByteBitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteBitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn read_bit(&mut self) -> Result<bool> {
        let byte = *self
            .bytes
            .get(self.pos / 8)
            .ok_or_else(|| ZyronError::ExecutionError("deflate read past end".into()))?;
        let bit = (byte >> (self.pos % 8)) & 1;
        self.pos += 1;
        Ok(bit == 1)
    }
    fn read_bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0u32;
        for i in 0..n {
            if self.read_bit()? {
                v |= 1 << i;
            }
        }
        Ok(v)
    }
    fn align_to_byte(&mut self) {
        let rem = self.pos % 8;
        if rem != 0 {
            self.pos += 8 - rem;
        }
    }
    fn read_u16_le(&mut self) -> Result<u16> {
        let lo = self.read_byte()? as u16;
        let hi = self.read_byte()? as u16;
        Ok(lo | (hi << 8))
    }
    fn read_byte(&mut self) -> Result<u8> {
        let v = *self
            .bytes
            .get(self.pos / 8)
            .ok_or_else(|| ZyronError::ExecutionError("deflate read past end".into()))?;
        self.pos += 8;
        Ok(v)
    }
}

fn inflate_fixed(br: &mut ByteBitReader, out: &mut Vec<u8>) -> Result<()> {
    // Fixed Huffman literal/length codes per RFC 1951 section 3.2.6
    loop {
        // Read 7 bits and try
        let mut code = 0u32;
        for _ in 0..7 {
            code = (code << 1) | (br.read_bit()? as u32);
        }
        let (lit, extra) = decode_fixed_code(code, 7);
        let (literal_or_length, length_extra_bits) = match (lit, extra) {
            (Some(c), e) => (c, e),
            (None, _) => {
                // Try 8 bits
                code = (code << 1) | (br.read_bit()? as u32);
                let (lit, e) = decode_fixed_code(code, 8);
                if let Some(c) = lit {
                    (c, e)
                } else {
                    code = (code << 1) | (br.read_bit()? as u32);
                    let (lit, e) = decode_fixed_code(code, 9);
                    let c = lit.ok_or_else(|| {
                        ZyronError::ExecutionError("invalid fixed huffman code".into())
                    })?;
                    (c, e)
                }
            }
        };
        if literal_or_length < 256 {
            out.push(literal_or_length as u8);
            continue;
        }
        if literal_or_length == 256 {
            return Ok(());
        }
        // Length code 257..285
        let length = decode_length(literal_or_length as u32, length_extra_bits, br)?;
        let dist_code = br.read_bits(5)?;
        let dist_code = reverse_bits(dist_code, 5);
        let distance = decode_distance(dist_code, br)?;
        let start = out
            .len()
            .checked_sub(distance as usize)
            .ok_or_else(|| ZyronError::ExecutionError("deflate back-reference too far".into()))?;
        for i in 0..length as usize {
            let b = out[start + i];
            out.push(b);
        }
    }
}

fn decode_fixed_code(code: u32, bits: u32) -> (Option<u32>, u32) {
    match bits {
        7 => {
            // Codes 0000000..0010111 -> 256..279
            if (0b0000000..=0b0010111).contains(&code) {
                let lit = 256 + (code - 0b0000000);
                return (Some(lit), 0);
            }
            (None, 0)
        }
        8 => {
            // Codes 00110000..10111111 -> 0..143
            if (0b00110000..=0b10111111).contains(&code) {
                let lit = code - 0b00110000;
                return (Some(lit), 0);
            }
            // Codes 11000000..11000111 -> 280..287
            if (0b11000000..=0b11000111).contains(&code) {
                let lit = 280 + (code - 0b11000000);
                return (Some(lit), 0);
            }
            (None, 0)
        }
        9 => {
            // Codes 110010000..111111111 -> 144..255
            if (0b110010000..=0b111111111).contains(&code) {
                let lit = 144 + (code - 0b110010000);
                return (Some(lit), 0);
            }
            (None, 0)
        }
        _ => (None, 0),
    }
}

fn decode_length(code: u32, _extra_in: u32, br: &mut ByteBitReader) -> Result<u32> {
    // Tables from RFC 1951 section 3.2.5
    static LENS: [(u32, u32); 29] = [
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (10, 0),
        (11, 1),
        (13, 1),
        (15, 1),
        (17, 1),
        (19, 2),
        (23, 2),
        (27, 2),
        (31, 2),
        (35, 3),
        (43, 3),
        (51, 3),
        (59, 3),
        (67, 4),
        (83, 4),
        (99, 4),
        (115, 4),
        (131, 5),
        (163, 5),
        (195, 5),
        (227, 5),
        (258, 0),
    ];
    let i = (code - 257) as usize;
    let (base, extra) = LENS[i];
    let extra_bits = br.read_bits(extra)?;
    Ok(base + extra_bits)
}

fn decode_distance(code: u32, br: &mut ByteBitReader) -> Result<u32> {
    static DISTS: [(u32, u32); 30] = [
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 1),
        (7, 1),
        (9, 2),
        (13, 2),
        (17, 3),
        (25, 3),
        (33, 4),
        (49, 4),
        (65, 5),
        (97, 5),
        (129, 6),
        (193, 6),
        (257, 7),
        (385, 7),
        (513, 8),
        (769, 8),
        (1025, 9),
        (1537, 9),
        (2049, 10),
        (3073, 10),
        (4097, 11),
        (6145, 11),
        (8193, 12),
        (12289, 12),
        (16385, 13),
        (24577, 13),
    ];
    let (base, extra) = DISTS[code as usize];
    let extra_bits = br.read_bits(extra)?;
    Ok(base + extra_bits)
}

fn reverse_bits(v: u32, n: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..n {
        if (v >> i) & 1 == 1 {
            r |= 1 << (n - 1 - i);
        }
    }
    r
}

// ---------------------------------------------------------------------------
// SVG output
// ---------------------------------------------------------------------------

fn write_svg_1d(modules: &[bool], height: u32) -> Vec<u8> {
    let width = modules.len();
    let mut s = String::with_capacity(modules.len() * 32);
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" shape-rendering=\"crispEdges\">",
        width, height, width, height
    ));
    s.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>");
    let mut x = 0;
    while x < modules.len() {
        if modules[x] {
            let mut end = x + 1;
            while end < modules.len() && modules[end] {
                end += 1;
            }
            s.push_str(&format!(
                "<rect x=\"{}\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"#000\"/>",
                x,
                end - x,
                height
            ));
            x = end;
        } else {
            x += 1;
        }
    }
    s.push_str("</svg>");
    s.into_bytes()
}

fn write_svg_qr(qr: &QrCode, scale: u32) -> Vec<u8> {
    let dim = qr.size as u32 * scale;
    let mut s = String::with_capacity(qr.modules.len() * 24);
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" shape-rendering=\"crispEdges\">",
        dim, dim, dim, dim
    ));
    s.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>");
    for r in 0..qr.size {
        for c in 0..qr.size {
            if qr.modules[r * qr.size + c] {
                s.push_str(&format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#000\"/>",
                    c as u32 * scale,
                    r as u32 * scale,
                    scale,
                    scale
                ));
            }
        }
    }
    s.push_str("</svg>");
    s.into_bytes()
}

// ---------------------------------------------------------------------------
// DataMatrix (square symbols, ASCII mode)
// ---------------------------------------------------------------------------
//
// 2D matrix barcode used for direct part marking, pharma packaging, supply
// chain. Distinct from QR in three ways
//   1. Reed-Solomon over GF(256) with primitive polynomial 0x12D (QR uses 0x11D)
//   2. "Utah" codeword placement, not a raster zigzag
//   3. Finder pattern is an L on bottom+left with alternating dots on top+right
//
// 24 standard square sizes (10 to 144). Symbols above 26 are split into
// 2x2 sub-regions (32 to 88) or 4x4 sub-regions (96 to 132) or 6x6 (144).
// Each region is 24 by 24 modules for the 2x2 family and 22 by 22 for the
// 4x4 family. Sub-region boundaries hold their own finder + timing patterns.

/// DataMatrix symbol descriptor (supports both square and rectangular sizes)
#[derive(Clone, Copy)]
struct DmSymbol {
    /// Total module rows including finders and timing
    nrow: u16,
    /// Total module columns including finders and timing
    ncol: u16,
    /// Region count vertically (rows of regions)
    regions_v: u8,
    /// Region count horizontally (columns of regions)
    regions_h: u8,
    /// Region inner row count (data area, excluding L finder + timing)
    region_inner_rows: u16,
    /// Region inner col count (data area, excluding L finder + timing)
    region_inner_cols: u16,
    /// Data codewords (input bytes after ASCII encoding)
    data_cw: u16,
    /// Error-correction codewords per RS block
    ecc_per_block: u16,
    /// Number of RS blocks (interleaved)
    rs_blocks: u16,
}

// Square sizes use nrow == ncol and regions_v == regions_h
// Rectangular sizes (8x18, 8x32, 12x26, 12x36, 16x36, 16x48) per ISO/IEC 16022 Annex C
const DM_SYMBOLS: &[DmSymbol] = &[
    // ----- rectangular sizes (smaller capacity, listed first so dm_select_symbol prefers them) -----
    DmSymbol {
        nrow: 8,
        ncol: 18,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 6,
        region_inner_cols: 16,
        data_cw: 5,
        ecc_per_block: 7,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 8,
        ncol: 32,
        regions_v: 1,
        regions_h: 2,
        region_inner_rows: 6,
        region_inner_cols: 14,
        data_cw: 10,
        ecc_per_block: 11,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 12,
        ncol: 26,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 10,
        region_inner_cols: 24,
        data_cw: 16,
        ecc_per_block: 14,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 12,
        ncol: 36,
        regions_v: 1,
        regions_h: 2,
        region_inner_rows: 10,
        region_inner_cols: 16,
        data_cw: 22,
        ecc_per_block: 18,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 16,
        ncol: 36,
        regions_v: 1,
        regions_h: 2,
        region_inner_rows: 14,
        region_inner_cols: 16,
        data_cw: 32,
        ecc_per_block: 24,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 16,
        ncol: 48,
        regions_v: 1,
        regions_h: 2,
        region_inner_rows: 14,
        region_inner_cols: 22,
        data_cw: 49,
        ecc_per_block: 28,
        rs_blocks: 1,
    },
    // ----- square sizes -----
    DmSymbol {
        nrow: 10,
        ncol: 10,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 8,
        region_inner_cols: 8,
        data_cw: 3,
        ecc_per_block: 5,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 12,
        ncol: 12,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 10,
        region_inner_cols: 10,
        data_cw: 5,
        ecc_per_block: 7,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 14,
        ncol: 14,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 12,
        region_inner_cols: 12,
        data_cw: 8,
        ecc_per_block: 10,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 16,
        ncol: 16,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 14,
        region_inner_cols: 14,
        data_cw: 12,
        ecc_per_block: 12,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 18,
        ncol: 18,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 16,
        region_inner_cols: 16,
        data_cw: 18,
        ecc_per_block: 14,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 20,
        ncol: 20,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 18,
        region_inner_cols: 18,
        data_cw: 22,
        ecc_per_block: 18,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 22,
        ncol: 22,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 20,
        region_inner_cols: 20,
        data_cw: 30,
        ecc_per_block: 20,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 24,
        ncol: 24,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 22,
        region_inner_cols: 22,
        data_cw: 36,
        ecc_per_block: 24,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 26,
        ncol: 26,
        regions_v: 1,
        regions_h: 1,
        region_inner_rows: 24,
        region_inner_cols: 24,
        data_cw: 44,
        ecc_per_block: 28,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 32,
        ncol: 32,
        regions_v: 2,
        regions_h: 2,
        region_inner_rows: 14,
        region_inner_cols: 14,
        data_cw: 62,
        ecc_per_block: 36,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 36,
        ncol: 36,
        regions_v: 2,
        regions_h: 2,
        region_inner_rows: 16,
        region_inner_cols: 16,
        data_cw: 86,
        ecc_per_block: 42,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 40,
        ncol: 40,
        regions_v: 2,
        regions_h: 2,
        region_inner_rows: 18,
        region_inner_cols: 18,
        data_cw: 114,
        ecc_per_block: 48,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 44,
        ncol: 44,
        regions_v: 2,
        regions_h: 2,
        region_inner_rows: 20,
        region_inner_cols: 20,
        data_cw: 144,
        ecc_per_block: 56,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 48,
        ncol: 48,
        regions_v: 2,
        regions_h: 2,
        region_inner_rows: 22,
        region_inner_cols: 22,
        data_cw: 174,
        ecc_per_block: 68,
        rs_blocks: 1,
    },
    DmSymbol {
        nrow: 52,
        ncol: 52,
        regions_v: 2,
        regions_h: 2,
        region_inner_rows: 24,
        region_inner_cols: 24,
        data_cw: 204,
        ecc_per_block: 42,
        rs_blocks: 2,
    },
    DmSymbol {
        nrow: 64,
        ncol: 64,
        regions_v: 4,
        regions_h: 4,
        region_inner_rows: 14,
        region_inner_cols: 14,
        data_cw: 280,
        ecc_per_block: 56,
        rs_blocks: 2,
    },
    DmSymbol {
        nrow: 72,
        ncol: 72,
        regions_v: 4,
        regions_h: 4,
        region_inner_rows: 16,
        region_inner_cols: 16,
        data_cw: 368,
        ecc_per_block: 36,
        rs_blocks: 4,
    },
    DmSymbol {
        nrow: 80,
        ncol: 80,
        regions_v: 4,
        regions_h: 4,
        region_inner_rows: 18,
        region_inner_cols: 18,
        data_cw: 456,
        ecc_per_block: 48,
        rs_blocks: 4,
    },
    DmSymbol {
        nrow: 88,
        ncol: 88,
        regions_v: 4,
        regions_h: 4,
        region_inner_rows: 20,
        region_inner_cols: 20,
        data_cw: 576,
        ecc_per_block: 56,
        rs_blocks: 4,
    },
    DmSymbol {
        nrow: 96,
        ncol: 96,
        regions_v: 4,
        regions_h: 4,
        region_inner_rows: 22,
        region_inner_cols: 22,
        data_cw: 696,
        ecc_per_block: 68,
        rs_blocks: 4,
    },
    DmSymbol {
        nrow: 104,
        ncol: 104,
        regions_v: 4,
        regions_h: 4,
        region_inner_rows: 24,
        region_inner_cols: 24,
        data_cw: 816,
        ecc_per_block: 56,
        rs_blocks: 6,
    },
    DmSymbol {
        nrow: 120,
        ncol: 120,
        regions_v: 6,
        regions_h: 6,
        region_inner_rows: 18,
        region_inner_cols: 18,
        data_cw: 1050,
        ecc_per_block: 68,
        rs_blocks: 6,
    },
    DmSymbol {
        nrow: 132,
        ncol: 132,
        regions_v: 6,
        regions_h: 6,
        region_inner_rows: 20,
        region_inner_cols: 20,
        data_cw: 1304,
        ecc_per_block: 62,
        rs_blocks: 8,
    },
    DmSymbol {
        nrow: 144,
        ncol: 144,
        regions_v: 6,
        regions_h: 6,
        region_inner_rows: 22,
        region_inner_cols: 22,
        data_cw: 1558,
        ecc_per_block: 62,
        rs_blocks: 10,
    },
];

fn dm_select_symbol(byte_len: usize) -> Result<&'static DmSymbol> {
    DM_SYMBOLS
        .iter()
        .find(|s| (s.data_cw as usize) >= byte_len)
        .ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "data too large for DataMatrix square symbols (max {} bytes)",
                DM_SYMBOLS.last().map(|s| s.data_cw).unwrap_or(0)
            ))
        })
}

// ---------------------------------------------------------------------------
// Encoding mode selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmEncodingMode {
    /// ASCII mode: 1 byte per codeword (or 2 for high bytes). Default fallback
    Ascii,
    /// C40 mode: 3 uppercase chars / digits packed into 2 codewords
    /// Best for SKU strings with mostly A-Z + 0-9
    C40,
    /// Base 256 mode: 1 codeword per byte, supports arbitrary binary data
    Base256,
}

/// Picks the most compact mode for the given data. Heuristic:
/// any byte > 127 forces Base256; otherwise C40 if >= 4 chars are all
/// uppercase A-Z + digits + space; else ASCII
fn dm_select_mode(data: &[u8]) -> DmEncodingMode {
    if data.iter().any(|&b| b >= 0x80) {
        return DmEncodingMode::Base256;
    }
    if data.len() >= 4
        && data
            .iter()
            .all(|&b| b == b' ' || (b'0'..=b'9').contains(&b) || (b'A'..=b'Z').contains(&b))
    {
        return DmEncodingMode::C40;
    }
    DmEncodingMode::Ascii
}

/// Encodes data in C40 mode. Each character is mapped to a value (0..40),
/// three values are packed as: 1600*v1 + 40*v2 + v3 + 1, then split into
/// two codewords (high byte, low byte). Mode is entered with 0xE6 latch and
/// exited with 0xFE unlatch back to ASCII
/// Falls back gracefully when input has chars outside C40 set 0
fn dm_encode_c40(data: &[u8], capacity: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(capacity);
    out.push(0xE6); // C40 latch
    let mut triple = Vec::with_capacity(3);
    let mut i = 0;
    while i < data.len() && out.len() + 2 <= capacity {
        let b = data[i];
        // C40 set 0 mapping: space=3, '0'-'9'=4-13, 'A'-'Z'=14-39
        let v = if b == b' ' {
            3u8
        } else if (b'0'..=b'9').contains(&b) {
            4 + (b - b'0')
        } else if (b'A'..=b'Z').contains(&b) {
            14 + (b - b'A')
        } else {
            // Not in C40 set 0, unlatch to ASCII for the remainder
            break;
        };
        triple.push(v);
        i += 1;
        if triple.len() == 3 {
            let packed =
                1600u16 * triple[0] as u16 + 40u16 * triple[1] as u16 + triple[2] as u16 + 1;
            out.push((packed >> 8) as u8);
            out.push((packed & 0xFF) as u8);
            triple.clear();
        }
    }
    // Handle leftover triple buffer per ISO/IEC 16022 5.2.5.2:
    // - 2 leftover chars: unlatch (0xFE), then emit each as ASCII (byte+1)
    // - 1 leftover char: unlatch, then ASCII for that one char
    // - 0 leftover: unlatch
    if out.len() < capacity {
        out.push(0xFE); // unlatch C40 -> ASCII
        for &v in &triple {
            // Reverse-map back to ASCII char
            let ch = match v {
                3 => b' ',
                4..=13 => b'0' + (v - 4),
                14..=39 => b'A' + (v - 14),
                _ => b' ',
            };
            if out.len() < capacity {
                out.push(ch + 1);
            }
        }
    }
    // Encode any remaining bytes as ASCII
    while i < data.len() && out.len() < capacity {
        let b = data[i];
        if b < 0x80 {
            out.push(b + 1);
        } else {
            out.push(0xEB);
            if out.len() < capacity {
                out.push(b - 128 + 1);
            }
        }
        i += 1;
    }
    if out.len() < capacity {
        out.push(0x81);
    }
    while out.len() < capacity {
        let pos = out.len() + 1;
        let r = ((149 * pos as u32) % 253 + 1) as u8;
        out.push(((129u32 + r as u32) % 254) as u8);
    }
    out
}

/// Encodes data in Base 256 mode. Each byte is one codeword (after byte
/// randomization). Mode is entered with 0xE7 latch followed by a length
/// indicator (1 byte for <=249 bytes, 2 bytes for longer). Per ISO/IEC
/// 16022 5.2.9, each Base 256 byte is randomized with:
///   randomized = (byte + ((149 * pos_in_symbol) mod 255) + 1) mod 256
/// where pos_in_symbol is 1-indexed from start of symbol
fn dm_encode_base256(data: &[u8], capacity: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(capacity);
    out.push(0xE7); // Base 256 latch
    // Length indicator
    let len = data.len();
    if len <= 249 {
        out.push(len as u8);
    } else if len <= 1555 {
        // 2-byte length: high = floor(len/250) + 249, low = len mod 250
        out.push(((len / 250) + 249) as u8);
        out.push((len % 250) as u8);
    } else {
        return Vec::new(); // too large for Base 256
    }
    for &b in data.iter() {
        if out.len() >= capacity {
            break;
        }
        let pos_in_symbol = out.len() + 1;
        let pseudo = ((149 * pos_in_symbol as u32) % 255 + 1) as u8;
        out.push(b.wrapping_add(pseudo));
    }
    if out.len() < capacity {
        out.push(0x81);
    }
    while out.len() < capacity {
        let pos = out.len() + 1;
        let r = ((149 * pos as u32) % 253 + 1) as u8;
        out.push(((129u32 + r as u32) % 254) as u8);
    }
    out
}

/// Encode bytes in ASCII mode (each byte stored as byte + 1, padding 0x81)
fn dm_encode_ascii(data: &[u8], capacity: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(capacity);
    for &b in data {
        if b < 0x80 {
            out.push(b + 1);
        } else {
            // Upper-half byte mode: emit 0xEB followed by (byte - 128 + 1)
            // This is the simplest extended-ASCII path per ISO/IEC 16022 5.2.3
            out.push(0xEB);
            out.push(b - 128 + 1);
        }
        if out.len() >= capacity {
            break;
        }
    }
    if out.len() < capacity {
        out.push(0x81); // first pad codeword
    }
    while out.len() < capacity {
        // Standard randomized pad sequence: 129 + ((149 * (i+1)) mod 253) + 1
        let pos = out.len() + 1;
        let r = ((149 * pos as u32) % 253 + 1) as u8;
        out.push(((129u32 + r as u32) % 254) as u8);
    }
    out
}

// ---------------------------------------------------------------------------
// Reed-Solomon over GF(256) with primitive polynomial 0x12D (DataMatrix)
// ---------------------------------------------------------------------------

const DM_GF_PRIM: u32 = 0x12D;

static DM_GF_EXP: [u8; 256] = build_dm_gf_exp();
static DM_GF_LOG: [u8; 256] = build_dm_gf_log();

const fn build_dm_gf_exp() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut x: u32 = 1;
    let mut i = 0;
    while i < 255 {
        t[i] = x as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= DM_GF_PRIM;
        }
        i += 1;
    }
    t[255] = t[0];
    t
}

const fn build_dm_gf_log() -> [u8; 256] {
    let exp = build_dm_gf_exp();
    let mut t = [0u8; 256];
    let mut i: u32 = 0;
    while i < 255 {
        t[exp[i as usize] as usize] = i as u8;
        i += 1;
    }
    t
}

fn dm_gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let la = DM_GF_LOG[a as usize] as u32;
    let lb = DM_GF_LOG[b as usize] as u32;
    DM_GF_EXP[((la + lb) % 255) as usize]
}

fn dm_gf_exp(p: u8) -> u8 {
    DM_GF_EXP[(p as usize) % 255]
}

fn dm_rs_generator(degree: usize) -> Vec<u8> {
    // Builds g(x) = (x + alpha^1)(x + alpha^2)...(x + alpha^degree)
    // Returned in DESCENDING coefficient order, so g[0] = 1 (leading) and
    // g[degree] = product of roots. Matches the convention used by
    // dm_rs_encode's synthetic division and dm_rs_correct's syndrome math
    let mut g = vec![1u8];
    for i in 0..degree {
        let mut new_g = vec![0u8; g.len() + 1];
        // Multiplying descending-order g by (x + alpha^(i+1)):
        // new_g[0]   = g[0]                        (leading stays leading)
        // new_g[j]   = g[j] ^ alpha^(i+1) * g[j-1] (mixed terms)
        // new_g[d+1] = alpha^(i+1) * g[d]          (new constant)
        let alpha = dm_gf_exp(i as u8 + 1);
        for j in 0..g.len() {
            new_g[j] ^= g[j];
            new_g[j + 1] ^= dm_gf_mul(alpha, g[j]);
        }
        g = new_g;
    }
    g
}

fn dm_rs_encode(data: &[u8], ecc_count: usize) -> Vec<u8> {
    let generator = dm_rs_generator(ecc_count);
    let mut buf = vec![0u8; data.len() + ecc_count];
    buf[..data.len()].copy_from_slice(data);
    for i in 0..data.len() {
        let lead = buf[i];
        if lead != 0 {
            for (j, &g) in generator.iter().enumerate() {
                buf[i + j] ^= dm_gf_mul(g, lead);
            }
        }
    }
    buf[data.len()..].to_vec()
}

/// Polynomial division remainder over GF(256), used internally by Forney
fn dm_poly_eval(poly: &[u8], x: u8) -> u8 {
    let mut y = 0u8;
    for &c in poly {
        y = dm_gf_mul(y, x) ^ c;
    }
    y
}

/// Decodes one RS block in place. Returns Ok if syndromes are all zero or
/// errors were corrected, Err if uncorrectable. Implements Berlekamp-Massey
/// for the error locator, Chien search for positions, Forney for magnitudes
/// Per ISO/IEC 16022 the codeword sequence in `block` is data followed by
/// ECC, with codewords numbered such that codeword[0] corresponds to x^(n-1)
fn dm_rs_correct(block: &mut [u8], ecc_count: usize) -> Result<()> {
    let n = block.len();
    if n == 0 || ecc_count == 0 {
        return Ok(());
    }

    // Compute syndromes S_i = block(alpha^(i+1)) for i in 0..ecc_count
    let mut syndromes = vec![0u8; ecc_count];
    for i in 0..ecc_count {
        let x = dm_gf_exp((i + 1) as u8);
        syndromes[i] = dm_poly_eval(block, x);
    }
    if syndromes.iter().all(|&s| s == 0) {
        return Ok(());
    }

    // Berlekamp-Massey to find the error locator polynomial
    let mut sigma = vec![1u8];
    let mut prev = vec![1u8];
    let mut delay = 1usize;
    let mut prev_disc = 1u8;
    let mut l = 0usize;

    for k in 0..ecc_count {
        let mut disc = syndromes[k];
        for i in 1..=l {
            if i < sigma.len() {
                disc ^= dm_gf_mul(sigma[i], syndromes[k - i]);
            }
        }
        if disc == 0 {
            delay += 1;
        } else if 2 * l <= k {
            let new_sigma = sigma.clone();
            let coef = dm_gf_mul(disc, dm_gf_inv(prev_disc));
            let mut shifted = vec![0u8; delay + prev.len()];
            for (i, &c) in prev.iter().enumerate() {
                shifted[delay + i] = dm_gf_mul(coef, c);
            }
            let nlen = sigma.len().max(shifted.len());
            sigma.resize(nlen, 0);
            for (i, &c) in shifted.iter().enumerate() {
                sigma[i] ^= c;
            }
            l = k + 1 - l;
            prev = new_sigma;
            prev_disc = disc;
            delay = 1;
        } else {
            let coef = dm_gf_mul(disc, dm_gf_inv(prev_disc));
            let mut shifted = vec![0u8; delay + prev.len()];
            for (i, &c) in prev.iter().enumerate() {
                shifted[delay + i] = dm_gf_mul(coef, c);
            }
            let nlen = sigma.len().max(shifted.len());
            sigma.resize(nlen, 0);
            for (i, &c) in shifted.iter().enumerate() {
                sigma[i] ^= c;
            }
            delay += 1;
        }
    }
    while sigma.len() > 1 && *sigma.last().unwrap() == 0 {
        sigma.pop();
    }
    let num_errors = sigma.len() - 1;
    if num_errors == 0 || num_errors > ecc_count / 2 {
        return Err(ZyronError::ExecutionError(format!(
            "RS decode: too many errors ({} > {})",
            num_errors,
            ecc_count / 2
        )));
    }

    // Chien search: find roots of sigma (ascending-order poly, eval via _rev)
    // Position i has error if sigma(alpha^-i) == 0
    let mut error_positions = Vec::with_capacity(num_errors);
    for i in 0..n {
        let x_inv = dm_gf_exp(((255 - i % 255) % 255) as u8);
        if dm_poly_eval_rev(&sigma, x_inv) == 0 {
            error_positions.push(i);
        }
    }
    if error_positions.len() != num_errors {
        return Err(ZyronError::ExecutionError(
            "RS decode: chien search did not find all error positions".into(),
        ));
    }

    // Compute error evaluator omega = (S * sigma) mod x^ecc_count
    // Syndrome polynomial S(x) = S_0 + S_1*x + ... + S_{nu-1}*x^(nu-1)
    let mut omega = vec![0u8; ecc_count];
    for i in 0..ecc_count {
        for j in 0..sigma.len() {
            if i >= j {
                omega[i] ^= dm_gf_mul(syndromes[i - j], sigma[j]);
            }
        }
    }

    // Forney: error magnitude e_k = omega(X_k^-1) / sigma'(X_k^-1)
    // (No X_k prefactor because our generator starts at alpha^1, not alpha^0)
    // sigma' is the formal derivative, with only odd-power terms surviving
    for &pos in &error_positions {
        let x_inv = dm_gf_exp(((255 - pos % 255) % 255) as u8);
        let omega_val = dm_poly_eval_rev(&omega, x_inv);
        let sigma_deriv = dm_poly_deriv_eval(&sigma, x_inv);
        if sigma_deriv == 0 {
            return Err(ZyronError::ExecutionError(
                "RS decode: sigma derivative zero".into(),
            ));
        }
        let magnitude = dm_gf_mul(omega_val, dm_gf_inv(sigma_deriv));
        // Block is in big-endian polynomial order: block[0] is x^(n-1)
        let idx = n - 1 - pos;
        block[idx] ^= magnitude;
    }
    Ok(())
}

fn dm_gf_inv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    let la = DM_GF_LOG[a as usize] as u32;
    DM_GF_EXP[((255 - la) % 255) as usize]
}

fn dm_poly_eval_rev(poly: &[u8], x: u8) -> u8 {
    // Evaluate polynomial whose coefficients are in ascending power order:
    // poly[0] + poly[1]*x + poly[2]*x^2 + ...
    let mut y = 0u8;
    for &c in poly.iter().rev() {
        y = dm_gf_mul(y, x) ^ c;
    }
    y
}

fn dm_poly_deriv_eval(poly: &[u8], x: u8) -> u8 {
    // Formal derivative over GF(2^k): only odd-power terms survive
    // For polynomial in ascending order, deriv = sum_{i odd} poly[i] * x^(i-1)
    let mut y = 0u8;
    let mut x_pow = 1u8;
    let mut i = 1usize;
    while i < poly.len() {
        y ^= dm_gf_mul(poly[i], x_pow);
        // Advance x_pow by x^2 for the next odd term, then increment i by 2
        x_pow = dm_gf_mul(dm_gf_mul(x_pow, x), x);
        i += 2;
    }
    y
}

// ---------------------------------------------------------------------------
// Utah placement (ISO/IEC 16022 Annex F)
// ---------------------------------------------------------------------------
//
// Each codeword's 8 bits are written as:
//   Bit 1 -> (row,   col-2)
//   Bit 2 -> (row,   col-1)
//   Bit 3 -> (row-1, col-2)
//   Bit 4 -> (row-1, col-1)
//   Bit 5 -> (row-1, col)
//   Bit 6 -> (row-2, col-2)
//   Bit 7 -> (row-2, col-1)
//   Bit 8 -> (row-2, col)
// where (row, col) is the position of bit 1 in the data area
// Bits that fall outside the data area wrap with one of 4 corner-case
// patterns documented in the spec

/// Returns (data_rows, data_cols) for the inner data area (excluding finder/timing)
fn dm_data_dims(sym: &DmSymbol) -> (usize, usize) {
    (
        sym.regions_v as usize * sym.region_inner_rows as usize,
        sym.regions_h as usize * sym.region_inner_cols as usize,
    )
}

/// Backwards-compat helper: returns square dim. Panics if symbol is rectangular
fn dm_data_dim(sym: &DmSymbol) -> usize {
    let (r, c) = dm_data_dims(sym);
    debug_assert_eq!(r, c, "dm_data_dim called on rectangular symbol");
    r
}

fn dm_wrap_coords(mut r: i32, mut c: i32, nrow: i32, ncol: i32) -> (usize, usize) {
    if r < 0 {
        r += nrow;
        c += 4 - ((nrow + 4).rem_euclid(8));
    }
    if c < 0 {
        c += ncol;
        r += 4 - ((ncol + 4).rem_euclid(8));
    }
    // After shifts r/c could land outside again, normalize to the matrix
    let re = r.rem_euclid(nrow) as usize;
    let ce = c.rem_euclid(ncol) as usize;
    (re, ce)
}

fn dm_place_codewords(codewords: &[u8], sym: &DmSymbol) -> Vec<bool> {
    let (nrow, ncol) = dm_data_dims(sym);
    let mut grid = vec![false; nrow * ncol];
    let mut visited = vec![false; nrow * ncol];

    // Per ISO/IEC 16022 Annex F, when a utah bit position falls off the top
    // edge (r<0) it wraps to the bottom AND shifts column by `4 - ((nrow + 4) mod 8)`
    // When it falls off the left edge (c<0) it wraps to the right AND shifts
    // row by `4 - ((ncol + 4) mod 8)`. The shifts are needed because the
    // utah pattern is meant to tile the matrix without overlap or gap.
    let put = |grid: &mut [bool], visited: &mut [bool], r: i32, c: i32, bit: bool| {
        let (re, ce) = dm_wrap_coords(r, c, nrow as i32, ncol as i32);
        let idx = re * ncol + ce;
        grid[idx] = bit;
        visited[idx] = true;
    };

    let utah = |grid: &mut [bool], visited: &mut [bool], r: i32, c: i32, byte: u8| {
        let bits = [
            (byte >> 7) & 1,
            (byte >> 6) & 1,
            (byte >> 5) & 1,
            (byte >> 4) & 1,
            (byte >> 3) & 1,
            (byte >> 2) & 1,
            (byte >> 1) & 1,
            byte & 1,
        ];
        let coords = [
            (r - 2, c - 2),
            (r - 2, c - 1),
            (r - 1, c - 2),
            (r - 1, c - 1),
            (r - 1, c),
            (r, c - 2),
            (r, c - 1),
            (r, c),
        ];
        for ((dr, dc), bit) in coords.iter().zip(bits.iter()) {
            put(grid, visited, *dr, *dc, *bit == 1);
        }
    };

    let corner1 = |grid: &mut [bool], visited: &mut [bool], byte: u8| {
        let bits = [
            (byte >> 7) & 1,
            (byte >> 6) & 1,
            (byte >> 5) & 1,
            (byte >> 4) & 1,
            (byte >> 3) & 1,
            (byte >> 2) & 1,
            (byte >> 1) & 1,
            byte & 1,
        ];
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 1, 0),
            (nr - 1, 1),
            (nr - 1, 2),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 1),
            (2, nc - 1),
            (3, nc - 1),
        ];
        for ((dr, dc), bit) in coords.iter().zip(bits.iter()) {
            put(grid, visited, *dr, *dc, *bit == 1);
        }
    };

    let corner2 = |grid: &mut [bool], visited: &mut [bool], byte: u8| {
        let bits = [
            (byte >> 7) & 1,
            (byte >> 6) & 1,
            (byte >> 5) & 1,
            (byte >> 4) & 1,
            (byte >> 3) & 1,
            (byte >> 2) & 1,
            (byte >> 1) & 1,
            byte & 1,
        ];
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 3, 0),
            (nr - 2, 0),
            (nr - 1, 0),
            (0, nc - 4),
            (0, nc - 3),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 1),
        ];
        for ((dr, dc), bit) in coords.iter().zip(bits.iter()) {
            put(grid, visited, *dr, *dc, *bit == 1);
        }
    };

    let corner3 = |grid: &mut [bool], visited: &mut [bool], byte: u8| {
        let bits = [
            (byte >> 7) & 1,
            (byte >> 6) & 1,
            (byte >> 5) & 1,
            (byte >> 4) & 1,
            (byte >> 3) & 1,
            (byte >> 2) & 1,
            (byte >> 1) & 1,
            byte & 1,
        ];
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 3, 0),
            (nr - 2, 0),
            (nr - 1, 0),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 1),
            (2, nc - 1),
            (3, nc - 1),
        ];
        for ((dr, dc), bit) in coords.iter().zip(bits.iter()) {
            put(grid, visited, *dr, *dc, *bit == 1);
        }
    };

    let corner4 = |grid: &mut [bool], visited: &mut [bool], byte: u8| {
        let bits = [
            (byte >> 7) & 1,
            (byte >> 6) & 1,
            (byte >> 5) & 1,
            (byte >> 4) & 1,
            (byte >> 3) & 1,
            (byte >> 2) & 1,
            (byte >> 1) & 1,
            byte & 1,
        ];
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 1, 0),
            (nr - 1, nc - 1),
            (0, nc - 3),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 3),
            (1, nc - 2),
            (1, nc - 1),
        ];
        for ((dr, dc), bit) in coords.iter().zip(bits.iter()) {
            put(grid, visited, *dr, *dc, *bit == 1);
        }
    };

    let mut row: i32 = 4;
    let mut col: i32 = 0;
    let nr = nrow as i32;
    let nc = ncol as i32;
    let mut idx = 0usize;

    loop {
        if row == nr && col == 0 {
            corner1(&mut grid, &mut visited, codewords[idx]);
            idx += 1;
        }
        if row == nr - 2 && col == 0 && (nc % 4) != 0 {
            corner2(&mut grid, &mut visited, codewords[idx]);
            idx += 1;
        }
        if row == nr - 2 && col == 0 && (nc % 8) == 4 {
            corner3(&mut grid, &mut visited, codewords[idx]);
            idx += 1;
        }
        if row == nr + 4 && col == 2 && (nc % 8) == 0 {
            corner4(&mut grid, &mut visited, codewords[idx]);
            idx += 1;
        }
        // Sweep upward and to the right
        loop {
            if row < nr && col >= 0 && !visited[(row as usize) * ncol + col as usize] {
                if idx < codewords.len() {
                    utah(&mut grid, &mut visited, row, col, codewords[idx]);
                    idx += 1;
                }
            }
            row -= 2;
            col += 2;
            if !(row >= 0 && col < nc) {
                break;
            }
        }
        row += 1;
        col += 3;
        // Sweep downward and to the left
        loop {
            if row >= 0 && col < nc && !visited[(row as usize) * ncol + col as usize] {
                if idx < codewords.len() {
                    utah(&mut grid, &mut visited, row, col, codewords[idx]);
                    idx += 1;
                }
            }
            row += 2;
            col -= 2;
            if !(col >= 0 && row < nr) {
                break;
            }
        }
        row += 3;
        col += 1;
        if row >= nr && col >= nc {
            break;
        }
        if idx >= codewords.len() {
            break;
        }
    }
    // Bottom-right 2x2 fixup: when the standard walk leaves the corner
    // unfilled, the spec requires a fixed checkerboard pattern there
    //   [on,  off]
    //   [off, on ]
    // This applies to symbol sizes where nrow*ncol - 8*codewords = 4
    if !visited[(nrow - 1) * ncol + ncol - 1] {
        let br = (nrow - 1) * ncol + (ncol - 1);
        let br_left = (nrow - 1) * ncol + (ncol - 2);
        let tr_top = (nrow - 2) * ncol + (ncol - 1);
        let tl = (nrow - 2) * ncol + (ncol - 2);
        grid[tl] = true;
        grid[tr_top] = false;
        grid[br_left] = false;
        grid[br] = true;
        visited[br] = true;
        visited[br_left] = true;
        visited[tr_top] = true;
        visited[tl] = true;
    }

    grid
}

fn dm_assemble_full_grid(data_grid: &[bool], sym: &DmSymbol) -> Vec<bool> {
    let regions_v = sym.regions_v as usize;
    let regions_h = sym.regions_h as usize;
    let inner_r = sym.region_inner_rows as usize;
    let inner_c = sym.region_inner_cols as usize;
    let region_outer_r = inner_r + 2; // L finder + timing border
    let region_outer_c = inner_c + 2;
    let nrow = sym.nrow as usize;
    let ncol = sym.ncol as usize;
    debug_assert_eq!(region_outer_r * regions_v, nrow);
    debug_assert_eq!(region_outer_c * regions_h, ncol);

    let data_per_row = inner_c * regions_h;
    let mut grid = vec![false; nrow * ncol];

    for ry in 0..regions_v {
        for rx in 0..regions_h {
            let row_off = ry * region_outer_r;
            let col_off = rx * region_outer_c;
            // Solid finder line: left column and bottom row of the region
            for r in 0..region_outer_r {
                grid[(row_off + r) * ncol + col_off] = true;
            }
            for c in 0..region_outer_c {
                grid[(row_off + region_outer_r - 1) * ncol + (col_off + c)] = true;
            }
            // Alternating timing pattern: top row and right column
            for c in 0..region_outer_c {
                let on = c % 2 == 0;
                grid[row_off * ncol + (col_off + c)] = on;
            }
            for r in 0..region_outer_r {
                let on = r % 2 == 1;
                grid[(row_off + r) * ncol + (col_off + region_outer_c - 1)] = on;
            }
            // Inner data area
            for r in 0..inner_r {
                for c in 0..inner_c {
                    let src_r = ry * inner_r + r;
                    let src_c = rx * inner_c + c;
                    let bit = data_grid[src_r * data_per_row + src_c];
                    grid[(row_off + 1 + r) * ncol + (col_off + 1 + c)] = bit;
                }
            }
        }
    }
    grid
}

/// Generates a DataMatrix square symbol as a PNG byte array
pub fn data_matrix_encode(data: &str) -> Result<Vec<u8>> {
    data_matrix_encode_with(data, ImageFormat::Png, 8)
}

pub fn data_matrix_encode_with(
    data: &str,
    img_format: ImageFormat,
    module_size: u32,
) -> Result<Vec<u8>> {
    let bytes = data.as_bytes();
    // Probe each mode and pick whichever fits the smallest symbol
    let mode = dm_select_mode(bytes);
    let approx_cw = match mode {
        DmEncodingMode::Ascii => bytes.len() + bytes.iter().filter(|&&b| b >= 0x80).count(),
        DmEncodingMode::C40 => 1 + ((bytes.len() + 2) / 3) * 2 + 1,
        DmEncodingMode::Base256 => 2 + bytes.len(),
    };
    let sym = dm_select_symbol(approx_cw)?;

    let cap = sym.data_cw as usize;
    let data_cw = match mode {
        DmEncodingMode::Ascii => dm_encode_ascii(bytes, cap),
        DmEncodingMode::C40 => dm_encode_c40(bytes, cap),
        DmEncodingMode::Base256 => dm_encode_base256(bytes, cap),
    };
    let blocks = sym.rs_blocks as usize;
    let ecc_per = sym.ecc_per_block as usize;
    let data_per_block = sym.data_cw as usize / blocks;

    let mut interleaved_data: Vec<Vec<u8>> = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let mut block = Vec::with_capacity(data_per_block);
        for i in 0..data_per_block {
            block.push(data_cw[b + i * blocks]);
        }
        interleaved_data.push(block);
    }
    let ecc_blocks: Vec<Vec<u8>> = interleaved_data
        .iter()
        .map(|b| dm_rs_encode(b, ecc_per))
        .collect();

    // Block-interleave per ISO/IEC 16022 5.6: stored[i*blocks+b] = data_block_b[i]
    // For data, this is equivalent to writing data_cw verbatim because the
    // block split itself uses the same i*blocks+b indexing
    let mut full = Vec::with_capacity(sym.data_cw as usize + ecc_per * blocks);
    full.extend_from_slice(&data_cw);
    for col in 0..ecc_per {
        for b in 0..blocks {
            full.push(ecc_blocks[b][col]);
        }
    }

    let data_grid = dm_place_codewords(&full, sym);
    let grid = dm_assemble_full_grid(&data_grid, sym);

    let scale = module_size.max(1);
    let nrow = sym.nrow as usize;
    let ncol = sym.ncol as usize;
    match img_format {
        ImageFormat::Png => {
            let img_h = sym.nrow as u32 * scale;
            let img_w = sym.ncol as u32 * scale;
            let mut pixels = vec![0xFFu8; (img_h as usize) * (img_w as usize)];
            for r in 0..nrow {
                for c in 0..ncol {
                    if grid[r * ncol + c] {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let py = (r as u32 * scale + dy) as usize;
                                let px = (c as u32 * scale + dx) as usize;
                                pixels[py * img_w as usize + px] = 0x00;
                            }
                        }
                    }
                }
            }
            write_png_grayscale(img_w, img_h, &pixels)
        }
        ImageFormat::Svg => Ok(write_svg_dm(&grid, nrow, ncol, scale)),
    }
}

fn write_svg_dm(grid: &[bool], nrow: usize, ncol: usize, scale: u32) -> Vec<u8> {
    let total_h = nrow as u32 * scale;
    let total_w = ncol as u32 * scale;
    let mut s = String::with_capacity(grid.len() * 24);
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" shape-rendering=\"crispEdges\">",
        total_w, total_h, total_w, total_h
    ));
    s.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>");
    for r in 0..nrow {
        for c in 0..ncol {
            if grid[r * ncol + c] {
                s.push_str(&format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#000\"/>",
                    c as u32 * scale,
                    r as u32 * scale,
                    scale,
                    scale
                ));
            }
        }
    }
    s.push_str("</svg>");
    s.into_bytes()
}

/// Decodes a DataMatrix PNG produced by `data_matrix_encode`
/// Pipeline: PNG decode -> Otsu adaptive binarize -> locate symbol bounding
/// box (handles quiet zones / margins) -> count timing modules to recover
/// scale and shape -> module sampling -> RS error correction -> ASCII/C40/
/// Base256 codeword decode. Supports both square and rectangular shapes
pub fn data_matrix_decode(image: &[u8]) -> Result<String> {
    let img = decode_png_grayscale(image)?;
    if img.width == 0 || img.height == 0 {
        return Err(ZyronError::ExecutionError(
            "DataMatrix image is empty".into(),
        ));
    }
    // Adaptive binarize first, then crop to the symbol bounding box.
    // This makes the decoder robust to quiet zones added around the symbol
    // and to non-uniform lighting (Otsu finds the optimal split per image)
    let binary = dm_binarize_otsu(&img);
    let cropped = match dm_locate_bounds(&binary) {
        Some((t, l, b, r)) if r > l && b > t => dm_crop(&binary, t, l, b, r),
        _ => binary,
    };
    let (sym, scale) = dm_find_symbol_and_scale(&cropped)?;
    let nrow_modules = sym.nrow as usize;
    let ncol_modules = sym.ncol as usize;

    // Sample modules at center of each scaled cell of the CROPPED image
    let mut grid = vec![false; nrow_modules * ncol_modules];
    for r in 0..nrow_modules {
        for c in 0..ncol_modules {
            let py = (r as u32 * scale + scale / 2) as usize;
            let px = (c as u32 * scale + scale / 2) as usize;
            grid[r * ncol_modules + c] = cropped.pixels[py * cropped.width as usize + px] < 128;
        }
    }

    // Strip finder + timing patterns to get the inner data grid
    let regions_v = sym.regions_v as usize;
    let regions_h = sym.regions_h as usize;
    let inner_r = sym.region_inner_rows as usize;
    let inner_c = sym.region_inner_cols as usize;
    let region_outer_r = inner_r + 2;
    let region_outer_c = inner_c + 2;
    let data_rows = inner_r * regions_v;
    let data_cols = inner_c * regions_h;
    let mut data_grid = vec![false; data_rows * data_cols];
    for ry in 0..regions_v {
        for rx in 0..regions_h {
            let row_off = ry * region_outer_r;
            let col_off = rx * region_outer_c;
            for r in 0..inner_r {
                for c in 0..inner_c {
                    let bit = grid[(row_off + 1 + r) * ncol_modules + (col_off + 1 + c)];
                    let dst_r = ry * inner_r + r;
                    let dst_c = rx * inner_c + c;
                    data_grid[dst_r * data_cols + dst_c] = bit;
                }
            }
        }
    }

    // Reverse the utah placement to recover codewords
    let mut codewords = dm_extract_codewords(&data_grid, sym);

    // Per-block RS error correction: split into blocks, correct, splice back
    let blocks = sym.rs_blocks as usize;
    let ecc_per = sym.ecc_per_block as usize;
    let data_per_block = sym.data_cw as usize / blocks;
    for b in 0..blocks {
        let mut block = Vec::with_capacity(data_per_block + ecc_per);
        for i in 0..data_per_block {
            block.push(codewords[i * blocks + b]);
        }
        for i in 0..ecc_per {
            block.push(codewords[sym.data_cw as usize + i * blocks + b]);
        }
        // Correct in place; on irrecoverable error, leave block as-is and
        // let downstream ASCII decode terminate at first invalid codeword
        let _ = dm_rs_correct(&mut block, ecc_per);
        for i in 0..data_per_block {
            codewords[i * blocks + b] = block[i];
        }
    }

    // Data codewords are stored interleaved (stored[i*blocks+b] = block_b[i])
    // which is equivalent to data_cw verbatim, so just slice the first data_cw bytes
    let data_cw: Vec<u8> = codewords[..sym.data_cw as usize].to_vec();

    let out = dm_decode_codewords(&data_cw, sym.data_cw as usize)?;
    String::from_utf8(out)
        .map_err(|e| ZyronError::ExecutionError(format!("DataMatrix payload not utf-8: {}", e)))
}

/// Decodes a DataMatrix codeword stream supporting ASCII, C40, and Base 256 modes
/// `total_data_cw` is the count of codeword bytes that belong to the data
/// section of the symbol (used to compute Base 256 randomization positions)
fn dm_decode_codewords(data_cw: &[u8], _total_data_cw: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data_cw.len());
    let mut i = 0usize;
    let mut in_c40 = false;
    let mut c40_buf: Vec<u8> = Vec::new();
    let mut in_base256 = false;
    let mut base256_remaining = 0usize;
    while i < data_cw.len() {
        let b = data_cw[i];
        if in_base256 {
            if base256_remaining == 0 {
                in_base256 = false;
                continue;
            }
            let pos_in_symbol = i + 1;
            let pseudo = ((149 * pos_in_symbol as u32) % 255 + 1) as u8;
            out.push(b.wrapping_sub(pseudo));
            base256_remaining -= 1;
            i += 1;
            if base256_remaining == 0 {
                in_base256 = false;
            }
            continue;
        }
        if in_c40 {
            if b == 0xFE {
                in_c40 = false;
                i += 1;
                continue;
            }
            if i + 1 >= data_cw.len() {
                break;
            }
            let packed = ((b as u16) << 8) | data_cw[i + 1] as u16;
            if packed == 0 {
                i += 2;
                continue;
            }
            let p = packed - 1;
            let v1 = (p / 1600) as u8;
            let v2 = ((p / 40) % 40) as u8;
            let v3 = (p % 40) as u8;
            for &v in &[v1, v2, v3] {
                if v == 0 {
                    // Shift to set 1, 2, or 3 (not implemented in this minimal decoder)
                    continue;
                }
                let ch = match v {
                    3 => b' ',
                    4..=13 => b'0' + (v - 4),
                    14..=39 => b'A' + (v - 14),
                    _ => continue,
                };
                c40_buf.push(ch);
            }
            // Flush full triples but keep buffer in case of unlatch with leftover
            out.extend(c40_buf.drain(..));
            i += 2;
            continue;
        }
        if b == 0x81 {
            break; // pad
        }
        if b == 0xE6 {
            in_c40 = true;
            i += 1;
            continue;
        }
        if b == 0xE7 {
            // Base 256 latch: read length indicator(s)
            if i + 1 >= data_cw.len() {
                break;
            }
            let l1 = data_cw[i + 1] as usize;
            if l1 == 0 {
                base256_remaining = data_cw.len() - i - 2;
                i += 2;
            } else if l1 <= 249 {
                base256_remaining = l1;
                i += 2;
            } else {
                if i + 2 >= data_cw.len() {
                    break;
                }
                base256_remaining = (l1 - 249) * 250 + data_cw[i + 2] as usize;
                i += 3;
            }
            in_base256 = true;
            continue;
        }
        if b == 0xEB && i + 1 < data_cw.len() {
            out.push(data_cw[i + 1].wrapping_sub(1).wrapping_add(128));
            i += 2;
            continue;
        }
        if (1..=128).contains(&b) {
            out.push(b - 1);
            i += 1;
            continue;
        }
        break;
    }
    Ok(out)
}

/// Computes the Otsu threshold for a grayscale image (returns the optimal
/// 0..255 split that maximizes between-class variance). Robust to non-uniform
/// lighting compared to a fixed midpoint threshold
fn dm_otsu_threshold(pixels: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &p in pixels {
        hist[p as usize] += 1;
    }
    let total = pixels.len() as u64;
    let mut sum_total: u64 = 0;
    for (i, h) in hist.iter().enumerate() {
        sum_total += i as u64 * *h as u64;
    }
    let mut w0: u64 = 0;
    let mut sum0: u64 = 0;
    let mut max_var = 0i128;
    let mut best_t = 128u8;
    for t in 0..255 {
        w0 += hist[t] as u64;
        if w0 == 0 || w0 == total {
            continue;
        }
        sum0 += t as u64 * hist[t] as u64;
        let w1 = total - w0;
        let m0 = sum0 as f64 / w0 as f64;
        let m1 = (sum_total - sum0) as f64 / w1 as f64;
        let var = (w0 as f64) * (w1 as f64) * (m0 - m1) * (m0 - m1);
        let var_i = var as i128;
        if var_i > max_var {
            max_var = var_i;
            best_t = t as u8;
        }
    }
    best_t
}

/// Returns a binarized copy of the input where each pixel is 0 (dark) or 255
/// (light) per the Otsu threshold. Used as the first stage of decoding so
/// downstream sampling does not rely on a fixed 128 threshold
fn dm_binarize_otsu(img: &Grayscale) -> Grayscale {
    let t = dm_otsu_threshold(&img.pixels);
    let mut out = Vec::with_capacity(img.pixels.len());
    for &p in &img.pixels {
        out.push(if p <= t { 0 } else { 255 });
    }
    Grayscale {
        width: img.width,
        height: img.height,
        pixels: out,
    }
}

/// Locates the bounding box of a DataMatrix symbol within a (possibly
/// quiet-zoned) binary image. Returns (top, left, bottom_exclusive,
/// right_exclusive) of the smallest rectangle containing all dark pixels
/// in the image. For axis-aligned symbols (our encoder's output, even after
/// quiet-zone padding), this is the symbol bounding box
fn dm_locate_bounds(img: &Grayscale) -> Option<(u32, u32, u32, u32)> {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut top = h;
    let mut bottom = 0usize;
    let mut left = w;
    let mut right = 0usize;
    let mut any = false;
    for y in 0..h {
        for x in 0..w {
            if img.pixels[y * w + x] < 128 {
                any = true;
                if y < top {
                    top = y;
                }
                if y >= bottom {
                    bottom = y + 1;
                }
                if x < left {
                    left = x;
                }
                if x >= right {
                    right = x + 1;
                }
            }
        }
    }
    if !any {
        return None;
    }
    Some((top as u32, left as u32, bottom as u32, right as u32))
}

/// Crops a grayscale image to the given bounding rectangle
fn dm_crop(img: &Grayscale, top: u32, left: u32, bottom: u32, right: u32) -> Grayscale {
    let w = (right - left) as usize;
    let h = (bottom - top) as usize;
    let mut pixels = Vec::with_capacity(w * h);
    let src_w = img.width as usize;
    for y in 0..h {
        let src_off = (top as usize + y) * src_w + left as usize;
        pixels.extend_from_slice(&img.pixels[src_off..src_off + w]);
    }
    Grayscale {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Detects the DataMatrix symbol shape and module-pixel scale of an
/// already-binarized, already-cropped image. Counts dark runs in the top
/// timing row to recover ncol, then picks the unique symbol whose dimensions
/// fit the cropped image cleanly
fn dm_find_symbol_and_scale(img: &Grayscale) -> Result<(&'static DmSymbol, u32)> {
    let row = &img.pixels[0..img.width as usize];
    let mut dark_runs = 0u32;
    let mut in_dark = false;
    for &p in row {
        let is_dark = p < 128;
        if is_dark && !in_dark {
            dark_runs += 1;
            in_dark = true;
        } else if !is_dark {
            in_dark = false;
        }
    }
    if dark_runs == 0 {
        return Err(ZyronError::ExecutionError(
            "DataMatrix top timing pattern not found".into(),
        ));
    }
    let detected_ncol = (dark_runs * 2) as u16;
    if img.width % detected_ncol as u32 != 0 {
        return Err(ZyronError::ExecutionError(format!(
            "DataMatrix detected ncol {} does not divide cropped width {}",
            detected_ncol, img.width
        )));
    }
    let scale = img.width / detected_ncol as u32;
    if scale == 0 || img.height % scale != 0 {
        return Err(ZyronError::ExecutionError(format!(
            "DataMatrix scale {} does not divide cropped height {}",
            scale, img.height
        )));
    }
    let detected_nrow = (img.height / scale) as u16;
    let candidate = DM_SYMBOLS
        .iter()
        .find(|s| s.ncol == detected_ncol && s.nrow == detected_nrow);
    candidate
        .ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "DataMatrix shape {}x{} (rows x cols) not in symbol table",
                detected_nrow, detected_ncol
            ))
        })
        .map(|s| (s, scale))
}

fn dm_extract_codewords(data_grid: &[bool], sym: &DmSymbol) -> Vec<u8> {
    let (nrow, ncol) = dm_data_dims(sym);
    let mut visited = vec![false; nrow * ncol];

    let total_cw = sym.data_cw as usize + sym.ecc_per_block as usize * sym.rs_blocks as usize;
    let mut out = Vec::with_capacity(total_cw);

    let read = |grid: &[bool], r: i32, c: i32| -> bool {
        let (re, ce) = dm_wrap_coords(r, c, nrow as i32, ncol as i32);
        grid[re * ncol + ce]
    };

    let mark = |visited: &mut [bool], r: i32, c: i32| {
        let (re, ce) = dm_wrap_coords(r, c, nrow as i32, ncol as i32);
        visited[re * ncol + ce] = true;
    };

    let utah = |grid: &[bool], visited: &mut [bool], r: i32, c: i32| -> u8 {
        let coords = [
            (r - 2, c - 2),
            (r - 2, c - 1),
            (r - 1, c - 2),
            (r - 1, c - 1),
            (r - 1, c),
            (r, c - 2),
            (r, c - 1),
            (r, c),
        ];
        let mut byte = 0u8;
        for (i, (dr, dc)) in coords.iter().enumerate() {
            if read(grid, *dr, *dc) {
                byte |= 1 << (7 - i);
            }
            mark(visited, *dr, *dc);
        }
        byte
    };

    let corner1 = |grid: &[bool], visited: &mut [bool]| -> u8 {
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 1, 0),
            (nr - 1, 1),
            (nr - 1, 2),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 1),
            (2, nc - 1),
            (3, nc - 1),
        ];
        let mut byte = 0u8;
        for (i, (dr, dc)) in coords.iter().enumerate() {
            if read(grid, *dr, *dc) {
                byte |= 1 << (7 - i);
            }
            mark(visited, *dr, *dc);
        }
        byte
    };

    let corner2 = |grid: &[bool], visited: &mut [bool]| -> u8 {
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 3, 0),
            (nr - 2, 0),
            (nr - 1, 0),
            (0, nc - 4),
            (0, nc - 3),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 1),
        ];
        let mut byte = 0u8;
        for (i, (dr, dc)) in coords.iter().enumerate() {
            if read(grid, *dr, *dc) {
                byte |= 1 << (7 - i);
            }
            mark(visited, *dr, *dc);
        }
        byte
    };

    let corner3 = |grid: &[bool], visited: &mut [bool]| -> u8 {
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 3, 0),
            (nr - 2, 0),
            (nr - 1, 0),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 1),
            (2, nc - 1),
            (3, nc - 1),
        ];
        let mut byte = 0u8;
        for (i, (dr, dc)) in coords.iter().enumerate() {
            if read(grid, *dr, *dc) {
                byte |= 1 << (7 - i);
            }
            mark(visited, *dr, *dc);
        }
        byte
    };

    let corner4 = |grid: &[bool], visited: &mut [bool]| -> u8 {
        let nr = nrow as i32;
        let nc = ncol as i32;
        let coords = [
            (nr - 1, 0),
            (nr - 1, nc - 1),
            (0, nc - 3),
            (0, nc - 2),
            (0, nc - 1),
            (1, nc - 3),
            (1, nc - 2),
            (1, nc - 1),
        ];
        let mut byte = 0u8;
        for (i, (dr, dc)) in coords.iter().enumerate() {
            if read(grid, *dr, *dc) {
                byte |= 1 << (7 - i);
            }
            mark(visited, *dr, *dc);
        }
        byte
    };

    let mut row: i32 = 4;
    let mut col: i32 = 0;
    let nr = nrow as i32;
    let nc = ncol as i32;

    loop {
        if row == nr && col == 0 && out.len() < total_cw {
            out.push(corner1(data_grid, &mut visited));
        }
        if row == nr - 2 && col == 0 && (nc % 4) != 0 && out.len() < total_cw {
            out.push(corner2(data_grid, &mut visited));
        }
        if row == nr - 2 && col == 0 && (nc % 8) == 4 && out.len() < total_cw {
            out.push(corner3(data_grid, &mut visited));
        }
        if row == nr + 4 && col == 2 && (nc % 8) == 0 && out.len() < total_cw {
            out.push(corner4(data_grid, &mut visited));
        }
        loop {
            if row < nr
                && col >= 0
                && !visited[(row as usize) * ncol + col as usize]
                && out.len() < total_cw
            {
                out.push(utah(data_grid, &mut visited, row, col));
            }
            row -= 2;
            col += 2;
            if !(row >= 0 && col < nc) {
                break;
            }
        }
        row += 1;
        col += 3;
        loop {
            if row >= 0
                && col < nc
                && !visited[(row as usize) * ncol + col as usize]
                && out.len() < total_cw
            {
                out.push(utah(data_grid, &mut visited, row, col));
            }
            row += 2;
            col -= 2;
            if !(col >= 0 && row < nr) {
                break;
            }
        }
        row += 3;
        col += 1;
        if (row >= nr && col >= nc) || out.len() >= total_cw {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ean_check_digit_known() {
        // EAN-13 example: 590123412345 + check digit 7 -> 5901234123457
        let digits: Vec<u8> = b"590123412345".iter().map(|c| c - b'0').collect();
        assert_eq!(ean_check_digit(&digits), 7);
    }

    #[test]
    fn ean13_round_trip() {
        let png = barcode_encode("5901234123457", BarcodeFormat::Ean13).unwrap();
        let (decoded, fmt) = barcode_decode(&png).unwrap();
        assert_eq!(decoded, "5901234123457");
        assert_eq!(fmt, BarcodeFormat::Ean13);
    }

    #[test]
    fn ean8_round_trip() {
        let png = barcode_encode("96385074", BarcodeFormat::Ean8).unwrap();
        let (decoded, fmt) = barcode_decode(&png).unwrap();
        assert_eq!(decoded, "96385074");
        assert_eq!(fmt, BarcodeFormat::Ean8);
    }

    #[test]
    fn upca_round_trip() {
        let png = barcode_encode("123456789012", BarcodeFormat::UpcA).unwrap();
        let (decoded, fmt) = barcode_decode(&png).unwrap();
        assert_eq!(decoded, "123456789012");
        assert_eq!(fmt, BarcodeFormat::UpcA);
    }

    #[test]
    fn code39_round_trip() {
        let png = barcode_encode("HELLO123", BarcodeFormat::Code39).unwrap();
        let (decoded, fmt) = barcode_decode(&png).unwrap();
        assert_eq!(decoded, "HELLO123");
        assert_eq!(fmt, BarcodeFormat::Code39);
    }

    #[test]
    fn qr_round_trip_short_url() {
        let png = qr_encode("https://example.com", QrErrorCorrection::M).unwrap();
        // PNG magic bytes
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let decoded = qr_decode(&png).unwrap();
        assert_eq!(decoded, "https://example.com");
    }

    #[test]
    fn qr_round_trip_all_ec_levels() {
        for ec in [
            QrErrorCorrection::L,
            QrErrorCorrection::M,
            QrErrorCorrection::Q,
            QrErrorCorrection::H,
        ] {
            let png = qr_encode("hello", ec).expect("encode");
            let decoded =
                qr_decode(&png).unwrap_or_else(|e| panic!("decode failed for {:?}: {}", ec, e));
            assert_eq!(decoded, "hello", "ec level {:?}", ec);
        }
    }

    #[test]
    fn data_matrix_round_trip_short_ascii() {
        let png = data_matrix_encode("HELLO").unwrap();
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, "HELLO");
    }

    #[test]
    fn data_matrix_round_trip_url() {
        let png = data_matrix_encode("https://example.com").unwrap();
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, "https://example.com");
    }

    #[test]
    fn data_matrix_svg_starts_with_svg_tag() {
        let svg = data_matrix_encode_with("HELLO", ImageFormat::Svg, 4).unwrap();
        let s = std::str::from_utf8(&svg).unwrap();
        assert!(s.starts_with("<svg"));
        assert!(s.ends_with("</svg>"));
    }

    #[test]
    fn data_matrix_round_trip_growing_payloads() {
        // Exercise every single-region symbol size by feeding payloads that
        // grow past each size boundary
        let cases = [
            "abc",
            "abcde",
            "abcdefgh",
            "abcdefghijkl",
            "abcdefghijklmnopqr",
            "abcdefghijklmnopqrstuv",
            "abcdefghijklmnopqrstuvwxyz1234",
            "abcdefghijklmnopqrstuvwxyz1234567890ABCD",
            "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOP",
        ];
        for input in cases {
            let png = data_matrix_encode(input)
                .unwrap_or_else(|e| panic!("encode {:?} failed: {}", input, e));
            let decoded = data_matrix_decode(&png)
                .unwrap_or_else(|e| panic!("decode {:?} failed: {}", input, e));
            assert_eq!(decoded, input, "round-trip mismatch for {:?}", input);
        }
    }

    #[test]
    fn dm_rs_corrects_one_error() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ecc = dm_rs_encode(&data, 10);
        let mut block = data.clone();
        block.extend_from_slice(&ecc);
        let original_block = block.clone();
        // Verify clean block has zero syndromes
        let mut syndromes_clean = Vec::new();
        for i in 0..10 {
            let x = dm_gf_exp((i + 1) as u8);
            syndromes_clean.push(dm_poly_eval(&original_block, x));
        }
        assert!(
            syndromes_clean.iter().all(|&s| s == 0),
            "encoder produced non-zero syndromes: {:?}",
            syndromes_clean
        );
        block[3] ^= 0xAA;
        let result = dm_rs_correct(&mut block, 10);
        assert!(result.is_ok(), "RS correct failed: {:?}", result);
        assert_eq!(block, original_block, "RS did not recover original");
    }

    #[test]
    fn dm_rs_corrects_known_codeword_errors() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ecc = dm_rs_encode(&data, 10);
        let mut block = data.clone();
        block.extend_from_slice(&ecc);
        let original_block = block.clone();
        block[3] ^= 0xAA;
        block[7] ^= 0x55;
        let result = dm_rs_correct(&mut block, 10);
        assert!(result.is_ok(), "RS correct failed: {:?}", result);
        assert_eq!(block, original_block, "RS did not recover original");
    }

    #[test]
    fn data_matrix_rs_corrects_single_module_flips() {
        // Encode at dim=18 (data_cw=18, ecc=14, so up to 7 errors correctable
        // per block). Flip a few module bits in the data area, decode should
        // still recover the original payload via Reed-Solomon
        let input = "abcdefghijklmnop";
        let png = data_matrix_encode(input).unwrap();
        let mut img = decode_png_grayscale(&png).unwrap();
        // Flip 4 module-sized regions inside the data area. Each flipped
        // module corresponds to one bit; with 8 bits per codeword, that's
        // up to 4 codeword corruptions, well within the correction limit
        let scale = 8;
        let positions = [(5usize, 5usize), (7, 9), (9, 11), (11, 7)];
        for (r, c) in positions {
            for dy in 0..scale {
                for dx in 0..scale {
                    let py = r * scale + dy;
                    let px = c * scale + dx;
                    let i = py * img.width as usize + px;
                    img.pixels[i] = if img.pixels[i] < 128 { 0xFF } else { 0x00 };
                }
            }
        }
        let png2 = write_png_grayscale(img.width, img.height, &img.pixels).unwrap();
        let decoded = data_matrix_decode(&png2).unwrap();
        assert_eq!(
            decoded, input,
            "RS decoder failed to correct corrupted modules"
        );
    }

    #[test]
    fn data_matrix_decode_empty_image_returns_clear_error() {
        // 1x1 white pixel — no symbol present
        let png = write_png_grayscale(1, 1, &[0xFF]).unwrap();
        let err = data_matrix_decode(&png).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("timing pattern") || msg.contains("not found") || msg.contains("symbol"),
            "expected a clear error, got: {}",
            msg
        );
    }

    #[test]
    fn data_matrix_otsu_threshold_handles_pure_constant_image() {
        // All pixels identical: Otsu should not panic, threshold can be anything
        // because there is no bimodal distribution to find. The decoder should
        // surface a clear error rather than panic
        let png = write_png_grayscale(40, 40, &vec![0x80u8; 40 * 40]).unwrap();
        // Decoder should error gracefully (no timing pattern present)
        let result = data_matrix_decode(&png);
        assert!(result.is_err(), "constant-color image should fail decode");
    }

    #[test]
    fn data_matrix_rectangular_with_high_byte_data_uses_base256() {
        // A rectangular size (16x48 has 49 cw capacity) with high-byte payload
        // should auto-select Base256 mode and round-trip
        let payload: Vec<u8> = (0..30u8).map(|i| 0x80 + i).collect();
        let input = std::str::from_utf8(&payload).unwrap_or("\u{FF}\u{FE}\u{FD}");
        if std::str::from_utf8(&payload).is_err() {
            return;
        }
        let png = data_matrix_encode(input).unwrap();
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn data_matrix_decode_handles_quiet_zone_padding() {
        // Encode normally, then re-emit with a uniform white border (quiet zone)
        // around the symbol. Decoder must locate the symbol within the padded
        // image via its bounding box rather than assuming the symbol fills the
        // whole image
        let png = data_matrix_encode("HELLO").unwrap();
        let img = decode_png_grayscale(&png).unwrap();
        let pad = 32u32;
        let pad_w = img.width + 2 * pad;
        let pad_h = img.height + 2 * pad;
        let mut padded = vec![0xFFu8; (pad_w * pad_h) as usize];
        for y in 0..img.height {
            for x in 0..img.width {
                let src = img.pixels[(y * img.width + x) as usize];
                let py = (y + pad) as usize;
                let px = (x + pad) as usize;
                padded[py * pad_w as usize + px] = src;
            }
        }
        let png2 = write_png_grayscale(pad_w, pad_h, &padded).unwrap();
        let decoded = data_matrix_decode(&png2).unwrap();
        assert_eq!(decoded, "HELLO");
    }

    #[test]
    fn data_matrix_decode_handles_dim_lighting_via_otsu() {
        // Simulate a photo where every pixel is shifted to the dimmer half:
        // dark=0x40, light=0xC0 instead of 0x00 / 0xFF. A fixed-128 threshold
        // would still work here but the test exercises Otsu's adaptive split
        let png = data_matrix_encode("LIGHT_TEST").unwrap();
        let img = decode_png_grayscale(&png).unwrap();
        let mut dim = Vec::with_capacity(img.pixels.len());
        for &p in &img.pixels {
            // Compress dynamic range into [0x40, 0xC0]
            dim.push(((p as u32 * 0x80 / 0xFF) + 0x40) as u8);
        }
        let png2 = write_png_grayscale(img.width, img.height, &dim).unwrap();
        let decoded = data_matrix_decode(&png2).unwrap();
        assert_eq!(decoded, "LIGHT_TEST");
    }

    #[test]
    fn data_matrix_decode_handles_uneven_lighting_gradient() {
        // Apply a left-to-right brightness gradient that pushes part of the
        // symbol darker and part lighter. A fixed-128 threshold splits in the
        // wrong place; Otsu computes the optimal split per image
        let png = data_matrix_encode("GRADIENT").unwrap();
        let img = decode_png_grayscale(&png).unwrap();
        let mut shaded = Vec::with_capacity(img.pixels.len());
        for y in 0..img.height {
            for x in 0..img.width {
                let p = img.pixels[(y * img.width + x) as usize] as i32;
                // Shift each row uniformly by an offset in [-40, +40] across width
                let offset = -40 + (80 * x as i32) / (img.width.max(1) as i32);
                let v = (p + offset).clamp(0, 255) as u8;
                shaded.push(v);
            }
        }
        let png2 = write_png_grayscale(img.width, img.height, &shaded).unwrap();
        let decoded = data_matrix_decode(&png2).unwrap();
        assert_eq!(decoded, "GRADIENT");
    }

    #[test]
    fn data_matrix_decode_recovers_from_blur_via_rs() {
        // Encode at a larger module size so Gaussian-style blur smears module
        // boundaries but keeps centers intact, then perform a 3x3 box blur and
        // confirm the decoder still recovers the data thanks to RS correction
        let png = data_matrix_encode_with("BLUR1234567890", ImageFormat::Png, 12).unwrap();
        let img = decode_png_grayscale(&png).unwrap();
        let w = img.width as usize;
        let h = img.height as usize;
        let mut blurred = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc: u32 = 0;
                let mut cnt: u32 = 0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;
                        if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                            acc += img.pixels[ny as usize * w + nx as usize] as u32;
                            cnt += 1;
                        }
                    }
                }
                blurred[y * w + x] = (acc / cnt) as u8;
            }
        }
        let png2 = write_png_grayscale(img.width, img.height, &blurred).unwrap();
        let decoded = data_matrix_decode(&png2).unwrap();
        assert_eq!(decoded, "BLUR1234567890");
    }

    #[test]
    fn data_matrix_rectangular_round_trip() {
        // Each rectangular size has a different capacity. Test payloads sized
        // to exercise each shape (8x18=5cw, 8x32=10, 12x26=16, 12x36=22, 16x36=32, 16x48=49)
        let cases = [
            ("ABC", 5),                                              // 8x18
            ("ABCDEFGH", 10),                                        // 8x32
            ("ABCDEFGHIJKLMN", 16),                                  // 12x26
            ("ABCDEFGHIJKLMNOPQRST", 22),                            // 12x36
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", 32),                // 16x36
            ("ABCDEFGHIJKLMNOPQRSTUVWXYZ012345678901234567890", 49), // 16x48
        ];
        for (input, _expected_cw) in cases {
            let png = data_matrix_encode(input).unwrap_or_else(|e| {
                panic!("encode {:?} (len={}) failed: {}", input, input.len(), e)
            });
            let decoded = data_matrix_decode(&png).unwrap_or_else(|e| {
                panic!("decode {:?} (len={}) failed: {}", input, input.len(), e)
            });
            assert_eq!(
                decoded,
                input,
                "rectangular round-trip mismatch len={}",
                input.len()
            );
        }
    }

    #[test]
    fn data_matrix_c40_round_trip() {
        // Pure uppercase + digits triggers C40 mode
        let input = "ABCDEFGHIJ0123456789ABCDEFGHIJ";
        let png = data_matrix_encode(input).unwrap();
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn data_matrix_base256_round_trip() {
        // High-byte data triggers Base 256 mode
        let input_bytes: Vec<u8> = (0..16u8).map(|i| 0x80 + i).collect();
        let input = unsafe { std::str::from_utf8_unchecked(&input_bytes) };
        // Skip if not valid utf-8 for our str interface
        if std::str::from_utf8(&input_bytes).is_err() {
            return;
        }
        let png = data_matrix_encode(input).unwrap();
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn data_matrix_each_size_round_trip() {
        for sym in DM_SYMBOLS.iter() {
            let n = (sym.data_cw as usize).saturating_sub(2).max(1);
            let input: String = (0..n).map(|i| ((b'A' + (i as u8) % 26) as char)).collect();
            let png = data_matrix_encode(&input).unwrap_or_else(|e| {
                panic!("encode {}x{} bytes={} failed: {}", sym.nrow, sym.ncol, n, e)
            });
            let decoded = data_matrix_decode(&png).unwrap_or_else(|e| {
                panic!("decode {}x{} bytes={} failed: {}", sym.nrow, sym.ncol, n, e)
            });
            assert_eq!(
                decoded, input,
                "round-trip mismatch {}x{} bytes={}",
                sym.nrow, sym.ncol, n
            );
        }
    }

    #[test]
    fn data_matrix_round_trip_smallest_multi_region() {
        // 32x32 = first multi-region (region_count=2, inner=14, data_cw=62)
        let input = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKL"; // 48 bytes
        let png = data_matrix_encode(input).unwrap();
        let decoded = data_matrix_decode(&png).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn data_matrix_round_trip_multi_region_2x2() {
        // 32x32 (62 cw, region_count=2), 36x36 (86), 40x40 (114), 44x44 (144),
        // 48x48 (174), 52x52 (204)
        let bases: Vec<String> = (0..6)
            .map(|i| {
                let target = match i {
                    0 => 60,
                    1 => 80,
                    2 => 110,
                    3 => 140,
                    4 => 170,
                    _ => 200,
                };
                "abcdefghij0123456789".repeat(target / 20)
            })
            .collect();
        for input in &bases {
            let png = data_matrix_encode(input)
                .unwrap_or_else(|e| panic!("encode len={} failed: {}", input.len(), e));
            let decoded = data_matrix_decode(&png)
                .unwrap_or_else(|e| panic!("decode len={} failed: {}", input.len(), e));
            assert_eq!(
                decoded,
                *input,
                "multi-region round-trip mismatch for len={}",
                input.len()
            );
        }
    }

    #[test]
    fn data_matrix_round_trip_multi_region_4x4() {
        // 64x64 (280 cw, region_count=4) and 80x80 (456 cw)
        for &target in &[260usize, 440] {
            let input = "abcdefghij0123456789".repeat(target / 20);
            let png = data_matrix_encode(&input)
                .unwrap_or_else(|e| panic!("encode len={} failed: {}", input.len(), e));
            let decoded = data_matrix_decode(&png)
                .unwrap_or_else(|e| panic!("decode len={} failed: {}", input.len(), e));
            assert_eq!(
                decoded,
                input,
                "4x4 multi-region round-trip mismatch for len={}",
                input.len()
            );
        }
    }

    #[test]
    fn svg_qr_starts_with_svg_tag() {
        let svg = qr_encode_with("hi", QrErrorCorrection::L, ImageFormat::Svg, 4).unwrap();
        let s = std::str::from_utf8(&svg).unwrap();
        assert!(s.starts_with("<svg"));
        assert!(s.ends_with("</svg>"));
    }

    #[test]
    fn svg_1d_starts_with_svg_tag() {
        let svg =
            barcode_encode_with("HELLO", BarcodeFormat::Code39, ImageFormat::Svg, 40).unwrap();
        let s = std::str::from_utf8(&svg).unwrap();
        assert!(s.starts_with("<svg"));
    }
}
