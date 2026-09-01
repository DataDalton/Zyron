//! Document metadata, text extraction and markdown conversion
//!
//! Supports pdf through a sequential object scan with FlateDecode handling,
//! docx through the zip container and word/document.xml, and plain text.
//! The pdf path avoids the xref table entirely, objects are discovered by
//! scanning, which tolerates files with damaged cross reference data

use std::io::Read;

use serde_json::{Value, json};

use crate::error::{MediaError, MediaResult};

const DOCX_MIME: &str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocKind {
    Pdf,
    Docx,
    Text,
}

fn detect_doc(bytes: &[u8]) -> MediaResult<DocKind> {
    let head = &bytes[..bytes.len().min(1024)];
    if find_bytes(head, b"%PDF-").is_some() {
        return Ok(DocKind::Pdf);
    }
    if bytes.len() >= 4 && &bytes[..4] == [0x50, 0x4B, 0x03, 0x04] {
        let archive_has = |name: &str| -> bool {
            zip::ZipArchive::new(std::io::Cursor::new(bytes))
                .ok()
                .map(|mut a| a.by_name(name).is_ok())
                .unwrap_or(false)
        };
        if archive_has("[Content_Types].xml") && archive_has("word/document.xml") {
            return Ok(DocKind::Docx);
        }
        return Err(MediaError::UnsupportedFormat(
            "zip archive is not a docx document, supported document formats are pdf, docx and plain text"
                .to_string(),
        ));
    }
    if std::str::from_utf8(bytes).is_ok() {
        return Ok(DocKind::Text);
    }
    Err(MediaError::UnsupportedFormat(
        "unrecognized document bytes, supported document formats are pdf, docx and plain text"
            .to_string(),
    ))
}

/// Extracts document level metadata as json
pub fn document_metadata(bytes: &[u8]) -> MediaResult<Value> {
    match detect_doc(bytes)? {
        DocKind::Pdf => {
            let parts = split_pdf(bytes);
            let page_count = count_pages(&parts);
            Ok(json!({
                "mime_type": "application/pdf",
                "page_count": page_count,
                "title": opt_str(pdf_info_value(&parts.outside, b"/Title")),
                "author": opt_str(pdf_info_value(&parts.outside, b"/Author")),
                "created": opt_str(pdf_info_value(&parts.outside, b"/CreationDate")),
                "modified": opt_str(pdf_info_value(&parts.outside, b"/ModDate")),
            }))
        }
        DocKind::Docx => {
            let core = read_zip_entry(bytes, "docProps/core.xml")?;
            let props = match core {
                Some(xml) => parse_core_properties(&xml)?,
                None => CoreProperties::default(),
            };
            Ok(json!({
                "mime_type": DOCX_MIME,
                "page_count": Value::Null,
                "title": opt_str(props.title),
                "author": opt_str(props.creator),
                "created": opt_str(props.created),
                "modified": opt_str(props.modified),
            }))
        }
        DocKind::Text => {
            let text = utf8_text(bytes)?;
            Ok(json!({
                "mime_type": "text/plain",
                "line_count": text.lines().count(),
                "char_count": text.chars().count(),
                "byte_len": bytes.len(),
            }))
        }
    }
}

/// Extracts the readable text of a document
pub fn document_extract_text(bytes: &[u8]) -> MediaResult<String> {
    match detect_doc(bytes)? {
        DocKind::Pdf => {
            let parts = split_pdf(bytes);
            let mut out = String::new();
            for stream in &parts.streams {
                extract_pdf_content_text(stream, &mut out);
            }
            Ok(out.trim_end().to_string())
        }
        DocKind::Docx => {
            let paragraphs = parse_docx_paragraphs(bytes)?;
            Ok(paragraphs
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"))
        }
        DocKind::Text => utf8_text(bytes),
    }
}

/// Converts a document to markdown
pub fn document_to_markdown(bytes: &[u8]) -> MediaResult<String> {
    match detect_doc(bytes)? {
        DocKind::Docx => {
            let paragraphs = parse_docx_paragraphs(bytes)?;
            let blocks: Vec<String> = paragraphs
                .iter()
                .filter(|p| !p.text.is_empty() || p.heading > 0)
                .map(|p| {
                    if p.heading >= 1 && p.heading <= 6 {
                        format!("{} {}", "#".repeat(p.heading as usize), p.text)
                    } else if p.list_item {
                        format!("- {}", p.text)
                    } else {
                        p.text.clone()
                    }
                })
                .collect();
            Ok(blocks.join("\n\n"))
        }
        DocKind::Pdf | DocKind::Text => {
            let text = document_extract_text(bytes)?;
            let blocks: Vec<&str> = text
                .lines()
                .map(str::trim_end)
                .filter(|l| !l.is_empty())
                .collect();
            Ok(blocks.join("\n\n"))
        }
    }
}

/// Real page count for pdf, clear errors for formats without stored pages
pub fn document_page_count(bytes: &[u8]) -> MediaResult<i64> {
    match detect_doc(bytes)? {
        DocKind::Pdf => {
            let parts = split_pdf(bytes);
            Ok(count_pages(&parts))
        }
        DocKind::Docx => Err(MediaError::UnsupportedFormat(
            "docx page count requires rendering the document layout, page boundaries are not stored in the file"
                .to_string(),
        )),
        DocKind::Text => Err(MediaError::UnsupportedFormat(
            "plain text has no page structure to count".to_string(),
        )),
    }
}

fn opt_str(value: Option<String>) -> Value {
    value.map(Value::String).unwrap_or(Value::Null)
}

fn utf8_text(bytes: &[u8]) -> MediaResult<String> {
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| MediaError::UnsupportedFormat("document bytes are not valid utf8".to_string()))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

// pdf handling

struct PdfParts {
    /// Everything outside stream bodies, dicts and structural tokens
    outside: Vec<u8>,
    /// Decoded stream bodies, raw when unfiltered, inflated for FlateDecode
    streams: Vec<Vec<u8>>,
    /// Decoded object streams, scanned for page objects and info values
    object_streams: Vec<Vec<u8>>,
}

fn split_pdf(bytes: &[u8]) -> PdfParts {
    let mut outside = Vec::new();
    let mut streams = Vec::new();
    let mut object_streams = Vec::new();
    let mut pos = 0usize;

    while pos < bytes.len() {
        let rel = match find_bytes(&bytes[pos..], b"stream") {
            Some(rel) => rel,
            None => {
                outside.extend_from_slice(&bytes[pos..]);
                break;
            }
        };
        let kw = pos + rel;
        // reject matches inside longer tokens such as endstream
        let standalone = kw == 0 || !bytes[kw - 1].is_ascii_alphanumeric();
        if !standalone {
            outside.extend_from_slice(&bytes[pos..kw + 6]);
            pos = kw + 6;
            continue;
        }
        outside.extend_from_slice(&bytes[pos..kw]);

        let mut content_start = kw + 6;
        if content_start < bytes.len() && bytes[content_start] == b'\r' {
            content_start += 1;
        }
        if content_start < bytes.len() && bytes[content_start] == b'\n' {
            content_start += 1;
        }
        let content_end = match find_bytes(&bytes[content_start..], b"endstream") {
            Some(rel_end) => content_start + rel_end,
            None => bytes.len(),
        };
        let mut content = &bytes[content_start..content_end];
        while let Some((&last, rest)) = content.split_last() {
            if last == b'\n' || last == b'\r' {
                content = rest;
            } else {
                break;
            }
        }

        let context_start = kw.saturating_sub(600);
        let context = &bytes[context_start..kw];
        let flate = find_bytes(context, b"/FlateDecode").is_some();
        let filtered = find_bytes(context, b"/Filter").is_some();
        let obj_stm = find_bytes(context, b"/ObjStm").is_some();

        let decoded: Option<Vec<u8>> = if flate {
            let mut inflated = Vec::new();
            let mut decoder = flate2::read::ZlibDecoder::new(content);
            match decoder.read_to_end(&mut inflated) {
                Ok(_) => Some(inflated),
                Err(_) => None,
            }
        } else if filtered {
            // a filter this scanner cannot decode, skipped best effort
            None
        } else {
            Some(content.to_vec())
        };
        if let Some(decoded) = decoded {
            if obj_stm {
                object_streams.push(decoded);
            } else {
                streams.push(decoded);
            }
        }

        pos = content_end;
        if let Some(rel_end) = find_bytes(&bytes[pos..], b"endstream") {
            pos += rel_end + 9;
        } else {
            pos = bytes.len();
        }
    }

    PdfParts {
        outside,
        streams,
        object_streams,
    }
}

/// Counts /Type /Page entries, excluding the /Pages tree nodes
fn count_pages(parts: &PdfParts) -> i64 {
    let mut count = count_page_markers(&parts.outside);
    for obj_stream in &parts.object_streams {
        count += count_page_markers(obj_stream);
    }
    count
}

fn count_page_markers(text: &[u8]) -> i64 {
    let mut count = 0i64;
    let mut pos = 0usize;
    while let Some(rel) = find_bytes(&text[pos..], b"/Type") {
        let mut cursor = pos + rel + 5;
        while cursor < text.len() && is_pdf_whitespace(text[cursor]) {
            cursor += 1;
        }
        if text[cursor..].starts_with(b"/Page") {
            let after = cursor + 5;
            let terminated = after >= text.len() || !text[after].is_ascii_alphanumeric();
            if terminated {
                count += 1;
            }
        }
        pos = pos + rel + 5;
    }
    count
}

fn is_pdf_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'\x0C' | b'\0')
}

/// Finds a name key in dict text and parses the following string value
fn pdf_info_value(text: &[u8], key: &[u8]) -> Option<String> {
    let mut pos = 0usize;
    while let Some(rel) = find_bytes(&text[pos..], key) {
        let after_key = pos + rel + key.len();
        // the key must end here, /Title must not match /TitleSort
        if after_key < text.len() && text[after_key].is_ascii_alphanumeric() {
            pos = after_key;
            continue;
        }
        let mut cursor = after_key;
        while cursor < text.len() && is_pdf_whitespace(text[cursor]) {
            cursor += 1;
        }
        if cursor < text.len() && text[cursor] == b'(' {
            let (raw, _) = parse_pdf_literal(text, cursor);
            return Some(pdf_bytes_to_string(&raw));
        }
        if cursor + 1 < text.len() && text[cursor] == b'<' && text[cursor + 1] != b'<' {
            let (raw, _) = parse_pdf_hex(text, cursor);
            return Some(pdf_bytes_to_string(&raw));
        }
        pos = after_key;
    }
    None
}

/// Parses a pdf literal string starting at the opening parenthesis
fn parse_pdf_literal(text: &[u8], open: usize) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    let mut depth = 1usize;
    let mut i = open + 1;
    while i < text.len() && depth > 0 {
        match text[i] {
            b'\\' if i + 1 < text.len() => {
                let esc = text[i + 1];
                i += 2;
                match esc {
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0C),
                    b'(' => out.push(b'('),
                    b')' => out.push(b')'),
                    b'\\' => out.push(b'\\'),
                    b'\r' => {
                        // line continuation, swallow an optional following newline
                        if i < text.len() && text[i] == b'\n' {
                            i += 1;
                        }
                    }
                    b'\n' => {}
                    d if d.is_ascii_digit() => {
                        let mut code = (d - b'0') as u32;
                        let mut digits = 1;
                        while digits < 3
                            && i < text.len()
                            && text[i].is_ascii_digit()
                            && text[i] < b'8'
                        {
                            code = code * 8 + (text[i] - b'0') as u32;
                            i += 1;
                            digits += 1;
                        }
                        out.push((code & 0xFF) as u8);
                    }
                    other => out.push(other),
                }
            }
            b'(' => {
                depth += 1;
                out.push(b'(');
                i += 1;
            }
            b')' => {
                depth -= 1;
                if depth > 0 {
                    out.push(b')');
                }
                i += 1;
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    (out, i)
}

/// Parses a pdf hex string starting at the opening angle bracket
fn parse_pdf_hex(text: &[u8], open: usize) -> (Vec<u8>, usize) {
    let mut nibbles = Vec::new();
    let mut i = open + 1;
    while i < text.len() && text[i] != b'>' {
        let c = text[i];
        if c.is_ascii_hexdigit() {
            let v = match c {
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                _ => c - b'A' + 10,
            };
            nibbles.push(v);
        }
        i += 1;
    }
    if nibbles.len() % 2 == 1 {
        nibbles.push(0);
    }
    let out = nibbles
        .chunks_exact(2)
        .map(|p| (p[0] << 4) | p[1])
        .collect();
    (out, i + 1)
}

fn pdf_bytes_to_string(raw: &[u8]) -> String {
    if raw.len() >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
        let units: Vec<u16> = raw[2..]
            .chunks_exact(2)
            .map(|p| u16::from_be_bytes([p[0], p[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    raw.iter().map(|&b| b as char).collect()
}

enum Operand {
    Str(Vec<u8>),
    Num(f64),
    Arr(Vec<ArrayItem>),
    Other,
}

enum ArrayItem {
    Str(Vec<u8>),
    Num(f64),
}

/// Tokenizes one content stream and appends the shown text
fn extract_pdf_content_text(content: &[u8], out: &mut String) {
    let mut operands: Vec<Operand> = Vec::new();
    let mut array_items: Vec<ArrayItem> = Vec::new();
    let mut in_array = false;
    let mut i = 0usize;

    while i < content.len() {
        let c = content[i];
        if is_pdf_whitespace(c) {
            i += 1;
            continue;
        }
        match c {
            b'%' => {
                while i < content.len() && content[i] != b'\n' && content[i] != b'\r' {
                    i += 1;
                }
            }
            b'(' => {
                let (raw, next) = parse_pdf_literal(content, i);
                if in_array {
                    array_items.push(ArrayItem::Str(raw));
                } else {
                    operands.push(Operand::Str(raw));
                }
                i = next;
            }
            b'<' => {
                if i + 1 < content.len() && content[i + 1] == b'<' {
                    operands.push(Operand::Other);
                    i += 2;
                } else {
                    let (raw, next) = parse_pdf_hex(content, i);
                    if in_array {
                        array_items.push(ArrayItem::Str(raw));
                    } else {
                        operands.push(Operand::Str(raw));
                    }
                    i = next;
                }
            }
            b'>' => {
                i += if i + 1 < content.len() && content[i + 1] == b'>' {
                    2
                } else {
                    1
                };
            }
            b'[' => {
                in_array = true;
                array_items.clear();
                i += 1;
            }
            b']' => {
                in_array = false;
                operands.push(Operand::Arr(std::mem::take(&mut array_items)));
                i += 1;
            }
            b'/' => {
                i += 1;
                while i < content.len() && is_regular_char(content[i]) {
                    i += 1;
                }
                if !in_array {
                    operands.push(Operand::Other);
                }
            }
            b'{' | b'}' => i += 1,
            b'+' | b'-' | b'.' | b'0'..=b'9' => {
                let start = i;
                i += 1;
                while i < content.len()
                    && (content[i].is_ascii_digit() || content[i] == b'.' || content[i] == b'-')
                {
                    i += 1;
                }
                let num = std::str::from_utf8(&content[start..i])
                    .ok()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                if in_array {
                    array_items.push(ArrayItem::Num(num));
                } else {
                    operands.push(Operand::Num(num));
                }
            }
            _ => {
                let start = i;
                while i < content.len() && is_operator_char(content[i]) {
                    i += 1;
                }
                if i == start {
                    i += 1;
                    continue;
                }
                dispatch_operator(&content[start..i], &operands, out);
                operands.clear();
            }
        }
    }
}

fn is_regular_char(b: u8) -> bool {
    !is_pdf_whitespace(b)
        && !matches!(
            b,
            b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
        )
}

fn is_operator_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'\'' | b'"' | b'*')
}

fn dispatch_operator(op: &[u8], operands: &[Operand], out: &mut String) {
    match op {
        b"Tj" => {
            if let Some(Operand::Str(raw)) =
                operands.iter().rev().find(|o| matches!(o, Operand::Str(_)))
            {
                out.push_str(&pdf_bytes_to_string(raw));
            }
        }
        b"TJ" => {
            if let Some(Operand::Arr(items)) =
                operands.iter().rev().find(|o| matches!(o, Operand::Arr(_)))
            {
                for item in items {
                    match item {
                        ArrayItem::Str(raw) => out.push_str(&pdf_bytes_to_string(raw)),
                        // large negative kerning approximates a word space
                        ArrayItem::Num(n) if *n <= -180.0 => out.push(' '),
                        ArrayItem::Num(_) => {}
                    }
                }
            }
        }
        b"'" | b"\"" => {
            push_newline(out);
            if let Some(Operand::Str(raw)) =
                operands.iter().rev().find(|o| matches!(o, Operand::Str(_)))
            {
                out.push_str(&pdf_bytes_to_string(raw));
            }
        }
        b"Td" | b"TD" => {
            // a vertical move starts a new line
            let nums: Vec<f64> = operands
                .iter()
                .filter_map(|o| match o {
                    Operand::Num(n) => Some(*n),
                    _ => None,
                })
                .collect();
            if nums.len() >= 2 && nums[nums.len() - 1] != 0.0 {
                push_newline(out);
            }
        }
        b"T*" | b"ET" => push_newline(out),
        _ => {}
    }
}

fn push_newline(out: &mut String) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
}

// docx handling

fn read_zip_entry(bytes: &[u8], name: &str) -> MediaResult<Option<Vec<u8>>> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| MediaError::CorruptObject(format!("docx zip open failed: {e}")))?;
    let mut file = match archive.by_name(name) {
        Ok(file) => file,
        Err(zip::result::ZipError::FileNotFound) => return Ok(None),
        Err(e) => {
            return Err(MediaError::CorruptObject(format!(
                "docx zip entry {name} failed: {e}"
            )));
        }
    };
    let mut out = Vec::new();
    file.read_to_end(&mut out).map_err(|e| {
        MediaError::CorruptObject(format!("docx zip entry {name} read failed: {e}"))
    })?;
    Ok(Some(out))
}

fn xml_local(name: &[u8]) -> &[u8] {
    match name.iter().rposition(|&b| b == b':') {
        Some(idx) => &name[idx + 1..],
        None => name,
    }
}

#[derive(Default)]
struct DocxParagraph {
    heading: u8,
    list_item: bool,
    text: String,
}

fn parse_docx_paragraphs(bytes: &[u8]) -> MediaResult<Vec<DocxParagraph>> {
    let xml = read_zip_entry(bytes, "word/document.xml")?
        .ok_or_else(|| MediaError::CorruptObject("docx has no word/document.xml".to_string()))?;
    let mut reader = quick_xml::Reader::from_reader(&xml[..]);
    let mut buf = Vec::new();
    let mut paragraphs = Vec::new();
    let mut current: Option<DocxParagraph> = None;
    let mut in_text = false;

    loop {
        let event = reader
            .read_event_into(&mut buf)
            .map_err(|e| MediaError::CorruptObject(format!("docx xml parse failed: {e}")))?;
        match event {
            quick_xml::events::Event::Start(ref e) | quick_xml::events::Event::Empty(ref e) => {
                let empty = matches!(event, quick_xml::events::Event::Empty(_));
                match xml_local(e.name().as_ref()) {
                    b"p" if !empty => current = Some(DocxParagraph::default()),
                    b"p" => paragraphs.push(DocxParagraph::default()),
                    b"pStyle" => {
                        if let Some(paragraph) = current.as_mut() {
                            for attr in e.attributes().flatten() {
                                if xml_local(attr.key.as_ref()) == b"val" {
                                    let val = String::from_utf8_lossy(&attr.value);
                                    if let Some(level) = val.strip_prefix("Heading")
                                        && let Ok(n) = level.parse::<u8>()
                                        && (1..=6).contains(&n)
                                    {
                                        paragraph.heading = n;
                                    }
                                }
                            }
                        }
                    }
                    b"numPr" => {
                        if let Some(paragraph) = current.as_mut() {
                            paragraph.list_item = true;
                        }
                    }
                    b"t" if !empty => in_text = true,
                    b"tab" => {
                        if let Some(paragraph) = current.as_mut() {
                            paragraph.text.push('\t');
                        }
                    }
                    b"br" => {
                        if let Some(paragraph) = current.as_mut() {
                            paragraph.text.push('\n');
                        }
                    }
                    _ => {}
                }
            }
            quick_xml::events::Event::End(ref e) => match xml_local(e.name().as_ref()) {
                b"p" => {
                    if let Some(paragraph) = current.take() {
                        paragraphs.push(paragraph);
                    }
                }
                b"t" => in_text = false,
                _ => {}
            },
            quick_xml::events::Event::Text(ref t) => {
                if in_text && let Some(paragraph) = current.as_mut() {
                    let text = t.unescape().map_err(|e| {
                        MediaError::CorruptObject(format!("docx xml text decode failed: {e}"))
                    })?;
                    paragraph.text.push_str(&text);
                }
            }
            quick_xml::events::Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(paragraphs)
}

#[derive(Default)]
struct CoreProperties {
    title: Option<String>,
    creator: Option<String>,
    created: Option<String>,
    modified: Option<String>,
}

fn parse_core_properties(xml: &[u8]) -> MediaResult<CoreProperties> {
    let mut reader = quick_xml::Reader::from_reader(xml);
    let mut buf = Vec::new();
    let mut props = CoreProperties::default();
    let mut current: Option<&'static str> = None;

    loop {
        let event = reader
            .read_event_into(&mut buf)
            .map_err(|e| MediaError::CorruptObject(format!("core.xml parse failed: {e}")))?;
        match event {
            quick_xml::events::Event::Start(ref e) => {
                current = match xml_local(e.name().as_ref()) {
                    b"title" => Some("title"),
                    b"creator" => Some("creator"),
                    b"created" => Some("created"),
                    b"modified" => Some("modified"),
                    _ => None,
                };
            }
            quick_xml::events::Event::Text(ref t) => {
                if let Some(field) = current {
                    let text = t
                        .unescape()
                        .map_err(|e| {
                            MediaError::CorruptObject(format!("core.xml text decode failed: {e}"))
                        })?
                        .into_owned();
                    match field {
                        "title" => props.title = Some(text),
                        "creator" => props.creator = Some(text),
                        "created" => props.created = Some(text),
                        _ => props.modified = Some(text),
                    }
                }
            }
            quick_xml::events::Event::End(_) => current = None,
            quick_xml::events::Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(props)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn sample_pdf() -> Vec<u8> {
        let content = b"BT /F1 12 Tf 72 720 Td (Hello Zyron media) Tj ET";
        let mut pdf = Vec::new();
        pdf.extend_from_slice(b"%PDF-1.4\n");
        pdf.extend_from_slice(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n");
        pdf.extend_from_slice(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n");
        pdf.extend_from_slice(
            b"3 0 obj << /Type /Page /Parent 2 0 R /Contents 4 0 R /MediaBox [0 0 612 792] >> endobj\n",
        );
        pdf.extend_from_slice(
            format!(
                "4 0 obj << /Length {} /Author (Zyron Tests) /Title (Media Fixture) >> stream\n",
                content.len()
            )
            .as_bytes(),
        );
        pdf.extend_from_slice(content);
        pdf.extend_from_slice(b"\nendstream endobj\n");
        pdf.extend_from_slice(b"trailer << /Root 1 0 R >>\n%%EOF\n");
        pdf
    }

    fn sample_docx() -> Vec<u8> {
        let content_types = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#;
        let document = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Zyron Media Title</w:t></w:r></w:p>
<w:p><w:r><w:t>Body paragraph text here</w:t></w:r></w:p>
<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/></w:numPr></w:pPr><w:r><w:t>First bullet</w:t></w:r></w:p>
</w:body>
</w:document>"#;
        let core = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Zyron Media Title</dc:title>
<dc:creator>Zyron Tests</dc:creator>
</cp:coreProperties>"#;

        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default();
        writer
            .start_file("[Content_Types].xml", options)
            .expect("start content types");
        writer
            .write_all(content_types.as_bytes())
            .expect("write content types");
        writer
            .start_file("word/document.xml", options)
            .expect("start document");
        writer
            .write_all(document.as_bytes())
            .expect("write document");
        writer
            .start_file("docProps/core.xml", options)
            .expect("start core");
        writer.write_all(core.as_bytes()).expect("write core");
        writer.finish().expect("finish zip").into_inner()
    }

    #[test]
    fn pdf_text_and_page_count() {
        let pdf = sample_pdf();
        let text = document_extract_text(&pdf).expect("extract text");
        assert!(text.contains("Hello Zyron media"), "text was: {text}");
        assert_eq!(document_page_count(&pdf).expect("page count"), 1);
    }

    #[test]
    fn pdf_metadata_fields() {
        let pdf = sample_pdf();
        let meta = document_metadata(&pdf).expect("metadata");
        assert_eq!(meta["mime_type"], "application/pdf");
        assert_eq!(meta["page_count"], 1);
        assert_eq!(meta["author"], "Zyron Tests");
        assert_eq!(meta["title"], "Media Fixture");
    }

    #[test]
    fn docx_text_and_markdown() {
        let docx = sample_docx();
        let text = document_extract_text(&docx).expect("extract text");
        assert!(text.contains("Zyron Media Title"));
        assert!(text.contains("Body paragraph text here"));
        let markdown = document_to_markdown(&docx).expect("markdown");
        assert!(
            markdown.contains("# Zyron Media Title"),
            "markdown was: {markdown}"
        );
        assert!(markdown.contains("- First bullet"));
    }

    #[test]
    fn docx_metadata_and_page_count() {
        let docx = sample_docx();
        let meta = document_metadata(&docx).expect("metadata");
        assert_eq!(meta["mime_type"], DOCX_MIME);
        assert_eq!(meta["page_count"], Value::Null);
        assert_eq!(meta["title"], "Zyron Media Title");
        assert_eq!(meta["author"], "Zyron Tests");
        let err = document_page_count(&docx).expect_err("must fail");
        assert!(err.to_string().contains("rendering"));
    }

    #[test]
    fn text_stats_and_page_count_error() {
        let text = b"line one\nline two\nline three";
        let meta = document_metadata(text).expect("metadata");
        assert_eq!(meta["mime_type"], "text/plain");
        assert_eq!(meta["line_count"], 3);
        assert_eq!(
            document_extract_text(text).expect("text"),
            "line one\nline two\nline three"
        );
        assert!(document_page_count(text).is_err());
    }

    #[test]
    fn binary_garbage_errors() {
        let garbage = [0u8, 159, 146, 150, 255, 254, 1, 2, 3];
        assert!(document_metadata(&garbage).is_err());
    }
}
