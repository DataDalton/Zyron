//! Markdown, HTML, and document processing
//!
//! markdown_to_html implements a pragmatic CommonMark subset plus GFM
//! extensions (tables, task lists, strikethrough) using a single-pass byte
//! scanner so README-sized inputs stay in L1 cache. The block scanner emits
//! events that drive an inline parser; HTML escaping is enforced on raw text
//!
//! html_to_text strips tags and decodes the standard named entities
//! html_to_markdown maps a small set of tags back to their Markdown spelling
//! sanitize_html applies an allowlist over tags and drops dangerous URI
//! schemes and event handler attributes

// ---------------------------------------------------------------------------
// HTML escaping
// ---------------------------------------------------------------------------

fn html_escape(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
}

fn html_escape_attr(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
}

// ---------------------------------------------------------------------------
// Block-level Markdown parser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum BlockKind {
    Heading(u8),
    Paragraph,
    BlockQuote,
    UnorderedList,
    OrderedList(u32),
    CodeFence(String),
    ThematicBreak,
    Table,
}

/// CommonMark subset plus GFM extensions: tables, task lists, strikethrough,
/// autolinks. Returns rendered HTML
pub fn markdown_to_html(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + md.len() / 4);
    let lines: Vec<&str> = md.split('\n').collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_end_matches('\r');
        // Skip blank lines
        if trimmed.trim().is_empty() {
            i += 1;
            continue;
        }
        // Thematic break
        if is_thematic_break(trimmed) {
            out.push_str("<hr />\n");
            i += 1;
            continue;
        }
        // ATX heading
        if let Some((level, text)) = parse_atx_heading(trimmed) {
            out.push_str(&format!("<h{}>", level));
            render_inlines(&text, &mut out);
            out.push_str(&format!("</h{}>\n", level));
            i += 1;
            continue;
        }
        // Setext heading: line followed by ===== or -----
        if i + 1 < lines.len() {
            let next = lines[i + 1].trim_end_matches('\r');
            if !trimmed.trim().is_empty() && is_setext_underline(next) {
                let level: u8 = if next.starts_with('=') { 1 } else { 2 };
                out.push_str(&format!("<h{}>", level));
                render_inlines(trimmed.trim(), &mut out);
                out.push_str(&format!("</h{}>\n", level));
                i += 2;
                continue;
            }
        }
        // Fenced code block
        if let Some(info) = parse_fence_open(trimmed) {
            i += 1;
            let mut code = String::new();
            while i < lines.len() {
                let l = lines[i].trim_end_matches('\r');
                if is_fence_close(l) {
                    i += 1;
                    break;
                }
                code.push_str(l);
                code.push('\n');
                i += 1;
            }
            out.push_str("<pre><code");
            if !info.is_empty() {
                out.push_str(" class=\"language-");
                html_escape_attr(&info, &mut out);
                out.push('"');
            }
            out.push('>');
            html_escape(&code, &mut out);
            out.push_str("</code></pre>\n");
            continue;
        }
        // Block quote
        if trimmed.starts_with('>') {
            let mut quoted = String::new();
            while i < lines.len() {
                let l = lines[i].trim_end_matches('\r');
                if let Some(rest) = l.strip_prefix('>') {
                    quoted.push_str(rest.strip_prefix(' ').unwrap_or(rest));
                    quoted.push('\n');
                    i += 1;
                } else {
                    break;
                }
            }
            out.push_str("<blockquote>\n");
            out.push_str(&markdown_to_html(&quoted));
            out.push_str("</blockquote>\n");
            continue;
        }
        // GFM table: line with pipes followed by separator line of dashes/pipes
        if i + 1 < lines.len() && is_table_separator(lines[i + 1].trim_end_matches('\r')) {
            let header_line = trimmed;
            let sep_line = lines[i + 1].trim_end_matches('\r');
            i += 2;
            let mut body_lines: Vec<&str> = Vec::new();
            while i < lines.len() {
                let l = lines[i].trim_end_matches('\r');
                if l.trim().is_empty() {
                    break;
                }
                body_lines.push(l);
                i += 1;
            }
            render_table(header_line, sep_line, &body_lines, &mut out);
            continue;
        }
        // List items. Detect bullet or ordered marker
        if let Some((ordered, start, _)) = parse_list_marker(trimmed) {
            let (consumed, items) = collect_list_items(&lines[i..], ordered);
            if ordered {
                if start == 1 {
                    out.push_str("<ol>\n");
                } else {
                    out.push_str(&format!("<ol start=\"{}\">\n", start));
                }
            } else {
                out.push_str("<ul>\n");
            }
            for item in items {
                out.push_str("<li>");
                if let Some(checked) = strip_task_marker(&item) {
                    if checked.0 {
                        out.push_str("<input type=\"checkbox\" checked disabled /> ");
                    } else {
                        out.push_str("<input type=\"checkbox\" disabled /> ");
                    }
                    render_inlines(checked.1.trim(), &mut out);
                } else {
                    render_inlines(item.trim(), &mut out);
                }
                out.push_str("</li>\n");
            }
            out.push_str(if ordered { "</ol>\n" } else { "</ul>\n" });
            i += consumed;
            continue;
        }
        // Paragraph: gather consecutive non-blank lines that don't start a new
        // block
        let mut para = String::new();
        while i < lines.len() {
            let l = lines[i].trim_end_matches('\r');
            if l.trim().is_empty()
                || is_thematic_break(l)
                || parse_atx_heading(l).is_some()
                || parse_fence_open(l).is_some()
                || l.starts_with('>')
                || parse_list_marker(l).is_some()
            {
                break;
            }
            if !para.is_empty() {
                para.push('\n');
            }
            para.push_str(l);
            i += 1;
        }
        out.push_str("<p>");
        render_inlines(para.trim(), &mut out);
        out.push_str("</p>\n");
    }
    out
}

fn is_thematic_break(line: &str) -> bool {
    let t = line.trim();
    if t.len() < 3 {
        return false;
    }
    let c = t.as_bytes()[0];
    if c != b'-' && c != b'*' && c != b'_' {
        return false;
    }
    t.chars().all(|x| x == c as char || x == ' ' || x == '\t')
        && t.chars().filter(|x| *x == c as char).count() >= 3
}

fn parse_atx_heading(line: &str) -> Option<(u8, String)> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &t[hashes..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let text = rest.trim().trim_end_matches('#').trim().to_string();
    Some((hashes as u8, text))
}

fn is_setext_underline(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    let c = t.as_bytes()[0];
    if c != b'=' && c != b'-' {
        return false;
    }
    t.bytes().all(|b| b == c)
}

fn parse_fence_open(line: &str) -> Option<String> {
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix("```") {
        return Some(rest.trim().to_string());
    }
    if let Some(rest) = t.strip_prefix("~~~") {
        return Some(rest.trim().to_string());
    }
    None
}

fn is_fence_close(line: &str) -> bool {
    let t = line.trim();
    t == "```" || t == "~~~" || t.starts_with("````") || t.starts_with("~~~~")
}

fn parse_list_marker(line: &str) -> Option<(bool, u32, usize)> {
    let t = line;
    let mut indent = 0;
    while indent < t.len() && (t.as_bytes()[indent] == b' ' || t.as_bytes()[indent] == b'\t') {
        indent += 1;
    }
    let rest = &t[indent..];
    if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
        return Some((false, 1, indent + 2));
    }
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let after = &rest[digits.len()..];
        if after.starts_with(". ") || after.starts_with(") ") {
            if let Ok(n) = digits.parse::<u32>() {
                return Some((true, n, indent + digits.len() + 2));
            }
        }
    }
    None
}

fn collect_list_items(lines: &[&str], ordered: bool) -> (usize, Vec<String>) {
    let mut items: Vec<String> = Vec::new();
    let mut consumed = 0usize;
    let mut current = String::new();
    while consumed < lines.len() {
        let l = lines[consumed].trim_end_matches('\r');
        if l.trim().is_empty() {
            // Blank line ends the list unless followed by another list item
            if consumed + 1 < lines.len() {
                let next = lines[consumed + 1].trim_end_matches('\r');
                if let Some((next_ordered, _, _)) = parse_list_marker(next) {
                    if next_ordered == ordered {
                        consumed += 1;
                        continue;
                    }
                }
            }
            break;
        }
        if let Some((next_ordered, _, after)) = parse_list_marker(l) {
            if next_ordered != ordered {
                break;
            }
            if !current.is_empty() {
                items.push(std::mem::take(&mut current));
            }
            current.push_str(&l[after..]);
            consumed += 1;
        } else {
            // Continuation line (not a new item, not blank)
            if current.is_empty() {
                break;
            }
            current.push('\n');
            current.push_str(l.trim_start());
            consumed += 1;
        }
    }
    if !current.is_empty() {
        items.push(current);
    }
    (consumed, items)
}

fn strip_task_marker(item: &str) -> Option<(bool, String)> {
    let t = item.trim_start();
    if t.starts_with("[ ] ") {
        return Some((false, t[4..].to_string()));
    }
    if t.starts_with("[x] ") || t.starts_with("[X] ") {
        return Some((true, t[4..].to_string()));
    }
    None
}

// ---------------------------------------------------------------------------
// GFM table rendering
// ---------------------------------------------------------------------------

fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    let cells = split_table_row(t);
    if cells.len() < 1 {
        return false;
    }
    cells.iter().all(|c| {
        let s = c.trim();
        !s.is_empty()
            && s.chars()
                .all(|ch| ch == '-' || ch == ':' || ch == ' ')
            && s.contains('-')
    })
}

fn split_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let body = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or_else(|| trimmed.strip_prefix('|').unwrap_or(trimmed));
    body.split('|').map(|c| c.to_string()).collect()
}

fn render_table(header: &str, separator: &str, body: &[&str], out: &mut String) {
    let header_cells = split_table_row(header);
    let sep_cells = split_table_row(separator);
    let aligns: Vec<&'static str> = sep_cells
        .iter()
        .map(|c| {
            let s = c.trim();
            let left = s.starts_with(':');
            let right = s.ends_with(':');
            match (left, right) {
                (true, true) => "center",
                (false, true) => "right",
                (true, false) => "left",
                _ => "",
            }
        })
        .collect();
    out.push_str("<table>\n<thead>\n<tr>");
    for (i, cell) in header_cells.iter().enumerate() {
        let a = aligns.get(i).copied().unwrap_or("");
        if a.is_empty() {
            out.push_str("<th>");
        } else {
            out.push_str(&format!("<th align=\"{}\">", a));
        }
        render_inlines(cell.trim(), out);
        out.push_str("</th>");
    }
    out.push_str("</tr>\n</thead>\n<tbody>\n");
    for row in body {
        let cells = split_table_row(row);
        out.push_str("<tr>");
        for (i, cell) in cells.iter().enumerate() {
            let a = aligns.get(i).copied().unwrap_or("");
            if a.is_empty() {
                out.push_str("<td>");
            } else {
                out.push_str(&format!("<td align=\"{}\">", a));
            }
            render_inlines(cell.trim(), out);
            out.push_str("</td>");
        }
        out.push_str("</tr>\n");
    }
    out.push_str("</tbody>\n</table>\n");
}

// ---------------------------------------------------------------------------
// Inline parser
// ---------------------------------------------------------------------------

fn render_inlines(text: &str, out: &mut String) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => {
                // Backslash escape: emit the next byte literally if it is
                // ASCII punctuation
                let next = bytes[i + 1];
                if next.is_ascii_punctuation() {
                    out.push(next as char);
                    i += 2;
                } else {
                    out.push('\\');
                    i += 1;
                }
            }
            b'`' => {
                let close = find_byte(bytes, i + 1, b'`');
                if let Some(end) = close {
                    out.push_str("<code>");
                    let inner = std::str::from_utf8(&bytes[i + 1..end]).unwrap_or("");
                    html_escape(inner, out);
                    out.push_str("</code>");
                    i = end + 1;
                } else {
                    out.push('`');
                    i += 1;
                }
            }
            b'!' if i + 1 < bytes.len() && bytes[i + 1] == b'[' => {
                if let Some((alt, url, end)) = parse_link(bytes, i + 1) {
                    out.push_str("<img src=\"");
                    html_escape_attr(&url, out);
                    out.push_str("\" alt=\"");
                    html_escape_attr(&alt, out);
                    out.push_str("\" />");
                    i = end;
                } else {
                    out.push('!');
                    i += 1;
                }
            }
            b'[' => {
                if let Some((text, url, end)) = parse_link(bytes, i) {
                    if is_safe_url(&url) {
                        out.push_str("<a href=\"");
                        html_escape_attr(&url, out);
                        out.push_str("\">");
                        render_inlines(&text, out);
                        out.push_str("</a>");
                    } else {
                        render_inlines(&text, out);
                    }
                    i = end;
                } else {
                    out.push('[');
                    i += 1;
                }
            }
            b'<' => {
                // Autolink: <http(s)://...> or <user@example.com>
                if let Some(end) = find_byte(bytes, i + 1, b'>') {
                    let inner = std::str::from_utf8(&bytes[i + 1..end]).unwrap_or("");
                    if inner.starts_with("http://") || inner.starts_with("https://") {
                        out.push_str("<a href=\"");
                        html_escape_attr(inner, out);
                        out.push_str("\">");
                        html_escape(inner, out);
                        out.push_str("</a>");
                        i = end + 1;
                        continue;
                    }
                    if inner.contains('@') && !inner.contains(' ') {
                        out.push_str("<a href=\"mailto:");
                        html_escape_attr(inner, out);
                        out.push_str("\">");
                        html_escape(inner, out);
                        out.push_str("</a>");
                        i = end + 1;
                        continue;
                    }
                }
                out.push_str("&lt;");
                i += 1;
            }
            b'*' | b'_' => {
                let c = bytes[i];
                let count = bytes[i..].iter().take_while(|b| **b == c).count();
                if count >= 2
                    && let Some(end) = find_double(bytes, i + count, c)
                {
                    out.push_str("<strong>");
                    let inner = std::str::from_utf8(&bytes[i + 2..end]).unwrap_or("");
                    render_inlines(inner, out);
                    out.push_str("</strong>");
                    i = end + 2;
                    continue;
                }
                if count >= 1
                    && let Some(end) = find_byte(bytes, i + 1, c)
                {
                    out.push_str("<em>");
                    let inner = std::str::from_utf8(&bytes[i + 1..end]).unwrap_or("");
                    render_inlines(inner, out);
                    out.push_str("</em>");
                    i = end + 1;
                    continue;
                }
                out.push(c as char);
                i += 1;
            }
            b'~' if i + 1 < bytes.len() && bytes[i + 1] == b'~' => {
                if let Some(end) = find_double(bytes, i + 2, b'~') {
                    out.push_str("<del>");
                    let inner = std::str::from_utf8(&bytes[i + 2..end]).unwrap_or("");
                    render_inlines(inner, out);
                    out.push_str("</del>");
                    i = end + 2;
                    continue;
                }
                out.push_str("~~");
                i += 2;
            }
            b'>' => {
                out.push_str("&gt;");
                i += 1;
            }
            b'&' => {
                out.push_str("&amp;");
                i += 1;
            }
            b => {
                out.push(b as char);
                i += 1;
            }
        }
    }
}

fn find_byte(bytes: &[u8], start: usize, target: u8) -> Option<usize> {
    bytes[start..].iter().position(|b| *b == target).map(|p| p + start)
}

fn find_double(bytes: &[u8], start: usize, target: u8) -> Option<usize> {
    let mut i = start;
    while i + 1 < bytes.len() {
        if bytes[i] == target && bytes[i + 1] == target {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn parse_link(bytes: &[u8], at: usize) -> Option<(String, String, usize)> {
    if bytes.get(at) != Some(&b'[') {
        return None;
    }
    let close_text = find_byte(bytes, at + 1, b']')?;
    if bytes.get(close_text + 1) != Some(&b'(') {
        return None;
    }
    let close_url = find_byte(bytes, close_text + 2, b')')?;
    let text = std::str::from_utf8(&bytes[at + 1..close_text]).ok()?.to_string();
    let url = std::str::from_utf8(&bytes[close_text + 2..close_url])
        .ok()?
        .to_string();
    Some((text, url, close_url + 1))
}

fn is_safe_url(url: &str) -> bool {
    let trimmed = url.trim().to_ascii_lowercase();
    !(trimmed.starts_with("javascript:")
        || trimmed.starts_with("data:")
        || trimmed.starts_with("vbscript:"))
}

// ---------------------------------------------------------------------------
// Extractors
// ---------------------------------------------------------------------------

/// Returns (level, text) for every ATX or setext heading
pub fn markdown_extract_headers(md: &str) -> Vec<(u8, String)> {
    let mut out = Vec::new();
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        if let Some((level, text)) = parse_atx_heading(l) {
            out.push((level, text));
            i += 1;
            continue;
        }
        if i + 1 < lines.len() && is_setext_underline(lines[i + 1]) && !l.trim().is_empty() {
            let level: u8 = if lines[i + 1].starts_with('=') { 1 } else { 2 };
            out.push((level, l.trim().to_string()));
            i += 2;
            continue;
        }
        i += 1;
    }
    out
}

/// Returns (text, url) for every inline link
pub fn markdown_extract_links(md: &str) -> Vec<(String, String)> {
    let bytes = md.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some((text, url, end)) = parse_link(bytes, i) {
                out.push((text, url));
                i = end;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Returns (language, code) for every fenced code block
pub fn markdown_extract_code_blocks(md: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = md.split('\n').collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i].trim_end_matches('\r');
        if let Some(info) = parse_fence_open(l) {
            i += 1;
            let mut code = String::new();
            while i < lines.len() {
                let cur = lines[i].trim_end_matches('\r');
                if is_fence_close(cur) {
                    i += 1;
                    break;
                }
                code.push_str(cur);
                code.push('\n');
                i += 1;
            }
            out.push((info, code));
            continue;
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// HTML to text
// ---------------------------------------------------------------------------

/// Strips tags and decodes named/numeric character entities. Collapses runs
/// of whitespace
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut i = 0;
    let mut in_tag = false;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'<' {
            in_tag = true;
            i += 1;
            continue;
        }
        if b == b'>' {
            in_tag = false;
            i += 1;
            continue;
        }
        if in_tag {
            i += 1;
            continue;
        }
        if b == b'&' {
            if let Some(end) = find_byte(bytes, i + 1, b';') {
                let entity = &bytes[i + 1..end];
                if let Some(decoded) = decode_entity(entity) {
                    out.push_str(&decoded);
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(b as char);
        i += 1;
    }
    collapse_whitespace(&out)
}

fn decode_entity(entity: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(entity).ok()?;
    match s {
        "amp" => Some("&".to_string()),
        "lt" => Some("<".to_string()),
        "gt" => Some(">".to_string()),
        "quot" => Some("\"".to_string()),
        "apos" => Some("'".to_string()),
        "nbsp" => Some("\u{00A0}".to_string()),
        "copy" => Some("\u{00A9}".to_string()),
        "reg" => Some("\u{00AE}".to_string()),
        s if s.starts_with('#') => {
            let body = &s[1..];
            let n: u32 = if let Some(hex) = body.strip_prefix(['x', 'X']) {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                body.parse().ok()?
            };
            char::from_u32(n).map(|c| c.to_string())
        }
        _ => None,
    }
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// HTML to markdown
// ---------------------------------------------------------------------------

/// Maps a small set of HTML tags back to Markdown. Lossy on unsupported tags
/// (their text content is preserved)
pub fn html_to_markdown(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if let Some(end) = find_byte(bytes, i + 1, b'>') {
                let tag = std::str::from_utf8(&bytes[i + 1..end]).unwrap_or("");
                let lowered = tag.trim().to_ascii_lowercase();
                if let Some(md) = map_tag_to_md(&lowered) {
                    out.push_str(&md);
                    i = end + 1;
                    continue;
                }
                // Unknown or self-closing tag, drop it but keep position
                i = end + 1;
                continue;
            }
        }
        if bytes[i] == b'&' {
            if let Some(end) = find_byte(bytes, i + 1, b';') {
                let entity = &bytes[i + 1..end];
                if let Some(decoded) = decode_entity(entity) {
                    out.push_str(&decoded);
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn map_tag_to_md(tag: &str) -> Option<String> {
    match tag {
        "h1" => Some("# ".to_string()),
        "h2" => Some("## ".to_string()),
        "h3" => Some("### ".to_string()),
        "h4" => Some("#### ".to_string()),
        "h5" => Some("##### ".to_string()),
        "h6" => Some("###### ".to_string()),
        "/h1" | "/h2" | "/h3" | "/h4" | "/h5" | "/h6" => Some("\n".to_string()),
        "p" => Some(String::new()),
        "/p" => Some("\n\n".to_string()),
        "br" | "br/" | "br /" => Some("\n".to_string()),
        "hr" | "hr/" | "hr /" => Some("\n---\n".to_string()),
        "strong" | "b" => Some("**".to_string()),
        "/strong" | "/b" => Some("**".to_string()),
        "em" | "i" => Some("*".to_string()),
        "/em" | "/i" => Some("*".to_string()),
        "code" => Some("`".to_string()),
        "/code" => Some("`".to_string()),
        "pre" => Some("```\n".to_string()),
        "/pre" => Some("\n```".to_string()),
        "ul" | "ol" => Some("\n".to_string()),
        "/ul" | "/ol" => Some("\n".to_string()),
        "li" => Some("- ".to_string()),
        "/li" => Some("\n".to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// sanitize_html
// ---------------------------------------------------------------------------

/// Returns html with disallowed tags stripped. Always strips <script>, <style>,
/// any tag whose first attribute name starts with on (event handlers), and any
/// href/src whose value uses javascript:, data:, or vbscript:
pub fn sanitize_html(html: &str, allowed_tags: &[&str]) -> String {
    let allowed_lower: Vec<String> = allowed_tags
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .collect();
    let mut out = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if let Some(end) = find_byte(bytes, i + 1, b'>') {
                let inner = std::str::from_utf8(&bytes[i + 1..end]).unwrap_or("");
                let trimmed = inner.trim();
                let is_close = trimmed.starts_with('/');
                let name_start = if is_close { 1 } else { 0 };
                let name_end = trimmed[name_start..]
                    .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
                    .map(|p| p + name_start)
                    .unwrap_or(trimmed.len());
                let name = trimmed[name_start..name_end].to_ascii_lowercase();
                if name == "script" || name == "style" {
                    // Skip entire element including nested content
                    if !is_close {
                        let close_marker = format!("</{}>", name);
                        if let Some(close_at) = find_subseq_ascii(
                            bytes,
                            end + 1,
                            close_marker.to_ascii_lowercase().as_bytes(),
                        ) {
                            i = close_at + close_marker.len();
                            continue;
                        }
                    }
                    i = end + 1;
                    continue;
                }
                if !allowed_lower.iter().any(|t| t == &name) {
                    // Drop the tag but keep the text content
                    i = end + 1;
                    continue;
                }
                // Strip event-handler attributes and dangerous URIs from
                // remaining attribute set
                let safe_attrs = strip_unsafe_attrs(&trimmed[name_end..]);
                out.push('<');
                if is_close {
                    out.push('/');
                }
                out.push_str(&name);
                if !safe_attrs.is_empty() {
                    out.push(' ');
                    out.push_str(&safe_attrs);
                }
                out.push('>');
                i = end + 1;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn find_subseq_ascii(haystack: &[u8], start: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || start >= haystack.len() {
        return None;
    }
    let mut i = start;
    while i + needle.len() <= haystack.len() {
        let mut ok = true;
        for j in 0..needle.len() {
            if haystack[i + j].to_ascii_lowercase() != needle[j] {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn strip_unsafe_attrs(attrs: &str) -> String {
    let mut out = String::new();
    let mut iter = attrs.split_whitespace().peekable();
    while let Some(attr) = iter.next() {
        let lower = attr.to_ascii_lowercase();
        if lower.starts_with("on") {
            // Event handler attribute, drop. If value follows on next token,
            // skip it too
            if !lower.contains('=') {
                let _ = iter.next();
            }
            continue;
        }
        if (lower.starts_with("href=") || lower.starts_with("src="))
            && (lower.contains("javascript:")
                || lower.contains("vbscript:")
                || lower.contains("data:"))
        {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(attr);
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h1_paragraph() {
        let html = markdown_to_html("# Hello\n\nParagraph");
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<p>Paragraph</p>"));
    }

    #[test]
    fn bold_and_italic() {
        let html = markdown_to_html("**bold** and *italic*");
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<em>italic</em>"));
    }

    #[test]
    fn code_span() {
        let html = markdown_to_html("`inline code`");
        assert!(html.contains("<code>inline code</code>"));
    }

    #[test]
    fn link_safe() {
        let html = markdown_to_html("[click](https://example.com)");
        assert!(html.contains("<a href=\"https://example.com\">click</a>"));
    }

    #[test]
    fn link_javascript_stripped() {
        let html = markdown_to_html("[bad](javascript:alert(1))");
        assert!(!html.contains("href"));
    }

    #[test]
    fn fenced_code_block() {
        let html = markdown_to_html("```rust\nfn main() {}\n```");
        assert!(html.contains("<pre><code class=\"language-rust\">"));
        assert!(html.contains("fn main()"));
    }

    #[test]
    fn unordered_list() {
        let html = markdown_to_html("- one\n- two\n- three");
        assert!(html.contains("<ul>"));
        assert!(html.contains("<li>one</li>"));
        assert!(html.contains("<li>three</li>"));
    }

    #[test]
    fn ordered_list_with_start() {
        let html = markdown_to_html("3. three\n4. four");
        assert!(html.contains("<ol start=\"3\">"));
        assert!(html.contains("<li>three</li>"));
    }

    #[test]
    fn task_list_gfm() {
        let html = markdown_to_html("- [x] done\n- [ ] todo");
        assert!(html.contains("checked"));
        assert!(html.matches("checkbox").count() == 2);
    }

    #[test]
    fn strikethrough_gfm() {
        let html = markdown_to_html("~~gone~~");
        assert!(html.contains("<del>gone</del>"));
    }

    #[test]
    fn table_gfm() {
        let md = "| a | b |\n|---|---|\n| 1 | 2 |";
        let html = markdown_to_html(md);
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>a</th>"));
        assert!(html.contains("<td>1</td>"));
    }

    #[test]
    fn extract_headers() {
        let h = markdown_extract_headers("# H1\n## H2\n### H3");
        assert_eq!(h, vec![(1, "H1".into()), (2, "H2".into()), (3, "H3".into())]);
    }

    #[test]
    fn extract_links() {
        let l = markdown_extract_links("[a](u1) and [b](u2)");
        assert_eq!(l.len(), 2);
        assert_eq!(l[0], ("a".into(), "u1".into()));
    }

    #[test]
    fn extract_code_blocks() {
        let c = markdown_extract_code_blocks("```rust\nlet x = 1;\n```\n```\nplain\n```");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].0, "rust");
        assert_eq!(c[1].0, "");
    }

    #[test]
    fn html_to_text_basic() {
        let t = html_to_text("<p>Hello <b>world</b></p>");
        assert_eq!(t, "Hello world");
    }

    #[test]
    fn html_to_text_entities() {
        let t = html_to_text("&amp; &lt; &gt; &#65;");
        assert_eq!(t, "& < > A");
    }

    #[test]
    fn html_to_markdown_basic() {
        let m = html_to_markdown("<h1>Title</h1><p>body</p>");
        assert!(m.contains("# Title"));
        assert!(m.contains("body"));
    }

    #[test]
    fn sanitize_strips_script() {
        let s = sanitize_html("<p>safe</p><script>alert(1)</script>", &["p"]);
        assert!(s.contains("<p>"));
        assert!(!s.contains("script"));
        assert!(!s.contains("alert"));
    }

    #[test]
    fn sanitize_drops_event_handlers() {
        let s = sanitize_html("<p onclick=\"x\">hi</p>", &["p"]);
        assert!(s.contains("<p>"));
        assert!(!s.to_lowercase().contains("onclick"));
    }
}
