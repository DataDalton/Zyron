//! String manipulation functions.
//!
//! Case conversions, slug generation, truncation, HTML stripping,
//! and pattern extraction (emails, URLs, phone numbers).
//! All operations are UTF-8 aware.

/// Capitalizes the first letter of each word.
pub fn initcap(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut capitalize_next = true;
    for c in text.chars() {
        if c.is_whitespace() || c == '-' || c == '_' {
            result.push(c);
            capitalize_next = true;
        } else if capitalize_next {
            for uc in c.to_uppercase() {
                result.push(uc);
            }
            capitalize_next = false;
        } else {
            for lc in c.to_lowercase() {
                result.push(lc);
            }
        }
    }
    result
}

/// Splits text into words by non-alphanumeric boundaries.
fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut prev_was_upper = false;
    let mut prev_was_lower = false;

    for c in text.chars() {
        if !c.is_alphanumeric() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            prev_was_upper = false;
            prev_was_lower = false;
        } else if c.is_uppercase() {
            // camelCase split: lowercase followed by uppercase starts a new word
            if prev_was_lower {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
            }
            current.push(c);
            prev_was_upper = true;
            prev_was_lower = false;
        } else {
            // Handle acronyms: "HTMLParser" -> "HTML" + "Parser"
            if prev_was_upper && current.len() > 1 && c.is_lowercase() {
                let last = current.pop().unwrap_or_default();
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                current.push(last);
            }
            current.push(c);
            prev_was_lower = c.is_lowercase();
            prev_was_upper = false;
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Converts text to camelCase.
pub fn camel_case(text: &str) -> String {
    let words = split_words(text);
    let mut result = String::with_capacity(text.len());
    for (i, word) in words.iter().enumerate() {
        if i == 0 {
            result.push_str(&word.to_lowercase());
        } else {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                for uc in first.to_uppercase() {
                    result.push(uc);
                }
                for c in chars {
                    for lc in c.to_lowercase() {
                        result.push(lc);
                    }
                }
            }
        }
    }
    result
}

/// Converts text to snake_case.
pub fn snake_case(text: &str) -> String {
    let words = split_words(text);
    words
        .iter()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join("_")
}

/// Converts text to kebab-case.
pub fn kebab_case(text: &str) -> String {
    let words = split_words(text);
    words
        .iter()
        .map(|w| w.to_lowercase())
        .collect::<Vec<_>>()
        .join("-")
}

/// Converts text to PascalCase.
pub fn pascal_case(text: &str) -> String {
    let words = split_words(text);
    let mut result = String::with_capacity(text.len());
    for word in &words {
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            for uc in first.to_uppercase() {
                result.push(uc);
            }
            for c in chars {
                for lc in c.to_lowercase() {
                    result.push(lc);
                }
            }
        }
    }
    result
}

/// English stop words for title case.
const TITLE_STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to", "by", "in", "of", "up",
    "as", "is", "if", "it", "so", "no", "not", "yet",
];

/// Converts text to Title Case with stop word handling.
/// First and last words are always capitalized. Stop words are lowercase
/// unless they start or end the string.
pub fn title_case(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }

    let mut parts = Vec::with_capacity(words.len());
    for (i, word) in words.iter().enumerate() {
        let lower = word.to_lowercase();
        if i == 0 || i == words.len() - 1 || !TITLE_STOP_WORDS.contains(&lower.as_str()) {
            parts.push(capitalize_first(word));
        } else {
            parts.push(lower);
        }
    }
    parts.join(" ")
}

fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let mut s = String::with_capacity(word.len());
            for uc in first.to_uppercase() {
                s.push(uc);
            }
            for c in chars {
                for lc in c.to_lowercase() {
                    s.push(lc);
                }
            }
            s
        }
    }
}

/// Generates a URL-safe slug: lowercase, hyphens, ASCII only.
/// Non-ASCII characters are transliterated where possible, otherwise removed.
pub fn slug(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_separator = true; // prevent leading hyphens

    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            result.push(c.to_ascii_lowercase());
            last_was_separator = false;
        } else if c.is_ascii() && !c.is_ascii_alphanumeric() {
            // Spaces, punctuation, etc -> hyphen (collapse multiple)
            if !last_was_separator && !result.is_empty() {
                result.push('-');
                last_was_separator = true;
            }
        } else {
            // Try basic transliteration for common accented characters
            let replacement = transliterate_char(c);
            if !replacement.is_empty() {
                for rc in replacement.chars() {
                    result.push(rc);
                }
                last_was_separator = false;
            }
        }
    }

    // Remove trailing hyphen
    if result.ends_with('-') {
        result.pop();
    }
    result
}

fn transliterate_char(c: char) -> &'static str {
    match c {
        '\u{00E0}' | '\u{00E1}' | '\u{00E2}' | '\u{00E3}' | '\u{00E4}' | '\u{00E5}' => "a",
        '\u{00C0}' | '\u{00C1}' | '\u{00C2}' | '\u{00C3}' | '\u{00C4}' | '\u{00C5}' => "a",
        '\u{00E8}' | '\u{00E9}' | '\u{00EA}' | '\u{00EB}' => "e",
        '\u{00C8}' | '\u{00C9}' | '\u{00CA}' | '\u{00CB}' => "e",
        '\u{00EC}' | '\u{00ED}' | '\u{00EE}' | '\u{00EF}' => "i",
        '\u{00CC}' | '\u{00CD}' | '\u{00CE}' | '\u{00CF}' => "i",
        '\u{00F2}' | '\u{00F3}' | '\u{00F4}' | '\u{00F5}' | '\u{00F6}' => "o",
        '\u{00D2}' | '\u{00D3}' | '\u{00D4}' | '\u{00D5}' | '\u{00D6}' => "o",
        '\u{00F9}' | '\u{00FA}' | '\u{00FB}' | '\u{00FC}' => "u",
        '\u{00D9}' | '\u{00DA}' | '\u{00DB}' | '\u{00DC}' => "u",
        '\u{00F1}' => "n",
        '\u{00D1}' => "n",
        '\u{00E7}' => "c",
        '\u{00C7}' => "c",
        '\u{00DF}' => "ss",
        '\u{00E6}' => "ae",
        '\u{00C6}' => "ae",
        '\u{00F8}' => "o",
        '\u{00D8}' => "o",
        _ => "",
    }
}

/// Truncates text at a word boundary to fit within `count` words.
pub fn truncate_words(text: &str, count: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= count {
        return text.to_string();
    }
    words[..count].join(" ")
}

/// Truncates text to `count` characters and appends the suffix (e.g., "...").
/// Counts unicode scalar values.
pub fn truncate_chars(text: &str, count: usize, suffix: &str) -> String {
    let char_count = text.chars().count();
    if char_count <= count {
        return text.to_string();
    }
    let truncated: String = text.chars().take(count).collect();
    format!("{}{}", truncated, suffix)
}

/// Removes HTML tags from a string, returning plain text.
/// Handles self-closing tags, attributes, nested tags, and HTML entities.
pub fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_entity = false;
    let mut entity = String::new();

    for c in html.chars() {
        if in_tag {
            if c == '>' {
                in_tag = false;
            }
        } else if in_entity {
            if c == ';' {
                let decoded = decode_entity(&entity);
                result.push_str(&decoded);
                entity.clear();
                in_entity = false;
            } else if entity.len() > 10 {
                // Entity too long, probably not a real entity
                result.push('&');
                result.push_str(&entity);
                result.push(c);
                entity.clear();
                in_entity = false;
            } else {
                entity.push(c);
            }
        } else if c == '<' {
            in_tag = true;
        } else if c == '&' {
            in_entity = true;
        } else {
            result.push(c);
        }
    }

    // Handle unterminated entity
    if in_entity {
        result.push('&');
        result.push_str(&entity);
    }

    result
}

fn decode_entity(entity: &str) -> String {
    match entity {
        "amp" => "&".to_string(),
        "lt" => "<".to_string(),
        "gt" => ">".to_string(),
        "quot" => "\"".to_string(),
        "apos" => "'".to_string(),
        "nbsp" => "\u{00A0}".to_string(),
        s if s.starts_with('#') => {
            let num_str = &s[1..];
            let code_point = if num_str.starts_with('x') || num_str.starts_with('X') {
                u32::from_str_radix(&num_str[1..], 16).ok()
            } else {
                num_str.parse::<u32>().ok()
            };
            code_point
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_else(|| format!("&{};", entity))
        }
        _ => format!("&{};", entity),
    }
}

/// Extracts email addresses from text using pattern matching.
pub fn extract_emails(text: &str) -> Vec<String> {
    let mut emails = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Find @ symbol
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }

        // Scan backwards for local part
        let mut local_start = i;
        while local_start > 0 {
            let c = bytes[local_start - 1];
            if c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'-' || c == b'+' {
                local_start -= 1;
            } else {
                break;
            }
        }

        // Scan forwards for domain part
        let mut domain_end = i + 1;
        let mut has_dot = false;
        while domain_end < len {
            let c = bytes[domain_end];
            if c.is_ascii_alphanumeric() || c == b'-' {
                domain_end += 1;
            } else if c == b'.'
                && domain_end + 1 < len
                && bytes[domain_end + 1].is_ascii_alphanumeric()
            {
                has_dot = true;
                domain_end += 1;
            } else {
                break;
            }
        }

        if local_start < i && domain_end > i + 1 && has_dot {
            let email = &text[local_start..domain_end];
            // Basic validation: local part not empty, domain has at least one dot
            if !email.starts_with('.') && !email.starts_with('@') {
                emails.push(email.to_string());
            }
        }

        i = domain_end;
    }

    emails
}

/// Extracts URLs from text (http://, https://, ftp://).
pub fn extract_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    let prefixes = ["https://", "http://", "ftp://"];

    for prefix in prefixes {
        let mut search_from = 0;
        while let Some(pos) = text[search_from..].find(prefix) {
            let start = search_from + pos;
            let mut end = start + prefix.len();

            // Extend URL until whitespace or certain terminating characters
            while end < text.len() {
                let c = text.as_bytes()[end];
                if c.is_ascii_whitespace()
                    || c == b'"'
                    || c == b'\''
                    || c == b'<'
                    || c == b'>'
                    || c == b')'
                    || c == b']'
                {
                    break;
                }
                end += 1;
            }

            // Remove trailing punctuation
            while end > start + prefix.len() {
                let last = text.as_bytes()[end - 1];
                if last == b'.' || last == b',' || last == b';' || last == b'!' || last == b'?' {
                    end -= 1;
                } else {
                    break;
                }
            }

            if end > start + prefix.len() {
                urls.push(text[start..end].to_string());
            }

            search_from = end;
        }
    }

    urls
}

/// Extracts phone numbers from text. Matches common formats:
/// +1-234-567-8901, (234) 567-8901, 234.567.8901, 234-567-8901
pub fn extract_phone_numbers(text: &str) -> Vec<String> {
    let mut phones = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Look for start of phone number: +, (, or digit
        let c = bytes[i];
        if c != b'+' && c != b'(' && !c.is_ascii_digit() {
            i += 1;
            continue;
        }

        let start = i;
        let mut digit_count = 0;
        let mut j = i;

        while j < len {
            let b = bytes[j];
            if b.is_ascii_digit() {
                digit_count += 1;
                j += 1;
            } else if b == b'+' || b == b'-' || b == b'.' || b == b' ' || b == b'(' || b == b')' {
                j += 1;
            } else {
                break;
            }
        }

        // Phone numbers typically have 7-15 digits
        if digit_count >= 7 && digit_count <= 15 {
            let candidate = text[start..j].trim();
            if !candidate.is_empty() {
                phones.push(candidate.to_string());
            }
        }

        i = j.max(i + 1);
    }

    phones
}

#[cfg(test)]
mod tests {
    use super::*;

    // initcap
    #[test]
    fn test_initcap() {
        assert_eq!(initcap("hello world"), "Hello World");
        assert_eq!(initcap("HELLO WORLD"), "Hello World");
        assert_eq!(initcap("hello-world"), "Hello-World");
    }

    #[test]
    fn test_initcap_empty() {
        assert_eq!(initcap(""), "");
    }

    // camelCase
    #[test]
    fn test_camel_case() {
        assert_eq!(camel_case("hello world"), "helloWorld");
        assert_eq!(camel_case("Hello World"), "helloWorld");
        assert_eq!(camel_case("some_variable_name"), "someVariableName");
        assert_eq!(camel_case("kebab-case-name"), "kebabCaseName");
    }

    // snake_case
    #[test]
    fn test_snake_case() {
        assert_eq!(snake_case("Hello World"), "hello_world");
        assert_eq!(snake_case("camelCase"), "camel_case");
        assert_eq!(snake_case("PascalCase"), "pascal_case");
        assert_eq!(snake_case("kebab-case"), "kebab_case");
    }

    // kebab-case
    #[test]
    fn test_kebab_case() {
        assert_eq!(kebab_case("Hello World"), "hello-world");
        assert_eq!(kebab_case("camelCase"), "camel-case");
        assert_eq!(kebab_case("snake_case_name"), "snake-case-name");
    }

    // PascalCase
    #[test]
    fn test_pascal_case() {
        assert_eq!(pascal_case("hello world"), "HelloWorld");
        assert_eq!(pascal_case("some_variable"), "SomeVariable");
        assert_eq!(pascal_case("camelCase"), "CamelCase");
    }

    // Title Case
    #[test]
    fn test_title_case() {
        assert_eq!(title_case("the quick brown fox"), "The Quick Brown Fox");
    }

    #[test]
    fn test_title_case_stop_words() {
        assert_eq!(title_case("war and peace"), "War and Peace");
    }

    #[test]
    fn test_title_case_first_last_capitalized() {
        assert_eq!(title_case("in the beginning"), "In the Beginning");
    }

    // slug
    #[test]
    fn test_slug_basic() {
        assert_eq!(slug("Hello, World!"), "hello-world");
    }

    #[test]
    fn test_slug_unicode() {
        assert_eq!(slug("caf\u{00E9} latt\u{00E9}"), "cafe-latte");
    }

    #[test]
    fn test_slug_multiple_spaces() {
        assert_eq!(slug("hello   world"), "hello-world");
    }

    #[test]
    fn test_slug_leading_trailing() {
        assert_eq!(slug("  hello  "), "hello");
    }

    #[test]
    fn test_slug_german() {
        assert_eq!(slug("Stra\u{00DF}e"), "strasse");
    }

    // truncate_words
    #[test]
    fn test_truncate_words() {
        assert_eq!(truncate_words("one two three four", 2), "one two");
    }

    #[test]
    fn test_truncate_words_short() {
        assert_eq!(truncate_words("hello", 5), "hello");
    }

    // truncate_chars
    #[test]
    fn test_truncate_chars() {
        assert_eq!(truncate_chars("Hello, World!", 5, "..."), "Hello...");
    }

    #[test]
    fn test_truncate_chars_no_truncation() {
        assert_eq!(truncate_chars("Hi", 10, "..."), "Hi");
    }

    #[test]
    fn test_truncate_chars_unicode() {
        assert_eq!(
            truncate_chars("caf\u{00E9} latt\u{00E9}", 4, "..."),
            "caf\u{00E9}..."
        );
    }

    // strip_html
    #[test]
    fn test_strip_html_basic() {
        assert_eq!(strip_html("<p>Hello</p>"), "Hello");
    }

    #[test]
    fn test_strip_html_nested() {
        assert_eq!(
            strip_html("<div><p>Hello <b>World</b></p></div>"),
            "Hello World"
        );
    }

    #[test]
    fn test_strip_html_entities() {
        assert_eq!(strip_html("A &amp; B"), "A & B");
        assert_eq!(strip_html("&lt;tag&gt;"), "<tag>");
    }

    #[test]
    fn test_strip_html_numeric_entity() {
        assert_eq!(strip_html("&#65;"), "A");
        assert_eq!(strip_html("&#x41;"), "A");
    }

    #[test]
    fn test_strip_html_no_tags() {
        assert_eq!(strip_html("plain text"), "plain text");
    }

    #[test]
    fn test_strip_html_self_closing() {
        assert_eq!(strip_html("line1<br/>line2"), "line1line2");
    }

    // extract_emails
    #[test]
    fn test_extract_emails() {
        let emails = extract_emails("Contact us at info@example.com or support@test.org");
        assert_eq!(emails.len(), 2);
        assert!(emails.contains(&"info@example.com".to_string()));
        assert!(emails.contains(&"support@test.org".to_string()));
    }

    #[test]
    fn test_extract_emails_with_plus() {
        let emails = extract_emails("user+tag@gmail.com");
        assert_eq!(emails, vec!["user+tag@gmail.com"]);
    }

    #[test]
    fn test_extract_emails_none() {
        assert!(extract_emails("no emails here").is_empty());
    }

    // extract_urls
    #[test]
    fn test_extract_urls() {
        let urls = extract_urls("Visit https://example.com or http://test.org/page");
        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"https://example.com".to_string()));
        assert!(urls.contains(&"http://test.org/page".to_string()));
    }

    #[test]
    fn test_extract_urls_with_path() {
        let urls = extract_urls("See https://example.com/path/to/page?q=1&b=2");
        assert_eq!(urls.len(), 1);
        assert!(urls[0].contains("path/to/page"));
    }

    #[test]
    fn test_extract_urls_trailing_punctuation() {
        let urls = extract_urls("Visit https://example.com.");
        assert_eq!(urls, vec!["https://example.com"]);
    }

    // extract_phone_numbers
    #[test]
    fn test_extract_phone_numbers() {
        let phones = extract_phone_numbers("Call +1-234-567-8901 or (234) 567-8901");
        assert!(phones.len() >= 2);
    }

    #[test]
    fn test_extract_phone_none() {
        assert!(extract_phone_numbers("no phone here").is_empty());
    }

    // Edge cases
    #[test]
    fn test_split_words_camel() {
        let words = split_words("camelCaseWord");
        assert_eq!(words, vec!["camel", "Case", "Word"]);
    }

    #[test]
    fn test_split_words_acronym() {
        let words = split_words("HTMLParser");
        assert_eq!(words, vec!["HTML", "Parser"]);
    }

    #[test]
    fn test_split_words_mixed() {
        let words = split_words("hello_world-test");
        assert_eq!(words, vec!["hello", "world", "test"]);
    }
}
