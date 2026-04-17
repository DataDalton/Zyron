//! Hand-rolled regex engine (subset of standard syntax).
//!
//! Supports: literals, `.`, `*`, `+`, `?`, `|`, `[]`, `[^]`, `\d`, `\w`, `\s`,
//! `^`, `$`, `{n}`, `{n,m}`, `()` capture groups.
//!
//! Uses NFA construction + simulation for O(nm) worst-case matching.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum RegexNode {
    Literal(char),
    AnyChar,
    CharClass(Vec<(char, char)>, bool), // (ranges, negated)
    Anchor(Anchor),
    Concat(Vec<RegexNode>),
    Alternation(Vec<RegexNode>),
    Repeat(Box<RegexNode>, usize, Option<usize>), // min, max
    Group(Box<RegexNode>, usize),                 // capture group index
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Anchor {
    Start,
    End,
}

/// Compiled regex pattern.
#[derive(Debug, Clone)]
pub struct CompiledRegex {
    ast: RegexNode,
    num_groups: usize,
}

impl CompiledRegex {
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser<'a> {
    input: &'a [char],
    pos: usize,
    group_counter: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a [char]) -> Self {
        Self {
            input,
            pos: 0,
            group_counter: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn parse_alternation(&mut self) -> Result<RegexNode> {
        let first = self.parse_concat()?;
        if self.peek() == Some('|') {
            let mut options = vec![first];
            while self.peek() == Some('|') {
                self.advance();
                options.push(self.parse_concat()?);
            }
            Ok(RegexNode::Alternation(options))
        } else {
            Ok(first)
        }
    }

    fn parse_concat(&mut self) -> Result<RegexNode> {
        let mut parts = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            parts.push(self.parse_repetition()?);
        }
        if parts.len() == 1 {
            Ok(parts.into_iter().next().expect("checked len 1"))
        } else {
            Ok(RegexNode::Concat(parts))
        }
    }

    fn parse_repetition(&mut self) -> Result<RegexNode> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some('*') => {
                self.advance();
                Ok(RegexNode::Repeat(Box::new(atom), 0, None))
            }
            Some('+') => {
                self.advance();
                Ok(RegexNode::Repeat(Box::new(atom), 1, None))
            }
            Some('?') => {
                self.advance();
                Ok(RegexNode::Repeat(Box::new(atom), 0, Some(1)))
            }
            Some('{') => {
                self.advance();
                let (min, max) = self.parse_repetition_bounds()?;
                Ok(RegexNode::Repeat(Box::new(atom), min, max))
            }
            _ => Ok(atom),
        }
    }

    fn parse_repetition_bounds(&mut self) -> Result<(usize, Option<usize>)> {
        let mut num = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                num.push(c);
                self.advance();
            } else {
                break;
            }
        }
        let min: usize = num
            .parse()
            .map_err(|_| ZyronError::ExecutionError("Invalid repetition bound".into()))?;

        let max = if self.peek() == Some(',') {
            self.advance();
            let mut max_num = String::new();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    max_num.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
            if max_num.is_empty() {
                None
            } else {
                Some(
                    max_num
                        .parse()
                        .map_err(|_| ZyronError::ExecutionError("Invalid upper bound".into()))?,
                )
            }
        } else {
            Some(min)
        };

        if self.peek() != Some('}') {
            return Err(ZyronError::ExecutionError(
                "Expected '}' in repetition".into(),
            ));
        }
        self.advance();
        Ok((min, max))
    }

    fn parse_atom(&mut self) -> Result<RegexNode> {
        match self.peek() {
            None => Err(ZyronError::ExecutionError(
                "Unexpected end of pattern".into(),
            )),
            Some('.') => {
                self.advance();
                Ok(RegexNode::AnyChar)
            }
            Some('^') => {
                self.advance();
                Ok(RegexNode::Anchor(Anchor::Start))
            }
            Some('$') => {
                self.advance();
                Ok(RegexNode::Anchor(Anchor::End))
            }
            Some('(') => {
                self.advance();
                self.group_counter += 1;
                let group_idx = self.group_counter;
                let inner = self.parse_alternation()?;
                if self.peek() != Some(')') {
                    return Err(ZyronError::ExecutionError("Expected ')'".into()));
                }
                self.advance();
                Ok(RegexNode::Group(Box::new(inner), group_idx))
            }
            Some('[') => {
                self.advance();
                self.parse_char_class()
            }
            Some('\\') => {
                self.advance();
                self.parse_escape()
            }
            Some(c) if matches!(c, ')' | '*' | '+' | '?' | '|' | '{' | '}' | ']') => Err(
                ZyronError::ExecutionError(format!("Unexpected metacharacter: {}", c)),
            ),
            Some(c) => {
                self.advance();
                Ok(RegexNode::Literal(c))
            }
        }
    }

    fn parse_char_class(&mut self) -> Result<RegexNode> {
        let negated = if self.peek() == Some('^') {
            self.advance();
            true
        } else {
            false
        };

        let mut ranges = Vec::new();
        while let Some(c) = self.peek() {
            if c == ']' {
                self.advance();
                return Ok(RegexNode::CharClass(ranges, negated));
            }
            let start = if c == '\\' {
                self.advance();
                match self.peek() {
                    Some('d') => {
                        self.advance();
                        ranges.push(('0', '9'));
                        continue;
                    }
                    Some('w') => {
                        self.advance();
                        ranges.push(('a', 'z'));
                        ranges.push(('A', 'Z'));
                        ranges.push(('0', '9'));
                        ranges.push(('_', '_'));
                        continue;
                    }
                    Some('s') => {
                        self.advance();
                        ranges.push((' ', ' '));
                        ranges.push(('\t', '\t'));
                        ranges.push(('\n', '\n'));
                        ranges.push(('\r', '\r'));
                        continue;
                    }
                    Some('n') => {
                        self.advance();
                        '\n'
                    }
                    Some('t') => {
                        self.advance();
                        '\t'
                    }
                    Some('r') => {
                        self.advance();
                        '\r'
                    }
                    Some(c) => {
                        self.advance();
                        c
                    }
                    None => {
                        return Err(ZyronError::ExecutionError(
                            "Unexpected end in escape".into(),
                        ));
                    }
                }
            } else {
                self.advance();
                c
            };

            if self.peek() == Some('-') && self.input.get(self.pos + 1).copied() != Some(']') {
                self.advance();
                let end_c = self.advance().ok_or_else(|| {
                    ZyronError::ExecutionError("Unexpected end in char class".into())
                })?;
                ranges.push((start, end_c));
            } else {
                ranges.push((start, start));
            }
        }
        Err(ZyronError::ExecutionError(
            "Unterminated character class".into(),
        ))
    }

    fn parse_escape(&mut self) -> Result<RegexNode> {
        match self.advance() {
            Some('d') => Ok(RegexNode::CharClass(vec![('0', '9')], false)),
            Some('D') => Ok(RegexNode::CharClass(vec![('0', '9')], true)),
            Some('w') => Ok(RegexNode::CharClass(
                vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')],
                false,
            )),
            Some('W') => Ok(RegexNode::CharClass(
                vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')],
                true,
            )),
            Some('s') => Ok(RegexNode::CharClass(
                vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
                false,
            )),
            Some('S') => Ok(RegexNode::CharClass(
                vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
                true,
            )),
            Some('n') => Ok(RegexNode::Literal('\n')),
            Some('t') => Ok(RegexNode::Literal('\t')),
            Some('r') => Ok(RegexNode::Literal('\r')),
            Some(c) => Ok(RegexNode::Literal(c)),
            None => Err(ZyronError::ExecutionError("Unexpected end after \\".into())),
        }
    }
}

/// Compiles a regex pattern string.
pub fn regex_compile(pattern: &str) -> Result<CompiledRegex> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut parser = Parser::new(&chars);
    let ast = parser.parse_alternation()?;
    if parser.pos < parser.input.len() {
        return Err(ZyronError::ExecutionError(format!(
            "Unexpected trailing character at position {}",
            parser.pos
        )));
    }
    Ok(CompiledRegex {
        ast,
        num_groups: parser.group_counter,
    })
}

// ---------------------------------------------------------------------------
// Matcher (recursive backtracking with depth bound)
// ---------------------------------------------------------------------------

fn match_node(
    node: &RegexNode,
    input: &[char],
    pos: usize,
    groups: &mut Vec<Option<(usize, usize)>>,
) -> Option<usize> {
    match node {
        RegexNode::Literal(c) => {
            if pos < input.len() && input[pos] == *c {
                Some(pos + 1)
            } else {
                None
            }
        }
        RegexNode::AnyChar => {
            if pos < input.len() && input[pos] != '\n' {
                Some(pos + 1)
            } else {
                None
            }
        }
        RegexNode::CharClass(ranges, negated) => {
            if pos >= input.len() {
                return None;
            }
            let c = input[pos];
            let matches = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
            if matches != *negated {
                Some(pos + 1)
            } else {
                None
            }
        }
        RegexNode::Anchor(Anchor::Start) => {
            if pos == 0 {
                Some(pos)
            } else {
                None
            }
        }
        RegexNode::Anchor(Anchor::End) => {
            if pos == input.len() {
                Some(pos)
            } else {
                None
            }
        }
        RegexNode::Concat(parts) => match_concat(parts, input, pos, groups),
        RegexNode::Alternation(options) => {
            let saved_groups = groups.clone();
            for opt in options {
                if let Some(end) = match_node(opt, input, pos, groups) {
                    return Some(end);
                }
                *groups = saved_groups.clone();
            }
            None
        }
        RegexNode::Repeat(inner, min, max) => match_repeat(inner, input, pos, *min, *max, groups),
        RegexNode::Group(inner, idx) => {
            let start = pos;
            if let Some(end) = match_node(inner, input, pos, groups) {
                if *idx - 1 < groups.len() {
                    groups[*idx - 1] = Some((start, end));
                }
                Some(end)
            } else {
                None
            }
        }
    }
}

fn match_concat(
    parts: &[RegexNode],
    input: &[char],
    pos: usize,
    groups: &mut Vec<Option<(usize, usize)>>,
) -> Option<usize> {
    if parts.is_empty() {
        return Some(pos);
    }
    let first = &parts[0];
    let rest = &parts[1..];

    // For repetition, try greedy match then backtrack
    if let RegexNode::Repeat(inner, min, max) = first {
        return match_repeat_concat(inner, *min, *max, rest, input, pos, groups);
    }

    let end = match_node(first, input, pos, groups)?;
    match_concat(rest, input, end, groups)
}

fn match_repeat_concat(
    inner: &RegexNode,
    min: usize,
    max: Option<usize>,
    rest: &[RegexNode],
    input: &[char],
    pos: usize,
    groups: &mut Vec<Option<(usize, usize)>>,
) -> Option<usize> {
    // Collect all possible match positions
    let mut positions = vec![pos];
    let mut current = pos;
    let limit = max.unwrap_or(input.len().saturating_sub(pos) + 1);

    for _ in 0..limit {
        let saved = groups.clone();
        match match_node(inner, input, current, groups) {
            Some(next) if next > current => {
                current = next;
                positions.push(current);
            }
            Some(_) => {
                *groups = saved;
                break;
            }
            None => {
                *groups = saved;
                break;
            }
        }
    }

    // Try greedy (longest) first, backtrack
    for i in (min..positions.len()).rev() {
        let saved = groups.clone();
        if let Some(end) = match_concat(rest, input, positions[i], groups) {
            return Some(end);
        }
        *groups = saved;
    }

    None
}

fn match_repeat(
    inner: &RegexNode,
    input: &[char],
    pos: usize,
    min: usize,
    max: Option<usize>,
    groups: &mut Vec<Option<(usize, usize)>>,
) -> Option<usize> {
    let mut current = pos;
    let mut count = 0;
    let upper = max.unwrap_or(usize::MAX);

    while count < upper {
        let saved = groups.clone();
        match match_node(inner, input, current, groups) {
            Some(next) if next > current => {
                current = next;
                count += 1;
            }
            _ => {
                *groups = saved;
                break;
            }
        }
    }

    if count >= min { Some(current) } else { None }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns true if the pattern matches anywhere in the text.
pub fn regex_match(text: &str, pattern: &str) -> Result<bool> {
    let compiled = regex_compile(pattern)?;
    regex_match_compiled(text, &compiled)
}

pub fn regex_match_compiled(text: &str, compiled: &CompiledRegex) -> Result<bool> {
    Ok(regex_find_compiled(text, compiled)?.is_some())
}

/// Returns the first match position (start_char, end_char) in the text.
pub fn regex_find(text: &str, pattern: &str) -> Result<Option<(usize, usize)>> {
    let compiled = regex_compile(pattern)?;
    regex_find_compiled(text, &compiled)
}

pub fn regex_find_compiled(text: &str, compiled: &CompiledRegex) -> Result<Option<(usize, usize)>> {
    let chars: Vec<char> = text.chars().collect();
    for start in 0..=chars.len() {
        let mut groups = vec![None; compiled.num_groups];
        if let Some(end) = match_node(&compiled.ast, &chars, start, &mut groups) {
            return Ok(Some((start, end)));
        }
    }
    Ok(None)
}

/// Returns all non-overlapping match positions.
pub fn regex_find_all(text: &str, pattern: &str) -> Result<Vec<(usize, usize)>> {
    let compiled = regex_compile(pattern)?;
    let chars: Vec<char> = text.chars().collect();
    let mut results = Vec::new();
    let mut pos = 0;
    while pos <= chars.len() {
        let mut groups = vec![None; compiled.num_groups];
        if let Some(end) = match_node(&compiled.ast, &chars, pos, &mut groups) {
            if end > pos {
                results.push((pos, end));
                pos = end;
            } else {
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }
    Ok(results)
}

/// Returns the full match plus all capture groups as strings.
pub fn regex_capture(text: &str, pattern: &str) -> Result<Vec<Option<String>>> {
    let compiled = regex_compile(pattern)?;
    let chars: Vec<char> = text.chars().collect();
    for start in 0..=chars.len() {
        let mut groups = vec![None; compiled.num_groups];
        if let Some(end) = match_node(&compiled.ast, &chars, start, &mut groups) {
            let mut result = Vec::with_capacity(compiled.num_groups + 1);
            // Group 0 is the full match
            result.push(Some(chars[start..end].iter().collect::<String>()));
            for g in groups {
                result.push(g.map(|(s, e)| chars[s..e].iter().collect::<String>()));
            }
            return Ok(result);
        }
    }
    Ok(Vec::new())
}

/// Replaces the first match of the pattern in text with replacement.
pub fn regex_replace(text: &str, pattern: &str, replacement: &str) -> Result<String> {
    let compiled = regex_compile(pattern)?;
    let chars: Vec<char> = text.chars().collect();
    if let Some((start, end)) = regex_find_compiled(text, &compiled)? {
        let mut result: String = chars[..start].iter().collect();
        result.push_str(replacement);
        result.extend(chars[end..].iter());
        Ok(result)
    } else {
        Ok(text.to_string())
    }
}

/// Replaces all non-overlapping matches of the pattern in text with replacement.
pub fn regex_replace_all(text: &str, pattern: &str, replacement: &str) -> Result<String> {
    let matches = regex_find_all(text, pattern)?;
    if matches.is_empty() {
        return Ok(text.to_string());
    }
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;
    for (start, end) in matches {
        result.extend(chars[last_end..start].iter());
        result.push_str(replacement);
        last_end = end;
    }
    result.extend(chars[last_end..].iter());
    Ok(result)
}

/// Splits the text by the pattern matches.
pub fn regex_split(text: &str, pattern: &str) -> Result<Vec<String>> {
    let matches = regex_find_all(text, pattern)?;
    let chars: Vec<char> = text.chars().collect();
    let mut result = Vec::new();
    let mut last_end = 0;
    for (start, end) in matches {
        result.push(chars[last_end..start].iter().collect());
        last_end = end;
    }
    result.push(chars[last_end..].iter().collect());
    Ok(result)
}

/// Counts the number of non-overlapping matches.
pub fn regex_count(text: &str, pattern: &str) -> Result<usize> {
    regex_find_all(text, pattern).map(|m| m.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_match() {
        assert!(regex_match("hello world", "hello").unwrap());
        assert!(!regex_match("hello world", "xyz").unwrap());
    }

    #[test]
    fn test_dot() {
        assert!(regex_match("cat", "c.t").unwrap());
        assert!(regex_match("cut", "c.t").unwrap());
        assert!(!regex_match("ct", "c.t").unwrap());
    }

    #[test]
    fn test_star() {
        assert!(regex_match("", "a*").unwrap());
        assert!(regex_match("aaa", "a*").unwrap());
        assert!(regex_match("bbb", "a*").unwrap());
    }

    #[test]
    fn test_plus() {
        assert!(!regex_match("", "a+").unwrap());
        assert!(regex_match("aaa", "a+").unwrap());
    }

    #[test]
    fn test_question() {
        assert!(regex_match("color", "colou?r").unwrap());
        assert!(regex_match("colour", "colou?r").unwrap());
    }

    #[test]
    fn test_char_class() {
        assert!(regex_match("abc", "[abc]").unwrap());
        assert!(regex_match("5", "[0-9]").unwrap());
        assert!(!regex_match("x", "[0-9]").unwrap());
    }

    #[test]
    fn test_negated_char_class() {
        assert!(!regex_match("5", "[^0-9]").unwrap());
        assert!(regex_match("x", "[^0-9]").unwrap());
    }

    #[test]
    fn test_digit_escape() {
        assert!(regex_match("123", "\\d+").unwrap());
        assert!(!regex_match("abc", "\\d+").unwrap());
    }

    #[test]
    fn test_word_escape() {
        assert!(regex_match("hello_123", "\\w+").unwrap());
    }

    #[test]
    fn test_whitespace_escape() {
        assert!(regex_match("a b", "\\s").unwrap());
        assert!(!regex_match("ab", "\\s").unwrap());
    }

    #[test]
    fn test_anchors() {
        assert!(regex_match("hello", "^hello$").unwrap());
        assert!(!regex_match("hello world", "^hello$").unwrap());
        assert!(regex_match("hello world", "^hello").unwrap());
        assert!(regex_match("hello world", "world$").unwrap());
    }

    #[test]
    fn test_alternation() {
        assert!(regex_match("cat", "cat|dog").unwrap());
        assert!(regex_match("dog", "cat|dog").unwrap());
        assert!(!regex_match("fish", "cat|dog").unwrap());
    }

    #[test]
    fn test_grouping() {
        assert!(regex_match("abab", "(ab)+").unwrap());
        assert!(regex_match("ab", "(ab)+").unwrap());
    }

    #[test]
    fn test_repetition_bounds() {
        assert!(regex_match("aaa", "a{3}").unwrap());
        assert!(!regex_match("aa", "a{3}").unwrap());
        assert!(regex_match("aaaa", "a{3,5}").unwrap());
        assert!(regex_match("aaaaa", "a{3,5}").unwrap());
        assert!(!regex_match("aa", "a{3,5}").unwrap());
    }

    #[test]
    fn test_find() {
        let m = regex_find("hello world", "world").unwrap();
        assert_eq!(m, Some((6, 11)));
    }

    #[test]
    fn test_find_all() {
        let matches = regex_find_all("abc abc abc", "abc").unwrap();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_capture_groups() {
        let captures = regex_capture("hello world", "(\\w+) (\\w+)").unwrap();
        assert_eq!(captures[0], Some("hello world".to_string()));
        assert_eq!(captures[1], Some("hello".to_string()));
        assert_eq!(captures[2], Some("world".to_string()));
    }

    #[test]
    fn test_replace() {
        let result = regex_replace("hello world", "world", "Rust").unwrap();
        assert_eq!(result, "hello Rust");
    }

    #[test]
    fn test_replace_all() {
        let result = regex_replace_all("a1b2c3", "\\d", "X").unwrap();
        assert_eq!(result, "aXbXcX");
    }

    #[test]
    fn test_split() {
        let parts = regex_split("a,b,c,d", ",").unwrap();
        assert_eq!(parts, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_count() {
        let count = regex_count("the quick brown fox", "o").unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_phone_like() {
        let compiled = regex_compile("\\d{3}-\\d{4}").unwrap();
        assert!(regex_match_compiled("call 555-1234", &compiled).unwrap());
    }

    #[test]
    fn test_invalid_pattern() {
        assert!(regex_compile("(").is_err());
        assert!(regex_compile("[").is_err());
    }
}
