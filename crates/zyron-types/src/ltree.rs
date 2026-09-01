//! LTREE hierarchical label path functions
//!
//! An ltree value is a dotted label path like Top.Science.Astronomy
//! Labels are 1 to 255 chars of [A-Za-z0-9_], a path holds 1 to 65535 labels
//! Every function validates its inputs and reports malformed values through
//! ZyronError::InvalidParameter

use zyron_common::{Result, ZyronError};

const MAX_LABEL_LEN: usize = 255;
const MAX_LABELS: usize = 65535;

fn invalid(name: &str, value: &str) -> ZyronError {
    ZyronError::InvalidParameter {
        name: name.to_string(),
        value: value.to_string(),
    }
}

fn valid_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= MAX_LABEL_LEN
        && label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Validates an ltree path and splits it into labels
pub fn parse_path(path: &str) -> Result<Vec<&str>> {
    if path.is_empty() {
        return Err(invalid("path", path));
    }
    let labels: Vec<&str> = path.split('.').collect();
    if labels.len() > MAX_LABELS {
        return Err(invalid("path", path));
    }
    for label in &labels {
        if !valid_label(label) {
            return Err(invalid("path", path));
        }
    }
    Ok(labels)
}

/// Returns the number of labels in the path
pub fn nlevel(path: &str) -> Result<i64> {
    Ok(parse_path(path)?.len() as i64)
}

/// Returns the subpath starting at offset with the given length
/// A negative offset counts from the end of the path, a negative len drops
/// that many labels from the end, out of range positions error
pub fn subpath(path: &str, offset: i64, len: Option<i64>) -> Result<String> {
    let labels = parse_path(path)?;
    let n = labels.len() as i64;
    let start = if offset < 0 { n + offset } else { offset };
    if start < 0 || start >= n {
        return Err(invalid("offset", &offset.to_string()));
    }
    let end = match len {
        None => n,
        Some(l) if l < 0 => n + l,
        Some(l) => (start + l).min(n),
    };
    if end <= start {
        let shown = match len {
            Some(l) => l.to_string(),
            None => String::new(),
        };
        return Err(invalid("len", &shown));
    }
    Ok(labels[start as usize..end as usize].join("."))
}

/// Returns the position of the first occurrence of subpath as a consecutive
/// label run inside path, or -1 when absent
pub fn index_of(path: &str, subpath: &str) -> Result<i64> {
    let labels = parse_path(path)?;
    let sub = parse_path(subpath)?;
    if sub.len() > labels.len() {
        return Ok(-1);
    }
    for start in 0..=(labels.len() - sub.len()) {
        if labels[start..start + sub.len()] == sub[..] {
            return Ok(start as i64);
        }
    }
    Ok(-1)
}

/// Returns the longest common ancestor of the given paths, or None when no
/// ancestor is shared. The ancestor is a proper prefix of every path, so
/// identical single label paths share nothing
pub fn lca(paths: &[&str]) -> Result<Option<String>> {
    if paths.is_empty() {
        return Err(invalid("paths", ""));
    }
    let parsed: Vec<Vec<&str>> = paths.iter().map(|p| parse_path(p)).collect::<Result<_>>()?;
    let limit = parsed.iter().map(|p| p.len() - 1).min().unwrap_or(0);
    let first = &parsed[0];
    let mut count = 0;
    while count < limit && parsed.iter().all(|p| p[count] == first[count]) {
        count += 1;
    }
    if count == 0 {
        Ok(None)
    } else {
        Ok(Some(first[..count].join(".")))
    }
}

/// Joins labels into a path, validating each label
pub fn build_path(labels: &[&str]) -> Result<String> {
    if labels.is_empty() || labels.len() > MAX_LABELS {
        return Err(invalid("labels", &labels.join(".")));
    }
    for label in labels {
        if !valid_label(label) {
            return Err(invalid("labels", label));
        }
    }
    Ok(labels.join("."))
}

/// Returns true when ancestor is a label prefix of descendant
/// Equal paths count as ancestors, matching the postgres @> operator
pub fn is_ancestor(ancestor: &str, descendant: &str) -> Result<bool> {
    let anc = parse_path(ancestor)?;
    let desc = parse_path(descendant)?;
    Ok(anc.len() <= desc.len() && anc[..] == desc[..anc.len()])
}

/// Returns true when descendant has ancestor as a label prefix
pub fn is_descendant(descendant: &str, ancestor: &str) -> Result<bool> {
    is_ancestor(ancestor, descendant)
}

/// Returns true when the path matches the lquery pattern
pub fn matches_lquery(path: &str, lquery: &str) -> Result<bool> {
    let labels = parse_path(path)?;
    let items = parse_lquery(lquery)?;
    Ok(matches_items(&items, &labels))
}

/// Returns true when the path matches any of the lquery patterns
/// Every pattern is validated before matching begins
pub fn matches_any_lquery(path: &str, lqueries: &[&str]) -> Result<bool> {
    let labels = parse_path(path)?;
    let parsed: Vec<Vec<LqueryItem>> = lqueries
        .iter()
        .map(|q| parse_lquery(q))
        .collect::<Result<_>>()?;
    Ok(parsed.iter().any(|items| matches_items(items, &labels)))
}

// ---------------------------------------------------------------------------
// lquery
// ---------------------------------------------------------------------------

enum LqueryLabel {
    // % matches any single label in its position
    Any,
    Exact {
        text: String,
        case_insensitive: bool,
    },
}

enum LqueryItem {
    // * with bounds, max None means unbounded
    Star {
        min: usize,
        max: Option<usize>,
    },
    // one label position, alternatives joined by |, negated inverts the whole set
    Position {
        negated: bool,
        alternatives: Vec<LqueryLabel>,
    },
}

fn parse_lquery(lquery: &str) -> Result<Vec<LqueryItem>> {
    if lquery.is_empty() {
        return Err(invalid("lquery", lquery));
    }
    let mut items = Vec::new();
    for raw in lquery.split('.') {
        items.push(parse_lquery_item(raw, lquery)?);
    }
    if items.len() > MAX_LABELS {
        return Err(invalid("lquery", lquery));
    }
    Ok(items)
}

fn parse_lquery_item(raw: &str, lquery: &str) -> Result<LqueryItem> {
    if raw.is_empty() {
        return Err(invalid("lquery", lquery));
    }
    if let Some(rest) = raw.strip_prefix('*') {
        let (min, max) = parse_star_bounds(rest, lquery)?;
        return Ok(LqueryItem::Star { min, max });
    }
    let (negated, body) = match raw.strip_prefix('!') {
        Some(rest) => (true, rest),
        None => (false, raw),
    };
    if body.is_empty() {
        return Err(invalid("lquery", lquery));
    }
    let mut alternatives = Vec::new();
    for alt in body.split('|') {
        alternatives.push(parse_lquery_label(alt, lquery)?);
    }
    Ok(LqueryItem::Position {
        negated,
        alternatives,
    })
}

fn parse_star_bounds(rest: &str, lquery: &str) -> Result<(usize, Option<usize>)> {
    if rest.is_empty() {
        return Ok((0, None));
    }
    let inner = rest
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| invalid("lquery", lquery))?;
    match inner.split_once(',') {
        None => {
            let exact = parse_star_bound(inner, lquery)?;
            Ok((exact, Some(exact)))
        }
        Some((lo, hi)) => {
            let min = parse_star_bound(lo, lquery)?;
            if hi.is_empty() {
                Ok((min, None))
            } else {
                let max = parse_star_bound(hi, lquery)?;
                if max < min {
                    return Err(invalid("lquery", lquery));
                }
                Ok((min, Some(max)))
            }
        }
    }
}

fn parse_star_bound(text: &str, lquery: &str) -> Result<usize> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid("lquery", lquery));
    }
    text.parse::<usize>().map_err(|_| invalid("lquery", lquery))
}

fn parse_lquery_label(alt: &str, lquery: &str) -> Result<LqueryLabel> {
    if alt == "%" {
        return Ok(LqueryLabel::Any);
    }
    let (text, case_insensitive) = match alt.strip_suffix('@') {
        Some(rest) => (rest, true),
        None => (alt, false),
    };
    if !valid_label(text) {
        return Err(invalid("lquery", lquery));
    }
    Ok(LqueryLabel::Exact {
        text: text.to_string(),
        case_insensitive,
    })
}

// backtracking over (query item, label) states with an explicit stack,
// the seen set keeps repeated star expansions from revisiting a state
fn matches_items(items: &[LqueryItem], labels: &[&str]) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![(0usize, 0usize)];
    while let Some((qi, li)) = stack.pop() {
        if !seen.insert((qi, li)) {
            continue;
        }
        if qi == items.len() {
            if li == labels.len() {
                return true;
            }
            continue;
        }
        match &items[qi] {
            LqueryItem::Star { min, max } => {
                let remaining = labels.len() - li;
                if *min > remaining {
                    continue;
                }
                let hi = match max {
                    Some(m) => (*m).min(remaining),
                    None => remaining,
                };
                for take in *min..=hi {
                    stack.push((qi + 1, li + take));
                }
            }
            LqueryItem::Position {
                negated,
                alternatives,
            } => {
                if li >= labels.len() {
                    continue;
                }
                let label = labels[li];
                let mut hit = alternatives.iter().any(|alt| label_matches(alt, label));
                if *negated {
                    hit = !hit;
                }
                if hit {
                    stack.push((qi + 1, li + 1));
                }
            }
        }
    }
    false
}

fn label_matches(alt: &LqueryLabel, label: &str) -> bool {
    match alt {
        LqueryLabel::Any => true,
        LqueryLabel::Exact {
            text,
            case_insensitive,
        } => {
            if *case_insensitive {
                text.eq_ignore_ascii_case(label)
            } else {
                text == label
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok<T>(r: Result<T>) -> T {
        r.expect("expected Ok")
    }

    #[test]
    fn parse_path_splits_labels() {
        assert_eq!(
            ok(parse_path("Top.Science.Astronomy")),
            vec!["Top", "Science", "Astronomy"]
        );
        assert_eq!(ok(parse_path("a")), vec!["a"]);
        assert_eq!(ok(parse_path("a_1.B_2")), vec!["a_1", "B_2"]);
    }

    #[test]
    fn parse_path_rejects_malformed() {
        assert!(parse_path("").is_err());
        assert!(parse_path("a..b").is_err());
        assert!(parse_path(".a").is_err());
        assert!(parse_path("a.").is_err());
        assert!(parse_path("a b").is_err());
        assert!(parse_path("a.b-c").is_err());
        let long = "x".repeat(256);
        assert!(parse_path(&long).is_err());
        let max_len = "x".repeat(255);
        assert!(parse_path(&max_len).is_ok());
    }

    #[test]
    fn nlevel_counts_labels() {
        assert_eq!(ok(nlevel("Top")), 1);
        assert_eq!(ok(nlevel("Top.Science.Astronomy")), 3);
        assert!(nlevel("").is_err());
        assert!(nlevel("a..b").is_err());
    }

    #[test]
    fn subpath_positive_offsets() {
        assert_eq!(ok(subpath("a.b.c.d", 0, None)), "a.b.c.d");
        assert_eq!(ok(subpath("a.b.c.d", 1, None)), "b.c.d");
        assert_eq!(ok(subpath("a.b.c.d", 1, Some(2))), "b.c");
        assert_eq!(ok(subpath("a.b.c.d", 3, Some(1))), "d");
        // len past the end takes the remainder
        assert_eq!(ok(subpath("a.b.c.d", 2, Some(10))), "c.d");
    }

    #[test]
    fn subpath_negative_offsets_and_lens() {
        assert_eq!(ok(subpath("a.b.c.d", -2, None)), "c.d");
        assert_eq!(ok(subpath("a.b.c.d", -4, None)), "a.b.c.d");
        assert_eq!(ok(subpath("a.b.c.d", 0, Some(-1))), "a.b.c");
        assert_eq!(ok(subpath("a.b.c.d", 1, Some(-1))), "b.c");
        assert_eq!(ok(subpath("a.b.c.d", -3, Some(-1))), "b.c");
    }

    #[test]
    fn subpath_out_of_range_errors() {
        assert!(subpath("a.b.c", 3, None).is_err());
        assert!(subpath("a.b.c", -4, None).is_err());
        assert!(subpath("a.b.c", 0, Some(0)).is_err());
        assert!(subpath("a.b.c", 2, Some(-2)).is_err());
        assert!(subpath("a.b.c", 0, Some(-3)).is_err());
        assert!(subpath("", 0, None).is_err());
    }

    #[test]
    fn index_of_finds_runs() {
        assert_eq!(ok(index_of("a.b.c.d", "a")), 0);
        assert_eq!(ok(index_of("a.b.c.d", "b.c")), 1);
        assert_eq!(ok(index_of("a.b.c.d", "a.b.c.d")), 0);
        assert_eq!(ok(index_of("a.b.a.b.c", "a.b.c")), 2);
    }

    #[test]
    fn index_of_absent_is_minus_one() {
        assert_eq!(ok(index_of("a.b.c", "x")), -1);
        assert_eq!(ok(index_of("a.b.c", "c.b")), -1);
        assert_eq!(ok(index_of("a.b", "a.b.c")), -1);
    }

    #[test]
    fn index_of_rejects_malformed() {
        assert!(index_of("a..b", "a").is_err());
        assert!(index_of("a.b", "").is_err());
    }

    #[test]
    fn lca_shared_prefix() {
        assert_eq!(ok(lca(&["a.b.c.d", "a.b.x.y"])), Some("a.b".to_string()));
        assert_eq!(ok(lca(&["a.b.c", "a.b.c.d.e"])), Some("a.b".to_string()));
        assert_eq!(
            ok(lca(&["a.b.c.d", "a.b.c.x", "a.b.y"])),
            Some("a.b".to_string())
        );
    }

    #[test]
    fn lca_none_when_nothing_shared() {
        assert_eq!(ok(lca(&["a.b", "c.d"])), None);
        assert_eq!(ok(lca(&["a", "a"])), None);
    }

    #[test]
    fn lca_identical_paths_yield_parent() {
        assert_eq!(ok(lca(&["a.b.c", "a.b.c"])), Some("a.b".to_string()));
        assert_eq!(ok(lca(&["a.b.c"])), Some("a.b".to_string()));
    }

    #[test]
    fn lca_rejects_bad_input() {
        assert!(lca(&[]).is_err());
        assert!(lca(&["a.b", "c..d"]).is_err());
    }

    #[test]
    fn build_path_joins_and_validates() {
        assert_eq!(ok(build_path(&["Top", "Science"])), "Top.Science");
        assert_eq!(ok(build_path(&["solo"])), "solo");
        assert!(build_path(&[]).is_err());
        assert!(build_path(&["a", ""]).is_err());
        assert!(build_path(&["a", "b.c"]).is_err());
        assert!(build_path(&["a", "b c"]).is_err());
    }

    #[test]
    fn ancestor_both_directions_and_equal() {
        assert!(ok(is_ancestor("a.b", "a.b.c")));
        assert!(ok(is_ancestor("a", "a.b.c")));
        assert!(ok(is_ancestor("a.b.c", "a.b.c")));
        assert!(!ok(is_ancestor("a.b.c", "a.b")));
        assert!(!ok(is_ancestor("a.x", "a.b.c")));
        // ab is not a label prefix of abc even though it is a string prefix
        assert!(!ok(is_ancestor("ab", "abc")));

        assert!(ok(is_descendant("a.b.c", "a.b")));
        assert!(ok(is_descendant("a.b.c", "a.b.c")));
        assert!(!ok(is_descendant("a.b", "a.b.c")));
    }

    #[test]
    fn ancestor_rejects_malformed() {
        assert!(is_ancestor("a..b", "a.b").is_err());
        assert!(is_ancestor("a.b", "").is_err());
    }

    #[test]
    fn lquery_exact_labels() {
        assert!(ok(matches_lquery("a.b.c", "a.b.c")));
        assert!(!ok(matches_lquery("a.b.c", "a.b")));
        assert!(!ok(matches_lquery("a.b", "a.b.c")));
        assert!(!ok(matches_lquery("a.b.c", "a.x.c")));
    }

    #[test]
    fn lquery_bare_star() {
        assert!(ok(matches_lquery("a.b.c", "*")));
        assert!(ok(matches_lquery("a.b.c", "a.*")));
        assert!(ok(matches_lquery("a.b.c", "*.c")));
        assert!(ok(matches_lquery("a.b.c", "a.*.c")));
        assert!(ok(matches_lquery("a.c", "a.*.c")));
        assert!(!ok(matches_lquery("a.b.c", "*.x")));
        assert!(ok(matches_lquery("a.b.c.d", "*.b.*.d")));
    }

    #[test]
    fn lquery_star_bounds() {
        assert!(ok(matches_lquery("a.b.c.d", "a.*{2}.d")));
        assert!(!ok(matches_lquery("a.b.c.d", "a.*{1}.d")));
        assert!(ok(matches_lquery("a.b.c.d", "a.*{1,}.d")));
        assert!(!ok(matches_lquery("a.d", "a.*{1,}.d")));
        assert!(ok(matches_lquery("a.b.c.d", "a.*{1,2}.d")));
        assert!(!ok(matches_lquery("a.b.c.x.d", "a.*{1,2}.d")));
        assert!(ok(matches_lquery("a.b", "*{2}")));
        assert!(ok(matches_lquery("a.b.c", "*{0,}")));
        assert!(!ok(matches_lquery("a.b.c", "*{4,}")));
    }

    #[test]
    fn lquery_alternation() {
        assert!(ok(matches_lquery("a.b.c", "a.b|x.c")));
        assert!(ok(matches_lquery("a.x.c", "a.b|x.c")));
        assert!(!ok(matches_lquery("a.y.c", "a.b|x.c")));
    }

    #[test]
    fn lquery_negation() {
        assert!(ok(matches_lquery("a.y.c", "a.!b.c")));
        assert!(!ok(matches_lquery("a.b.c", "a.!b.c")));
        assert!(!ok(matches_lquery("a.x.c", "a.!b|x.c")));
        assert!(ok(matches_lquery("a.y.c", "a.!b|x.c")));
        // a negated position still consumes exactly one label
        assert!(!ok(matches_lquery("a.c", "a.!b.c")));
    }

    #[test]
    fn lquery_percent_matches_any_single_label() {
        assert!(ok(matches_lquery("a.b.c", "a.%.c")));
        assert!(ok(matches_lquery("a.zzz.c", "a.%.c")));
        assert!(!ok(matches_lquery("a.c", "a.%.c")));
        assert!(!ok(matches_lquery("a.b.x.c", "a.%.c")));
        assert!(ok(matches_lquery("a.b.c", "%.%.%")));
    }

    #[test]
    fn lquery_case_flag() {
        assert!(ok(matches_lquery("a.FOO.c", "a.foo@.c")));
        assert!(ok(matches_lquery("a.foo.c", "a.FoO@.c")));
        assert!(!ok(matches_lquery("a.FOO.c", "a.foo.c")));
        assert!(ok(matches_lquery("A.b", "a@.b")));
    }

    #[test]
    fn lquery_malformed_errors() {
        assert!(matches_lquery("a.b", "").is_err());
        assert!(matches_lquery("a.b", "a..b").is_err());
        assert!(matches_lquery("a.b", "a.!").is_err());
        assert!(matches_lquery("a.b", "a.b|").is_err());
        assert!(matches_lquery("a.b", "a.*{").is_err());
        assert!(matches_lquery("a.b", "a.*{x}").is_err());
        assert!(matches_lquery("a.b", "a.*{2,1}").is_err());
        assert!(matches_lquery("a.b", "a.*{,2}").is_err());
        assert!(matches_lquery("a.b", "a.*x").is_err());
        assert!(matches_lquery("a.b", "a.b c").is_err());
    }

    #[test]
    fn lquery_malformed_path_errors() {
        assert!(matches_lquery("", "a").is_err());
        assert!(matches_lquery("a..b", "a.b").is_err());
    }

    #[test]
    fn matches_any_lquery_semantics() {
        assert!(ok(matches_any_lquery("a.b.c", &["x.y", "a.*"])));
        assert!(!ok(matches_any_lquery("a.b.c", &["x.y", "b.*"])));
        assert!(!ok(matches_any_lquery("a.b.c", &[])));
        // a malformed pattern errors even when another pattern matches
        assert!(matches_any_lquery("a.b.c", &["a.*", "!"]).is_err());
    }
}
