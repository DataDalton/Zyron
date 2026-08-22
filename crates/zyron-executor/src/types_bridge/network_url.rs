//! Dispatch arms for zyron_types::network (INET/CIDR/MACADDR) and
//! zyron_types::url_type functions.
//!
//! Physical representation follows the binder contract, Inet and Cidr cells
//! are the 18-byte network.rs form inside ColumnData::Binary, MacAddr cells
//! are 6 raw bytes, macaddr_oui emits 3 raw bytes as Bytea. Inet and mac
//! arguments also accept Utf8 columns, parsed per row with the module's own
//! parser. url_parse (Composite) emits a hand-built JSON object as Binary
//! text bytes since UrlParts does not derive Serialize. url_query_params
//! (Array) emits a JSON array of [key, value] pairs as Binary text bytes.
//! Parser and extractor failures yield NULL for that row, matching the
//! validator/parser convention in the parent module.

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_types::{network, url_type};

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        // ---------- network: parsers ----------
        "inet_parse" => parse_inet_text(args, "inet_parse"),
        "cidr_parse" => parse_inet_text(args, "cidr_parse"),
        "macaddr_parse" => macaddr_parse_impl(args),

        // ---------- network: inet transforms ----------
        "inet_network" => inet_unary(args, "inet_network", network::inet_network),
        "inet_broadcast" => inet_unary(args, "inet_broadcast", network::inet_broadcast),
        "inet_netmask" => inet_unary(args, "inet_netmask", network::inet_netmask),
        "inet_host" => inet_unary(args, "inet_host", network::inet_host),

        // ---------- network: inet extractors and predicates ----------
        "inet_format" => inet_format_impl(args),
        "inet_family" => inet_to_int32(args, "inet_family", network::inet_family),
        "inet_prefix" => inet_to_int32(args, "inet_prefix", network::inet_prefix),
        "inet_is_private" => inet_to_bool(args, "inet_is_private", network::inet_is_private),
        "inet_is_loopback" => inet_to_bool(args, "inet_is_loopback", network::inet_is_loopback),
        "inet_contains" => inet_contains_impl(args),

        // ---------- network: macaddr ----------
        "macaddr_format" => macaddr_format_impl(args),
        "macaddr_oui" => macaddr_oui_impl(args),

        // ---------- url_type: component extractors ----------
        "url_scheme" => url_one_string(args, "url_scheme", |s| url_type::url_scheme(s).ok()),
        "url_host" => url_one_string(args, "url_host", |s| url_type::url_host(s).ok()),
        "url_path" => url_one_string(args, "url_path", |s| url_type::url_path(s).ok()),
        "url_domain" => url_one_string(args, "url_domain", |s| url_type::url_domain(s).ok()),
        "url_tld" => url_one_string(args, "url_tld", |s| url_type::url_tld(s).ok()),
        "url_normalize" => {
            url_one_string(args, "url_normalize", |s| url_type::url_normalize(s).ok())
        }
        // Ok(None) means no fragment present, folded to SQL NULL
        "url_fragment" => url_one_string(args, "url_fragment", |s| {
            url_type::url_fragment(s).ok().flatten()
        }),
        "url_port" => url_port_impl(args),
        "url_is_absolute" => url_is_absolute_impl(args),
        "url_query_param" => url_two_string(args, "url_query_param", |u, k| {
            url_type::url_query_param(u, k).ok().flatten()
        }),
        "url_resolve" => {
            url_two_string(args, "url_resolve", |b, r| url_type::url_resolve(b, r).ok())
        }
        "url_parse" => url_parse_impl(args),
        "url_query_params" => url_query_params_impl(args),

        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// column readers, per row None means NULL input or an unusable cell
// ---------------------------------------------------------------------------

fn read_strings_opt<'a>(col: &'a Column, fn_name: &str) -> Result<Vec<Option<&'a str>>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok(v
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if col.nulls.is_null(i) {
                    None
                } else {
                    Some(s.as_str())
                }
            })
            .collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a text argument",
            fn_name
        ))),
    }
}

/// Accepts the 18-byte Binary inet form or Utf8 parsed per row
fn read_inets(col: &Column, fn_name: &str) -> Result<Vec<Option<[u8; 18]>>> {
    match &col.data {
        ColumnData::Binary(cells) => Ok(cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if col.nulls.is_null(i) {
                    None
                } else {
                    <[u8; 18]>::try_from(c.as_slice()).ok()
                }
            })
            .collect()),
        ColumnData::Utf8(strings) => Ok(strings
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if col.nulls.is_null(i) {
                    None
                } else {
                    network::inet_parse(s).ok()
                }
            })
            .collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects an inet argument (binary or text)",
            fn_name
        ))),
    }
}

/// Accepts the 6-byte Binary mac form or Utf8 parsed per row
fn read_macs(col: &Column, fn_name: &str) -> Result<Vec<Option<[u8; 6]>>> {
    match &col.data {
        ColumnData::Binary(cells) => Ok(cells
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if col.nulls.is_null(i) {
                    None
                } else {
                    <[u8; 6]>::try_from(c.as_slice()).ok()
                }
            })
            .collect()),
        ColumnData::Utf8(strings) => Ok(strings
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if col.nulls.is_null(i) {
                    None
                } else {
                    network::macaddr_parse(s).ok()
                }
            })
            .collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a macaddr argument (binary or text)",
            fn_name
        ))),
    }
}

// ---------------------------------------------------------------------------
// column builders, None slots become NULL rows
// ---------------------------------------------------------------------------

fn binary_out(values: Vec<Option<Vec<u8>>>, type_id: TypeId) -> Column {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = NullBitmap::none(values.len());
    for (i, v) in values.into_iter().enumerate() {
        match v {
            Some(b) => data.push(b),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Binary(data), nulls, type_id)
}

fn utf8_out(values: Vec<Option<String>>) -> Column {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = NullBitmap::none(values.len());
    for (i, v) in values.into_iter().enumerate() {
        match v {
            Some(s) => data.push(s),
            None => {
                data.push(String::new());
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Utf8(data), nulls, TypeId::Varchar)
}

fn bool_out(values: Vec<Option<bool>>) -> Column {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = NullBitmap::none(values.len());
    for (i, v) in values.into_iter().enumerate() {
        match v {
            Some(b) => data.push(b),
            None => {
                data.push(false);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Boolean(data), nulls, TypeId::Boolean)
}

fn int32_out(values: Vec<Option<i32>>) -> Column {
    let mut data = Vec::with_capacity(values.len());
    let mut nulls = NullBitmap::none(values.len());
    for (i, v) in values.into_iter().enumerate() {
        match v {
            Some(n) => data.push(n),
            None => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Int32(data), nulls, TypeId::Int32)
}

fn arg_count(args: &[Column], expected: usize, signature: &str) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} argument{}",
            signature,
            expected,
            if expected == 1 { "" } else { "s" }
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// network impls
// ---------------------------------------------------------------------------

/// Text to 18-byte inet cell, parse failure yields NULL for that row
fn parse_inet_text(args: &[Column], fn_name: &str) -> Result<Column> {
    arg_count(args, 1, &format!("{}(text)", fn_name))?;
    let strings = read_strings_opt(&args[0], fn_name)?;
    let values: Vec<Option<Vec<u8>>> = strings
        .iter()
        .map(|s| s.and_then(|t| network::inet_parse(t).ok().map(|a| a.to_vec())))
        .collect();
    Ok(binary_out(values, TypeId::Inet))
}

fn inet_unary(args: &[Column], fn_name: &str, f: fn(&[u8; 18]) -> [u8; 18]) -> Result<Column> {
    arg_count(args, 1, &format!("{}(inet)", fn_name))?;
    let inets = read_inets(&args[0], fn_name)?;
    let values: Vec<Option<Vec<u8>>> = inets
        .iter()
        .map(|a| a.as_ref().map(|v| f(v).to_vec()))
        .collect();
    Ok(binary_out(values, TypeId::Inet))
}

fn inet_to_int32(args: &[Column], fn_name: &str, f: fn(&[u8; 18]) -> u8) -> Result<Column> {
    arg_count(args, 1, &format!("{}(inet)", fn_name))?;
    let inets = read_inets(&args[0], fn_name)?;
    let values: Vec<Option<i32>> = inets
        .iter()
        .map(|a| a.as_ref().map(|v| f(v) as i32))
        .collect();
    Ok(int32_out(values))
}

fn inet_to_bool(args: &[Column], fn_name: &str, f: fn(&[u8; 18]) -> bool) -> Result<Column> {
    arg_count(args, 1, &format!("{}(inet)", fn_name))?;
    let inets = read_inets(&args[0], fn_name)?;
    let values: Vec<Option<bool>> = inets.iter().map(|a| a.as_ref().map(f)).collect();
    Ok(bool_out(values))
}

fn inet_format_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "inet_format(inet)")?;
    let inets = read_inets(&args[0], "inet_format")?;
    let values: Vec<Option<String>> = inets
        .iter()
        .map(|a| a.as_ref().map(network::inet_format))
        .collect();
    Ok(utf8_out(values))
}

fn inet_contains_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 2, "inet_contains(inet, inet)")?;
    let networks = read_inets(&args[0], "inet_contains")?;
    let addrs = read_inets(&args[1], "inet_contains")?;
    if networks.len() != addrs.len() {
        return Err(ZyronError::ExecutionError(
            "inet_contains: column length mismatch".to_string(),
        ));
    }
    let values: Vec<Option<bool>> = networks
        .iter()
        .zip(addrs.iter())
        .map(|(n, a)| match (n, a) {
            (Some(n), Some(a)) => Some(network::inet_contains(n, a)),
            _ => None,
        })
        .collect();
    Ok(bool_out(values))
}

/// Text to 6-byte mac cell, parse failure yields NULL for that row
fn macaddr_parse_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "macaddr_parse(text)")?;
    let strings = read_strings_opt(&args[0], "macaddr_parse")?;
    let values: Vec<Option<Vec<u8>>> = strings
        .iter()
        .map(|s| s.and_then(|t| network::macaddr_parse(t).ok().map(|m| m.to_vec())))
        .collect();
    Ok(binary_out(values, TypeId::MacAddr))
}

fn macaddr_format_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "macaddr_format(macaddr)")?;
    let macs = read_macs(&args[0], "macaddr_format")?;
    let values: Vec<Option<String>> = macs
        .iter()
        .map(|m| m.as_ref().map(network::macaddr_format))
        .collect();
    Ok(utf8_out(values))
}

fn macaddr_oui_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "macaddr_oui(macaddr)")?;
    let macs = read_macs(&args[0], "macaddr_oui")?;
    let values: Vec<Option<Vec<u8>>> = macs
        .iter()
        .map(|m| m.as_ref().map(|v| network::macaddr_oui(v).to_vec()))
        .collect();
    Ok(binary_out(values, TypeId::Bytea))
}

// ---------------------------------------------------------------------------
// url_type impls
// ---------------------------------------------------------------------------

fn url_one_string<F: Fn(&str) -> Option<String>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    arg_count(args, 1, &format!("{}(text)", fn_name))?;
    let strings = read_strings_opt(&args[0], fn_name)?;
    let values: Vec<Option<String>> = strings.iter().map(|s| s.and_then(&f)).collect();
    Ok(utf8_out(values))
}

fn url_two_string<F: Fn(&str, &str) -> Option<String>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    arg_count(args, 2, &format!("{}(text, text)", fn_name))?;
    let a = read_strings_opt(&args[0], fn_name)?;
    let b = read_strings_opt(&args[1], fn_name)?;
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(format!(
            "{}: column length mismatch",
            fn_name
        )));
    }
    let values: Vec<Option<String>> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => f(x, y),
            _ => None,
        })
        .collect();
    Ok(utf8_out(values))
}

/// Missing port or parse failure yields NULL for that row
fn url_port_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "url_port(text)")?;
    let strings = read_strings_opt(&args[0], "url_port")?;
    let values: Vec<Option<i32>> = strings
        .iter()
        .map(|s| {
            s.and_then(|t| url_type::url_port(t).ok().flatten())
                .map(|p| p as i32)
        })
        .collect();
    Ok(int32_out(values))
}

fn url_is_absolute_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "url_is_absolute(text)")?;
    let strings = read_strings_opt(&args[0], "url_is_absolute")?;
    let values: Vec<Option<bool>> = strings
        .iter()
        .map(|s| s.map(url_type::url_is_absolute))
        .collect();
    Ok(bool_out(values))
}

/// Composite output, JSON object text bytes in a Binary cell since UrlParts
/// has no Serialize derive, absent components serialize as JSON null
fn url_parse_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "url_parse(text)")?;
    let strings = read_strings_opt(&args[0], "url_parse")?;
    let values: Vec<Option<Vec<u8>>> = strings
        .iter()
        .map(|s| {
            s.and_then(|t| url_type::url_parse(t).ok()).map(|u| {
                serde_json::json!({
                    "scheme": u.scheme,
                    "user": u.user,
                    "password": u.password,
                    "host": u.host,
                    "port": u.port,
                    "path": u.path,
                    "query": u.query,
                    "fragment": u.fragment,
                })
                .to_string()
                .into_bytes()
            })
        })
        .collect();
    Ok(binary_out(values, TypeId::Composite))
}

/// Array output, JSON array of [key, value] pairs as text bytes in a Binary cell
fn url_query_params_impl(args: &[Column]) -> Result<Column> {
    arg_count(args, 1, "url_query_params(text)")?;
    let strings = read_strings_opt(&args[0], "url_query_params")?;
    let values: Vec<Option<Vec<u8>>> = strings
        .iter()
        .map(|s| {
            s.and_then(|t| url_type::url_query_params(t).ok())
                .map(|pairs| {
                    let arr: Vec<serde_json::Value> = pairs
                        .into_iter()
                        .map(|(k, v)| serde_json::json!([k, v]))
                        .collect();
                    serde_json::Value::Array(arr).to_string().into_bytes()
                })
        })
        .collect();
    Ok(binary_out(values, TypeId::Array))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(values: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(values.iter().map(|s| s.to_string()).collect()),
            TypeId::Text,
        )
    }

    fn utf8_col_with_null(values: &[&str], null_idx: usize) -> Column {
        let mut nulls = NullBitmap::none(values.len());
        nulls.set_null(null_idx);
        Column::with_nulls(
            ColumnData::Utf8(values.iter().map(|s| s.to_string()).collect()),
            nulls,
            TypeId::Text,
        )
    }

    #[test]
    fn inet_parse_format_round_trip() {
        let input = utf8_col(&["192.168.1.1", "10.0.0.0/8"]);
        let parsed = dispatch("inet_parse", &[input], 2).unwrap().unwrap();
        assert_eq!(parsed.type_id, TypeId::Inet);
        match &parsed.data {
            ColumnData::Binary(cells) => {
                assert_eq!(cells[0].len(), 18);
                assert_eq!(cells[0][0], 4);
            }
            other => panic!("expected Binary, got {:?}", other),
        }
        let formatted = dispatch("inet_format", &[parsed], 2).unwrap().unwrap();
        match &formatted.data {
            ColumnData::Utf8(v) => {
                assert_eq!(v[0], "192.168.1.1");
                assert_eq!(v[1], "10.0.0.0/8");
            }
            other => panic!("expected Utf8, got {:?}", other),
        }
    }

    #[test]
    fn inet_parse_invalid_and_null_rows_are_null() {
        let input = utf8_col_with_null(&["not an ip", "192.168.1.1", ""], 2);
        let parsed = dispatch("inet_parse", &[input], 3).unwrap().unwrap();
        assert!(parsed.nulls.is_null(0));
        assert!(!parsed.nulls.is_null(1));
        assert!(parsed.nulls.is_null(2));
    }

    #[test]
    fn inet_contains_accepts_text_inputs() {
        let networks = utf8_col(&["192.168.0.0/16", "192.168.0.0/16"]);
        let addrs = utf8_col(&["192.168.1.100", "10.0.0.1"]);
        let out = dispatch("inet_contains", &[networks, addrs], 2)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Boolean(v) => {
                assert!(v[0]);
                assert!(!v[1]);
            }
            other => panic!("expected Boolean, got {:?}", other),
        }
    }

    #[test]
    fn macaddr_parse_format_and_oui() {
        let input = utf8_col(&["aa:bb:cc:dd:ee:ff"]);
        let parsed = dispatch("macaddr_parse", &[input], 1).unwrap().unwrap();
        assert_eq!(parsed.type_id, TypeId::MacAddr);
        let oui = dispatch("macaddr_oui", &[parsed.clone()], 1)
            .unwrap()
            .unwrap();
        match &oui.data {
            ColumnData::Binary(cells) => assert_eq!(cells[0], vec![0xAA, 0xBB, 0xCC]),
            other => panic!("expected Binary, got {:?}", other),
        }
        let formatted = dispatch("macaddr_format", &[parsed], 1).unwrap().unwrap();
        match &formatted.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "AA:BB:CC:DD:EE:FF"),
            other => panic!("expected Utf8, got {:?}", other),
        }
    }

    #[test]
    fn url_port_present_and_missing() {
        let input = utf8_col(&["http://example.com:8080/", "http://example.com/"]);
        let out = dispatch("url_port", &[input], 2).unwrap().unwrap();
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v[0], 8080),
            other => panic!("expected Int32, got {:?}", other),
        }
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn url_scheme_null_propagates() {
        let input = utf8_col_with_null(&["https://example.com/", "https://other.com/"], 1);
        let out = dispatch("url_scheme", &[input], 2).unwrap().unwrap();
        match &out.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "https"),
            other => panic!("expected Utf8, got {:?}", other),
        }
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn url_parse_emits_json_composite() {
        let input = utf8_col(&["https://user:pass@example.com:8080/path?q=1#frag"]);
        let out = dispatch("url_parse", &[input], 1).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Composite);
        let cell = match &out.data {
            ColumnData::Binary(cells) => cells[0].clone(),
            other => panic!("expected Binary, got {:?}", other),
        };
        let v: serde_json::Value = serde_json::from_slice(&cell).unwrap();
        assert_eq!(v["scheme"], "https");
        assert_eq!(v["host"], "example.com");
        assert_eq!(v["port"], 8080);
        assert_eq!(v["path"], "/path");
        assert_eq!(v["query"], "q=1");
        assert_eq!(v["fragment"], "frag");
    }

    #[test]
    fn url_query_params_emits_json_array_of_pairs() {
        let input = utf8_col(&["https://example.com/?a=1&b=two"]);
        let out = dispatch("url_query_params", &[input], 1).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Array);
        let cell = match &out.data {
            ColumnData::Binary(cells) => cells[0].clone(),
            other => panic!("expected Binary, got {:?}", other),
        };
        let v: serde_json::Value = serde_json::from_slice(&cell).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], serde_json::json!(["a", "1"]));
        assert_eq!(arr[1], serde_json::json!(["b", "two"]));
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(dispatch("definitely_not_here", &[], 1).is_none());
    }
}
