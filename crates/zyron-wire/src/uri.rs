//! Zyron URI parser for client-facing connection strings.
//!
//! Accepts `zyron://user@host1:port,host2:port/database/schema.table?key=value`.
//! Either a `schema.table` path suffix or a `pub=name` query parameter may be
//! provided to identify the target. Hosts may omit the port, defaulting to 5432.

use std::collections::HashMap;

/// One host entry parsed from the URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UriHost {
    pub host: String,
    pub port: u16,
}

/// The identified target: either a sink table or a source publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZyronUriTarget {
    Table { schema: String, table: String },
    Publication { name: String },
    Database,
}

/// Parsed form of a zyron connection URI.
#[derive(Debug, Clone)]
pub struct ParsedZyronUri {
    pub hosts: Vec<UriHost>,
    pub user: String,
    pub password: Option<String>,
    pub database: String,
    pub target: ZyronUriTarget,
    pub query_params: HashMap<String, String>,
}

/// Error type for URI parsing.
#[derive(Debug, thiserror::Error)]
pub enum UriError {
    #[error("URI must start with zyron://")]
    MissingScheme,
    #[error("URI is missing the user component")]
    MissingUser,
    #[error("URI is missing at least one host")]
    MissingHost,
    #[error("Malformed host entry: {0}")]
    BadHost(String),
    #[error("Malformed URI: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, UriError>;

const DEFAULT_PORT: u16 = 5432;

/// Parses a zyron URI into its components.
pub fn parse_zyron_uri(uri: &str) -> Result<ParsedZyronUri> {
    let rest = uri
        .strip_prefix("zyron://")
        .ok_or(UriError::MissingScheme)?;

    // Split query string first.
    let (path_part, query_part) = match rest.split_once('?') {
        Some((a, b)) => (a, Some(b)),
        None => (rest, None),
    };

    // Split authority and path.
    let (authority, path) = match path_part.split_once('/') {
        Some((a, b)) => (a, b),
        None => (path_part, ""),
    };

    // Authority: user[:password]@host[:port][,host[:port]]*
    let (userinfo, hostlist) = authority.split_once('@').ok_or(UriError::MissingUser)?;
    let (user, password) = match userinfo.split_once(':') {
        Some((u, p)) => (u.to_string(), Some(p.to_string())),
        None => (userinfo.to_string(), None),
    };
    if user.is_empty() {
        return Err(UriError::MissingUser);
    }

    let mut hosts = Vec::new();
    for entry in hostlist.split(',') {
        if entry.is_empty() {
            continue;
        }
        let (host, port) = match entry.rsplit_once(':') {
            Some((h, p)) => {
                let port: u16 = p
                    .parse()
                    .map_err(|_| UriError::BadHost(entry.to_string()))?;
                (h.to_string(), port)
            }
            None => (entry.to_string(), DEFAULT_PORT),
        };
        if host.is_empty() {
            return Err(UriError::BadHost(entry.to_string()));
        }
        hosts.push(UriHost { host, port });
    }
    if hosts.is_empty() {
        return Err(UriError::MissingHost);
    }

    // Path: database[/schema.table]
    let mut path_segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let database = path_segments
        .first()
        .cloned()
        .ok_or_else(|| UriError::Malformed("missing database".into()))?
        .to_string();
    path_segments.remove(0);

    let mut query_params = HashMap::new();
    if let Some(q) = query_part {
        for pair in q.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            query_params.insert(k.to_string(), v.to_string());
        }
    }

    let target = if let Some(seg) = path_segments.first() {
        let (schema, table) = seg
            .split_once('.')
            .ok_or_else(|| UriError::Malformed("expected schema.table".into()))?;
        ZyronUriTarget::Table {
            schema: schema.to_string(),
            table: table.to_string(),
        }
    } else if let Some(pub_name) = query_params.get("pub") {
        ZyronUriTarget::Publication {
            name: pub_name.clone(),
        }
    } else {
        ZyronUriTarget::Database
    };

    Ok(ParsedZyronUri {
        hosts,
        user,
        password,
        database,
        target,
        query_params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal() {
        let u = parse_zyron_uri("zyron://alice@host1/db").unwrap();
        assert_eq!(u.user, "alice");
        assert_eq!(u.database, "db");
        assert_eq!(u.hosts.len(), 1);
        assert_eq!(u.hosts[0].port, 5432);
        assert!(matches!(u.target, ZyronUriTarget::Database));
    }

    #[test]
    fn test_parse_multi_host_with_ports() {
        let u = parse_zyron_uri("zyron://bob@h1:5433,h2:5434/mydb").unwrap();
        assert_eq!(u.hosts.len(), 2);
        assert_eq!(u.hosts[0].port, 5433);
        assert_eq!(u.hosts[1].port, 5434);
    }

    #[test]
    fn test_parse_table_target() {
        let u = parse_zyron_uri("zyron://u@h/db/public.events").unwrap();
        match u.target {
            ZyronUriTarget::Table { schema, table } => {
                assert_eq!(schema, "public");
                assert_eq!(table, "events");
            }
            _ => panic!("wrong target"),
        }
    }

    #[test]
    fn test_parse_publication_target() {
        let u = parse_zyron_uri("zyron://u@h/db?pub=orders_pub").unwrap();
        match u.target {
            ZyronUriTarget::Publication { name } => assert_eq!(name, "orders_pub"),
            _ => panic!("wrong target"),
        }
    }

    #[test]
    fn test_parse_query_params() {
        let u = parse_zyron_uri("zyron://u@h/db?tls=required&pool_size=4").unwrap();
        assert_eq!(u.query_params.get("tls").unwrap(), "required");
        assert_eq!(u.query_params.get("pool_size").unwrap(), "4");
    }

    #[test]
    fn test_parse_password() {
        let u = parse_zyron_uri("zyron://u:secret@h/db").unwrap();
        assert_eq!(u.password.as_deref(), Some("secret"));
    }

    #[test]
    fn test_missing_scheme() {
        assert!(parse_zyron_uri("postgres://u@h/db").is_err());
    }

    #[test]
    fn test_missing_user() {
        assert!(parse_zyron_uri("zyron://host/db").is_err());
    }

    #[test]
    fn test_missing_host() {
        assert!(parse_zyron_uri("zyron://u@/db").is_err());
    }

    #[test]
    fn test_bad_port() {
        assert!(parse_zyron_uri("zyron://u@h:notaport/db").is_err());
    }
}
