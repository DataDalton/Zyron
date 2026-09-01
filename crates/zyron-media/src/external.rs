//! External media references fetched through opendal
//!
//! Supports file, s3, gcs, azblob, http and https uris. Cloud schemes build
//! real operators with the default anonymous credential chain, credential
//! errors surface from opendal at call time. Fetches are cached per uri
//! with a time to live

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use opendal::{Operator, services};

use crate::error::{MediaError, MediaResult};

struct CacheEntry {
    bytes: Arc<Vec<u8>>,
    fetched_at: Instant,
}

/// Fetches and writes external media objects with a TTL read cache
pub struct ExternalFetcher {
    ttl: Duration,
    cache: scc::HashMap<String, CacheEntry>,
}

impl ExternalFetcher {
    pub fn new(ttl: Duration) -> Self {
        ExternalFetcher {
            ttl,
            cache: scc::HashMap::new(),
        }
    }

    /// Reads the object behind a uri, serving cached bytes within the ttl
    pub fn fetch(&self, uri: &str) -> MediaResult<Vec<u8>> {
        let mut cached: Option<Arc<Vec<u8>>> = None;
        self.cache.read_sync(uri, |_, entry| {
            if entry.fetched_at.elapsed() <= self.ttl {
                cached = Some(Arc::clone(&entry.bytes));
            }
        });
        if let Some(bytes) = cached {
            return Ok((*bytes).clone());
        }
        self.cache.remove_sync(uri);

        let (operator, path) = build_operator(uri)?;
        let bytes = run_async(async move {
            operator
                .read(&path)
                .await
                .map(|buffer| buffer.to_vec())
                .map_err(|e| MediaError::External(format!("read failed: {e}")))
        })?;

        let entry = CacheEntry {
            bytes: Arc::new(bytes.clone()),
            fetched_at: Instant::now(),
        };
        match self.cache.entry_sync(uri.to_string()) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.insert(entry);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(entry);
            }
        }
        Ok(bytes)
    }

    /// Writes bytes under a uri prefix, returning the full object uri
    pub fn put(&self, uri_prefix: &str, name: &str, bytes: &[u8]) -> MediaResult<String> {
        if name.is_empty() || name.contains('/') || name.contains('\\') {
            return Err(MediaError::InvalidArgument(format!(
                "invalid external object name {name}"
            )));
        }
        let trimmed = uri_prefix.trim_end_matches('/');
        let full_uri = format!("{trimmed}/{name}");
        let (operator, path) = build_operator(&full_uri)?;
        let payload = bytes.to_vec();
        run_async(async move {
            operator
                .write(&path, payload)
                .await
                .map_err(|e| MediaError::External(format!("write failed: {e}")))
        })?;
        Ok(full_uri)
    }
}

/// Builds the operator for a uri and returns the in operator object path
fn build_operator(uri: &str) -> MediaResult<(Operator, String)> {
    if let Some(rest) = strip_scheme(uri, "file://") {
        return file_operator(rest);
    }
    if let Some(rest) = strip_scheme(uri, "s3://") {
        let (bucket, key) = split_bucket_key(uri, rest)?;
        // region comes from the environment when set, us-east-1 otherwise,
        // credentials resolve through the default chain at call time
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        let builder = services::S3::default().bucket(bucket).region(&region);
        let operator = Operator::new(builder)
            .map_err(|e| MediaError::External(format!("s3 operator build failed: {e}")))?
            .finish();
        return Ok((operator, key.to_string()));
    }
    if let Some(rest) = strip_scheme(uri, "gcs://").or_else(|| strip_scheme(uri, "gs://")) {
        let (bucket, key) = split_bucket_key(uri, rest)?;
        let builder = services::Gcs::default().bucket(bucket);
        let operator = Operator::new(builder)
            .map_err(|e| MediaError::External(format!("gcs operator build failed: {e}")))?
            .finish();
        return Ok((operator, key.to_string()));
    }
    if let Some(rest) = strip_scheme(uri, "azblob://").or_else(|| strip_scheme(uri, "az://")) {
        let (container, key) = split_bucket_key(uri, rest)?;
        let builder = services::Azblob::default().container(container);
        let operator = Operator::new(builder)
            .map_err(|e| MediaError::External(format!("azblob operator build failed: {e}")))?
            .finish();
        return Ok((operator, key.to_string()));
    }
    if uri.starts_with("http://") || uri.starts_with("https://") {
        let scheme_len = if uri.starts_with("https://") { 8 } else { 7 };
        let rest = &uri[scheme_len..];
        let (host, key) = match rest.split_once('/') {
            Some((host, key)) if !host.is_empty() && !key.is_empty() => (host, key),
            _ => {
                return Err(MediaError::External(format!(
                    "http uri {uri} needs a host and an object path"
                )));
            }
        };
        let endpoint = format!("{}{}", &uri[..scheme_len], host);
        let builder = services::Http::default().endpoint(&endpoint);
        let operator = Operator::new(builder)
            .map_err(|e| MediaError::External(format!("http operator build failed: {e}")))?
            .finish();
        return Ok((operator, key.to_string()));
    }
    Err(MediaError::UnsupportedScheme {
        uri: uri.to_string(),
    })
}

fn file_operator(rest: &str) -> MediaResult<(Operator, String)> {
    // file:///C:/x arrives as /C:/x, strip the extra slash before a drive letter
    let path_str = if rest.len() >= 3
        && rest.starts_with('/')
        && rest.as_bytes()[2] == b':'
        && rest.as_bytes()[1].is_ascii_alphabetic()
    {
        &rest[1..]
    } else {
        rest
    };
    let path = Path::new(path_str);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| {
            MediaError::External(format!("file uri path {path_str} has no parent directory"))
        })?;
    let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        MediaError::External(format!("file uri path {path_str} has no file name"))
    })?;
    let root = parent.to_string_lossy().to_string();
    let builder = services::Fs::default().root(&root);
    let operator = Operator::new(builder)
        .map_err(|e| MediaError::External(format!("fs operator build failed: {e}")))?
        .finish();
    Ok((operator, name.to_string()))
}

fn strip_scheme<'a>(uri: &'a str, scheme: &str) -> Option<&'a str> {
    uri.strip_prefix(scheme)
}

fn split_bucket_key<'a>(uri: &str, rest: &'a str) -> MediaResult<(&'a str, &'a str)> {
    match rest.split_once('/') {
        Some((bucket, key)) if !bucket.is_empty() && !key.is_empty() => Ok((bucket, key)),
        _ => Err(MediaError::External(format!(
            "uri {uri} needs a bucket and an object key"
        ))),
    }
}

/// Runs a future to completion on a dedicated thread with its own runtime
///
/// A fresh thread avoids panics from nesting block_on inside whatever
/// runtime flavor the caller happens to be on
fn run_async<T, F>(future: F) -> MediaResult<T>
where
    T: Send,
    F: std::future::Future<Output = MediaResult<T>> + Send,
{
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| MediaError::External(format!("runtime build failed: {e}")))?;
            runtime.block_on(future)
        });
        handle
            .join()
            .map_err(|_| MediaError::External("external fetch thread panicked".to_string()))?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_put_fetch_and_cache_hit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let prefix = format!("file://{}", dir.path().display());
        let fetcher = ExternalFetcher::new(Duration::from_secs(60));

        let uri = fetcher
            .put(&prefix, "object.bin", b"original bytes")
            .expect("put");
        assert!(uri.ends_with("/object.bin"));
        assert_eq!(fetcher.fetch(&uri).expect("fetch"), b"original bytes");

        // mutate the file behind the cache, the ttl serves the old bytes
        let on_disk = dir.path().join("object.bin");
        std::fs::write(&on_disk, b"mutated on disk").expect("mutate");
        assert_eq!(
            fetcher.fetch(&uri).expect("cached fetch"),
            b"original bytes"
        );

        // a zero ttl fetcher sees the mutation
        let uncached = ExternalFetcher::new(Duration::ZERO);
        assert_eq!(
            uncached.fetch(&uri).expect("fresh fetch"),
            b"mutated on disk"
        );
    }

    #[test]
    fn missing_file_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let fetcher = ExternalFetcher::new(Duration::from_secs(60));
        let uri = format!("file://{}/absent.bin", dir.path().display());
        assert!(fetcher.fetch(&uri).is_err());
    }

    #[test]
    fn unsupported_scheme_errors() {
        let fetcher = ExternalFetcher::new(Duration::from_secs(60));
        let err = fetcher.fetch("ftp://host/file").expect_err("must fail");
        assert!(err.to_string().contains("ftp://host/file"));
    }

    #[test]
    fn cloud_uris_build_operators() {
        assert!(build_operator("s3://bucket/key.bin").is_ok());
        assert!(build_operator("https://example.com/key.bin").is_ok());
        assert!(build_operator("s3://only-bucket").is_err());
        assert!(build_operator("gcs://only-bucket").is_err());
        assert!(build_operator("azblob://only-container").is_err());
    }
}
