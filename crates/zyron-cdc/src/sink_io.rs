//! Shared network IO for CDC sinks: a dedicated async runtime that bridges the
//! synchronous CdcSink trait to async HTTP, an HTTP client, and AWS SigV4
//! request signing for S3.

use std::future::Future;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zyron_common::{Result, ZyronError};

type HmacSha256 = Hmac<Sha256>;

/// Dedicated multi-thread runtime for CDC sink network IO. The sink trait is
/// synchronous, so write_batch bridges to async by spawning onto this runtime
/// and blocking on the result. Running the future on a separate runtime,
/// rather than block_on on the caller's, avoids the nested-runtime panic when
/// the pump drives sinks from inside a tokio task.
fn io_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("cdc-sink-io")
            .build()
            .expect("build CDC sink IO runtime")
    })
}

/// Runs an async IO future to completion from a synchronous context, including
/// from within another tokio runtime's worker thread. The future resolves to a
/// Result so a cancelled or panicked IO task surfaces as a ZyronError on the
/// calling thread rather than unwinding it.
pub fn block_on_io<F, T>(fut: F) -> Result<T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    io_runtime().spawn(async move {
        let _ = tx.send(fut.await);
    });
    rx.recv().unwrap_or_else(|_| {
        Err(ZyronError::CdcIngestError(
            "CDC sink IO task was dropped before completing".into(),
        ))
    })
}

/// Shared HTTP client for webhook and S3 sinks. rustls-backed, pure Rust.
pub fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("build CDC HTTP client")
    })
}

// ---------------------------------------------------------------------------
// Webhook delivery
// ---------------------------------------------------------------------------

/// POSTs a body to a webhook URL with the configured headers. Returns an error
/// on any non-2xx response so the caller does not advance its checkpoint past
/// an undelivered batch.
pub fn webhook_post(url: &str, headers: &[(String, String)], body: Vec<u8>) -> Result<()> {
    let url = url.to_string();
    let headers = headers.to_vec();
    block_on_io(async move {
        let mut req = http_client()
            .post(&url)
            .header("content-type", "application/json")
            .body(body);
        for (k, v) in &headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req.send().await.map_err(|e| {
            ZyronError::CdcStreamError(format!("webhook POST to {url} failed: {e}"))
        })?;
        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(ZyronError::CdcStreamError(format!(
                "webhook {url} returned {status}: {detail}"
            )));
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// S3 delivery with AWS SigV4
// ---------------------------------------------------------------------------

/// AWS credentials resolved from the standard environment variables.
#[derive(Clone)]
pub struct S3Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
}

impl S3Credentials {
    pub fn from_env() -> Result<Self> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| {
            ZyronError::CdcStreamError("AWS_ACCESS_KEY_ID is not set for the S3 sink".into())
        })?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| {
            ZyronError::CdcStreamError("AWS_SECRET_ACCESS_KEY is not set for the S3 sink".into())
        })?;
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();
        Ok(Self {
            access_key,
            secret_key,
            session_token,
        })
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(msg);
    mac.finalize().into_bytes().to_vec()
}

/// Computes the SigV4 signature and Authorization header value from the fully
/// assembled canonical-request components. Pure and deterministic so it can be
/// validated against AWS's published test vectors.
#[allow(clippy::too_many_arguments)]
pub fn sigv4_authorization(
    method: &str,
    canonical_uri: &str,
    canonical_query: &str,
    canonical_headers: &str,
    signed_headers: &str,
    payload_hash: &str,
    amz_date: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
    access_key: &str,
    secret_key: &str,
) -> (String, String) {
    let canonical_request = format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let k_date = hmac_sha256(
        format!("AWS4{secret_key}").as_bytes(),
        date_stamp.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    );
    (signature, authorization)
}

/// Percent-encodes an S3 object key for use in a request path, preserving `/`
/// segment separators. Unreserved characters per RFC 3986 are left as-is.
fn encode_s3_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for &b in key.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Converts a day count since the Unix epoch to a civil (year, month, day)
/// using Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Formats a Unix timestamp as the SigV4 (amz_date, date_stamp) pair:
/// `YYYYMMDDTHHMMSSZ` and `YYYYMMDD`.
fn amz_times(unix_secs: i64) -> (String, String) {
    let days = unix_secs.div_euclid(86_400);
    let sod = unix_secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let h = sod / 3600;
    let mi = (sod % 3600) / 60;
    let s = sod % 60;
    (
        format!("{y:04}{m:02}{d:02}T{h:02}{mi:02}{s:02}Z"),
        format!("{y:04}{m:02}{d:02}"),
    )
}

/// PUTs an object to S3 using SigV4 (virtual-hosted-style addressing). Returns
/// an error on any non-2xx response.
pub fn s3_put(
    bucket: &str,
    region: &str,
    key: &str,
    body: Vec<u8>,
    content_type: &str,
) -> Result<()> {
    let creds = S3Credentials::from_env()?;
    let host = format!("{bucket}.s3.{region}.amazonaws.com");
    let encoded_key = encode_s3_key(key);
    let canonical_uri = format!("/{encoded_key}");
    let url = format!("https://{host}{canonical_uri}");

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let (amz_date, date_stamp) = amz_times(now);
    let payload_hash = sha256_hex(&body);

    // Canonical and signed headers, sorted by lowercase name. A session token,
    // when present, is part of the signed set.
    let mut canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
    let mut signed_headers = String::from("host;x-amz-content-sha256;x-amz-date");
    if let Some(ref token) = creds.session_token {
        canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed_headers.push_str(";x-amz-security-token");
    }

    let (_sig, authorization) = sigv4_authorization(
        "PUT",
        &canonical_uri,
        "",
        &canonical_headers,
        &signed_headers,
        &payload_hash,
        &amz_date,
        &date_stamp,
        region,
        "s3",
        &creds.access_key,
        &creds.secret_key,
    );

    let content_type = content_type.to_string();
    block_on_io(async move {
        let mut req = http_client()
            .put(&url)
            .header("x-amz-date", amz_date)
            .header("x-amz-content-sha256", payload_hash)
            .header("authorization", authorization)
            .header("content-type", content_type)
            .body(body);
        if let Some(token) = creds.session_token {
            req = req.header("x-amz-security-token", token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ZyronError::CdcStreamError(format!("S3 PUT to {url} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(ZyronError::CdcStreamError(format!(
                "S3 PUT {url} returned {status}: {detail}"
            )));
        }
        Ok(())
    })
}

/// Percent-encodes a query parameter value per RFC 3986 (AWS canonical query
/// encoding): unreserved characters pass through, everything else is `%XX`.
fn encode_query_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for &b in value.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Signs and sends a GET to S3 with the given canonical URI and canonical query
/// string, returning the response body on success.
fn s3_signed_get(
    bucket: &str,
    region: &str,
    canonical_uri: &str,
    canonical_query: &str,
) -> Result<Vec<u8>> {
    let creds = S3Credentials::from_env()?;
    let host = format!("{bucket}.s3.{region}.amazonaws.com");
    let query_suffix = if canonical_query.is_empty() {
        String::new()
    } else {
        format!("?{canonical_query}")
    };
    let url = format!("https://{host}{canonical_uri}{query_suffix}");

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let (amz_date, date_stamp) = amz_times(now);
    let payload_hash = sha256_hex(b"");

    let mut canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
    let mut signed_headers = String::from("host;x-amz-content-sha256;x-amz-date");
    if let Some(ref token) = creds.session_token {
        canonical_headers.push_str(&format!("x-amz-security-token:{token}\n"));
        signed_headers.push_str(";x-amz-security-token");
    }

    let (_sig, authorization) = sigv4_authorization(
        "GET",
        canonical_uri,
        canonical_query,
        &canonical_headers,
        &signed_headers,
        &payload_hash,
        &amz_date,
        &date_stamp,
        region,
        "s3",
        &creds.access_key,
        &creds.secret_key,
    );

    block_on_io(async move {
        let mut req = http_client()
            .get(&url)
            .header("x-amz-date", amz_date)
            .header("x-amz-content-sha256", payload_hash)
            .header("authorization", authorization);
        if let Some(token) = creds.session_token {
            req = req.header("x-amz-security-token", token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| ZyronError::CdcIngestError(format!("S3 GET {url} failed: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            return Err(ZyronError::CdcIngestError(format!(
                "S3 GET {url} returned {status}: {detail}"
            )));
        }
        let bytes = resp.bytes().await.map_err(|e| {
            ZyronError::CdcIngestError(format!("S3 GET {url} body read failed: {e}"))
        })?;
        Ok(bytes.to_vec())
    })
}

/// Fetches a single S3 object's bytes via a SigV4-signed GET.
pub fn s3_get(bucket: &str, region: &str, key: &str) -> Result<Vec<u8>> {
    let canonical_uri = format!("/{}", encode_s3_key(key));
    s3_signed_get(bucket, region, &canonical_uri, "")
}

/// Lists object keys under a prefix via ListObjectsV2, returning keys greater
/// than `start_after` in lexicographic order, capped at `max_keys`. The
/// canonical query parameters are sorted by name as SigV4 requires.
pub fn s3_list_objects_v2(
    bucket: &str,
    region: &str,
    prefix: &str,
    start_after: &str,
    max_keys: u32,
) -> Result<Vec<String>> {
    // SigV4 canonical query: parameters sorted by name, values RFC 3986 encoded.
    let mut params: Vec<(String, String)> = vec![
        ("list-type".to_string(), "2".to_string()),
        ("max-keys".to_string(), max_keys.to_string()),
    ];
    if !prefix.is_empty() {
        params.push(("prefix".to_string(), prefix.to_string()));
    }
    if !start_after.is_empty() {
        params.push(("start-after".to_string(), start_after.to_string()));
    }
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let canonical_query = params
        .iter()
        .map(|(k, v)| format!("{}={}", encode_query_value(k), encode_query_value(v)))
        .collect::<Vec<_>>()
        .join("&");

    let body = s3_signed_get(bucket, region, "/", &canonical_query)?;
    let text = String::from_utf8_lossy(&body);
    Ok(parse_s3_keys(&text))
}

/// Extracts `<Key>...</Key>` values from a ListObjectsV2 XML response in order.
fn parse_s3_keys(xml: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<Key>") {
        let after = &rest[start + 5..];
        if let Some(end) = after.find("</Key>") {
            keys.push(xml_unescape(&after[..end]));
            rest = &after[end + 6..];
        } else {
            break;
        }
    }
    keys
}

/// Unescapes the five XML predefined entities that appear in S3 object keys.
fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_s3_keys_extracts_in_order() {
        let xml = "<ListBucketResult><Contents><Key>a/1.json</Key></Contents>\
                   <Contents><Key>a/2.json</Key></Contents></ListBucketResult>";
        assert_eq!(parse_s3_keys(xml), vec!["a/1.json", "a/2.json"]);
    }

    #[test]
    fn parse_s3_keys_unescapes_entities() {
        let xml = "<Contents><Key>a&amp;b.json</Key></Contents>";
        assert_eq!(parse_s3_keys(xml), vec!["a&b.json"]);
    }

    #[test]
    fn encode_query_value_encodes_reserved() {
        assert_eq!(encode_query_value("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn sha256_hex_empty_matches_known_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // AWS SigV4 documented example: a GET to the IAM endpoint. Validates the
    // canonical-request hashing, the signing-key derivation chain, and the
    // final signature against the value published by AWS.
    #[test]
    fn sigv4_matches_aws_published_vector() {
        let payload_hash = sha256_hex(b"");
        let canonical_headers = "content-type:application/x-www-form-urlencoded; charset=utf-8\nhost:iam.amazonaws.com\nx-amz-date:20150830T123600Z\n";
        let (signature, authorization) = sigv4_authorization(
            "GET",
            "/",
            "Action=ListUsers&Version=2010-05-08",
            canonical_headers,
            "content-type;host;x-amz-date",
            &payload_hash,
            "20150830T123600Z",
            "20150830",
            "us-east-1",
            "iam",
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        );
        assert_eq!(
            signature,
            "5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
        );
        assert!(
            authorization.contains("Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request")
        );
        assert!(authorization.contains("SignedHeaders=content-type;host;x-amz-date"));
    }

    #[test]
    fn amz_times_formats_known_instant() {
        // 2015-08-30T12:36:00Z = 1440938160 seconds since the epoch.
        let (amz_date, date_stamp) = amz_times(1_440_938_160);
        assert_eq!(amz_date, "20150830T123600Z");
        assert_eq!(date_stamp, "20150830");
    }

    #[test]
    fn encode_s3_key_preserves_slashes_and_encodes_specials() {
        assert_eq!(encode_s3_key("a/b c/d.json"), "a/b%20c/d.json");
    }
}
