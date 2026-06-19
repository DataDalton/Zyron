//! Integration tests for the CDC webhook sink against a local mock HTTP
//! server. Verifies the batch is delivered as a JSON array with the configured
//! headers, and that a non-2xx response surfaces as an error (no silent loss).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;

use bytes::Bytes;
use zyron_cdc::cdc_stream::{CdcSink, CdcSinkConfig, WebhookSink};

/// Reads a full HTTP/1.1 request (headers + Content-Length body) from a stream.
fn read_http_request(stream: &mut TcpStream) -> String {
    let mut data = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        // Stop once headers are complete and the declared body is fully read.
        if let Some(hdr_end) = find_subslice(&data, b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&data[..hdr_end]).to_lowercase();
            let content_len = headers
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if data.len() >= hdr_end + 4 + content_len {
                break;
            }
        }
        let n = stream.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
    }
    String::from_utf8_lossy(&data).to_string()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Starts a one-shot mock HTTP server that replies with `status_line`, captures
/// the request, and returns the bound address plus a receiver for the request.
fn spawn_mock_server(status_line: &'static str) -> (String, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let req = read_http_request(&mut stream);
            let body = b"ok";
            let resp = format!(
                "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
            let _ = tx.send(req);
        }
    });
    (format!("http://{addr}/hook"), rx)
}

#[test]
fn webhook_delivers_batch_as_json_array_with_headers() {
    let (url, rx) = spawn_mock_server("HTTP/1.1 200 OK");
    let sink = WebhookSink::new(
        CdcSinkConfig::Webhook {
            url,
            headers: vec![("x-zyron-stream".into(), "orders".into())],
            batch_size: 10,
        },
        "orders".into(),
    );

    let changes = vec![
        Bytes::from_static(b"{\"id\":1}"),
        Bytes::from_static(b"{\"id\":2}"),
    ];
    sink.write_batch(&changes).expect("webhook delivery");

    let req = rx.recv().expect("server captured request");
    assert!(
        req.starts_with("POST /hook"),
        "method/path: {}",
        &req[..req.find('\r').unwrap_or(req.len())]
    );
    assert!(
        req.to_lowercase().contains("x-zyron-stream: orders"),
        "custom header present"
    );
    assert!(
        req.contains("[{\"id\":1},{\"id\":2}]"),
        "JSON array body present"
    );
}

#[test]
fn webhook_non_2xx_is_an_error() {
    let (url, _rx) = spawn_mock_server("HTTP/1.1 500 Internal Server Error");
    let sink = WebhookSink::new(
        CdcSinkConfig::Webhook {
            url,
            headers: vec![],
            batch_size: 10,
        },
        "orders".into(),
    );
    let changes = vec![Bytes::from_static(b"{\"id\":1}")];
    let result = sink.write_batch(&changes);
    assert!(result.is_err(), "a 500 response must surface as an error");
}

#[test]
fn webhook_empty_batch_is_noop() {
    let sink = WebhookSink::new(
        CdcSinkConfig::Webhook {
            url: "http://127.0.0.1:1/never".into(),
            headers: vec![],
            batch_size: 10,
        },
        "orders".into(),
    );
    // No connection attempted for an empty batch.
    sink.write_batch(&[]).expect("empty batch is a no-op");
}
