//! Media columns end to end over SQL: IMAGE, VIDEO, AUDIO, DOCUMENT and
//! EXTERNAL_REF round trips, metadata extraction, storage mode selection by
//! payload size with inline tuples, compressed toast objects and
//! uncompressed content addressed objects, refcounted dedup, presigned
//! URLs, pure Rust image operations, actionable errors for unconfigured
//! external tools, and aggregate rejection over media values
//!
//! Run: cargo test -p zyron-wire --test media_columns_test

mod common;

use std::sync::Arc;

use common::{
    create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_error, query_values,
};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

// ---------------------------------------------------------------------------
// Generated media fixtures, no binary files are checked in
// ---------------------------------------------------------------------------

fn png_chunk(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(12 + body.len());
    chunk.extend_from_slice(&(body.len() as u32).to_be_bytes());
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(body);
    let mut crc_input = Vec::with_capacity(4 + body.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(body);
    chunk.extend_from_slice(&zyron_types::encoding::crc32(&crc_input).to_be_bytes());
    chunk
}

/// Valid rgb8 png built by hand, the zlib stream uses stored deflate blocks
fn sample_png(width: u32, height: u32) -> Vec<u8> {
    png_from_pixel(width, height, |x, y| {
        [(x % 256) as u8, (y % 256) as u8, 128]
    })
}

/// One flat color, so the encoded bytes stay compressible by the store's
/// lz4 pass. The gradient fixture defeats lz4, its three byte pixel stride
/// with an incrementing channel never yields a four byte match
fn solid_png(width: u32, height: u32) -> Vec<u8> {
    png_from_pixel(width, height, |_, _| [7, 42, 128])
}

fn png_from_pixel(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

    let mut raw = Vec::new();
    for y in 0..height {
        raw.push(0u8);
        for x in 0..width {
            raw.extend_from_slice(&pixel(x, y));
        }
    }
    let mut zlib = vec![0x78, 0x01];
    let mut offset = 0usize;
    while offset < raw.len() {
        let end = (offset + 65_535).min(raw.len());
        let block = &raw[offset..end];
        zlib.push(if end == raw.len() { 1 } else { 0 });
        zlib.extend_from_slice(&(block.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(block);
        offset = end;
    }
    zlib.extend_from_slice(&zyron_types::checksum::adler32(&raw).to_be_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
    png.extend_from_slice(&png_chunk(b"IDAT", &zlib));
    png.extend_from_slice(&png_chunk(b"IEND", &[]));
    png
}

/// PCM wav with a real fmt and data chunk
fn sample_wav(sample_rate: u32, channels: u16, seconds: f64) -> Vec<u8> {
    let bits: u16 = 16;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (byte_rate as f64 * seconds) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend(std::iter::repeat_n(0u8, data_len as usize));
    out
}

fn mp4_box(typ: &[u8; 4], payload: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
    out.extend_from_slice(typ);
    out.extend_from_slice(&payload);
    out
}

/// ISO-BMFF mp4 with one avc1 video track, 320x240, 5 seconds, 120 samples
fn minimal_mp4() -> Vec<u8> {
    let mvhd = {
        let mut p = vec![0u8; 100];
        p[12..16].copy_from_slice(&1000u32.to_be_bytes());
        p[16..20].copy_from_slice(&5000u32.to_be_bytes());
        mp4_box(b"mvhd", p)
    };
    let tkhd = {
        let mut p = vec![0u8; 84];
        p[76..80].copy_from_slice(&(320u32 << 16).to_be_bytes());
        p[80..84].copy_from_slice(&(240u32 << 16).to_be_bytes());
        mp4_box(b"tkhd", p)
    };
    let mdhd = {
        let mut p = vec![0u8; 24];
        p[12..16].copy_from_slice(&1000u32.to_be_bytes());
        p[16..20].copy_from_slice(&5000u32.to_be_bytes());
        mp4_box(b"mdhd", p)
    };
    let hdlr = {
        let mut p = vec![0u8; 24];
        p[8..12].copy_from_slice(b"vide");
        mp4_box(b"hdlr", p)
    };
    let stsd = {
        let mut p = vec![0u8; 8];
        p[4..8].copy_from_slice(&1u32.to_be_bytes());
        let mut entry = vec![0u8; 86];
        entry[..4].copy_from_slice(&86u32.to_be_bytes());
        entry[4..8].copy_from_slice(b"avc1");
        p.extend_from_slice(&entry);
        mp4_box(b"stsd", p)
    };
    let stts = {
        let mut p = vec![0u8; 8];
        p[4..8].copy_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&120u32.to_be_bytes());
        p.extend_from_slice(&250u32.to_be_bytes());
        mp4_box(b"stts", p)
    };
    let ftyp = mp4_box(b"ftyp", b"isom\0\0\x02\0isomiso2avc1mp41".to_vec());
    let mdat = mp4_box(b"mdat", vec![0u8; 4000]);
    let stbl = mp4_box(b"stbl", [stsd, stts].concat());
    let minf = mp4_box(b"minf", stbl);
    let mdia = mp4_box(b"mdia", [mdhd, hdlr, minf].concat());
    let trak = mp4_box(b"trak", [tkhd, mdia].concat());
    let moov = mp4_box(b"moov", [mvhd, trak].concat());
    [ftyp, mdat, moov].concat()
}

/// Single page pdf with one text stream plus Author and Title info values
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

// ---------------------------------------------------------------------------
// SQL and result helpers
// ---------------------------------------------------------------------------

/// Inserts one row of (id, payload) through the hex_decode path, which is
/// how raw bytes travel in a SQL literal
async fn insert_media(server: &Arc<ServerState>, table: &str, id: i32, payload: &[u8]) {
    let sql = format!(
        "INSERT INTO {table} VALUES ({id}, hex_decode('{}'))",
        zyron_types::encoding::hex_encode(payload)
    );
    exec_dml(server, &sql).await;
}

fn one_binary(rows: &[Vec<ScalarValue>]) -> Vec<u8> {
    assert_eq!(rows.len(), 1, "expected one row, got {}", rows.len());
    match &rows[0][0] {
        ScalarValue::Binary(b) => b.clone(),
        other => panic!("expected binary, got {other:?}"),
    }
}

fn one_text(rows: &[Vec<ScalarValue>]) -> String {
    assert_eq!(rows.len(), 1, "expected one row, got {}", rows.len());
    match &rows[0][0] {
        ScalarValue::Utf8(s) => s.clone(),
        other => panic!("expected text, got {other:?}"),
    }
}

fn one_json(rows: &[Vec<ScalarValue>]) -> serde_json::Value {
    serde_json::from_str(&one_text(rows)).expect("metadata parses as json")
}

fn one_int(rows: &[Vec<ScalarValue>]) -> i64 {
    assert_eq!(rows.len(), 1, "expected one row, got {}", rows.len());
    match &rows[0][0] {
        ScalarValue::Int32(v) => *v as i64,
        ScalarValue::Int64(v) => *v,
        other => panic!("expected an integer, got {other:?}"),
    }
}

/// Subdirectories of the store's media dir. Object files land in two hex
/// character shards, so a store holding nothing has none
fn shard_dirs(media_dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(media_dir)
        .expect("read media dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect()
}

/// On disk object file for a content hash
fn object_path(media_dir: &std::path::Path, sha: &[u8; 32]) -> std::path::PathBuf {
    let hex = zyron_types::encoding::hex_encode(sha);
    media_dir.join(&hex[..2]).join(hex)
}

// ---------------------------------------------------------------------------
// Task 30, IMAGE end to end
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_image_round_trips_and_reports_metadata() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE pics (id INT, img IMAGE)",
    )
    .await
    .expect("create");

    let png = sample_png(16, 16);
    insert_media(&server, "pics", 1, &png).await;

    let back = one_binary(&query_values(&server, "SELECT img FROM pics").await);
    assert_eq!(back, png, "stored image reads back byte identical");

    let meta = one_json(&query_values(&server, "SELECT image_metadata(img) FROM pics").await);
    assert_eq!(meta["width"], 16);
    assert_eq!(meta["height"], 16);
    assert_eq!(meta["format"], "png");
    assert_eq!(meta["color_space"], "rgb");
    assert_eq!(meta["bit_depth"], 8);
}

/// FORMAT hint enforcement on write, a payload of another format is refused
/// and a matching one is stored
#[tokio::test]
async fn test_image_format_hint_checked_on_write() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hinted (id INT, img IMAGE FORMAT 'jpeg')",
    )
    .await
    .expect("create hinted");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hinted_ok (id INT, img IMAGE FORMAT 'png')",
    )
    .await
    .expect("create hinted_ok");

    let png = sample_png(8, 8);
    let hex = zyron_types::encoding::hex_encode(&png);

    let err = exec_dml_result(
        &server,
        &format!("INSERT INTO hinted VALUES (1, hex_decode('{hex}'))"),
    )
    .await
    .expect_err("a png into a FORMAT 'jpeg' column must be refused");
    let msg = err.to_string();
    assert!(msg.contains("FORMAT 'jpeg'"), "message was: {msg}");
    assert!(msg.contains("png"), "message was: {msg}");

    exec_dml(
        &server,
        &format!("INSERT INTO hinted_ok VALUES (1, hex_decode('{hex}'))"),
    )
    .await;
    let back = one_binary(&query_values(&server, "SELECT img FROM hinted_ok").await);
    assert_eq!(back, png);
}

// ---------------------------------------------------------------------------
// Task 31, VIDEO container metadata and the ffmpeg skeleton
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_video_metadata_parses_mp4_and_tools_error_actionably() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE vids (id INT, vid VIDEO)",
    )
    .await
    .expect("create");

    let mp4 = minimal_mp4();
    insert_media(&server, "vids", 1, &mp4).await;

    let back = one_binary(&query_values(&server, "SELECT vid FROM vids").await);
    assert_eq!(back, mp4, "stored video reads back byte identical");

    // The box tree parses in pure Rust, no external tool involved
    let meta = one_json(&query_values(&server, "SELECT video_metadata(vid) FROM vids").await);
    assert_eq!(meta["duration_seconds"], 5.0);
    assert_eq!(meta["resolution"]["width"], 320);
    assert_eq!(meta["resolution"]["height"], 240);
    assert_eq!(meta["codec"], "avc1");
    assert_eq!(meta["framerate"], 24.0);
    assert_eq!(meta["audio_streams"], 0);
    assert_eq!(meta["bitrate"], 6400);

    // Frame extraction needs ffmpeg, the refusal names the tool and the key
    let err = query_error(&server, "SELECT video_thumbnail(vid) FROM vids").await;
    assert!(err.contains("ffmpeg"), "error was: {err}");
    assert!(err.contains("media.ffmpeg_path"), "error was: {err}");
    assert!(err.contains("zyron.toml"), "error was: {err}");
}

// ---------------------------------------------------------------------------
// Task 32, AUDIO wav header metadata and the ffmpeg skeleton
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_audio_metadata_parses_wav_and_transcode_errors_actionably() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE tracks (id INT, aud AUDIO)",
    )
    .await
    .expect("create");

    let wav = sample_wav(8000, 1, 0.25);
    insert_media(&server, "tracks", 1, &wav).await;

    let back = one_binary(&query_values(&server, "SELECT aud FROM tracks").await);
    assert_eq!(back, wav, "stored audio reads back byte identical");

    // The wav header parses in pure Rust
    let meta = one_json(&query_values(&server, "SELECT audio_metadata(aud) FROM tracks").await);
    assert_eq!(meta["codec"], "pcm");
    assert_eq!(meta["sample_rate"], 8000);
    assert_eq!(meta["channels"], 1);
    assert_eq!(meta["bits_per_sample"], 16);
    assert_eq!(meta["duration_seconds"], 0.25);
    assert_eq!(meta["bitrate"], 128_000);

    // Transcoding needs ffmpeg, the refusal names the tool and the key
    let err = query_error(
        &server,
        "SELECT audio_transcode(aud, 'flac', 128000) FROM tracks",
    )
    .await;
    assert!(err.contains("ffmpeg"), "error was: {err}");
    assert!(err.contains("media.ffmpeg_path"), "error was: {err}");
}

// ---------------------------------------------------------------------------
// Task 33, DOCUMENT text extraction and the tesseract skeleton
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_document_text_extracts_and_ocr_errors_actionably() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, doc DOCUMENT)",
    )
    .await
    .expect("create");

    let pdf = sample_pdf();
    insert_media(&server, "docs", 1, &pdf).await;

    // The pdf content stream parses in pure Rust
    let text =
        one_text(&query_values(&server, "SELECT document_extract_text(doc) FROM docs").await);
    assert!(
        text.contains("Hello Zyron media"),
        "extracted text was: {text}"
    );

    let pages = one_int(&query_values(&server, "SELECT document_page_count(doc) FROM docs").await);
    assert_eq!(pages, 1);

    let meta = one_json(&query_values(&server, "SELECT document_metadata(doc) FROM docs").await);
    assert_eq!(meta["mime_type"], "application/pdf");
    assert_eq!(meta["page_count"], 1);
    assert_eq!(meta["author"], "Zyron Tests");
    assert_eq!(meta["title"], "Media Fixture");

    // OCR needs tesseract, the refusal names the tool and the key
    let err = query_error(&server, "SELECT image_ocr(doc) FROM docs").await;
    assert!(err.contains("tesseract"), "error was: {err}");
    assert!(err.contains("media.tesseract_path"), "error was: {err}");
    assert!(err.contains("zyron.toml"), "error was: {err}");
}

// ---------------------------------------------------------------------------
// Task 34, EXTERNAL_REF uri values
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_external_ref_stores_uri_and_resolves_through_the_fetcher() {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE refs (id INT, loc EXTERNAL_REF)",
    )
    .await
    .expect("create");

    // A real object behind a file uri, written outside the media store
    let src_dir = tmp.path().join("external_src");
    std::fs::create_dir_all(&src_dir).expect("create source dir");
    let object_bytes = b"external object bytes for the reference test".to_vec();
    let object_file = src_dir.join("object.bin");
    std::fs::write(&object_file, &object_bytes).expect("write source object");
    let uri = format!("file://{}", object_file.display()).replace('\\', "/");

    exec_dml(&server, &format!("INSERT INTO refs VALUES (1, '{uri}')")).await;

    // The uri lands as an ExternalUri descriptor, nothing enters the
    // content addressed store
    assert!(
        shard_dirs(&tmp.path().join("media")).is_empty(),
        "an external reference must not copy the object into the store"
    );

    // A select resolves the reference through the external fetcher and
    // answers the bytes behind the uri
    let back = one_binary(&query_values(&server, "SELECT loc FROM refs").await);
    assert_eq!(back, object_bytes);

    // A scheme the fetcher does not speak errors with the uri and the
    // supported scheme list, no configuration exists that would allow it
    exec_dml(
        &server,
        "INSERT INTO refs VALUES (2, 'ftp://host/object.bin')",
    )
    .await;
    let err = query_error(&server, "SELECT loc FROM refs").await;
    assert!(
        err.contains("unsupported external uri scheme"),
        "error was: {err}"
    );
    assert!(err.contains("ftp://host/object.bin"), "error was: {err}");
    assert!(err.contains("file, s3"), "error was: {err}");
}

// ---------------------------------------------------------------------------
// Task 35, payloads under 8KB stay inline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_small_payload_stays_inline_and_the_store_stays_empty() {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE inline_docs (id INT, doc DOCUMENT)",
    )
    .await
    .expect("create");

    let mut text = String::new();
    let mut line = 0u32;
    while text.len() < 2000 {
        text.push_str(&format!("inline document payload line {line}\n"));
        line += 1;
    }
    let payload = text.into_bytes();
    assert!(
        payload.len() < 8 * 1024,
        "fixture must stay under the inline limit"
    );
    insert_media(&server, "inline_docs", 1, &payload).await;

    let back = one_binary(&query_values(&server, "SELECT doc FROM inline_docs").await);
    assert_eq!(back, payload, "inline payload reads back byte identical");

    // The payload lives in the tuple, the content addressed store holds
    // nothing for it
    let sha = zyron_types::crypto::sha256(&payload);
    assert!(
        !server.media_store.contains(&sha),
        "an inline payload must not enter the content addressed store"
    );
    assert!(
        shard_dirs(&tmp.path().join("media")).is_empty(),
        "the media directory must hold no object shards"
    );
}

// ---------------------------------------------------------------------------
// Task 36, payloads between 8KB and 1MB toast into the store compressed
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_medium_payload_toasts_compressed_and_round_trips() {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE toasted (id INT, img IMAGE)",
    )
    .await
    .expect("create");

    let png = solid_png(200, 160);
    assert!(
        png.len() > 8 * 1024 && png.len() <= 1024 * 1024,
        "fixture must sit in the toast band, got {} bytes",
        png.len()
    );
    insert_media(&server, "toasted", 1, &png).await;

    let sha = zyron_types::crypto::sha256(&png);
    assert!(
        server.media_store.contains(&sha),
        "a toast payload lands in the content addressed store"
    );
    assert_eq!(server.media_store.count_of(&sha), 1);
    assert_eq!(
        server.media_store.len_of(&sha).expect("stored length"),
        png.len() as u64
    );

    // The object file header records lz4 compression, which is what
    // separates toast from the uncompressed external mode
    let raw = std::fs::read(object_path(&tmp.path().join("media"), &sha))
        .expect("object file exists on disk");
    assert_eq!(&raw[..4], b"ZYMO");
    assert_eq!(raw[5], 1, "toast objects store compressed");

    let back = one_binary(&query_values(&server, "SELECT img FROM toasted").await);
    assert_eq!(back, png, "toast payload reads back byte identical");

    let meta = one_json(&query_values(&server, "SELECT image_metadata(img) FROM toasted").await);
    assert_eq!(meta["width"], 200);
    assert_eq!(meta["height"], 160);
}

// ---------------------------------------------------------------------------
// Task 37, payloads over 1MB store uncompressed with refcounted dedup
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_large_payload_goes_external_and_dedups_by_content() {
    let (server, _schema, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE blobs (id INT, doc DOCUMENT)",
    )
    .await
    .expect("create");

    let mut text = String::with_capacity(1_100_000);
    let mut line = 0u32;
    while text.len() <= 1024 * 1024 {
        text.push_str(&format!(
            "external payload line {line} for the content addressed store\n"
        ));
        line += 1;
    }
    let payload = text.into_bytes();
    assert!(
        payload.len() > 1024 * 1024,
        "fixture must exceed the toast limit"
    );

    insert_media(&server, "blobs", 1, &payload).await;
    insert_media(&server, "blobs", 2, &payload).await;

    let sha = zyron_types::crypto::sha256(&payload);
    let media_dir = tmp.path().join("media");
    let object = object_path(&media_dir, &sha);
    assert!(
        object.is_file(),
        "object file exists under media/<2hex>/<sha256>"
    );
    assert!(server.media_store.contains(&sha));

    // Two rows of the same bytes share one object with two references
    assert_eq!(
        server.media_store.count_of(&sha),
        2,
        "two inserts of the same bytes hold two references on one object"
    );
    let hex = zyron_types::encoding::hex_encode(&sha);
    let shard_files = std::fs::read_dir(media_dir.join(&hex[..2]))
        .expect("read shard dir")
        .count();
    assert_eq!(shard_files, 1, "the shard holds one object file, not two");

    // External objects store uncompressed
    let raw = std::fs::read(&object).expect("read object file");
    assert_eq!(&raw[..4], b"ZYMO");
    assert_eq!(raw[6..], payload[..], "the object body is the raw payload");
    assert_eq!(raw[5], 0, "external objects store uncompressed");

    let rows = query_values(&server, "SELECT doc FROM blobs ORDER BY id").await;
    assert_eq!(rows.len(), 2);
    for row in &rows {
        match &row[0] {
            ScalarValue::Binary(b) => assert_eq!(b, &payload, "row reads back byte identical"),
            other => panic!("expected binary, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Task 38, presigned URLs over stored media
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_presigned_url_signs_the_content_address() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE signed_pics (id INT, img IMAGE)",
    )
    .await
    .expect("create");

    let png = sample_png(16, 16);
    insert_media(&server, "signed_pics", 1, &png).await;
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs() as i64;

    let url =
        one_text(&query_values(&server, "SELECT presigned_url(img, '1h') FROM signed_pics").await);
    let expected_resource = format!(
        "media/{}",
        zyron_types::encoding::hex_encode(&zyron_types::crypto::sha256(&png))
    );
    assert!(
        url.starts_with(&format!("{expected_resource}?")),
        "url was: {url}"
    );
    assert!(url.contains("zx="), "url carries an expiry, was: {url}");
    assert!(url.contains("zm=GET"), "url carries the method, was: {url}");
    assert!(url.contains("zs="), "url carries a signature, was: {url}");

    // The signature verifies against the secret the harness installs, so
    // the deterministic pieces are provable rather than just present
    let handle = zyron_media::presign::verify(&[7u8; 32], &url, before)
        .expect("signed url verifies with the installed secret");
    assert_eq!(handle.resource, expected_resource);
    assert_eq!(handle.method, "GET");
    assert!(
        handle.expires_at_unix >= before + 3590 && handle.expires_at_unix <= before + 3620,
        "expiry {} is not about an hour after {before}",
        handle.expires_at_unix
    );

    // The method argument travels into the signature
    let post_url = one_text(
        &query_values(
            &server,
            "SELECT presigned_url(img, '30m', 'POST') FROM signed_pics",
        )
        .await,
    );
    assert!(post_url.contains("zm=POST"), "url was: {post_url}");
    let post_handle =
        zyron_media::presign::verify(&[7u8; 32], &post_url, before).expect("post url verifies");
    assert_eq!(post_handle.method, "POST");

    // A url signed with another secret does not verify
    assert!(zyron_media::presign::verify(&[8u8; 32], &url, before).is_err());
}

// ---------------------------------------------------------------------------
// Task 39, pure Rust image operations
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_image_ops_produce_the_expected_dimensions() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE art (id INT, img IMAGE)",
    )
    .await
    .expect("create");
    insert_media(&server, "art", 1, &sample_png(64, 48)).await;

    let dims = |meta: serde_json::Value| {
        (
            meta["width"].as_u64().expect("width"),
            meta["height"].as_u64().expect("height"),
        )
    };

    // Fit preserves aspect ratio inside the box
    let fit = one_json(
        &query_values(
            &server,
            "SELECT image_metadata(image_resize(img, 32, 32)) FROM art",
        )
        .await,
    );
    assert_eq!(dims(fit), (32, 24));

    // Cover fills the box exactly
    let cover = one_json(
        &query_values(
            &server,
            "SELECT image_metadata(image_resize(img, 32, 32, 'cover')) FROM art",
        )
        .await,
    );
    assert_eq!(dims(cover), (32, 32));

    let cropped = one_json(
        &query_values(
            &server,
            "SELECT image_metadata(image_crop(img, 10, 10, 20, 20)) FROM art",
        )
        .await,
    );
    assert_eq!(dims(cropped), (20, 20));

    // A quarter turn swaps the dimensions
    let rotated = one_json(
        &query_values(
            &server,
            "SELECT image_metadata(image_rotate(img, 90)) FROM art",
        )
        .await,
    );
    assert_eq!(dims(rotated), (48, 64));

    // Conversion re encodes into the target format at the same size
    let converted = one_json(
        &query_values(
            &server,
            "SELECT image_metadata(image_format(img, 'jpeg')) FROM art",
        )
        .await,
    );
    assert_eq!(converted["format"], "jpeg");
    assert_eq!(dims(converted), (64, 48));
}

// ---------------------------------------------------------------------------
// Task 40, arithmetic aggregates reject media values at bind
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_arithmetic_aggregates_reject_media_at_planning() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE agg_pics (id INT, img IMAGE)",
    )
    .await
    .expect("create");
    insert_media(&server, "agg_pics", 1, &sample_png(8, 8)).await;

    for func in ["SUM", "AVG", "MIN", "MAX"] {
        let err = query_error(&server, &format!("SELECT {func}({}) FROM agg_pics", "img")).await;
        assert!(
            err.contains("does not apply"),
            "{func}(img) error was: {err}"
        );
        assert!(
            err.to_lowercase().contains(&func.to_lowercase()),
            "{func}(img) error names the aggregate, was: {err}"
        );
        assert!(
            err.to_lowercase().contains("image"),
            "{func}(img) error names the type, was: {err}"
        );
    }

    // COUNT stays legal over media, the rejection is specific to arithmetic
    let counted = one_int(&query_values(&server, "SELECT COUNT(img) FROM agg_pics").await);
    assert_eq!(counted, 1);
}
