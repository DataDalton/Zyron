//! Media operation scalar functions bridged to the zyron-media engine
//!
//! Payload cells arrive as Binary columns and every function applies per
//! row with NULL passthrough. Engine failures abort the call carrying the
//! engine's actionable message, which names the missing tool and config
//! key for operations backed by external binaries. Option arguments are
//! literals and are read from row zero of their column

use crate::column::{Column, ColumnData, NullBitmap};
use crate::media_runtime;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_media::image_ops::ResizeMode;

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "image_metadata" => metadata_impl(
            "image_metadata(image)",
            args,
            num_rows,
            zyron_media::image_meta::image_metadata,
        ),
        "video_metadata" => metadata_impl(
            "video_metadata(video)",
            args,
            num_rows,
            zyron_media::video_meta::video_metadata,
        ),
        "audio_metadata" => metadata_impl(
            "audio_metadata(audio)",
            args,
            num_rows,
            zyron_media::audio_meta::audio_metadata,
        ),
        "document_metadata" => metadata_impl(
            "document_metadata(document)",
            args,
            num_rows,
            zyron_media::document::document_metadata,
        ),
        "image_resize" => image_resize_impl(args, num_rows),
        "image_crop" => image_crop_impl(args, num_rows),
        "image_rotate" => image_rotate_impl(args, num_rows),
        "image_format" => image_format_impl(args, num_rows),
        "image_ocr" => image_ocr_impl(args, num_rows),
        "image_embed" => image_embed_impl(args, num_rows),
        "video_transcode" => video_transcode_impl(args, num_rows),
        "video_extract_audio" => video_extract_audio_impl(args, num_rows),
        "video_extract_frame" => video_extract_frame_impl(args, num_rows),
        "video_thumbnail" => video_thumbnail_impl(args, num_rows),
        "audio_transcribe" => audio_transcribe_impl(args, num_rows),
        "audio_transcode" => audio_transcode_impl(args, num_rows),
        "audio_trim" => audio_trim_impl(args, num_rows),
        "document_extract_text" => document_text_impl(
            "document_extract_text(document)",
            args,
            num_rows,
            zyron_media::document::document_extract_text,
        ),
        "document_to_markdown" => document_text_impl(
            "document_to_markdown(document)",
            args,
            num_rows,
            zyron_media::document::document_to_markdown,
        ),
        "document_page_count" => document_page_count_impl(args, num_rows),
        "presigned_url" => presigned_url_impl(args, num_rows),
        "presigned_verify" => presigned_verify_impl(args, num_rows),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Argument access
// ---------------------------------------------------------------------------

fn exec_err(message: String) -> ZyronError {
    ZyronError::ExecutionError(message)
}

fn arity_exact(sig: &str, args: &[Column], expected: usize) -> Result<()> {
    if args.len() != expected {
        return Err(exec_err(format!(
            "{sig} takes exactly {expected} arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

fn arity_range(sig: &str, args: &[Column], min: usize, max: usize) -> Result<()> {
    if args.len() < min || args.len() > max {
        return Err(exec_err(format!(
            "{sig} takes between {min} and {max} arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

/// Per row view over one argument column. Literal arguments may arrive
/// shorter than the batch, reads past the end broadcast the last value
struct Rows<'a, T: Copy> {
    values: Vec<T>,
    nulls: &'a NullBitmap,
}

impl<'a, T: Copy> Rows<'a, T> {
    fn at(&self, row: usize) -> T {
        let idx = row.min(self.values.len() - 1);
        self.values[idx]
    }

    fn null_at(&self, row: usize) -> bool {
        if self.values.is_empty() {
            return true;
        }
        self.nulls.is_null(row.min(self.values.len() - 1))
    }
}

fn build_rows<'a, T: Copy>(
    sig: &str,
    what: &str,
    values: Vec<T>,
    nulls: &'a NullBitmap,
    num_rows: usize,
) -> Result<Rows<'a, T>> {
    if values.is_empty() && num_rows > 0 {
        return Err(exec_err(format!("{sig}: {what} column carries no values")));
    }
    Ok(Rows { values, nulls })
}

fn bytes_rows<'a>(
    sig: &str,
    what: &str,
    col: &'a Column,
    num_rows: usize,
) -> Result<Rows<'a, &'a [u8]>> {
    let values: Vec<&[u8]> = match &col.data {
        ColumnData::Binary(v) => v.iter().map(|b| b.as_slice()).collect(),
        ColumnData::Utf8(v) => v.iter().map(|s| s.as_bytes()).collect(),
        _ => {
            return Err(exec_err(format!("{sig}: {what} must be a binary payload")));
        }
    };
    build_rows(sig, what, values, &col.nulls, num_rows)
}

fn text_rows<'a>(
    sig: &str,
    what: &str,
    col: &'a Column,
    num_rows: usize,
) -> Result<Rows<'a, &'a str>> {
    let values: Vec<&str> = match &col.data {
        ColumnData::Utf8(v) => v.iter().map(|s| s.as_str()).collect(),
        _ => return Err(exec_err(format!("{sig}: {what} must be a string"))),
    };
    build_rows(sig, what, values, &col.nulls, num_rows)
}

fn int_rows<'a>(sig: &str, what: &str, col: &'a Column, num_rows: usize) -> Result<Rows<'a, i64>> {
    let values: Vec<i64> = match &col.data {
        ColumnData::Int64(v) => v.clone(),
        ColumnData::Int32(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Int16(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Int8(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::UInt32(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::UInt64(v) => v.iter().map(|&x| x as i64).collect(),
        _ => return Err(exec_err(format!("{sig}: {what} must be an integer"))),
    };
    build_rows(sig, what, values, &col.nulls, num_rows)
}

fn float_rows<'a>(
    sig: &str,
    what: &str,
    col: &'a Column,
    num_rows: usize,
) -> Result<Rows<'a, f64>> {
    let values: Vec<f64> = match &col.data {
        ColumnData::Float64(v) => v.clone(),
        ColumnData::Float32(v) => v.iter().map(|&x| x as f64).collect(),
        ColumnData::Int64(v) => v.iter().map(|&x| x as f64).collect(),
        ColumnData::Int32(v) => v.iter().map(|&x| x as f64).collect(),
        _ => return Err(exec_err(format!("{sig}: {what} must be numeric"))),
    };
    build_rows(sig, what, values, &col.nulls, num_rows)
}

/// Option text argument read from row zero, absent or NULL falls back to
/// the default. The rows are returned so NULL propagates to the output
fn option_text<'a>(
    sig: &str,
    what: &str,
    col: Option<&'a Column>,
    num_rows: usize,
    default: &str,
) -> Result<(String, Option<Rows<'a, &'a str>>)> {
    match col {
        Some(col) => {
            let rows = text_rows(sig, what, col, num_rows)?;
            let value = if num_rows > 0 && !rows.null_at(0) {
                rows.at(0).to_string()
            } else {
                default.to_string()
            };
            Ok((value, Some(rows)))
        }
        None => Ok((default.to_string(), None)),
    }
}

/// Option numeric argument read from row zero, same fallback rules
fn option_float<'a>(
    sig: &str,
    what: &str,
    col: Option<&'a Column>,
    num_rows: usize,
    default: f64,
) -> Result<(f64, Option<Rows<'a, f64>>)> {
    match col {
        Some(col) => {
            let rows = float_rows(sig, what, col, num_rows)?;
            let value = if num_rows > 0 && !rows.null_at(0) {
                rows.at(0)
            } else {
                default
            };
            Ok((value, Some(rows)))
        }
        None => Ok((default, None)),
    }
}

fn option_null_at(rows: &Option<Rows<'_, &str>>, row: usize) -> bool {
    rows.as_ref().is_some_and(|r| r.null_at(row))
}

fn to_u32(sig: &str, what: &str, value: i64) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        exec_err(format!(
            "{sig}: {what} {value} is out of range for an unsigned 32 bit value"
        ))
    })
}

// ---------------------------------------------------------------------------
// Output builders
// ---------------------------------------------------------------------------

struct Utf8Out {
    data: Vec<String>,
    nulls: NullBitmap,
}

impl Utf8Out {
    fn new(num_rows: usize) -> Self {
        Self {
            data: Vec::with_capacity(num_rows),
            nulls: NullBitmap::none(num_rows),
        }
    }

    fn push(&mut self, value: String) {
        self.data.push(value);
    }

    fn push_null(&mut self) {
        self.nulls.set_null(self.data.len());
        self.data.push(String::new());
    }

    fn finish(self, type_id: TypeId) -> Column {
        Column::with_nulls(ColumnData::Utf8(self.data), self.nulls, type_id)
    }
}

struct BinaryOut {
    data: Vec<Vec<u8>>,
    nulls: NullBitmap,
}

impl BinaryOut {
    fn new(num_rows: usize) -> Self {
        Self {
            data: Vec::with_capacity(num_rows),
            nulls: NullBitmap::none(num_rows),
        }
    }

    fn push(&mut self, value: Vec<u8>) {
        self.data.push(value);
    }

    fn push_null(&mut self) {
        self.nulls.set_null(self.data.len());
        self.data.push(Vec::new());
    }

    fn finish(self, type_id: TypeId) -> Column {
        Column::with_nulls(ColumnData::Binary(self.data), self.nulls, type_id)
    }
}

struct Int64Out {
    data: Vec<i64>,
    nulls: NullBitmap,
}

impl Int64Out {
    fn new(num_rows: usize) -> Self {
        Self {
            data: Vec::with_capacity(num_rows),
            nulls: NullBitmap::none(num_rows),
        }
    }

    fn push(&mut self, value: i64) {
        self.data.push(value);
    }

    fn push_null(&mut self) {
        self.nulls.set_null(self.data.len());
        self.data.push(0);
    }

    fn finish(self) -> Column {
        Column::with_nulls(ColumnData::Int64(self.data), self.nulls, TypeId::Int64)
    }
}

// ---------------------------------------------------------------------------
// Metadata and document functions
// ---------------------------------------------------------------------------

fn metadata_impl(
    sig: &str,
    args: &[Column],
    num_rows: usize,
    f: fn(&[u8]) -> zyron_media::MediaResult<serde_json::Value>,
) -> Result<Column> {
    arity_exact(sig, args, 1)?;
    let payload = bytes_rows(sig, "the media payload", &args[0], num_rows)?;
    let mut out = Utf8Out::new(num_rows);
    for row in 0..num_rows {
        if payload.null_at(row) {
            out.push_null();
            continue;
        }
        let value = f(payload.at(row)).map_err(ZyronError::from)?;
        out.push(value.to_string());
    }
    Ok(out.finish(TypeId::Jsonb))
}

fn document_text_impl(
    sig: &str,
    args: &[Column],
    num_rows: usize,
    f: fn(&[u8]) -> zyron_media::MediaResult<String>,
) -> Result<Column> {
    arity_exact(sig, args, 1)?;
    let payload = bytes_rows(sig, "the document payload", &args[0], num_rows)?;
    let mut out = Utf8Out::new(num_rows);
    for row in 0..num_rows {
        if payload.null_at(row) {
            out.push_null();
            continue;
        }
        out.push(f(payload.at(row)).map_err(ZyronError::from)?);
    }
    Ok(out.finish(TypeId::Text))
}

fn document_page_count_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "document_page_count(document)";
    arity_exact(SIG, args, 1)?;
    let payload = bytes_rows(SIG, "the document payload", &args[0], num_rows)?;
    let mut out = Int64Out::new(num_rows);
    for row in 0..num_rows {
        if payload.null_at(row) {
            out.push_null();
            continue;
        }
        out.push(
            zyron_media::document::document_page_count(payload.at(row))
                .map_err(ZyronError::from)?,
        );
    }
    Ok(out.finish())
}

// ---------------------------------------------------------------------------
// Image transformations
// ---------------------------------------------------------------------------

fn parse_resize_mode(sig: &str, mode: &str) -> Result<ResizeMode> {
    match mode.to_ascii_lowercase().as_str() {
        "fit" => Ok(ResizeMode::Fit),
        "cover" => Ok(ResizeMode::Cover),
        "stretch" => Ok(ResizeMode::Stretch),
        other => Err(exec_err(format!(
            "{sig}: resize mode {other} is not recognized, supported modes are fit, cover and stretch"
        ))),
    }
}

fn image_resize_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "image_resize(image, width, height, mode)";
    arity_range(SIG, args, 3, 4)?;
    let image = bytes_rows(SIG, "the image payload", &args[0], num_rows)?;
    let width = int_rows(SIG, "width", &args[1], num_rows)?;
    let height = int_rows(SIG, "height", &args[2], num_rows)?;
    let (mode_text, mode_rows) = option_text(SIG, "mode", args.get(3), num_rows, "fit")?;
    let mode = parse_resize_mode(SIG, &mode_text)?;
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if image.null_at(row)
            || width.null_at(row)
            || height.null_at(row)
            || option_null_at(&mode_rows, row)
        {
            out.push_null();
            continue;
        }
        let w = to_u32(SIG, "width", width.at(row))?;
        let h = to_u32(SIG, "height", height.at(row))?;
        out.push(
            zyron_media::image_ops::resize(image.at(row), w, h, mode).map_err(ZyronError::from)?,
        );
    }
    Ok(out.finish(TypeId::Image))
}

fn image_crop_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "image_crop(image, x, y, width, height)";
    arity_exact(SIG, args, 5)?;
    let image = bytes_rows(SIG, "the image payload", &args[0], num_rows)?;
    let x = int_rows(SIG, "x", &args[1], num_rows)?;
    let y = int_rows(SIG, "y", &args[2], num_rows)?;
    let width = int_rows(SIG, "width", &args[3], num_rows)?;
    let height = int_rows(SIG, "height", &args[4], num_rows)?;
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if image.null_at(row)
            || x.null_at(row)
            || y.null_at(row)
            || width.null_at(row)
            || height.null_at(row)
        {
            out.push_null();
            continue;
        }
        let cropped = zyron_media::image_ops::crop(
            image.at(row),
            to_u32(SIG, "x", x.at(row))?,
            to_u32(SIG, "y", y.at(row))?,
            to_u32(SIG, "width", width.at(row))?,
            to_u32(SIG, "height", height.at(row))?,
        )
        .map_err(ZyronError::from)?;
        out.push(cropped);
    }
    Ok(out.finish(TypeId::Image))
}

fn image_rotate_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "image_rotate(image, degrees)";
    arity_exact(SIG, args, 2)?;
    let image = bytes_rows(SIG, "the image payload", &args[0], num_rows)?;
    let degrees = int_rows(SIG, "degrees", &args[1], num_rows)?;
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if image.null_at(row) || degrees.null_at(row) {
            out.push_null();
            continue;
        }
        // negative angles map to their positive quarter turn equivalent
        let turned = degrees.at(row).rem_euclid(360) as u32;
        out.push(zyron_media::image_ops::rotate(image.at(row), turned).map_err(ZyronError::from)?);
    }
    Ok(out.finish(TypeId::Image))
}

fn image_format_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "image_format(image, target)";
    arity_exact(SIG, args, 2)?;
    let image = bytes_rows(SIG, "the image payload", &args[0], num_rows)?;
    let target = text_rows(SIG, "target", &args[1], num_rows)?;
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if image.null_at(row) || target.null_at(row) {
            out.push_null();
            continue;
        }
        let converted = zyron_media::image_ops::convert_format(image.at(row), target.at(row))
            .map_err(ZyronError::from)?;
        out.push(converted);
    }
    Ok(out.finish(TypeId::Image))
}

fn image_ocr_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "image_ocr(image, lang)";
    arity_range(SIG, args, 1, 2)?;
    let image = bytes_rows(SIG, "the image payload", &args[0], num_rows)?;
    let (lang, lang_rows) = option_text(SIG, "lang", args.get(1), num_rows, "eng")?;
    let config = media_runtime::tool_config();
    let mut out = Utf8Out::new(num_rows);
    for row in 0..num_rows {
        if image.null_at(row) || option_null_at(&lang_rows, row) {
            out.push_null();
            continue;
        }
        let text = zyron_media::skeleton::image_ocr(&config, image.at(row), &lang)
            .map_err(ZyronError::from)?;
        out.push(text);
    }
    Ok(out.finish(TypeId::Text))
}

fn image_embed_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "image_embed(image, model)";
    arity_range(SIG, args, 1, 2)?;
    let image = bytes_rows(SIG, "the image payload", &args[0], num_rows)?;
    // the model name defaults to clip, embedding requires a registered
    // model so the engine call errors until a model runtime exists
    let (_model, model_rows) = option_text(SIG, "model", args.get(1), num_rows, "clip")?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if image.null_at(row) || option_null_at(&model_rows, row) {
            out.push_null();
            continue;
        }
        let embedding = zyron_media::skeleton::image_embed(&config, image.at(row), false)
            .map_err(ZyronError::from)?;
        out.push(pack_f32(&embedding));
    }
    Ok(out.finish(TypeId::Vector))
}

/// Vector cells are packed little endian f32
fn pack_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Video and audio tool operations
// ---------------------------------------------------------------------------

fn positive_bitrate(sig: &str, bitrate: i64) -> Result<()> {
    if bitrate <= 0 {
        return Err(exec_err(format!(
            "{sig}: bitrate {bitrate} must be a positive bits per second value"
        )));
    }
    Ok(())
}

fn video_transcode_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "video_transcode(video, codec, bitrate, resolution)";
    arity_exact(SIG, args, 4)?;
    let video = bytes_rows(SIG, "the video payload", &args[0], num_rows)?;
    let codec = text_rows(SIG, "codec", &args[1], num_rows)?;
    let bitrate = int_rows(SIG, "bitrate", &args[2], num_rows)?;
    let resolution = text_rows(SIG, "resolution", &args[3], num_rows)?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if video.null_at(row)
            || codec.null_at(row)
            || bitrate.null_at(row)
            || resolution.null_at(row)
        {
            out.push_null();
            continue;
        }
        // Codec, bitrate and resolution all reach the encoder, and the
        // container follows from the codec
        positive_bitrate(SIG, bitrate.at(row))?;
        if resolution.at(row).is_empty() {
            return Err(exec_err(format!("{SIG}: resolution must not be empty")));
        }
        let transcoded = zyron_media::skeleton::video_transcode(
            &config,
            video.at(row),
            codec.at(row),
            bitrate.at(row),
            resolution.at(row),
        )
        .map_err(ZyronError::from)?;
        out.push(transcoded);
    }
    Ok(out.finish(TypeId::Video))
}

fn video_extract_audio_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "video_extract_audio(video)";
    arity_exact(SIG, args, 1)?;
    let video = bytes_rows(SIG, "the video payload", &args[0], num_rows)?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if video.null_at(row) {
            out.push_null();
            continue;
        }
        // audio is emitted as mp3, a container every ffmpeg build muxes to a pipe
        let audio = zyron_media::skeleton::video_extract_audio(&config, video.at(row), "mp3")
            .map_err(ZyronError::from)?;
        out.push(audio);
    }
    Ok(out.finish(TypeId::Audio))
}

fn video_extract_frame_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "video_extract_frame(video, at_seconds)";
    arity_exact(SIG, args, 2)?;
    let video = bytes_rows(SIG, "the video payload", &args[0], num_rows)?;
    let at_seconds = float_rows(SIG, "at_seconds", &args[1], num_rows)?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if video.null_at(row) || at_seconds.null_at(row) {
            out.push_null();
            continue;
        }
        let frame =
            zyron_media::skeleton::video_extract_frame(&config, video.at(row), at_seconds.at(row))
                .map_err(ZyronError::from)?;
        out.push(frame);
    }
    Ok(out.finish(TypeId::Image))
}

fn video_thumbnail_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "video_thumbnail(video, at_seconds)";
    arity_range(SIG, args, 1, 2)?;
    let video = bytes_rows(SIG, "the video payload", &args[0], num_rows)?;
    let (at_seconds, at_rows) = option_float(SIG, "at_seconds", args.get(1), num_rows, 0.0)?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if video.null_at(row) || at_rows.as_ref().is_some_and(|r| r.null_at(row)) {
            out.push_null();
            continue;
        }
        let thumbnail = zyron_media::skeleton::video_thumbnail(&config, video.at(row), at_seconds)
            .map_err(ZyronError::from)?;
        out.push(thumbnail);
    }
    Ok(out.finish(TypeId::Image))
}

fn audio_transcribe_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "audio_transcribe(audio, model)";
    arity_range(SIG, args, 1, 2)?;
    let audio = bytes_rows(SIG, "the audio payload", &args[0], num_rows)?;
    // the model name defaults to whisper-small, transcription requires a
    // registered model so the engine call errors until a model runtime exists
    let (_model, model_rows) = option_text(SIG, "model", args.get(1), num_rows, "whisper-small")?;
    let config = media_runtime::tool_config();
    let mut out = Utf8Out::new(num_rows);
    for row in 0..num_rows {
        if audio.null_at(row) || option_null_at(&model_rows, row) {
            out.push_null();
            continue;
        }
        let text = zyron_media::skeleton::audio_transcribe(&config, audio.at(row), false)
            .map_err(ZyronError::from)?;
        out.push(text);
    }
    Ok(out.finish(TypeId::Text))
}

fn audio_transcode_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "audio_transcode(audio, codec, bitrate)";
    arity_exact(SIG, args, 3)?;
    let audio = bytes_rows(SIG, "the audio payload", &args[0], num_rows)?;
    let codec = text_rows(SIG, "codec", &args[1], num_rows)?;
    let bitrate = int_rows(SIG, "bitrate", &args[2], num_rows)?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if audio.null_at(row) || codec.null_at(row) || bitrate.null_at(row) {
            out.push_null();
            continue;
        }
        // Codec and bitrate both reach the encoder, and the container
        // follows from the codec
        positive_bitrate(SIG, bitrate.at(row))?;
        let transcoded = zyron_media::skeleton::audio_transcode(
            &config,
            audio.at(row),
            codec.at(row),
            bitrate.at(row),
        )
        .map_err(ZyronError::from)?;
        out.push(transcoded);
    }
    Ok(out.finish(TypeId::Audio))
}

/// Trim keeps the source container, identified from the payload's magic bytes
fn sniff_audio_container(bytes: &[u8]) -> Result<&'static str> {
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return Ok("wav");
    }
    if bytes.len() >= 4 && &bytes[..4] == b"fLaC" {
        return Ok("flac");
    }
    if bytes.len() >= 3 && &bytes[..3] == b"ID3" {
        return Ok("mp3");
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0 {
        return Ok("mp3");
    }
    if bytes.len() >= 4 && &bytes[..4] == b"OggS" {
        return Ok("ogg");
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return Ok("mp4");
    }
    Err(exec_err(
        "audio_trim writes the trimmed audio in the source container, which was not recognized, \
         supported containers are wav, flac, mp3, ogg and mp4"
            .to_string(),
    ))
}

fn audio_trim_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "audio_trim(audio, start_seconds, end_seconds)";
    arity_exact(SIG, args, 3)?;
    let audio = bytes_rows(SIG, "the audio payload", &args[0], num_rows)?;
    let start = float_rows(SIG, "start_seconds", &args[1], num_rows)?;
    let end = float_rows(SIG, "end_seconds", &args[2], num_rows)?;
    let config = media_runtime::tool_config();
    let mut out = BinaryOut::new(num_rows);
    for row in 0..num_rows {
        if audio.null_at(row) || start.null_at(row) || end.null_at(row) {
            out.push_null();
            continue;
        }
        let payload = audio.at(row);
        let container = sniff_audio_container(payload)?;
        let trimmed = zyron_media::skeleton::audio_trim(
            &config,
            payload,
            start.at(row),
            end.at(row),
            container,
        )
        .map_err(ZyronError::from)?;
        out.push(trimmed);
    }
    Ok(out.finish(TypeId::Audio))
}

// ---------------------------------------------------------------------------
// Presigned URLs
// ---------------------------------------------------------------------------

fn unix_now() -> Result<i64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| exec_err("system clock is before the unix epoch".to_string()))?;
    i64::try_from(now.as_secs())
        .map_err(|_| exec_err("system clock overflows a signed 64 bit unix timestamp".to_string()))
}

/// Parses a duration like 1h, 30m, 45s or 1500ms into whole seconds,
/// rounding up so the handle lives at least the requested time
fn parse_expiry_seconds(sig: &str, text: &str) -> Result<i64> {
    let trimmed = text.trim();
    let (digits, unit_millis) = if let Some(d) = trimmed.strip_suffix("ms") {
        (d, 1i64)
    } else if let Some(d) = trimmed.strip_suffix('s') {
        (d, 1_000i64)
    } else if let Some(d) = trimmed.strip_suffix('m') {
        (d, 60_000i64)
    } else if let Some(d) = trimmed.strip_suffix('h') {
        (d, 3_600_000i64)
    } else {
        return Err(exec_err(format!(
            "{sig}: expiry {trimmed} needs a unit suffix, supported suffixes are ms, s, m and h"
        )));
    };
    let count: i64 = digits.trim().parse().map_err(|_| {
        exec_err(format!(
            "{sig}: expiry count {digits} is not a whole number"
        ))
    })?;
    if count <= 0 {
        return Err(exec_err(format!(
            "{sig}: expiry {trimmed} must be a positive duration"
        )));
    }
    let millis = count.checked_mul(unit_millis).ok_or_else(|| {
        exec_err(format!(
            "{sig}: expiry {trimmed} overflows the expiry timestamp"
        ))
    })?;
    let rounded_up = millis.checked_add(999).ok_or_else(|| {
        exec_err(format!(
            "{sig}: expiry {trimmed} overflows the expiry timestamp"
        ))
    })?;
    Ok(rounded_up / 1_000)
}

fn presigned_url_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "presigned_url(payload, expires_in, method)";
    arity_range(SIG, args, 2, 3)?;
    let payload = bytes_rows(SIG, "the payload", &args[0], num_rows)?;
    let expires_in = text_rows(SIG, "expires_in", &args[1], num_rows)?;
    let (method, method_rows) = option_text(SIG, "method", args.get(2), num_rows, "GET")?;
    let secret = media_runtime::presign_secret()?;
    let now = unix_now()?;
    let mut out = Utf8Out::new(num_rows);
    for row in 0..num_rows {
        if payload.null_at(row) || expires_in.null_at(row) || option_null_at(&method_rows, row) {
            out.push_null();
            continue;
        }
        let lifetime = parse_expiry_seconds(SIG, expires_in.at(row))?;
        let expires_at = now
            .checked_add(lifetime)
            .ok_or_else(|| exec_err(format!("{SIG}: expiry timestamp overflows")))?;
        // the resource is the store's content address for the payload
        let digest = zyron_types::crypto::sha256(payload.at(row));
        let resource = format!("media/{}", zyron_types::encoding::hex_encode(&digest));
        let url = zyron_media::presign::sign(secret, &resource, &method, expires_at)
            .map_err(ZyronError::from)?;
        out.push(url);
    }
    Ok(out.finish(TypeId::Text))
}

fn presigned_verify_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "presigned_verify(url)";
    arity_exact(SIG, args, 1)?;
    let url = text_rows(SIG, "url", &args[0], num_rows)?;
    let secret = media_runtime::presign_secret()?;
    let now = unix_now()?;
    let mut data = Vec::with_capacity(num_rows);
    let mut nulls = NullBitmap::none(num_rows);
    for row in 0..num_rows {
        if url.null_at(row) {
            nulls.set_null(data.len());
            data.push(false);
            continue;
        }
        // expiry and signature failures answer false, they are the question asked
        data.push(zyron_media::presign::verify(secret, url.at(row), now).is_ok());
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_col(values: Vec<Vec<u8>>) -> Column {
        Column::new(ColumnData::Binary(values), TypeId::Bytea)
    }

    fn int_col(values: Vec<i64>) -> Column {
        Column::new(ColumnData::Int64(values), TypeId::Int64)
    }

    fn text_col(values: Vec<&str>) -> Column {
        Column::new(
            ColumnData::Utf8(values.into_iter().map(String::from).collect()),
            TypeId::Varchar,
        )
    }

    fn run(name: &str, args: &[Column], num_rows: usize) -> Result<Column> {
        dispatch(name, args, num_rows).expect("media function dispatched")
    }

    fn utf8_row(col: &Column, row: usize) -> String {
        match &col.data {
            ColumnData::Utf8(v) => v[row].clone(),
            other => panic!("expected utf8 column, got {other:?}"),
        }
    }

    fn binary_row(col: &Column, row: usize) -> Vec<u8> {
        match &col.data {
            ColumnData::Binary(v) => v[row].clone(),
            other => panic!("expected binary column, got {other:?}"),
        }
    }

    fn int64_rows(col: &Column) -> Vec<i64> {
        match &col.data {
            ColumnData::Int64(v) => v.clone(),
            other => panic!("expected int64 column, got {other:?}"),
        }
    }

    fn bool_rows(col: &Column) -> Vec<bool> {
        match &col.data {
            ColumnData::Boolean(v) => v.clone(),
            other => panic!("expected boolean column, got {other:?}"),
        }
    }

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

    /// Valid rgb8 png built by hand, zlib stream uses stored deflate blocks
    fn sample_png(width: u32, height: u32) -> Vec<u8> {
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);

        let mut raw = Vec::new();
        for y in 0..height {
            raw.push(0u8);
            for x in 0..width {
                raw.push((x % 256) as u8);
                raw.push((y % 256) as u8);
                raw.push(128);
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

    /// Single page pdf with one text stream, mirrors the zyron-media fixture
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
            format!("4 0 obj << /Length {} >> stream\n", content.len()).as_bytes(),
        );
        pdf.extend_from_slice(content);
        pdf.extend_from_slice(b"\nendstream endobj\n");
        pdf.extend_from_slice(b"trailer << /Root 1 0 R >>\n%%EOF\n");
        pdf
    }

    fn dims(bytes: &[u8]) -> (u64, u64) {
        let meta = zyron_media::image_meta::image_metadata(bytes).expect("metadata");
        (
            meta["width"].as_u64().expect("width"),
            meta["height"].as_u64().expect("height"),
        )
    }

    #[test]
    fn image_metadata_reports_dimensions() {
        let png = sample_png(64, 48);
        let out = run("image_metadata", &[bin_col(vec![png])], 1).expect("metadata");
        assert_eq!(out.type_id, TypeId::Jsonb);
        let parsed: serde_json::Value =
            serde_json::from_str(&utf8_row(&out, 0)).expect("json parses");
        assert_eq!(parsed["width"], 64);
        assert_eq!(parsed["height"], 48);
        assert_eq!(parsed["format"], "png");
    }

    #[test]
    fn image_resize_changes_dimensions() {
        let png = sample_png(64, 48);
        let fit = run(
            "image_resize",
            &[
                bin_col(vec![png.clone()]),
                int_col(vec![32]),
                int_col(vec![32]),
            ],
            1,
        )
        .expect("resize fit");
        assert_eq!(fit.type_id, TypeId::Image);
        assert_eq!(dims(&binary_row(&fit, 0)), (32, 24));

        let cover = run(
            "image_resize",
            &[
                bin_col(vec![png]),
                int_col(vec![32]),
                int_col(vec![32]),
                text_col(vec!["cover"]),
            ],
            1,
        )
        .expect("resize cover");
        assert_eq!(dims(&binary_row(&cover, 0)), (32, 32));
    }

    #[test]
    fn image_rotate_quarter_turn_swaps_dimensions() {
        let png = sample_png(64, 48);
        let out = run("image_rotate", &[bin_col(vec![png]), int_col(vec![90])], 1).expect("rotate");
        assert_eq!(out.type_id, TypeId::Image);
        assert_eq!(dims(&binary_row(&out, 0)), (48, 64));
    }

    #[test]
    fn document_extract_text_reads_pdf() {
        let pdf = sample_pdf();
        let out =
            run("document_extract_text", &[bin_col(vec![pdf.clone()])], 1).expect("extract text");
        assert_eq!(out.type_id, TypeId::Text);
        assert!(utf8_row(&out, 0).contains("Hello Zyron media"));

        let pages = run("document_page_count", &[bin_col(vec![pdf])], 1).expect("page count");
        assert_eq!(int64_rows(&pages), vec![1]);
    }

    #[test]
    fn video_transcode_without_ffmpeg_names_the_config_key() {
        let err = run(
            "video_transcode",
            &[
                bin_col(vec![b"not a real video".to_vec()]),
                text_col(vec!["webm"]),
                int_col(vec![1_000_000]),
                text_col(vec!["1280x720"]),
            ],
            1,
        )
        .expect_err("transcode without ffmpeg");
        let message = err.to_string();
        assert!(message.contains("ffmpeg"), "message was: {message}");
        assert!(
            message.contains("media.ffmpeg_path"),
            "message was: {message}"
        );
    }

    #[test]
    fn presigned_url_round_trips_and_expired_verifies_false() {
        media_runtime::install_presign_secret([7u8; 32]);
        let out = run(
            "presigned_url",
            &[
                bin_col(vec![b"payload bytes".to_vec()]),
                text_col(vec!["1h"]),
            ],
            1,
        )
        .expect("sign");
        let url = utf8_row(&out, 0);
        assert!(url.starts_with("media/"), "url was: {url}");
        assert!(url.contains("zm=GET"), "url was: {url}");

        let verified = run("presigned_verify", &[text_col(vec![&url])], 1).expect("verify");
        assert_eq!(bool_rows(&verified), vec![true]);

        let secret = media_runtime::presign_secret().expect("secret installed");
        let expired =
            zyron_media::presign::sign(secret, "media/deadbeef", "GET", 1_000).expect("sign past");
        let rejected = run("presigned_verify", &[text_col(vec![&expired])], 1).expect("verify");
        assert_eq!(bool_rows(&rejected), vec![false]);
    }

    #[test]
    fn presigned_url_rejects_expiry_without_unit() {
        media_runtime::install_presign_secret([7u8; 32]);
        let err = run(
            "presigned_url",
            &[bin_col(vec![b"payload".to_vec()]), text_col(vec!["soon"])],
            1,
        )
        .expect_err("bad expiry");
        assert!(err.to_string().contains("suffix"));
    }

    #[test]
    fn null_rows_pass_through_image_metadata() {
        let png = sample_png(8, 8);
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let col = Column::with_nulls(
            ColumnData::Binary(vec![png, Vec::new()]),
            nulls,
            TypeId::Bytea,
        );
        let out = run("image_metadata", &[col], 2).expect("metadata with null row");
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        assert_eq!(utf8_row(&out, 1), "");
    }
}
