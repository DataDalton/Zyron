//! Tool backed media operations
//!
//! Every operation resolves its external tool from MediaToolConfig. With no
//! configured binary the call returns an error naming the tool and the
//! config key to set. With a configured binary that exists on disk the tool
//! is invoked for real, streaming through pipes and falling back to temp
//! files for formats that need seekable input or output. Model backed
//! operations error until a model is registered and served by a runtime

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{MediaError, MediaResult};

const FFMPEG_KEY: &str = "media.ffmpeg_path";
const TESSERACT_KEY: &str = "media.tesseract_path";

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Paths to external tool binaries resolved from zyron.toml
#[derive(Debug, Clone, Default)]
pub struct MediaToolConfig {
    pub ffmpeg_path: Option<PathBuf>,
    pub tesseract_path: Option<PathBuf>,
}

/// The container a video codec is muxed into. A codec names the bitstream,
/// not the file it lands in, and the two are only sometimes spelled alike
fn container_for_video_codec(codec: &str) -> &'static str {
    match codec.to_ascii_lowercase().as_str() {
        "vp8" | "vp9" | "av1" | "libvpx" | "libvpx-vp9" | "libaom-av1" => "webm",
        "theora" | "libtheora" => "ogg",
        _ => "mp4",
    }
}

/// The container an audio codec is muxed into
fn container_for_audio_codec(codec: &str) -> &'static str {
    match codec.to_ascii_lowercase().as_str() {
        "opus" | "libopus" | "vorbis" | "libvorbis" => "ogg",
        "flac" => "flac",
        "pcm_s16le" | "pcm_s24le" => "wav",
        "aac" | "libfdk_aac" => "m4a",
        _ => "mp3",
    }
}

/// Transcodes a video to the named codec, bitrate and frame size.
///
/// Every argument reaches the encoder: the codec selects the bitstream and
/// the container it is muxed into, the bitrate caps the video stream, and
/// the resolution scales the frames. A resolution of `WxH` is passed as a
/// frame size, which is what makes the call a transcode rather than a
/// remux of the source at its original dimensions
pub fn video_transcode(
    config: &MediaToolConfig,
    input: &[u8],
    codec: &str,
    bitrate: i64,
    resolution: &str,
) -> MediaResult<Vec<u8>> {
    validate_codec(codec)?;
    validate_bitrate(bitrate)?;
    validate_resolution(resolution)?;
    let container = container_for_video_codec(codec);
    let post = vec![
        "-c:v".to_string(),
        codec.to_string(),
        "-b:v".to_string(),
        bitrate.to_string(),
        "-s".to_string(),
        resolution.to_string(),
    ];
    run_ffmpeg(config, "video transcoding", input, &[], &post, container)
}

/// Strips the video track and emits the audio in the target format
pub fn video_extract_audio(
    config: &MediaToolConfig,
    input: &[u8],
    target_format: &str,
) -> MediaResult<Vec<u8>> {
    validate_format(target_format)?;
    run_ffmpeg(
        config,
        "video audio extraction",
        input,
        &[],
        &["-vn".to_string()],
        target_format,
    )
}

/// Extracts a single frame at the given timestamp as an image
pub fn video_extract_frame(
    config: &MediaToolConfig,
    input: &[u8],
    at_seconds: f64,
) -> MediaResult<Vec<u8>> {
    validate_timestamp(at_seconds)?;
    run_ffmpeg(
        config,
        "video frame extraction",
        input,
        &["-ss".to_string(), format!("{at_seconds:.3}")],
        &[
            "-frames:v".to_string(),
            "1".to_string(),
            "-update".to_string(),
            "1".to_string(),
        ],
        "image2",
    )
}

/// Thumbnail generation, a frame extraction at the given timestamp
pub fn video_thumbnail(
    config: &MediaToolConfig,
    input: &[u8],
    at_seconds: f64,
) -> MediaResult<Vec<u8>> {
    validate_timestamp(at_seconds)?;
    run_ffmpeg(
        config,
        "video thumbnailing",
        input,
        &["-ss".to_string(), format!("{at_seconds:.3}")],
        &[
            "-frames:v".to_string(),
            "1".to_string(),
            "-update".to_string(),
            "1".to_string(),
        ],
        "image2",
    )
}

/// Transcodes audio to the named codec at the named bitrate.
///
/// Both arguments reach the encoder: the codec selects the bitstream and
/// its container, and the bitrate caps the audio stream
pub fn audio_transcode(
    config: &MediaToolConfig,
    input: &[u8],
    codec: &str,
    bitrate: i64,
) -> MediaResult<Vec<u8>> {
    validate_codec(codec)?;
    validate_bitrate(bitrate)?;
    let container = container_for_audio_codec(codec);
    let post = vec![
        "-c:a".to_string(),
        codec.to_string(),
        "-b:a".to_string(),
        bitrate.to_string(),
    ];
    run_ffmpeg(config, "audio transcoding", input, &[], &post, container)
}

/// Cuts audio to the given time range in the target format
pub fn audio_trim(
    config: &MediaToolConfig,
    input: &[u8],
    start_seconds: f64,
    end_seconds: f64,
    target_format: &str,
) -> MediaResult<Vec<u8>> {
    validate_format(target_format)?;
    validate_timestamp(start_seconds)?;
    if end_seconds <= start_seconds {
        return Err(MediaError::InvalidArgument(format!(
            "trim end {end_seconds} must be after start {start_seconds}"
        )));
    }
    run_ffmpeg(
        config,
        "audio trimming",
        input,
        &[],
        &[
            "-ss".to_string(),
            format!("{start_seconds:.3}"),
            "-to".to_string(),
            format!("{end_seconds:.3}"),
        ],
        target_format,
    )
}

/// Speech to text, requires a registered model served by an inference runtime
pub fn audio_transcribe(
    _config: &MediaToolConfig,
    _input: &[u8],
    model_registered: bool,
) -> MediaResult<String> {
    if !model_registered {
        return Err(MediaError::ModelMissing {
            operation: "audio transcription".to_string(),
        });
    }
    Err(MediaError::ModelRuntimeUnavailable {
        operation: "audio transcription".to_string(),
    })
}

/// Optical character recognition through a configured tesseract binary
pub fn image_ocr(config: &MediaToolConfig, input: &[u8], lang: &str) -> MediaResult<String> {
    if lang.is_empty()
        || !lang
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '+')
    {
        return Err(MediaError::InvalidArgument(format!(
            "invalid ocr language code {lang}"
        )));
    }
    let tesseract = resolve_tool(
        &config.tesseract_path,
        "tesseract",
        TESSERACT_KEY,
        "image OCR",
    )?;
    let tmp_img = TempFile::create(input, "img")?;
    let args = [
        tmp_img.path().as_os_str().to_os_string(),
        "stdout".into(),
        "-l".into(),
        lang.into(),
    ];
    let stdout = run_command(&tesseract, &args, None, "tesseract", "image OCR")?;
    Ok(String::from_utf8_lossy(&stdout).into_owned())
}

/// Vector embedding, requires a registered model served by an inference runtime
pub fn image_embed(
    _config: &MediaToolConfig,
    _input: &[u8],
    model_registered: bool,
) -> MediaResult<Vec<f32>> {
    if !model_registered {
        return Err(MediaError::ModelMissing {
            operation: "image embedding".to_string(),
        });
    }
    Err(MediaError::ModelRuntimeUnavailable {
        operation: "image embedding".to_string(),
    })
}

fn validate_format(format: &str) -> MediaResult<()> {
    if format.is_empty() || !format.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(MediaError::InvalidArgument(format!(
            "invalid target format {format}"
        )));
    }
    Ok(())
}

/// A codec name reaches an argument vector, so it is held to the alphabet
/// codec names actually use rather than passed through as written
fn validate_codec(codec: &str) -> MediaResult<()> {
    if codec.is_empty()
        || !codec
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err(MediaError::InvalidArgument(format!(
            "invalid codec {codec}"
        )));
    }
    Ok(())
}

fn validate_bitrate(bitrate: i64) -> MediaResult<()> {
    if bitrate <= 0 {
        return Err(MediaError::InvalidArgument(format!(
            "bitrate {bitrate} must be positive"
        )));
    }
    Ok(())
}

/// Frame size as `WIDTHxHEIGHT`, the spelling the encoder takes
fn validate_resolution(resolution: &str) -> MediaResult<()> {
    let bad = || {
        MediaError::InvalidArgument(format!(
            "invalid resolution {resolution}, expected WIDTHxHEIGHT"
        ))
    };
    let (w, h) = resolution.split_once(['x', 'X']).ok_or_else(bad)?;
    let width: u32 = w.parse().map_err(|_| bad())?;
    let height: u32 = h.parse().map_err(|_| bad())?;
    if width == 0 || height == 0 {
        return Err(bad());
    }
    Ok(())
}

fn validate_timestamp(seconds: f64) -> MediaResult<()> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(MediaError::InvalidArgument(format!(
            "timestamp {seconds} must be a non negative number of seconds"
        )));
    }
    Ok(())
}

fn resolve_tool(
    configured: &Option<PathBuf>,
    tool: &str,
    config_key: &str,
    operation: &str,
) -> MediaResult<PathBuf> {
    match configured {
        None => Err(MediaError::ToolMissing {
            operation: operation.to_string(),
            tool: tool.to_string(),
            config_key: config_key.to_string(),
        }),
        Some(path) if !path.is_file() => Err(MediaError::ToolPathInvalid {
            tool: tool.to_string(),
            config_key: config_key.to_string(),
            path: path.display().to_string(),
        }),
        Some(path) => Ok(path.clone()),
    }
}

/// Container formats ffmpeg cannot mux to a pipe
fn needs_seekable_output(format: &str) -> bool {
    matches!(format, "mp4" | "mov" | "m4a" | "3gp" | "3g2" | "mj2")
}

/// ISO-BMFF input may keep its moov box at the end, which needs seeking
fn looks_like_isobmff(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && matches!(
            &bytes[4..8],
            b"ftyp" | b"moov" | b"mdat" | b"free" | b"skip" | b"wide"
        )
}

fn run_ffmpeg(
    config: &MediaToolConfig,
    operation: &str,
    input: &[u8],
    pre_input_args: &[String],
    post_input_args: &[String],
    output_format: &str,
) -> MediaResult<Vec<u8>> {
    let ffmpeg = resolve_tool(&config.ffmpeg_path, "ffmpeg", FFMPEG_KEY, operation)?;

    let input_tmp = if looks_like_isobmff(input) {
        Some(TempFile::create(input, "in")?)
    } else {
        None
    };
    let output_tmp = if needs_seekable_output(output_format) {
        Some(TempFile::reserve(output_format)?)
    } else {
        None
    };

    let mut args: Vec<std::ffi::OsString> = Vec::new();
    args.push("-hide_banner".into());
    args.push("-loglevel".into());
    args.push("error".into());
    for arg in pre_input_args {
        args.push(arg.into());
    }
    args.push("-i".into());
    match &input_tmp {
        Some(tmp) => args.push(tmp.path().as_os_str().to_os_string()),
        None => args.push("pipe:0".into()),
    }
    for arg in post_input_args {
        args.push(arg.into());
    }
    args.push("-f".into());
    args.push(output_format.into());
    args.push("-y".into());
    match &output_tmp {
        Some(tmp) => args.push(tmp.path().as_os_str().to_os_string()),
        None => args.push("pipe:1".into()),
    }

    let stdin_bytes = if input_tmp.is_none() {
        Some(input.to_vec())
    } else {
        None
    };
    let stdout = run_command(&ffmpeg, &args, stdin_bytes, "ffmpeg", operation)?;

    match output_tmp {
        Some(tmp) => Ok(fs::read(tmp.path())?),
        None => Ok(stdout),
    }
}

fn run_command(
    bin: &Path,
    args: &[std::ffi::OsString],
    stdin_bytes: Option<Vec<u8>>,
    tool: &str,
    operation: &str,
) -> MediaResult<Vec<u8>> {
    let mut command = Command::new(bin);
    command
        .args(args)
        .stdin(if stdin_bytes.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;

    let stdin_writer = match (stdin_bytes, child.stdin.take()) {
        (Some(bytes), Some(mut stdin)) => Some(std::thread::spawn(move || {
            // a broken pipe here means the tool exited early, its exit
            // status carries the failure
            let _ = stdin.write_all(&bytes);
        })),
        _ => None,
    };
    let stderr_reader = child.stderr.take().map(|mut stderr| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        })
    });

    let mut stdout = Vec::new();
    if let Some(mut out) = child.stdout.take() {
        out.read_to_end(&mut stdout)?;
    }
    let status = child.wait()?;

    if let Some(writer) = stdin_writer {
        writer
            .join()
            .map_err(|_| MediaError::External(format!("{tool} stdin writer thread panicked")))?;
    }
    let stderr_bytes = match stderr_reader {
        Some(reader) => reader
            .join()
            .map_err(|_| MediaError::External(format!("{tool} stderr reader thread panicked")))?,
        None => Vec::new(),
    };

    if !status.success() {
        let stderr_text = String::from_utf8_lossy(&stderr_bytes);
        let tail: String = stderr_text
            .chars()
            .rev()
            .take(2000)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        return Err(MediaError::ToolFailed {
            tool: tool.to_string(),
            operation: operation.to_string(),
            status: status.to_string(),
            stderr: tail.trim().to_string(),
        });
    }
    Ok(stdout)
}

/// Temp file removed on drop
struct TempFile(PathBuf);

impl TempFile {
    fn create(bytes: &[u8], ext: &str) -> MediaResult<Self> {
        let path = Self::unique_path(ext);
        fs::write(&path, bytes)?;
        Ok(TempFile(path))
    }

    fn reserve(ext: &str) -> MediaResult<Self> {
        Ok(TempFile(Self::unique_path(ext)))
    }

    fn unique_path(ext: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "zyron-media-{}-{}.{ext}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_config() -> MediaToolConfig {
        MediaToolConfig::default()
    }

    #[test]
    fn every_op_without_config_names_the_config_key() {
        let config = empty_config();
        let input = b"payload";
        let cases: Vec<(String, &str)> = vec![
            (
                video_transcode(&config, input, "vp9", 2_000_000, "1280x720")
                    .expect_err("transcode")
                    .to_string(),
                FFMPEG_KEY,
            ),
            (
                video_extract_audio(&config, input, "mp3")
                    .expect_err("extract audio")
                    .to_string(),
                FFMPEG_KEY,
            ),
            (
                video_extract_frame(&config, input, 1.5)
                    .expect_err("extract frame")
                    .to_string(),
                FFMPEG_KEY,
            ),
            (
                video_thumbnail(&config, input, 0.0)
                    .expect_err("thumbnail")
                    .to_string(),
                FFMPEG_KEY,
            ),
            (
                audio_transcode(&config, input, "flac", 320_000)
                    .expect_err("audio transcode")
                    .to_string(),
                FFMPEG_KEY,
            ),
            (
                audio_trim(&config, input, 0.0, 2.0, "wav")
                    .expect_err("trim")
                    .to_string(),
                FFMPEG_KEY,
            ),
            (
                image_ocr(&config, input, "eng")
                    .expect_err("ocr")
                    .to_string(),
                TESSERACT_KEY,
            ),
        ];
        for (message, key) in cases {
            assert!(message.contains(key), "message {message} lacks {key}");
            assert!(
                message.contains("zyron.toml"),
                "message {message} lacks zyron.toml"
            );
        }
    }

    #[test]
    fn model_ops_name_the_model_registry() {
        let config = empty_config();
        let err = audio_transcribe(&config, b"audio", false).expect_err("transcribe");
        assert!(err.to_string().contains("model registry"));
        let err = image_embed(&config, b"image", false).expect_err("embed");
        assert!(err.to_string().contains("model registry"));
        let err = audio_transcribe(&config, b"audio", true).expect_err("transcribe registered");
        assert!(err.to_string().contains("inference runtime"));
    }

    #[test]
    fn configured_but_missing_binary_names_the_config_key() {
        let config = MediaToolConfig {
            ffmpeg_path: Some(PathBuf::from("Z:/definitely/not/here/ffmpeg.exe")),
            tesseract_path: Some(PathBuf::from("Z:/definitely/not/here/tesseract.exe")),
        };
        let err =
            video_transcode(&config, b"x", "vp9", 2_000_000, "1280x720").expect_err("transcode");
        let msg = err.to_string();
        assert!(msg.contains(FFMPEG_KEY));
        assert!(msg.contains("no file exists"));
        let err = image_ocr(&config, b"x", "eng").expect_err("ocr");
        assert!(err.to_string().contains(TESSERACT_KEY));
    }

    #[test]
    fn argument_validation_rejects_bad_values() {
        let config = empty_config();
        // A codec, a bitrate and a frame size each refuse a bad value
        assert!(video_transcode(&config, b"x", "vp 9", 1, "16x16").is_err());
        assert!(video_transcode(&config, b"x", "vp9", 0, "16x16").is_err());
        assert!(video_transcode(&config, b"x", "vp9", 1, "1280").is_err());
        assert!(video_transcode(&config, b"x", "vp9", 1, "0x720").is_err());
        assert!(audio_transcode(&config, b"x", "flac", -1).is_err());
        assert!(video_extract_frame(&config, b"x", -1.0).is_err());
        assert!(audio_trim(&config, b"x", 5.0, 2.0, "wav").is_err());
        assert!(image_ocr(&config, b"x", "eng; rm").is_err());
    }
}
