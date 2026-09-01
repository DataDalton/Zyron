//! Audio metadata for wav, flac and mp3 parsed in pure Rust

use serde_json::{Value, json};

use crate::error::{MediaError, MediaResult};

const FFMPEG_HINT: &str = "set media.ffmpeg_path in zyron.toml to the ffmpeg binary";

/// Extracts codec, bitrate, sample rate, channels and duration
pub fn audio_metadata(bytes: &[u8]) -> MediaResult<Value> {
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        return wav_metadata(bytes);
    }
    if bytes.len() >= 4 && &bytes[..4] == b"fLaC" {
        return flac_metadata(bytes);
    }
    if looks_like_mp3(bytes) {
        return mp3_metadata(bytes);
    }
    if bytes.len() >= 4 && &bytes[..4] == b"OggS" {
        return Err(MediaError::UnsupportedFormat(format!(
            "ogg containers are unsupported without ffmpeg, {FFMPEG_HINT}"
        )));
    }
    Err(MediaError::UnsupportedFormat(format!(
        "unrecognized audio bytes, wav, flac and mp3 parse natively, other formats require ffmpeg, {FFMPEG_HINT}"
    )))
}

fn wav_metadata(bytes: &[u8]) -> MediaResult<Value> {
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u32, u16)> = None;
    let mut data_len: Option<u64> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        if id == b"fmt " {
            if body_start + 16 > bytes.len() {
                return Err(MediaError::CorruptObject(
                    "wav fmt chunk is truncated".to_string(),
                ));
            }
            let b = &bytes[body_start..];
            fmt = Some((
                u16::from_le_bytes([b[0], b[1]]),
                u16::from_le_bytes([b[2], b[3]]),
                u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
                u32::from_le_bytes([b[8], b[9], b[10], b[11]]),
                u16::from_le_bytes([b[14], b[15]]),
            ));
        } else if id == b"data" {
            data_len = Some(size as u64);
        }
        // chunks are padded to even lengths
        pos = body_start + size + (size & 1);
    }

    let (audio_format, channels, sample_rate, byte_rate, bits) =
        fmt.ok_or_else(|| MediaError::CorruptObject("wav file has no fmt chunk".to_string()))?;
    let codec = match audio_format {
        1 => "pcm".to_string(),
        3 => "ieee_float".to_string(),
        other => format!("wav_format_{other}"),
    };
    let duration = match (data_len, byte_rate) {
        (Some(dl), br) if br > 0 => json!((dl as f64 / br as f64 * 1000.0).round() / 1000.0),
        _ => Value::Null,
    };
    Ok(json!({
        "codec": codec,
        "bitrate": byte_rate * 8,
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits,
        "duration_seconds": duration,
    }))
}

fn flac_metadata(bytes: &[u8]) -> MediaResult<Value> {
    // first metadata block must be STREAMINFO, 34 bytes after a 4 byte header
    if bytes.len() < 4 + 4 + 34 {
        return Err(MediaError::CorruptObject(
            "flac file is too short for a STREAMINFO block".to_string(),
        ));
    }
    let block_type = bytes[4] & 0x7F;
    if block_type != 0 {
        return Err(MediaError::CorruptObject(
            "flac first metadata block is not STREAMINFO".to_string(),
        ));
    }
    let info = &bytes[8..8 + 34];
    let sample_rate =
        ((info[10] as u32) << 12) | ((info[11] as u32) << 4) | ((info[12] as u32) >> 4);
    let channels = (((info[12] >> 1) & 0x07) + 1) as u32;
    let bits_per_sample = ((((info[12] & 0x01) << 4) | (info[13] >> 4)) + 1) as u32;
    let total_samples = (((info[13] & 0x0F) as u64) << 32)
        | ((info[14] as u64) << 24)
        | ((info[15] as u64) << 16)
        | ((info[16] as u64) << 8)
        | (info[17] as u64);

    let duration = if sample_rate > 0 && total_samples > 0 {
        json!((total_samples as f64 / sample_rate as f64 * 1000.0).round() / 1000.0)
    } else {
        Value::Null
    };
    let bitrate = match &duration {
        Value::Number(n) => {
            let secs = n.as_f64().unwrap_or(0.0);
            if secs > 0.0 {
                json!((bytes.len() as f64 * 8.0 / secs).round() as i64)
            } else {
                Value::Null
            }
        }
        _ => Value::Null,
    };
    Ok(json!({
        "codec": "flac",
        "bitrate": bitrate,
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
        "duration_seconds": duration,
    }))
}

fn looks_like_mp3(bytes: &[u8]) -> bool {
    if bytes.len() >= 3 && &bytes[..3] == b"ID3" {
        return true;
    }
    find_mp3_frame(bytes, 0).is_some()
}

struct Mp3Frame {
    offset: usize,
    bitrate_bps: u32,
    sample_rate: u32,
    channels: u32,
    samples_per_frame: u32,
    side_info_len: usize,
}

fn find_mp3_frame(bytes: &[u8], start: usize) -> Option<Mp3Frame> {
    if bytes.len() < 4 {
        return None;
    }
    let limit = (bytes.len() - 4).min(start.saturating_add(64 * 1024));
    let mut i = start;
    while i <= limit {
        if bytes[i] == 0xFF && (bytes[i + 1] & 0xE0) == 0xE0 {
            if let Some(frame) = parse_mp3_header(bytes, i) {
                return Some(frame);
            }
        }
        i += 1;
    }
    None
}

fn parse_mp3_header(bytes: &[u8], offset: usize) -> Option<Mp3Frame> {
    let b1 = bytes[offset + 1];
    let b2 = bytes[offset + 2];
    let b3 = bytes[offset + 3];
    let version_bits = (b1 >> 3) & 0x03;
    let layer_bits = (b1 >> 1) & 0x03;
    // layer III only
    if version_bits == 1 || layer_bits != 1 {
        return None;
    }
    let mpeg1 = version_bits == 3;
    let bitrate_idx = (b2 >> 4) as usize;
    let sr_idx = ((b2 >> 2) & 0x03) as usize;
    if bitrate_idx == 0 || bitrate_idx == 15 || sr_idx == 3 {
        return None;
    }
    const MPEG1_L3: [u32; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const MPEG2_L3: [u32; 16] = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ];
    const SR_MPEG1: [u32; 3] = [44100, 48000, 32000];
    let bitrate_kbps = if mpeg1 {
        MPEG1_L3[bitrate_idx]
    } else {
        MPEG2_L3[bitrate_idx]
    };
    let base_rate = SR_MPEG1[sr_idx];
    let sample_rate = match version_bits {
        3 => base_rate,
        2 => base_rate / 2,
        0 => base_rate / 4,
        _ => return None,
    };
    let mode = (b3 >> 6) & 0x03;
    let channels = if mode == 3 { 1 } else { 2 };
    let samples_per_frame = if mpeg1 { 1152 } else { 576 };
    let side_info_len = match (mpeg1, channels) {
        (true, 1) => 17,
        (true, _) => 32,
        (false, 1) => 9,
        (false, _) => 17,
    };
    Some(Mp3Frame {
        offset,
        bitrate_bps: bitrate_kbps * 1000,
        sample_rate,
        channels,
        samples_per_frame,
        side_info_len,
    })
}

fn mp3_metadata(bytes: &[u8]) -> MediaResult<Value> {
    let audio_start = if bytes.len() >= 10 && &bytes[..3] == b"ID3" {
        let size = ((bytes[6] as usize & 0x7F) << 21)
            | ((bytes[7] as usize & 0x7F) << 14)
            | ((bytes[8] as usize & 0x7F) << 7)
            | (bytes[9] as usize & 0x7F);
        10 + size
    } else {
        0
    };
    let frame = find_mp3_frame(bytes, audio_start.min(bytes.len()))
        .ok_or_else(|| MediaError::CorruptObject("no valid mp3 frame header found".to_string()))?;

    let audio_bytes = (bytes.len() - frame.offset) as f64;
    let mut duration = if frame.bitrate_bps > 0 {
        audio_bytes * 8.0 / frame.bitrate_bps as f64
    } else {
        0.0
    };
    let mut bitrate = frame.bitrate_bps as f64;

    // Xing or VBRI headers carry the exact frame count for vbr files
    let tag_offset = frame.offset + 4 + frame.side_info_len;
    if tag_offset + 12 <= bytes.len() {
        let tag = &bytes[tag_offset..tag_offset + 4];
        if tag == b"Xing" || tag == b"Info" {
            let flags = u32::from_be_bytes([
                bytes[tag_offset + 4],
                bytes[tag_offset + 5],
                bytes[tag_offset + 6],
                bytes[tag_offset + 7],
            ]);
            if flags & 1 == 1 {
                let frames = u32::from_be_bytes([
                    bytes[tag_offset + 8],
                    bytes[tag_offset + 9],
                    bytes[tag_offset + 10],
                    bytes[tag_offset + 11],
                ]);
                if frame.sample_rate > 0 {
                    duration =
                        frames as f64 * frame.samples_per_frame as f64 / frame.sample_rate as f64;
                    if duration > 0.0 {
                        bitrate = audio_bytes * 8.0 / duration;
                    }
                }
            }
        }
    }
    let vbri_offset = frame.offset + 4 + 32;
    if vbri_offset + 18 <= bytes.len() && &bytes[vbri_offset..vbri_offset + 4] == b"VBRI" {
        let frames = u32::from_be_bytes([
            bytes[vbri_offset + 14],
            bytes[vbri_offset + 15],
            bytes[vbri_offset + 16],
            bytes[vbri_offset + 17],
        ]);
        if frame.sample_rate > 0 {
            duration = frames as f64 * frame.samples_per_frame as f64 / frame.sample_rate as f64;
            if duration > 0.0 {
                bitrate = audio_bytes * 8.0 / duration;
            }
        }
    }

    Ok(json!({
        "codec": "mp3",
        "bitrate": bitrate.round() as i64,
        "sample_rate": frame.sample_rate,
        "channels": frame.channels,
        "duration_seconds": (duration * 1000.0).round() / 1000.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn wav_fields() {
        let wav = sample_wav(44100, 2, 1.0);
        let meta = audio_metadata(&wav).expect("wav metadata");
        assert_eq!(meta["codec"], "pcm");
        assert_eq!(meta["sample_rate"], 44100);
        assert_eq!(meta["channels"], 2);
        assert_eq!(meta["bitrate"], 44100 * 4 * 8);
        assert_eq!(meta["duration_seconds"], 1.0);
    }

    #[test]
    fn mp3_frame_header_fields() {
        // MPEG1 layer III, 128 kbps, 44100 Hz, stereo
        let frame_len = 144 * 128000 / 44100;
        let mut mp3 = Vec::new();
        for _ in 0..10 {
            let mut frame = vec![0u8; frame_len];
            frame[0] = 0xFF;
            frame[1] = 0xFB;
            frame[2] = 0x90;
            frame[3] = 0x00;
            mp3.extend_from_slice(&frame);
        }
        let meta = audio_metadata(&mp3).expect("mp3 metadata");
        assert_eq!(meta["codec"], "mp3");
        assert_eq!(meta["bitrate"], 128000);
        assert_eq!(meta["sample_rate"], 44100);
        assert_eq!(meta["channels"], 2);
        let secs = meta["duration_seconds"].as_f64().expect("duration");
        let expected = (10 * frame_len) as f64 * 8.0 / 128000.0;
        assert!((secs - expected).abs() < 0.05);
    }

    #[test]
    fn flac_streaminfo_fields() {
        let mut flac = Vec::new();
        flac.extend_from_slice(b"fLaC");
        // last block flag set, type 0, length 34
        flac.push(0x80);
        flac.extend_from_slice(&[0, 0, 34]);
        let mut info = vec![0u8; 34];
        // min and max block size
        info[0..2].copy_from_slice(&4096u16.to_be_bytes());
        info[2..4].copy_from_slice(&4096u16.to_be_bytes());
        let sample_rate: u32 = 44100;
        let channels: u32 = 2;
        let bps: u32 = 16;
        let total_samples: u64 = 44100;
        info[10] = (sample_rate >> 12) as u8;
        info[11] = (sample_rate >> 4) as u8;
        info[12] = (((sample_rate & 0x0F) << 4) as u8)
            | (((channels - 1) << 1) as u8)
            | (((bps - 1) >> 4) as u8);
        info[13] = ((((bps - 1) & 0x0F) << 4) as u8) | ((total_samples >> 32) as u8 & 0x0F);
        info[14] = (total_samples >> 24) as u8;
        info[15] = (total_samples >> 16) as u8;
        info[16] = (total_samples >> 8) as u8;
        info[17] = total_samples as u8;
        flac.extend_from_slice(&info);
        let meta = audio_metadata(&flac).expect("flac metadata");
        assert_eq!(meta["codec"], "flac");
        assert_eq!(meta["sample_rate"], 44100);
        assert_eq!(meta["channels"], 2);
        assert_eq!(meta["bits_per_sample"], 16);
        assert_eq!(meta["duration_seconds"], 1.0);
    }

    #[test]
    fn unknown_format_errors_with_hint() {
        let err = audio_metadata(b"not audio bytes in any recognizable way").expect_err("fail");
        assert!(err.to_string().contains("media.ffmpeg_path"));
    }
}
