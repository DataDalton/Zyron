//! Video metadata from ISO-BMFF containers parsed in pure Rust
//!
//! Walks the box tree of mp4 and mov files, reading mvhd for duration,
//! tkhd for resolution, hdlr and stsd for codec and stream kinds, stts
//! for sample counts, and mdat sizes for a bitrate estimate. Containers
//! other than ISO-BMFF need ffmpeg

use serde_json::{Value, json};

use crate::error::{MediaError, MediaResult};

const FFMPEG_HINT: &str = "set media.ffmpeg_path in zyron.toml to the ffmpeg binary";

#[derive(Default)]
struct TrackInfo {
    handler: [u8; 4],
    width: f64,
    height: f64,
    codec: Option<String>,
    timescale: u32,
    duration: u64,
    sample_count: u64,
}

/// Extracts codec, bitrate, resolution, framerate, duration and audio stream count
pub fn video_metadata(bytes: &[u8]) -> MediaResult<Value> {
    classify_container(bytes)?;

    let mut mdat_bytes: u64 = 0;
    let mut movie_timescale: u32 = 0;
    let mut movie_duration: u64 = 0;
    let mut tracks: Vec<TrackInfo> = Vec::new();

    for (typ, payload) in boxes(bytes)? {
        match &typ {
            b"mdat" => mdat_bytes += payload.len() as u64,
            b"moov" => {
                for (mtyp, mpayload) in boxes(payload)? {
                    match &mtyp {
                        b"mvhd" => {
                            let (ts, dur) = parse_time_header(mpayload, "mvhd")?;
                            movie_timescale = ts;
                            movie_duration = dur;
                        }
                        b"trak" => tracks.push(parse_track(mpayload)?),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if movie_timescale == 0 {
        return Err(MediaError::CorruptObject(
            "isobmff container has no mvhd movie header".to_string(),
        ));
    }

    let duration_seconds = movie_duration as f64 / movie_timescale as f64;
    let video = tracks.iter().find(|t| &t.handler == b"vide");
    let audio_streams = tracks.iter().filter(|t| &t.handler == b"soun").count();

    let resolution = match video {
        Some(v) if v.width > 0.0 && v.height > 0.0 => json!({
            "width": v.width.round() as i64,
            "height": v.height.round() as i64,
        }),
        _ => Value::Null,
    };
    let codec = video
        .and_then(|v| v.codec.clone())
        .map(Value::String)
        .unwrap_or(Value::Null);
    let framerate = match video {
        Some(v) if v.sample_count > 0 && v.timescale > 0 && v.duration > 0 => {
            let track_seconds = v.duration as f64 / v.timescale as f64;
            json!((v.sample_count as f64 / track_seconds * 1000.0).round() / 1000.0)
        }
        _ => Value::Null,
    };
    let bitrate = if duration_seconds > 0.0 && mdat_bytes > 0 {
        json!((mdat_bytes as f64 * 8.0 / duration_seconds).round() as i64)
    } else {
        Value::Null
    };

    Ok(json!({
        "codec": codec,
        "bitrate": bitrate,
        "resolution": resolution,
        "framerate": framerate,
        "duration_seconds": (duration_seconds * 1000.0).round() / 1000.0,
        "audio_streams": audio_streams,
    }))
}

fn classify_container(bytes: &[u8]) -> MediaResult<()> {
    if bytes.len() >= 12 {
        let first_type = &bytes[4..8];
        if matches!(
            first_type,
            b"ftyp" | b"moov" | b"mdat" | b"free" | b"skip" | b"wide"
        ) {
            return Ok(());
        }
    }
    if bytes.len() >= 4 && bytes[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        return Err(MediaError::UnsupportedFormat(format!(
            "matroska or webm containers are unsupported without ffmpeg, {FFMPEG_HINT}"
        )));
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"AVI " {
        return Err(MediaError::UnsupportedFormat(format!(
            "avi containers are unsupported without ffmpeg, {FFMPEG_HINT}"
        )));
    }
    if bytes.len() >= 189 && bytes[0] == 0x47 && bytes[188] == 0x47 {
        return Err(MediaError::UnsupportedFormat(format!(
            "mpeg transport streams are unsupported without ffmpeg, {FFMPEG_HINT}"
        )));
    }
    Err(MediaError::UnsupportedFormat(format!(
        "bytes do not match a known video container, only ISO-BMFF mp4 and mov parse natively, {FFMPEG_HINT}"
    )))
}

/// Splits a byte range into (box type, payload) pairs
fn boxes(data: &[u8]) -> MediaResult<Vec<([u8; 4], &[u8])>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        if pos + 8 > data.len() {
            return Err(MediaError::CorruptObject(format!(
                "isobmff box header truncated at offset {pos}"
            )));
        }
        let size32 =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as u64;
        let mut typ = [0u8; 4];
        typ.copy_from_slice(&data[pos + 4..pos + 8]);
        let (payload_start, box_end) = match size32 {
            0 => (pos + 8, data.len()),
            1 => {
                if pos + 16 > data.len() {
                    return Err(MediaError::CorruptObject(format!(
                        "isobmff large box header truncated at offset {pos}"
                    )));
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&data[pos + 8..pos + 16]);
                let size64 = u64::from_be_bytes(b);
                if size64 < 16 || pos as u64 + size64 > data.len() as u64 {
                    return Err(MediaError::CorruptObject(format!(
                        "isobmff large box size {size64} at offset {pos} exceeds the buffer"
                    )));
                }
                (pos + 16, pos + size64 as usize)
            }
            s if s < 8 => {
                return Err(MediaError::CorruptObject(format!(
                    "isobmff box size {s} at offset {pos} is below the header size"
                )));
            }
            s => {
                if pos as u64 + s > data.len() as u64 {
                    return Err(MediaError::CorruptObject(format!(
                        "isobmff box size {s} at offset {pos} exceeds the buffer"
                    )));
                }
                (pos + 8, pos + s as usize)
            }
        };
        out.push((typ, &data[payload_start..box_end]));
        pos = box_end;
    }
    Ok(out)
}

/// Reads timescale and duration from mvhd or mdhd payloads
fn parse_time_header(payload: &[u8], name: &str) -> MediaResult<(u32, u64)> {
    if payload.is_empty() {
        return Err(MediaError::CorruptObject(format!("{name} box is empty")));
    }
    match payload[0] {
        0 => {
            if payload.len() < 20 {
                return Err(MediaError::CorruptObject(format!(
                    "{name} version 0 payload is truncated"
                )));
            }
            let ts = u32::from_be_bytes([payload[12], payload[13], payload[14], payload[15]]);
            let dur =
                u32::from_be_bytes([payload[16], payload[17], payload[18], payload[19]]) as u64;
            Ok((ts, dur))
        }
        1 => {
            if payload.len() < 32 {
                return Err(MediaError::CorruptObject(format!(
                    "{name} version 1 payload is truncated"
                )));
            }
            let ts = u32::from_be_bytes([payload[20], payload[21], payload[22], payload[23]]);
            let mut b = [0u8; 8];
            b.copy_from_slice(&payload[24..32]);
            Ok((ts, u64::from_be_bytes(b)))
        }
        v => Err(MediaError::CorruptObject(format!(
            "{name} has unknown version {v}"
        ))),
    }
}

fn parse_track(trak_payload: &[u8]) -> MediaResult<TrackInfo> {
    let mut track = TrackInfo::default();
    for (typ, payload) in boxes(trak_payload)? {
        match &typ {
            b"tkhd" => parse_tkhd(payload, &mut track)?,
            b"mdia" => parse_mdia(payload, &mut track)?,
            _ => {}
        }
    }
    Ok(track)
}

fn parse_tkhd(payload: &[u8], track: &mut TrackInfo) -> MediaResult<()> {
    if payload.is_empty() {
        return Err(MediaError::CorruptObject("tkhd box is empty".to_string()));
    }
    let dim_offset = match payload[0] {
        0 => 76usize,
        1 => 88usize,
        v => {
            return Err(MediaError::CorruptObject(format!(
                "tkhd has unknown version {v}"
            )));
        }
    };
    if payload.len() < dim_offset + 8 {
        return Err(MediaError::CorruptObject(
            "tkhd payload is truncated".to_string(),
        ));
    }
    // width and height are 16.16 fixed point
    let w = u32::from_be_bytes([
        payload[dim_offset],
        payload[dim_offset + 1],
        payload[dim_offset + 2],
        payload[dim_offset + 3],
    ]);
    let h = u32::from_be_bytes([
        payload[dim_offset + 4],
        payload[dim_offset + 5],
        payload[dim_offset + 6],
        payload[dim_offset + 7],
    ]);
    track.width = w as f64 / 65536.0;
    track.height = h as f64 / 65536.0;
    Ok(())
}

fn parse_mdia(payload: &[u8], track: &mut TrackInfo) -> MediaResult<()> {
    for (typ, inner) in boxes(payload)? {
        match &typ {
            b"mdhd" => {
                let (ts, dur) = parse_time_header(inner, "mdhd")?;
                track.timescale = ts;
                track.duration = dur;
            }
            b"hdlr" => {
                if inner.len() >= 12 {
                    track.handler.copy_from_slice(&inner[8..12]);
                }
            }
            b"minf" => {
                for (mtyp, minner) in boxes(inner)? {
                    if &mtyp == b"stbl" {
                        parse_stbl(minner, track)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_stbl(payload: &[u8], track: &mut TrackInfo) -> MediaResult<()> {
    for (typ, inner) in boxes(payload)? {
        match &typ {
            b"stsd" => {
                // version and flags, entry count, then the first sample entry
                if inner.len() >= 16 {
                    let entry_count = u32::from_be_bytes([inner[4], inner[5], inner[6], inner[7]]);
                    if entry_count > 0 {
                        let fourcc = &inner[12..16];
                        let name: String = fourcc
                            .iter()
                            .map(|&b| if b.is_ascii_graphic() { b as char } else { '_' })
                            .collect();
                        track.codec = Some(name);
                    }
                }
            }
            b"stts" => {
                if inner.len() >= 8 {
                    let entry_count =
                        u32::from_be_bytes([inner[4], inner[5], inner[6], inner[7]]) as usize;
                    let mut total: u64 = 0;
                    for i in 0..entry_count {
                        let off = 8 + i * 8;
                        if off + 8 > inner.len() {
                            return Err(MediaError::CorruptObject(
                                "stts entries are truncated".to_string(),
                            ));
                        }
                        total += u32::from_be_bytes([
                            inner[off],
                            inner[off + 1],
                            inner[off + 2],
                            inner[off + 3],
                        ]) as u64;
                    }
                    track.sample_count = total;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(typ: &[u8; 4], payload: Vec<u8>) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
        out.extend_from_slice(typ);
        out.extend_from_slice(&payload);
        out
    }

    fn mvhd(timescale: u32, duration: u32) -> Vec<u8> {
        let mut p = vec![0u8; 100];
        p[12..16].copy_from_slice(&timescale.to_be_bytes());
        p[16..20].copy_from_slice(&duration.to_be_bytes());
        boxed(b"mvhd", p)
    }

    fn tkhd(width: u32, height: u32) -> Vec<u8> {
        let mut p = vec![0u8; 84];
        p[76..80].copy_from_slice(&(width << 16).to_be_bytes());
        p[80..84].copy_from_slice(&(height << 16).to_be_bytes());
        boxed(b"tkhd", p)
    }

    fn mdhd(timescale: u32, duration: u32) -> Vec<u8> {
        let mut p = vec![0u8; 24];
        p[12..16].copy_from_slice(&timescale.to_be_bytes());
        p[16..20].copy_from_slice(&duration.to_be_bytes());
        boxed(b"mdhd", p)
    }

    fn hdlr(handler: &[u8; 4]) -> Vec<u8> {
        let mut p = vec![0u8; 24];
        p[8..12].copy_from_slice(handler);
        boxed(b"hdlr", p)
    }

    fn stsd(fourcc: &[u8; 4]) -> Vec<u8> {
        let mut p = vec![0u8; 8];
        p[4..8].copy_from_slice(&1u32.to_be_bytes());
        let mut entry = vec![0u8; 86];
        entry[..4].copy_from_slice(&86u32.to_be_bytes());
        entry[4..8].copy_from_slice(fourcc);
        p.extend_from_slice(&entry);
        boxed(b"stsd", p)
    }

    fn stts(sample_count: u32, delta: u32) -> Vec<u8> {
        let mut p = vec![0u8; 8];
        p[4..8].copy_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&sample_count.to_be_bytes());
        p.extend_from_slice(&delta.to_be_bytes());
        boxed(b"stts", p)
    }

    fn minimal_mp4() -> Vec<u8> {
        let ftyp = boxed(b"ftyp", b"isom\0\0\x02\0isomiso2avc1mp41".to_vec());
        let mdat = boxed(b"mdat", vec![0u8; 4000]);
        let stbl_payload = [stsd(b"avc1"), stts(120, 250)].concat();
        let stbl = boxed(b"stbl", stbl_payload);
        let minf = boxed(b"minf", stbl);
        let mdia_payload = [mdhd(1000, 5000), hdlr(b"vide"), minf].concat();
        let mdia = boxed(b"mdia", mdia_payload);
        let trak_payload = [tkhd(320, 240), mdia].concat();
        let trak = boxed(b"trak", trak_payload);
        let moov_payload = [mvhd(1000, 5000), trak].concat();
        let moov = boxed(b"moov", moov_payload);
        [ftyp, mdat, moov].concat()
    }

    #[test]
    fn extracts_mp4_metadata() {
        let mp4 = minimal_mp4();
        let meta = video_metadata(&mp4).expect("metadata");
        assert_eq!(meta["duration_seconds"], 5.0);
        assert_eq!(meta["resolution"]["width"], 320);
        assert_eq!(meta["resolution"]["height"], 240);
        assert_eq!(meta["codec"], "avc1");
        assert_eq!(meta["framerate"], 24.0);
        assert_eq!(meta["audio_streams"], 0);
        assert_eq!(meta["bitrate"], 6400);
    }

    #[test]
    fn garbage_bytes_error_clearly() {
        let err = video_metadata(b"this is not a video at all, just text").expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("media.ffmpeg_path"));
    }

    #[test]
    fn matroska_names_the_container() {
        let mut ebml = vec![0x1A, 0x45, 0xDF, 0xA3];
        ebml.extend_from_slice(&[0u8; 32]);
        let err = video_metadata(&ebml).expect_err("must fail");
        let msg = err.to_string();
        assert!(msg.contains("matroska"));
        assert!(msg.contains("media.ffmpeg_path"));
    }

    #[test]
    fn truncated_box_errors() {
        let mut mp4 = minimal_mp4();
        mp4.truncate(mp4.len() - 10);
        assert!(video_metadata(&mp4).is_err());
    }
}
