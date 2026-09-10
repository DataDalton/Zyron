# video_transcode

Decodes and re-encodes the whole stream, so the cost grows with duration rather than with file size. Backed by an external tool, so the call fails naming the tool and the configuration key when none is configured.

## Syntax

```sql
video_transcode(video, codec, bitrate, resolution)
```

## Returns

VIDEO. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `codec` | Video codec to encode with. | Not applicable. |
| `bitrate` | Target bitrate in bits per second. | Not applicable. |
| `resolution` | Target resolution as width by height. | Not applicable. |

## Examples

```sql
SELECT video_transcode(clip, 'h264', 1000000, '1280x720') FROM zyron_test.uploads
```

Each clip re-encoded as 720p H.264 at one megabit.

## Refused

- No transcoding tool is configured.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [video_metadata](video-metadata.md)
- [video_extract_audio](video-extract-audio.md)
- [audio_transcode](audio-transcode.md)
