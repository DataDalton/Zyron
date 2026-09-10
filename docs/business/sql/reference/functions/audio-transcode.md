# audio_transcode

Decodes and re-encodes the whole stream. Re-encoding a lossy source to another lossy format compounds the loss, so transcode from the original where one is held.

## Syntax

```sql
audio_transcode(audio, codec, bitrate)
```

## Returns

AUDIO. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `codec` | Audio codec to encode with. | Not applicable. |
| `bitrate` | Target bitrate in bits per second. | Not applicable. |

## Examples

```sql
SELECT audio_transcode(track, 'aac', 128000) FROM zyron_test.recordings
```

Each track re-encoded as AAC at 128 kilobits.

## Refused

- No transcoding tool is configured.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [audio_trim](audio-trim.md)
- [audio_metadata](audio-metadata.md)
- [video_transcode](video-transcode.md)
