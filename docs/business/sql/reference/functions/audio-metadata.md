# audio_metadata

Reads the container header and returns what it states, including duration, channel count and sample rate where the format records them.

## Syntax

```sql
audio_metadata(audio)
```

## Returns

COMPOSITE as JSON text. NULL when the payload is NULL.

## Examples

```sql
SELECT audio_metadata(track) FROM zyron_test.recordings
```

An object naming the duration, channels and sample rate of each track.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [audio_transcode](audio-transcode.md)
- [audio_trim](audio-trim.md)
- [video_metadata](video-metadata.md)
