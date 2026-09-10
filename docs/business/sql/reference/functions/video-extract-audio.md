# video_extract_audio

Separates the audio stream and writes it as MP3. The output format is fixed rather than following the source, so a lossless source track is re-encoded.

## Syntax

```sql
video_extract_audio(video)
```

## Returns

AUDIO holding MP3. NULL when the payload is NULL.

## Examples

```sql
SELECT audio_metadata(video_extract_audio(clip)) FROM zyron_test.uploads
```

The duration and sample rate of each clip's audio.

## Refused

- No extraction tool is configured.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [audio_transcribe](audio-transcribe.md)
- [video_metadata](video-metadata.md)
- [audio_transcode](audio-transcode.md)
