# audio_transcribe

Runs speech recognition over the audio and returns the transcript. Backed by an external model, so the call fails naming the tool and the configuration key when none is configured. The model is read from the first row and applies to every row.

## Syntax

```sql
audio_transcribe(audio [, model])
```

## Returns

TEXT. NULL when the payload is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `model` | Recognition model to use. | whisper-small. |

## Examples

```sql
SELECT audio_transcribe(video_extract_audio(clip)) FROM zyron_test.uploads
```

The spoken words in each clip as text.

## Refused

- No recognition model is configured.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [video_extract_audio](video-extract-audio.md)
- [image_ocr](image-ocr.md)
- [audio_metadata](audio-metadata.md)
