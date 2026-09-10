# audio_trim

Keeps the span from the start offset to the end offset and drops the rest. Both offsets are in seconds and may be fractional. An end offset at or before the start leaves nothing to keep and is an error.

## Syntax

```sql
audio_trim(audio, start_seconds, end_seconds)
```

## Returns

AUDIO. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `start_seconds` | Offset the kept span begins at. | Not applicable. |
| `end_seconds` | Offset the kept span ends at. | Not applicable. |

## Examples

```sql
SELECT audio_trim(track, 0, 30) FROM zyron_test.recordings
```

The first 30 seconds of each track.

## Refused

- An offset argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [audio_transcode](audio-transcode.md)
- [audio_metadata](audio-metadata.md)
