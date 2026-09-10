# video_metadata

Reads the container header and returns what it states. The figures come from the header rather than from a full scan, so a file whose header disagrees with its stream reports the header's numbers.

## Syntax

```sql
video_metadata(video)
```

## Returns

COMPOSITE as JSON text. NULL when the payload is NULL.

## Examples

```sql
SELECT video_metadata(clip) FROM zyron_test.uploads
```

An object naming the duration, dimensions and codecs of each clip.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [video_transcode](video-transcode.md)
- [video_thumbnail](video-thumbnail.md)
- [audio_metadata](audio-metadata.md)
