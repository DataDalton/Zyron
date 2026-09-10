# video_thumbnail

Takes a frame and returns it as an image. The offset is optional, unlike video_extract_frame, so a thumbnail can be taken without knowing the duration.

## Syntax

```sql
video_thumbnail(video [, at_seconds])
```

## Returns

IMAGE. NULL when the payload is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `at_seconds` | Offset from the start, in seconds. | An offset near the start is used. |

## Examples

```sql
SELECT video_thumbnail(clip) FROM zyron_test.uploads
```

One still image per clip.

## Refused

- The offset argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [video_extract_frame](video-extract-frame.md)
- [image_resize](image-resize.md)
