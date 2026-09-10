# video_extract_frame

Takes the frame at the given offset in seconds, which may be fractional. An offset past the end of the video is an error rather than returning the last frame.

## Syntax

```sql
video_extract_frame(video, at_seconds)
```

## Returns

IMAGE. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `at_seconds` | Offset from the start, in seconds. | Not applicable. |

## Examples

```sql
SELECT video_extract_frame(clip, 1.5) FROM zyron_test.uploads
```

The frame one and a half seconds into each clip.

## Refused

- The offset argument is not numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [video_thumbnail](video-thumbnail.md)
- [image_resize](image-resize.md)
- [video_metadata](video-metadata.md)
