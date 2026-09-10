# image_metadata

Reads the image header and returns what it states, without decoding the pixels. Works on the payload bytes rather than a file path, so the image can come from a column, a parameter or another function.

## Syntax

```sql
image_metadata(image)
```

## Returns

COMPOSITE as JSON text. NULL when the payload is NULL.

## Examples

```sql
SELECT image_metadata(thumbnail) FROM zyron_test.photos
```

An object naming the format, width and height of each image.

## Refused

- The argument is not a binary or text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [image_resize](image-resize.md)
- [video_metadata](video-metadata.md)
- [detect_mime_type](detect-mime-type.md)
