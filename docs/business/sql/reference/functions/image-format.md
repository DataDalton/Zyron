# image_format

Decodes the image and writes it out in the target format. Converting to a format without transparency flattens the alpha channel, and converting to a lossy format loses detail that a second conversion cannot recover.

## Syntax

```sql
image_format(image, target)
```

## Returns

IMAGE in the target format. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `target` | Format name to write. | Not applicable. |

## Examples

```sql
SELECT image_format(photo, 'webp') FROM zyron_test.photos
```

Each photo re-encoded as WebP.

## Refused

- The target argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [image_resize](image-resize.md)
- [image_metadata](image-metadata.md)
- [detect_mime_type](detect-mime-type.md)
