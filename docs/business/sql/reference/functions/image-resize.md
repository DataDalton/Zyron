# image_resize

Decodes the image, scales it and re-encodes it in the same format. The mode decides what happens when the target shape differs from the source shape. The mode is read from the first row and applies to every row, so it cannot vary per image.

## Syntax

```sql
image_resize(image, width, height [, mode])
```

## Returns

IMAGE in the source format. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `mode` | fit to stay inside the box and keep the aspect ratio, cover to fill the box and crop the overflow, stretch to match the box exactly and distort the image. | fit. |

## Examples

```sql
SELECT image_resize(photo, 200, 200, 'cover') FROM zyron_test.photos
```

Each photo filled to a 200 by 200 square, with the overflow cropped.

## Refused

- The mode is not fit, cover or stretch.
- A dimension is negative or above the 32-bit range.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [image_crop](image-crop.md)
- [image_format](image-format.md)
- [image_metadata](image-metadata.md)
