# image_crop

Takes the rectangle at the given offset, measured in pixels from the top left corner. A rectangle reaching past the image edge is an error rather than being clipped to what is there.

## Syntax

```sql
image_crop(image, x, y, width, height)
```

## Returns

IMAGE in the source format. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `x` | Pixels from the left edge to the rectangle's left side. | Not applicable. |
| `y` | Pixels from the top edge to the rectangle's top side. | Not applicable. |

## Examples

```sql
SELECT image_crop(photo, 0, 0, 100, 100) FROM zyron_test.photos
```

The top left 100 by 100 pixels of each photo.

## Refused

- An offset or dimension is negative or above the 32-bit range.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [image_resize](image-resize.md)
- [image_rotate](image-rotate.md)
