# image_rotate

Rotates by the given angle, reduced into 0 to 359 first, so 450 turns a quarter turn and -90 turns three quarters. An angle that is not a multiple of 90 enlarges the canvas to hold the turned image.

## Syntax

```sql
image_rotate(image, degrees)
```

## Returns

IMAGE in the source format. NULL when either argument is NULL.

## Examples

```sql
SELECT image_rotate(photo, 90) FROM zyron_test.photos
```

Each photo turned a quarter turn clockwise.

## Refused

- The angle argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [image_crop](image-crop.md)
- [image_resize](image-resize.md)
