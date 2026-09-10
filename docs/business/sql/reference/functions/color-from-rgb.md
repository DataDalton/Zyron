# color_from_rgb

Packs red into the high byte, then green, then blue, with the alpha byte set to 255 for fully opaque. Each channel is truncated to its low eight bits rather than clamped, so 256 reads as 0. A NULL channel reads as 0 rather than producing NULL.

## Syntax

```sql
color_from_rgb(r, g, b)
```

## Returns

COLOR, a packed RGBA value. Never NULL.

## Examples

```sql
SELECT color_to_hex(color_from_rgb(255, 136, 0))
```

#ff8800.

## Refused

- A channel argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_rgba](color-from-rgba.md)
- [color_from_hex](color-from-hex.md)
- [color_to_hex](color-to-hex.md)
- [color_rgb](color-rgb.md)
