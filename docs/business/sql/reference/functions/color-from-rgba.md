# color_from_rgba

Packs red, green, blue and alpha into one value, red in the high byte and alpha in the low byte. Each channel is clamped into 0 to 255 rather than truncated, unlike color_from_rgb. An alpha of 255 is fully opaque and 0 fully transparent.

## Syntax

```sql
color_from_rgba(r, g, b, a)
```

## Returns

COLOR, a packed RGBA value. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `a` | Alpha channel from 0 for fully transparent to 255 for fully opaque. | Not applicable. |

## Examples

```sql
SELECT color_to_hex(color_from_rgba(255, 136, 0, 128))
```

#ff880080, with the alpha byte kept because it is not 255.

## Refused

- A channel argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_rgb](color-from-rgb.md)
- [color_to_hex](color-to-hex.md)
- [color_blend](color-blend.md)
- [color_rgba](color-rgba.md)
