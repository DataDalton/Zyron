# color_from_hsl

Converts hue in degrees with saturation and lightness as fractions into a packed colour with alpha 255. A saturation of 0 gives a grey at the given lightness whatever the hue. Channel values are rounded to the nearest byte, so a round trip through color_to_hsl does not always return the same numbers.

## Syntax

```sql
color_from_hsl(h, s, l)
```

## Returns

COLOR, a packed RGBA value with alpha 255. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `h` | Hue in degrees from 0 to 360. | Not applicable. |
| `s` | Saturation as a fraction from 0.0 for grey to 1.0 for full colour. | Not applicable. |
| `l` | Lightness as a fraction from 0.0 for black to 1.0 for white. | Not applicable. |

## Examples

```sql
SELECT color_to_hex(color_from_hsl(0, 1, 0.5))
```

#ff0000.

## Refused

- An argument is not a numeric column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_to_hsl](color-to-hsl.md)
- [color_lighten](color-lighten.md)
- [color_palette](color-palette.md)
- [color_hsl](color-hsl.md)
