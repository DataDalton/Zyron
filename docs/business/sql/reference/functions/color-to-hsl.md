# color_to_hsl

Returns the three values in that order, with hue in degrees and the other two as fractions. A grey has no meaningful hue and reports 0. Alpha is not part of the result, so a blend performed in HSL space loses it unless it is carried separately.

## Syntax

```sql
color_to_hsl(color)
```

## Returns

ARRAY of three numbers as JSON text, hue then saturation then lightness. NULL when the colour is NULL.

## Examples

```sql
SELECT color_to_hsl(color_from_rgb(255, 0, 0))
```

[0.0,1.0,0.5].

## Refused

- The argument is not a colour or integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_hsl](color-from-hsl.md)
- [color_to_hex](color-to-hex.md)
- [color_lighten](color-lighten.md)
