# color_hsl

Names the same function as color_from_hsl. color_to_hsl is the one that reads a colour back out as hue, saturation and lightness.

## Syntax

```sql
color_hsl(h, s, l)
```

## Returns

COLOR, a packed RGBA value with alpha 255. NULL when any argument is NULL.

## Examples

```sql
SELECT color_to_hex(color_hsl(120, 1, 0.5))
```

#00ff00.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_hsl](color-from-hsl.md)
- [color_to_hsl](color-to-hsl.md)
