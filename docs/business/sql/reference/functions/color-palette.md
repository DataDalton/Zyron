# color_palette

Rotates the base colour's hue by the offsets the scheme names, holding saturation, lightness and alpha fixed, and returns the results as hex strings. The base colour is part of every scheme's output.

## Syntax

```sql
color_palette(color, scheme)
```

## Returns

ARRAY of hex strings as JSON text. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `scheme` | complementary for the base and its opposite, analogous for the base and the hues 30 degrees either side, triadic for three hues 120 degrees apart, and split-complementary or split_complementary for the base and the two hues either side of its opposite. Matched without regard to case. | Not applicable. |

## Examples

```sql
SELECT color_palette(color_from_rgb(255, 0, 0), 'complementary')
```

["#ff0000","#00ffff"].

## Refused

- The scheme is not one of the four names.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_hsl](color-from-hsl.md)
- [color_blend](color-blend.md)
- [wcag_contrast_ratio](wcag-contrast-ratio.md)
