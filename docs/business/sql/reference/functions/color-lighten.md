# color_lighten

Converts to HSL, adds the amount to the lightness, clamps it into 0.0 to 1.0 and converts back, keeping the original alpha. An amount of 0.0 leaves the colour alone and 1.0 returns white. Hue and saturation are unchanged, so lightening stays on the same colour rather than washing toward white through grey.

## Syntax

```sql
color_lighten(color, amount)
```

## Returns

COLOR carrying the input's alpha. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `amount` | Lightness to add, as a fraction. | Not applicable. |

## Examples

```sql
SELECT color_to_hex(color_lighten(color_from_rgb(0, 0, 0), 1))
```

#ffffff.

## Refused

- The colour argument is not a colour or integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_darken](color-darken.md)
- [color_blend](color-blend.md)
- [color_to_hsl](color-to-hsl.md)
