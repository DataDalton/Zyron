# color_blend

Interpolates each channel, alpha included, between the two colours. A ratio of 0.0 returns the first colour and 1.0 the second. The ratio is clamped into that span, so a value outside it behaves as the nearer end. Mixing happens in the packed channel space rather than in HSL, so two saturated colours can blend through a duller midpoint.

## Syntax

```sql
color_blend(a, b, ratio)
```

## Returns

COLOR. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `ratio` | How far from the first colour toward the second, from 0.0 to 1.0. | Not applicable. |

## Examples

```sql
SELECT color_to_hex(color_blend(color_from_rgb(0, 0, 0), color_from_rgb(255, 255, 255), 0.5))
```

#808080.

## Refused

- A colour argument is not a colour or integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_lighten](color-lighten.md)
- [color_darken](color-darken.md)
- [color_palette](color-palette.md)
