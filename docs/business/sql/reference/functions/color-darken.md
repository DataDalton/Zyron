# color_darken

Converts to HSL, subtracts the amount from the lightness, clamps it into 0.0 to 1.0 and converts back, keeping the original alpha. An amount of 1.0 returns black. Darkening then lightening by the same amount does not always return the original colour, because the clamp and the channel rounding both lose information.

## Syntax

```sql
color_darken(color, amount)
```

## Returns

COLOR carrying the input's alpha. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `amount` | Lightness to remove, as a fraction. | Not applicable. |

## Examples

```sql
SELECT color_to_hex(color_darken(color_from_rgb(255, 255, 255), 1))
```

#000000.

## Refused

- The colour argument is not a colour or integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_lighten](color-lighten.md)
- [color_blend](color-blend.md)
