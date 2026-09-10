# color_rgb

Names the same function as color_from_rgb, with the same arguments and the same result.

## Syntax

```sql
color_rgb(r, g, b)
```

## Returns

COLOR, a packed RGBA value. Never NULL.

## Examples

```sql
SELECT color_to_hex(color_rgb(0, 0, 255))
```

#0000ff.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_rgb](color-from-rgb.md)
