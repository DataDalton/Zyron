# color_rgba

Names the same function as color_from_rgba, with the same arguments and the same result.

## Syntax

```sql
color_rgba(r, g, b, a)
```

## Returns

COLOR, a packed RGBA value. NULL when any argument is NULL.

## Examples

```sql
SELECT color_to_hex(color_rgba(0, 0, 0, 0))
```

#00000000.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_rgba](color-from-rgba.md)
