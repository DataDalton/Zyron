# color_hex

Names the same function as color_from_hex. The name reads as a formatter, so color_to_hex is the one that writes hex text.

## Syntax

```sql
color_hex(text)
```

## Returns

COLOR, a packed RGBA value. NULL when the text is NULL.

## Examples

```sql
SELECT color_to_hex(color_hex('000000'))
```

#000000, parsed without a leading hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_hex](color-from-hex.md)
- [color_to_hex](color-to-hex.md)
