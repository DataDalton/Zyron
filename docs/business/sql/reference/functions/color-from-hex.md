# color_from_hex

Accepts the three-digit, four-digit, six-digit and eight-digit forms, with or without a leading hash, where the short forms double each digit. A four or eight digit form carries alpha and the others set it to 255. Text that does not parse gives 0, which is fully transparent black, rather than an error.

## Syntax

```sql
color_from_hex(text)
```

## Returns

COLOR, a packed RGBA value, 0 when the text does not parse. NULL when the text is NULL.

## Examples

```sql
SELECT color_to_hex(color_from_hex('#f80'))
```

#ff8800, with each short digit doubled.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_to_hex](color-to-hex.md)
- [color_from_rgb](color-from-rgb.md)
- [color_hex](color-hex.md)
