# color_to_hex

Writes six lower-case hex digits behind a hash, adding two more for alpha only when the alpha is not 255. A round trip through color_from_hex therefore preserves the value, while a round trip through the six-digit form alone loses a non-opaque alpha.

## Syntax

```sql
color_to_hex(color)
```

## Returns

VARCHAR. NULL when the colour is NULL.

## Examples

```sql
SELECT color_to_hex(color_from_rgba(17, 34, 51, 255))
```

#112233, with no alpha digits because the colour is opaque.

## Refused

- The argument is not a colour or integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [color_from_hex](color-from-hex.md)
- [color_to_hsl](color-to-hsl.md)
