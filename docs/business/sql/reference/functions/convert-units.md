# convert_units

Converts a bare number between two units of the same dimension, without building a QUANTITY. Use it where the unit is known from context and does not need to travel with the value. Units of differing dimensions are an error.

## Syntax

```sql
convert_units(value, from_unit, to_unit)
```

## Returns

DOUBLE PRECISION. NULL when any argument is NULL.

## Examples

```sql
SELECT convert_units(1, 'km', 'm')
```

1000.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_convert](quantity-convert.md)
