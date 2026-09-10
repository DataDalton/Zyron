# quantity_convert

Restates a quantity in another unit measuring the same dimension, applying the fixed ratio between them. Unlike money_convert no rate is supplied, because unit ratios are definitions rather than market prices. A target unit of a different dimension is an error.

## Syntax

```sql
quantity_convert(quantity, unit)
```

## Returns

QUANTITY in the target unit. NULL when either argument is NULL.

## Examples

```sql
SELECT quantity_convert(quantity_create(1, 'km'), 'm')
```

A QUANTITY of 1000 metres.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [convert_units](convert-units.md)
- [quantity_dimension](quantity-dimension.md)
