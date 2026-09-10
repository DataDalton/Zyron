# quantity_create

Builds a QUANTITY carrying a number and the unit it is measured in. The unit travels with the value, so arithmetic between incompatible dimensions is refused rather than producing a number whose unit nobody can name.

## Syntax

```sql
quantity_create(value, unit)
```

## Returns

QUANTITY. NULL when either argument is NULL.

## Examples

```sql
SELECT quantity_create(5, 'km')
```

A QUANTITY of 5 kilometres.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_add](quantity-add.md)
- [quantity_convert](quantity-convert.md)
- [quantity_dimension](quantity-dimension.md)
