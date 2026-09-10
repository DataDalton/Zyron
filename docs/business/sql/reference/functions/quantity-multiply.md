# quantity_multiply

Multiplies a quantity by a plain number and keeps its unit. Multiplying two quantities would produce a compound dimension, such as a length times a length giving an area, and is not supported here.

## Syntax

```sql
quantity_multiply(quantity, factor)
```

## Returns

QUANTITY in the input's unit. NULL when either argument is NULL.

## Examples

```sql
SELECT quantity_multiply(quantity_create(5, 'km'), 3)
```

A QUANTITY of 15 kilometres.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_scale](quantity-scale.md)
- [quantity_add](quantity-add.md)
