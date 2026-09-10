# quantity_scale

Scales a quantity's value by a factor, keeping its unit and dimension.

## Syntax

```sql
quantity_scale(quantity, factor)
```

## Returns

QUANTITY in the input's unit. NULL when either argument is NULL.

## Examples

```sql
SELECT quantity_scale(quantity_create(2, 'kg'), 2.5)
```

A QUANTITY of 5 kilograms.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_multiply](quantity-multiply.md)
