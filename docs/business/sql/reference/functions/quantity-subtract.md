# quantity_subtract

Subtracts the second quantity from the first, converting to the first's unit. Quantities of different dimensions are an error. The result may be negative.

## Syntax

```sql
quantity_subtract(quantity, quantity)
```

## Returns

QUANTITY in the first argument's unit. NULL when either argument is NULL.

## Examples

```sql
SELECT quantity_subtract(quantity_create(1, 'km'), quantity_create(500, 'm'))
```

A QUANTITY of 0.5 kilometres.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_add](quantity-add.md)
