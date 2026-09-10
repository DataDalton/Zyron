# quantity_add

Adds two quantities, converting the second to the first's unit where the two measure the same dimension. Quantities of different dimensions, such as a length and a mass, are an error.

## Syntax

```sql
quantity_add(quantity, quantity)
```

## Returns

QUANTITY in the first argument's unit. NULL when either argument is NULL.

## Examples

```sql
SELECT quantity_add(quantity_create(1, 'km'), quantity_create(500, 'm'))
```

A QUANTITY of 1.5 kilometres.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_subtract](quantity-subtract.md)
- [quantity_convert](quantity-convert.md)
