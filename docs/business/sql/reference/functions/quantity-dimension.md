# quantity_dimension

Returns the dimension, such as length, mass or time. Two quantities may be added or compared when their dimensions match, whatever units they are held in, so grouping by dimension finds the values that can be combined.

## Syntax

```sql
quantity_dimension(quantity)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT quantity_dimension(quantity_create(5, 'km'))
```

The text length.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_convert](quantity-convert.md)
- [quantity_unit_name](quantity-unit-name.md)
