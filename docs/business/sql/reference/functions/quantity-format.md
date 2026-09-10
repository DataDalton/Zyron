# quantity_format

Writes the value followed by its unit's symbol.

## Syntax

```sql
quantity_format(quantity)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT quantity_format(quantity_create(5, 'km'))
```

The text 5 km.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_unit_name](quantity-unit-name.md)
