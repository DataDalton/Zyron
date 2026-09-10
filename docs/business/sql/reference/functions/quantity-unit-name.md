# quantity_unit_name

Returns the unit's name spelled out, such as kilometre rather than km. Suits display where a symbol would be ambiguous.

## Syntax

```sql
quantity_unit_name(quantity)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT quantity_unit_name(quantity_create(5, 'km'))
```

The text kilometre.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [quantity_dimension](quantity-dimension.md)
- [quantity_format](quantity-format.md)
