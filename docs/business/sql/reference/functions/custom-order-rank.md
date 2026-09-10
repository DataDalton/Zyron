# custom_order_rank

Returns where the value sits in the given list, so ORDER BY over this sorts by a declared sequence rather than alphabetically, as a status column usually needs. A value absent from the list goes first or last according to the third argument.

## Syntax

```sql
custom_order_rank(value, order [, unknown])
```

## Returns

INTEGER. NULL when the value or the order is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `order` | JSON array of values, in the order wanted. | Not applicable. |
| `unknown` | Where a value absent from the list goes, either first or last. | last. |

## Examples

```sql
SELECT custom_order_rank('high', '["low","medium","high"]')
```

The position of high in the list.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [natural_compare](natural-compare.md)
- [collated_compare](collated-compare.md)
