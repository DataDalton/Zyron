# matrix_create

Reads the values in row-major order, filling the first row before the second. The value count must equal rows times cols exactly. Every other matrix function takes the encoding this returns, so a matrix literal is written through this function rather than as text.

## Syntax

```sql
matrix_create(rows, cols, values)
```

## Returns

MATRIX. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `values` | JSON array of numbers, row-major, holding rows times cols values. | Not applicable. |

## Examples

```sql
SELECT matrix_determinant(matrix_create(2, 2, '[1,2,3,4]'))
```

-2, the determinant of the 2 by 2 matrix holding rows 1 2 and 3 4.

## Refused

- The value count does not match the stated shape.
- The values argument is not a JSON array of numbers.
- A row or column count is negative or above the 32-bit range.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_identity](matrix-identity.md)
- [matrix_multiply](matrix-multiply.md)
- [matrix_determinant](matrix-determinant.md)
