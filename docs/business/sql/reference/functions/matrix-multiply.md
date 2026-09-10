# matrix_multiply

Multiplies row by column, so the column count of the first matrix must equal the row count of the second, and the result carries the rows of the first and the columns of the second. The order matters, because the product of two matrices is not the same either way round. A 4 by 4 product runs a fixed-size path that reads both operands from their encoded bytes without decoding them into vectors first.

## Syntax

```sql
matrix_multiply(a, b)
```

## Returns

MATRIX. NULL when either matrix is NULL.

## Examples

```sql
SELECT matrix_determinant(matrix_multiply(matrix_create(2, 2, '[1,2,3,4]'), matrix_identity(2)))
```

-2, unchanged by multiplying through the identity.

## Refused

- The column count of the first does not equal the row count of the second.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_add](matrix-add.md)
- [matrix_transpose](matrix-transpose.md)
- [matrix_inverse](matrix-inverse.md)
