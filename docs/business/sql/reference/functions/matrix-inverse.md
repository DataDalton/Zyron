# matrix_inverse

Solves for the matrix that multiplies with the input to give the identity, using Gauss-Jordan elimination with partial pivoting. A matrix whose pivot falls below 1e-12 in absolute value is treated as singular and refused, so a matrix that is invertible only in exact arithmetic is refused here.

## Syntax

```sql
matrix_inverse(matrix)
```

## Returns

MATRIX. NULL when the matrix is NULL.

## Examples

```sql
SELECT matrix_determinant(matrix_inverse(matrix_create(2, 2, '[4,7,2,6]')))
```

Approximately 0.1, the reciprocal of the original determinant of 10.

## Refused

- The matrix is not square.
- The matrix is singular or numerically indistinguishable from singular.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_determinant](matrix-determinant.md)
- [matrix_multiply](matrix-multiply.md)
- [matrix_identity](matrix-identity.md)
