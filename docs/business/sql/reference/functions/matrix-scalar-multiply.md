# matrix_scalar_multiply

Scales each element by the same factor, leaving the shape unchanged. A determinant scales by the factor raised to the power of the row count, not by the factor itself.

## Syntax

```sql
matrix_scalar_multiply(matrix, scalar)
```

## Returns

MATRIX. NULL when the matrix or the scalar is NULL.

## Examples

```sql
SELECT matrix_trace(matrix_scalar_multiply(matrix_identity(3), 2))
```

6.

## Refused

- The matrix argument holds bytes that are not a matrix.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_add](matrix-add.md)
- [matrix_norm](matrix-norm.md)
