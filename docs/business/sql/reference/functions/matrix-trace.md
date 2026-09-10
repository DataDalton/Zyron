# matrix_trace

Adds the elements where the row index equals the column index. The trace equals the sum of the eigenvalues, which makes it a cheap check against an eigenvalue result.

## Syntax

```sql
matrix_trace(matrix)
```

## Returns

DOUBLE PRECISION. NULL when the matrix is NULL.

## Examples

```sql
SELECT matrix_trace(matrix_create(2, 2, '[1,2,3,4]'))
```

5.

## Refused

- The matrix is not square.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_determinant](matrix-determinant.md)
- [eigenvalues](eigenvalues.md)
