# matrix_determinant

Factors the matrix with partial pivoting and multiplies the diagonal, tracking the sign of the row swaps. A pivot below 1e-12 in absolute value returns 0 rather than an error, so a singular matrix reports as singular here and is refused by matrix_inverse.

## Syntax

```sql
matrix_determinant(matrix)
```

## Returns

DOUBLE PRECISION. NULL when the matrix is NULL.

## Examples

```sql
SELECT matrix_determinant(matrix_create(2, 2, '[1,2,2,4]'))
```

0, because the second row is twice the first.

## Refused

- The matrix is not square.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_inverse](matrix-inverse.md)
- [matrix_trace](matrix-trace.md)
- [eigenvalues](eigenvalues.md)
