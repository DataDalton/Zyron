# eigenvalues

Runs Jacobi rotations for a fixed 100 sweeps and returns the resulting diagonal in descending order. The method assumes a symmetric matrix. A matrix that is square but not symmetric returns numbers without failing, and those numbers are not its eigenvalues.

## Syntax

```sql
eigenvalues(matrix)
```

## Returns

ARRAY as JSON text, one number per row, descending. NULL when the matrix is NULL.

## Examples

```sql
SELECT eigenvalues(matrix_create(2, 2, '[2,0,0,3]'))
```

[3.0,2.0], the diagonal in descending order.

## Refused

- The matrix is not square.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [svd](svd.md)
- [pca](pca.md)
- [matrix_trace](matrix-trace.md)
