# svd

Factors the matrix into the three parts named u, s and vt, returned as a JSON object holding one matrix under each key. The factorisation goes through the eigendecomposition of the matrix multiplied by its own transpose, which is accurate on small matrices and loses precision as the matrix grows or its singular values spread.

## Syntax

```sql
svd(matrix)
```

## Returns

COMPOSITE as JSON text with the keys u, s and vt, each a matrix holding rows, cols and data. NULL when the matrix is NULL.

## Examples

```sql
SELECT svd(matrix_create(2, 2, '[2,0,0,3]'))
```

An object holding u, s and vt, with s carrying the singular values 3 and 2.

## Refused

- The argument holds bytes that are not a matrix.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [pca](pca.md)
- [eigenvalues](eigenvalues.md)
- [matrix_norm](matrix-norm.md)
