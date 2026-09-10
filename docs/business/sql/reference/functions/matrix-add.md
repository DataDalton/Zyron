# matrix_add

Adds each element of the first matrix to the element at the same position in the second. Both matrices must carry the same row and column counts, and the result carries them too.

## Syntax

```sql
matrix_add(a, b)
```

## Returns

MATRIX. NULL when either matrix is NULL.

## Examples

```sql
SELECT matrix_trace(matrix_add(matrix_identity(2), matrix_identity(2)))
```

4.

## Refused

- The two matrices differ in shape.
- An argument holds bytes that are not a matrix.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_subtract](matrix-subtract.md)
- [matrix_scalar_multiply](matrix-scalar-multiply.md)
- [matrix_multiply](matrix-multiply.md)
