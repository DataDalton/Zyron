# matrix_subtract

Subtracts each element of the second matrix from the element at the same position in the first. Both matrices must carry the same row and column counts.

## Syntax

```sql
matrix_subtract(a, b)
```

## Returns

MATRIX. NULL when either matrix is NULL.

## Examples

```sql
SELECT matrix_trace(matrix_subtract(matrix_identity(3), matrix_identity(3)))
```

0.

## Refused

- The two matrices differ in shape.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_add](matrix-add.md)
- [matrix_scalar_multiply](matrix-scalar-multiply.md)
