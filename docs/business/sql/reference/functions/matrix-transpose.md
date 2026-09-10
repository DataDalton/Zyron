# matrix_transpose

Writes the element at row i and column j to row j and column i, so an r by c matrix becomes c by r. Transposing twice returns the original.

## Syntax

```sql
matrix_transpose(matrix)
```

## Returns

MATRIX. NULL when the matrix is NULL.

## Examples

```sql
SELECT matrix_trace(matrix_transpose(matrix_create(2, 2, '[1,2,3,4]')))
```

5, because the diagonal is unchanged by a transpose.

## Refused

- The argument holds bytes that are not a matrix.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_multiply](matrix-multiply.md)
- [matrix_inverse](matrix-inverse.md)
