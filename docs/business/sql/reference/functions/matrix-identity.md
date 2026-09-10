# matrix_identity

Builds a square matrix holding 1 on the diagonal and 0 elsewhere, which leaves any conformable matrix unchanged under multiplication. Size is capped at 4096 to bound the allocation, because the matrix holds n squared eight-byte values.

## Syntax

```sql
matrix_identity(n)
```

## Returns

MATRIX. NULL when the size is NULL.

## Examples

```sql
SELECT matrix_trace(matrix_identity(3))
```

3, the sum of three diagonal ones.

## Refused

- The size is negative or above 4096.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_create](matrix-create.md)
- [matrix_multiply](matrix-multiply.md)
- [matrix_inverse](matrix-inverse.md)
