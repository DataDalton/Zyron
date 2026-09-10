# matrix_norm

Reduces a matrix to one number, with the norm chosen by name. The kind is matched without regard to case. Unlike the other matrix functions this accepts a non-square matrix under every kind.

## Syntax

```sql
matrix_norm(matrix, kind)
```

## Returns

DOUBLE PRECISION. NULL when the matrix or the kind is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `kind` | frobenius or fro for the square root of the summed squares, l1 or 1 for the largest column sum of absolute values, inf or infinity for the largest row sum, l2 or 2 or spectral for the largest singular value by power iteration. | Not applicable. |

## Examples

```sql
SELECT matrix_norm(matrix_create(2, 2, '[3,0,0,4]'), 'frobenius')
```

5.

## Refused

- The kind is not one of the four names.
- The kind argument is not text.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [matrix_scalar_multiply](matrix-scalar-multiply.md)
- [eigenvalues](eigenvalues.md)
- [svd](svd.md)
