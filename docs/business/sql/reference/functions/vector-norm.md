# vector_norm

Returns the square root of the summed squares, which is the vector's Euclidean length. A vector of all zeros has length 0 and cannot be normalized.

## Syntax

```sql
vector_norm(vector)
```

## Returns

DOUBLE PRECISION. NULL when the vector is NULL.

## Examples

```sql
SELECT vector_norm(vector_normalize(embedding)) FROM zyron_test.documents
```

1 for every non-zero embedding.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vector_normalize](vector-normalize.md)
- [vector_dot](vector-dot.md)
- [matrix_norm](matrix-norm.md)
