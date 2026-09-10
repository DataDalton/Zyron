# vector_dot

Multiplies the two vectors position by position and adds the results. This takes the VECTOR column type, where dot_product takes JSON arrays of numbers.

## Syntax

```sql
vector_dot(a, b)
```

## Returns

DOUBLE PRECISION. NULL when either vector is NULL.

## Examples

```sql
SELECT vector_dot(embedding, embedding) FROM zyron_test.documents
```

Each embedding's squared length.

## Refused

- The two vectors hold different numbers of elements.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vector_norm](vector-norm.md)
- [vector_angle](vector-angle.md)
- [dot_product](dot-product.md)
