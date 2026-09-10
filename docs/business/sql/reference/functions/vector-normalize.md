# vector_normalize

Divides each element by the vector's length, keeping its direction and discarding its magnitude. Two normalized vectors compared by dot product give their cosine similarity directly, which is why an index over normalized vectors can use the cheaper measure.

## Syntax

```sql
vector_normalize(vector)
```

## Returns

VECTOR of the same width. NULL when the vector is NULL.

## Examples

```sql
SELECT vector_norm(vector_normalize(embedding)) FROM zyron_test.documents
```

1 for every non-zero embedding.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vector_norm](vector-norm.md)
- [vector_angle](vector-angle.md)
- [cosine_similarity](cosine-similarity.md)
