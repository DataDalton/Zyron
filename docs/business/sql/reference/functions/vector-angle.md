# vector_angle

Returns the angle from the arc cosine of the cosine similarity, so two vectors pointing the same way give 0 and opposite ones give pi. Unlike a similarity the figure grows as the vectors diverge, which makes it usable directly as a distance.

## Syntax

```sql
vector_angle(a, b)
```

## Returns

DOUBLE PRECISION between 0 and pi. NULL when either vector is NULL.

## Examples

```sql
SELECT vector_angle(embedding, embedding) FROM zyron_test.documents
```

0, because a vector is at no angle to itself.

## Refused

- The two vectors hold different numbers of elements.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cosine_similarity](cosine-similarity.md)
- [vector_dot](vector-dot.md)
- [vector_normalize](vector-normalize.md)
