# vector_cross

Returns the vector at right angles to both inputs. Both must hold exactly three elements, because the cross product is defined in three dimensions. Swapping the arguments negates the result.

## Syntax

```sql
vector_cross(a, b)
```

## Returns

VECTOR of three elements. NULL when either vector is NULL.

## Examples

```sql
SELECT vector_norm(vector_cross(a, b)) FROM zyron_test.frames
```

The area of the parallelogram each pair spans.

## Refused

- Either vector holds a number of elements other than three.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [vector_dot](vector-dot.md)
- [cross_product](cross-product.md)
