# cross_product

Returns the three-element vector at right angles to both inputs, whose length is the area of the parallelogram they span. Both inputs must hold exactly three elements. Swapping the arguments negates the result.

## Syntax

```sql
cross_product(a, b)
```

## Returns

ARRAY of three numbers as JSON text. NULL when either vector is NULL.

## Examples

```sql
SELECT cross_product('[1,0,0]', '[0,1,0]')
```

[0.0,0.0,1.0].

## Refused

- Either vector holds a number of elements other than three.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [dot_product](dot-product.md)
