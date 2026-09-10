# dot_product

Multiplies the two vectors position by position and adds the results. Both must hold the same number of elements. The arguments are JSON arrays of numbers rather than the matrix encoding.

## Syntax

```sql
dot_product(a, b)
```

## Returns

DOUBLE PRECISION. NULL when either vector is NULL.

## Examples

```sql
SELECT dot_product('[1,2,3]', '[4,5,6]')
```

32.

## Refused

- The two vectors differ in length.
- An argument is not a JSON array of numbers.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cross_product](cross-product.md)
- [matrix_multiply](matrix-multiply.md)
