# array_filter

Keeps the elements a lambda answers true for, in the order the array held them. The lambda names one element and yields a boolean. It is not evaluated once per element: every element of every row in a batch is gathered into one column and the body runs over that column in a single pass.

## Syntax

```sql
array_filter(arr, x -> predicate)
```

## Returns

An array of the same element type.

## Examples

```sql
SELECT array_filter(ARRAY[1, 2, 3], x -> x > 1)
```

An array holding 2 and 3.

## Refused

- The lambda yields something other than a boolean.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [array_transform](array-transform.md)
