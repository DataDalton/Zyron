# h3_distance

Returns the larger of the two axis differences, counting a diagonal step as one. Both indexes must carry the same resolution, because a cell count means nothing across two grid sizes. The figure counts cells rather than ground distance, which st_distance measures.

## Syntax

```sql
h3_distance(a, b)
```

## Returns

INTEGER counting cells. NULL when either index is NULL.

## Examples

```sql
SELECT h3_distance(h3_from_point(0, 0, 4), h3_from_point(90, 0, 4))
```

4, four cells apart along longitude at resolution 4.

## Refused

- The two indexes carry different resolutions.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [h3_from_point](h3-from-point.md)
- [h3_to_boundary](h3-to-boundary.md)
- [st_distance](st-distance.md)
