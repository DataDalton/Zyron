# st_area

Sums the signed area of the exterior ring and subtracts the holes. The calculation is planar and carries the square of the coordinate unit, so a polygon in degrees returns square degrees rather than square metres, and the figure shrinks with latitude for the same ground area. A ring of fewer than three points contributes nothing.

## Syntax

```sql
st_area(polygon)
```

## Returns

DOUBLE PRECISION in squared coordinate units. NULL when the geometry is NULL.

## Examples

```sql
SELECT st_area(st_geom_from_text('POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))'))
```

9, in square coordinate units.

## Refused

- The argument is not a polygon.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_centroid](st-centroid.md)
- [st_buffer](st-buffer.md)
- [st_contains](st-contains.md)
