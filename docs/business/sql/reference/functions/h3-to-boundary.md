# h3_to_boundary

Unpacks the resolution and cell numbers from the index and returns the cell outline as a four-cornered polygon at SRID 4326. The cells are rectangles in degrees, so a boundary is not a hexagon and its ground width narrows toward the poles.

## Syntax

```sql
h3_to_boundary(index)
```

## Returns

GEOMETRY, a polygon with SRID 4326. NULL when the index is NULL.

## Examples

```sql
SELECT st_area(h3_to_boundary(h3_from_point(0, 0, 1)))
```

16200, the square degrees of one cell at resolution 1.

## Refused

- The argument is not an integer column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [h3_from_point](h3-from-point.md)
- [h3_distance](h3-distance.md)
- [st_area](st-area.md)
