# st_contains

Casts a ray from the point and counts crossings of the exterior ring, then rejects the point when it falls inside a hole. The polygon comes first. Only a polygon against a point is supported, so a polygon inside another polygon is tested through st_intersects.

## Syntax

```sql
st_contains(polygon, point)
```

## Returns

BOOLEAN. NULL when either geometry is NULL.

## Examples

```sql
SELECT st_contains(st_geom_from_text('POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))'), st_make_point(2, 2))
```

true.

## Refused

- The arguments are not a polygon followed by a point.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_intersects](st-intersects.md)
- [st_dwithin](st-dwithin.md)
- [st_area](st-area.md)
