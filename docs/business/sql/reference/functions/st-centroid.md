# st_centroid

Returns a point carrying the input's SRID. A point returns itself, a linestring returns the mean of its vertices rather than the midpoint along its length, and a polygon returns the area-weighted centre of its exterior ring, ignoring holes. A polygon whose ring encloses no area falls back to the mean of its vertices.

## Syntax

```sql
st_centroid(geometry)
```

## Returns

GEOMETRY, a point carrying the input's SRID. NULL when the geometry is NULL.

## Examples

```sql
SELECT st_as_text(st_centroid(st_geom_from_text('POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))')))
```

POINT(1 1).

## Refused

- The geometry is a collection or another unsupported type.
- A polygon exterior carries fewer than three points.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_area](st-area.md)
- [st_make_point](st-make-point.md)
