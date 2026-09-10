# st_union

Returns a geometry collection carrying both inputs under the first one's SRID. Overlapping areas are not dissolved into a single boundary, so the result holds two members whose areas overlap rather than one merged outline, and st_area does not accept it.

## Syntax

```sql
st_union(a, b)
```

## Returns

GEOMETRY, a collection of two members. NULL when either geometry is NULL.

## Examples

```sql
SELECT st_as_text(st_union(st_make_point(0, 0), st_make_point(1, 1)))
```

A collection holding both points.

## Refused

- Either argument holds bytes that are not a geometry.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_intersects](st-intersects.md)
- [st_centroid](st-centroid.md)
