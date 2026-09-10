# st_intersects

Tests a point against a polygon through st_contains, two points for equality within 1e-9, and two polygons by whether either holds a vertex of the other. Two polygons that cross without either holding a vertex of the other, such as two overlapping rectangles forming a cross, report false.

## Syntax

```sql
st_intersects(a, b)
```

## Returns

BOOLEAN. NULL when either geometry is NULL.

## Examples

```sql
SELECT st_intersects(st_geom_from_text('POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))'), st_make_point(1, 1))
```

true.

## Refused

- The pair of types is neither point nor polygon on both sides.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_contains](st-contains.md)
- [st_union](st-union.md)
- [st_dwithin](st-dwithin.md)
