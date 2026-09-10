# st_buffer

Builds a closed ring of 32 evenly spaced vertices at the given distance from the point. The distance is added in coordinate units, so a buffer around a geographic point is a circle in degrees and covers less ground east to west as latitude rises. Only a point is supported.

## Syntax

```sql
st_buffer(point, distance)
```

## Returns

GEOMETRY, a polygon of 32 sides. NULL when the geometry or the distance is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `distance` | Radius in coordinate units, which is degrees for a geometry at SRID 4326. | Not applicable. |

## Examples

```sql
SELECT st_contains(st_buffer(st_make_point(0, 0), 1), st_make_point(0.5, 0))
```

true.

## Refused

- The geometry is not a point.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_dwithin](st-dwithin.md)
- [st_area](st-area.md)
- [st_contains](st-contains.md)
