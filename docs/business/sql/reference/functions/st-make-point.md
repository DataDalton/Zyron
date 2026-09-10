# st_make_point

Builds a point carrying SRID 4326, so distances against it are measured on the earth rather than on a plane. Longitude comes first, matching the x before y ordering the geometry encoding uses and the opposite of how a coordinate pair is usually spoken.

## Syntax

```sql
st_make_point(lon, lat)
```

## Returns

GEOMETRY, a point with SRID 4326. NULL when either coordinate is NULL.

## Examples

```sql
SELECT st_as_text(st_make_point(-122.4, 37.8))
```

POINT(-122.4 37.8).

## Refused

- Either coordinate is text or binary rather than numeric.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_geom_from_text](st-geom-from-text.md)
- [st_distance](st-distance.md)
- [st_as_text](st-as-text.md)
