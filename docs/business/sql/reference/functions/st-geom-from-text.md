# st_geom_from_text

Reads POINT, LINESTRING and POLYGON in well-known text and assigns SRID 4326. A POLYGON may carry holes after its exterior ring. The type word is matched in upper case only, and any other type word is refused.

## Syntax

```sql
st_geom_from_text(wkt)
```

## Returns

GEOMETRY with SRID 4326. NULL when the text is NULL.

## Examples

```sql
SELECT st_area(st_geom_from_text('POLYGON((0 0, 0 2, 2 2, 2 0, 0 0))'))
```

4.

## Refused

- The type word is not POINT, LINESTRING or POLYGON.
- A POINT carries fewer than two coordinates.
- A POLYGON carries no ring.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_as_text](st-as-text.md)
- [st_geom_from_geojson](st-geom-from-geojson.md)
- [st_make_point](st-make-point.md)
