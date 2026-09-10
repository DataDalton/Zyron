# st_as_text

Writes the geometry in the same well-known text form st_geom_from_text reads. The SRID is not part of the output, so a round trip through text reads back at SRID 4326 whatever the original carried.

## Syntax

```sql
st_as_text(geometry)
```

## Returns

TEXT. NULL when the geometry is NULL.

## Examples

```sql
SELECT st_as_text(st_make_point(1, 2))
```

POINT(1 2).

## Refused

- The argument holds bytes that are not a geometry.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_geom_from_text](st-geom-from-text.md)
- [st_as_geojson](st-as-geojson.md)
