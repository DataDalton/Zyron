# st_as_geojson

Writes a GeoJSON geometry object holding a type and a coordinates member. The result is the geometry alone, without the Feature wrapper or the properties member a GeoJSON consumer may expect.

## Syntax

```sql
st_as_geojson(geometry)
```

## Returns

TEXT holding a JSON object. NULL when the geometry is NULL.

## Examples

```sql
SELECT st_as_geojson(st_make_point(1, 2))
```

An object with type Point and coordinates [1.0,2.0].

## Refused

- The argument holds bytes that are not a geometry.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_geom_from_geojson](st-geom-from-geojson.md)
- [st_as_text](st-as-text.md)
