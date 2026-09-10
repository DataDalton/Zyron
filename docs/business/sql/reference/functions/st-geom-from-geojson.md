# st_geom_from_geojson

Reads a GeoJSON object holding a type and a coordinates member, covering Point, LineString and Polygon. The argument is one geometry object, not a Feature and not a FeatureCollection, so a member wrapping the geometry has to be selected out first.

## Syntax

```sql
st_geom_from_geojson(json)
```

## Returns

GEOMETRY with SRID 4326. NULL when the text is NULL.

## Examples

```sql
SELECT st_as_text(st_geom_from_geojson('{"type":"Point","coordinates":[1,2]}'))
```

POINT(1 2).

## Refused

- The text is not a JSON object.
- The object carries no type member.
- A Point carries fewer than two coordinates.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_as_geojson](st-as-geojson.md)
- [st_geom_from_text](st-geom-from-text.md)
