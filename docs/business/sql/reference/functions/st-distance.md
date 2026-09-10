# st_distance

Measures along the earth's surface by the haversine formula, in metres, when either point carries SRID 4326, and as a straight line in coordinate units otherwise. Points parsed from text or built by st_make_point carry 4326, so the metre reading is the usual one. Only a point against a point is supported.

## Syntax

```sql
st_distance(a, b)
```

## Returns

DOUBLE PRECISION in metres at SRID 4326, otherwise in coordinate units. NULL when either geometry is NULL.

## Examples

```sql
SELECT st_distance(st_make_point(0, 0), st_make_point(1, 0))
```

Approximately 111195 metres.

## Refused

- Either argument is not a point.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_dwithin](st-dwithin.md)
- [st_centroid](st-centroid.md)
- [h3_distance](h3-distance.md)
