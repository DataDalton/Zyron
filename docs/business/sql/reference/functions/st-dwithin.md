# st_dwithin

Compares st_distance against the radius, inclusive at exactly the radius. The radius carries the unit st_distance returns, which is metres for points at SRID 4326. Only a point against a point is supported.

## Syntax

```sql
st_dwithin(a, b, radius)
```

## Returns

BOOLEAN. NULL when either geometry or the radius is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `radius` | Largest distance counted as within, in the unit st_distance returns. | Not applicable. |

## Examples

```sql
SELECT st_dwithin(st_make_point(0, 0), st_make_point(0.001, 0), 200)
```

true, because the points are about 111 metres apart.

## Refused

- Either argument is not a point.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [st_distance](st-distance.md)
- [st_contains](st-contains.md)
- [st_intersects](st-intersects.md)
