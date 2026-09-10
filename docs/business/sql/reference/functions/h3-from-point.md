# h3_from_point

Divides the longitude and latitude ranges into 2 to the power of the resolution steps each and packs the resolution and the two cell numbers into one 64-bit index. The grid is a regular rectangular division of the coordinate ranges, not the icosahedral H3 grid, so an index from this function does not match an index from an H3 library and must not be stored as one.

## Syntax

```sql
h3_from_point(lon, lat, resolution)
```

## Returns

BIGINT holding the packed index. NULL when any argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `resolution` | Grid level between 0 and 15. Each step halves the cell width on both axes. | Not applicable. |

## Examples

```sql
SELECT h3_distance(h3_from_point(0, 0, 6), h3_from_point(0, 0, 6))
```

0, because both points fall in one cell.

## Refused

- The resolution is negative or above 15.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [h3_to_boundary](h3-to-boundary.md)
- [h3_distance](h3-distance.md)
- [st_make_point](st-make-point.md)
