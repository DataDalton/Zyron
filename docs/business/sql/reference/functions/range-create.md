# range_create

Encodes the two bounds as 8-byte order keys whose byte order matches numeric order, so a range compares and indexes without being decoded. A NULL bound means unbounded on that side rather than an unknown range. A lower bound above the upper, or two equal bounds with either side exclusive, produces the empty range rather than an error.

## Syntax

```sql
range_create(lower, upper, lower_inclusive, upper_inclusive)
```

## Returns

RANGE. NULL when either inclusivity flag is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `lower` | Lower bound. NULL leaves the range unbounded below. | Not applicable. |
| `upper` | Upper bound. NULL leaves the range unbounded above. | Not applicable. |
| `lower_inclusive` | Whether the lower bound is part of the range. | Not applicable. |
| `upper_inclusive` | Whether the upper bound is part of the range. | Not applicable. |

## Examples

```sql
SELECT range_contains_value(range_create(1, 10, true, false), 10)
```

false, because the upper bound is excluded.

## Refused

- A bound argument is not an integer column.
- An inclusivity argument is not a boolean column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [range_contains_value](range-contains-value.md)
- [range_is_empty](range-is-empty.md)
- [range_union](range-union.md)
