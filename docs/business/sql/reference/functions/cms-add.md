# cms_add

Adds the count to one counter in each row, chosen by that row's hash of the value. Byte forms match hll_add. Distinct values sharing a counter accumulate together, which is why cms_estimate can overshoot and never undershoots.

## Syntax

```sql
cms_add(cms, value [, count])
```

## Returns

COUNTMINSKETCH. NULL when the sketch, the value or the count is NULL, or the sketch bytes do not parse.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `count` | How much to add for this value. | 1. |

## Examples

```sql
SELECT cms_estimate(cms_add(cms_create(256, 4), 'GET /health', 12), 'GET /health')
```

12.

## Refused

- The count is negative.
- Called with fewer than two or more than three arguments.
- The value is an interval, which has no byte form to hash.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cms_create](cms-create.md)
- [cms_estimate](cms-estimate.md)
- [cms_merge](cms-merge.md)
