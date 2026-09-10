# cms_create

Allocates depth rows of width eight-byte counters behind a nine-byte header, so the sketch holds 9 plus width times depth times 8 bytes and does not grow. Width bounds how far an estimate can overshoot, and depth bounds how often it overshoots that far.

## Syntax

```sql
cms_create(width, depth)
```

## Returns

COUNTMINSKETCH. NULL when either argument is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `width` | Counters per row, between 1 and 4294967295. | Not applicable. |
| `depth` | Number of rows, between 1 and 16. Each row hashes independently. | Not applicable. |

## Examples

```sql
SELECT cms_create(2048, 5)
```

An empty sketch of 81929 bytes.

## Refused

- The width is zero or above 4294967295.
- The depth is zero or above 16.
- Called with any count of arguments other than two.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cms_add](cms-add.md)
- [cms_estimate](cms-estimate.md)
- [cms_merge](cms-merge.md)
