# cms_estimate

Reads the counter the value addresses in each row and returns the smallest. Every counter holds the value's own count plus whatever collided with it, so the minimum is an upper bound that is never below the true count. Sketch bytes that do not parse give 0.

## Syntax

```sql
cms_estimate(sketch, value)
```

## Returns

BIGINT, never NULL. 0 when the sketch bytes do not parse.

## Examples

```sql
SELECT cms_estimate(cms_add(cms_add(cms_create(256, 4), 'a'), 'a'), 'a')
```

2.

## Refused

- Either argument is not text or binary.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [cms_create](cms-create.md)
- [cms_add](cms-add.md)
- [bloom_contains](bloom-contains.md)
