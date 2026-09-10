# ip_compare

Returns -1, 0 or 1, comparing the addresses numerically rather than as text, so 10.0.0.2 sorts before 10.0.0.10. An address that does not parse sorts after every one that does, and two that do not parse compare as plain text.

## Syntax

```sql
ip_compare(a, b)
```

## Returns

INTEGER, one of -1, 0 or 1. NULL when either address is NULL.

## Examples

```sql
SELECT ip_compare('10.0.0.2', '10.0.0.10')
```

-1.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ip_sort_key](ip-sort-key.md)
- [inet_parse](inet-parse.md)
- [natural_compare](natural-compare.md)
