# ip_sort_key

Encodes the address so that comparing two keys as bytes orders them by value, putting 10.0.0.2 before 10.0.0.10. An address that does not parse has no key and gives NULL, which is why ip_compare sorts those last rather than by key.

## Syntax

```sql
ip_sort_key(text)
```

## Returns

BYTEA. NULL when the text is NULL or is not an address.

## Examples

```sql
SELECT ip_sort_key('10.0.0.2') < ip_sort_key('10.0.0.10')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [ip_compare](ip-compare.md)
- [inet_parse](inet-parse.md)
- [natural_sort_key](natural-sort-key.md)
