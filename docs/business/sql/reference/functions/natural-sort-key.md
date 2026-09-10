# natural_sort_key

Encodes digit runs with a length prefix and text runs as their bytes, so comparing two keys as bytes gives the same answer as natural_compare on the originals. ORDER BY over this key sorts naturally without calling a comparison function per pair, and an index on it holds that order.

## Syntax

```sql
natural_sort_key(text)
```

## Returns

BYTEA. NULL when the text is NULL.

## Examples

```sql
SELECT natural_sort_key('item2') < natural_sort_key('item10')
```

true.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [natural_compare](natural-compare.md)
- [ip_sort_key](ip-sort-key.md)
- [collation_sort_key](collation-sort-key.md)
