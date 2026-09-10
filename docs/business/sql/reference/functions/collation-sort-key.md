# collation_sort_key

Encodes the string so that comparing two keys as bytes gives what collated_compare gives for the originals. ORDER BY over this key needs no comparison call per pair, and an index on it holds the locale's order rather than byte order.

## Syntax

```sql
collation_sort_key(text, locale, provider, case_sensitive)
```

## Returns

BYTEA. NULL when the text is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `case_sensitive` | Whether the key distinguishes two strings differing only in case. | Not applicable. |

## Examples

```sql
SELECT collation_sort_key('a', 'en-US', 'icu', false) < collation_sort_key('B', 'en-US', 'icu', false)
```

true.

## Refused

- Called with any count of arguments other than four.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [collated_compare](collated-compare.md)
- [natural_sort_key](natural-sort-key.md)
