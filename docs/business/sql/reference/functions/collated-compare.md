# collated_compare

Orders the two strings by the rules of the locale rather than by byte value, so accented and case-varying letters sort where the language puts them. The locale, provider and case flag are read from the first row and apply to every row.

## Syntax

```sql
collated_compare(a, b, locale, provider, case_sensitive)
```

## Returns

INTEGER, one of -1, 0 or 1. NULL when either string is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `locale` | Locale whose ordering rules apply. | Not applicable. |
| `provider` | Collation provider to ask. | Not applicable. |
| `case_sensitive` | Whether two strings differing only in case compare as different. | Not applicable. |

## Examples

```sql
SELECT collated_compare('a', 'B', 'en-US', 'icu', false)
```

-1, because a sorts before b when case is ignored.

## Refused

- Called with any count of arguments other than five.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [collation_sort_key](collation-sort-key.md)
- [natural_compare](natural-compare.md)
