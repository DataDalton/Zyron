# format_ordinal

Appends st, nd, rd or th according to the English rules, including the exception that 11, 12 and 13 take th rather than following their last digit. The suffixes are English only.

## Syntax

```sql
format_ordinal(n)
```

## Returns

VARCHAR. NULL when the number is NULL.

## Examples

```sql
SELECT format_ordinal(12)
```

12th, not 12nd.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [format_number](format-number.md)
- [format_bytes](format-bytes.md)
