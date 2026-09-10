# format_bytes

Scales the count by 1000 per step and labels it B, KB, MB, GB, TB, PB or EB, so 1500 reads as 1.5 KB. The scaling is decimal rather than by 1024, which makes the figures match a disk manufacturer's rather than an operating system's. Digits shown fall as the value grows, and a count below 1000 is written in whole bytes.

## Syntax

```sql
format_bytes(bytes)
```

## Returns

VARCHAR. NULL when the count is NULL.

## Examples

```sql
SELECT format_bytes(1500)
```

1.5 KB.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [format_number](format-number.md)
- [format_duration](format-duration.md)
