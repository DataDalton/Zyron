# format_duration

Writes the largest units that fit, so a long span reads in hours and minutes rather than in seconds. A negative count is written with a leading minus and a count of zero reads as 0s.

## Syntax

```sql
format_duration(seconds)
```

## Returns

VARCHAR. NULL when the count is NULL.

## Examples

```sql
SELECT format_duration(0)
```

0s.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [parse_natural_duration](parse-natural-duration.md)
- [format_bytes](format-bytes.md)
