# parse_natural_duration

Accepts a sequence of counts and units, joined by spaces or by the word and, covering units from picosecond to millennium, as well as the colon form an interval literal uses. Months and years stay separate from days in the result, because neither has a fixed number of days. Text that does not parse gives NULL for that row.

## Syntax

```sql
parse_natural_duration(text)
```

## Returns

INTERVAL. NULL when the text is NULL or is not a duration.

## Examples

```sql
SELECT parse_natural_duration('2 hours and 30 minutes')
```

An interval of 2 hours 30 minutes.

## Refused

- The argument is not a text column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [parse_natural_date](parse-natural-date.md)
