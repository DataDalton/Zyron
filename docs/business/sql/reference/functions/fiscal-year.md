# fiscal_year

Returns the calendar year the fiscal year began in, so a date before the start month belongs to the year before it. A fiscal year named for the year it ends in is this figure plus one.

## Syntax

```sql
fiscal_year(date, fy_start_month)
```

## Returns

INTEGER. NULL when either argument is NULL or the start month is outside 1 to 12.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `fy_start_month` | Calendar month the fiscal year begins, 1 through 12. | Not applicable. |

## Examples

```sql
SELECT fiscal_year(0, 7)
```

1969, because January 1970 falls in the year that began in July 1969.

## Refused

- An argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fiscal_quarter](fiscal-quarter.md)
- [week_of_fiscal_year](week-of-fiscal-year.md)
