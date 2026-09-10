# week_of_fiscal_year

Counts whole weeks from the first day of the fiscal year and adds one, so week 1 always starts on the fiscal year's first day whatever weekday that is. The weeks do not align to calendar weeks, and a fiscal year spanning a leap day reaches week 53.

## Syntax

```sql
week_of_fiscal_year(date, fy_start_month)
```

## Returns

INTEGER between 1 and 53. NULL when either argument is NULL or the start month is outside 1 to 12.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `fy_start_month` | Calendar month the fiscal year begins, 1 through 12. | Not applicable. |

## Examples

```sql
SELECT week_of_fiscal_year(0, 1)
```

1, the first week of a January fiscal year.

## Refused

- An argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fiscal_year](fiscal-year.md)
- [fiscal_quarter](fiscal-quarter.md)
