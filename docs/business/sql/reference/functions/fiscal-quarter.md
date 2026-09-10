# fiscal_quarter

Counts months from the fiscal year's first month and divides by three, giving 1 through 4. The quarters are three calendar months each, so a fiscal year starting mid-month is not supported.

## Syntax

```sql
fiscal_quarter(date, fy_start_month)
```

## Returns

INTEGER between 1 and 4. NULL when either argument is NULL or the start month is outside 1 to 12.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `fy_start_month` | Calendar month the fiscal year begins, 1 through 12. | Not applicable. |

## Examples

```sql
SELECT fiscal_quarter(0, 7)
```

3, because January is the third quarter of a July fiscal year.

## Refused

- An argument is not an integer or date column.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [fiscal_year](fiscal-year.md)
- [week_of_fiscal_year](week-of-fiscal-year.md)
