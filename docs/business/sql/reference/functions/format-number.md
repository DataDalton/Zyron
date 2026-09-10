# format_number

Writes a number with the group and decimal separators a locale uses. A locale that groups by ten-thousands or separates decimals with a comma is followed.

## Syntax

```sql
format_number(value [, locale])
```

## Returns

TEXT. NULL when the value is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `locale` | The locale whose separators the number follows. | The server's default locale is used. |

## Examples

```sql
SELECT format_number(1234567.89)
```

The text 1,234,567.89.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [format_currency](format-currency.md)
- [format_percentage](format-percentage.md)
- [parse_number](parse-number.md)
