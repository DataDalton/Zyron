# format_currency

Writes a plain number as a currency amount, placing the symbol and choosing the group and decimal separators by locale. The locale decides both: 1.234,56 € and €1,234.56 are the same amount written for different readers.

## Syntax

```sql
format_currency(value, currency [, locale])
```

## Returns

TEXT. NULL when the value or the currency is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `locale` | The locale whose conventions the number follows. | The server's default locale is used. |

## Examples

```sql
SELECT format_currency(1234.56, 'USD')
```

The text $1,234.56.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_format](money-format.md)
- [format_number](format-number.md)
