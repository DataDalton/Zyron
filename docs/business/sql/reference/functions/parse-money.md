# parse_money

Reads a symbol or code, group separators and a decimal separator according to the locale, so the same digits read differently under two locales. A currency named in the text wins, and the locale's currency is assumed when the text names none.

## Syntax

```sql
parse_money(text [, locale])
```

## Returns

MONEY. NULL when the text is NULL.

## Clauses

| Clause | What it does | Left out |
| --- | --- | --- |
| `locale` | Locale whose separators and default currency apply. | The server's default locale is used. |

## Examples

```sql
SELECT money_currency_code(parse_money('$1,234.56'))
```

USD, taken from the symbol.

## Refused

- The text does not read as an amount under the locale.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_format](money-format.md)
- [money_create](money-create.md)
- [parse_number](parse-number.md)
