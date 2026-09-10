# money_format

Writes the value with its currency's symbol and its correct number of decimal places, so a Japanese yen amount carries none and a US dollar amount carries two.

## Syntax

```sql
money_format(money)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT money_format(money_create(1999, 'USD'))
```

The text $19.99.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [format_currency](format-currency.md)
- [money_minor_digits](money-minor-digits.md)
