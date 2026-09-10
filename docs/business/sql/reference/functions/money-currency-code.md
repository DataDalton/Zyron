# money_currency_code

Returns the currency's alphabetic ISO 4217 code, such as USD or JPY. Grouping by this separates amounts that must not be summed together.

## Syntax

```sql
money_currency_code(money)
```

## Returns

TEXT, three characters. NULL when the argument is NULL.

## Examples

```sql
SELECT money_currency_code(money_create(1999, 'USD'))
```

The text USD.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_currency_symbol](money-currency-symbol.md)
- [currency_lookup](currency-lookup.md)
