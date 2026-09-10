# money_currency_symbol

Returns the currency's symbol, such as $ or ¥. Symbols are not unique across currencies, so a symbol identifies a currency for display and the ISO code identifies it for comparison.

## Syntax

```sql
money_currency_symbol(money)
```

## Returns

TEXT. NULL when the argument is NULL.

## Examples

```sql
SELECT money_currency_symbol(money_create(1999, 'USD'))
```

The text $.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_currency_code](money-currency-code.md)
- [money_format](money-format.md)
