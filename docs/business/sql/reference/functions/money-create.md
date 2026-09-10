# money_create

Builds a MONEY value carrying both the amount and the currency it is denominated in. The amount is held in the currency's minor units as an exact integer, never as a float, so repeated arithmetic does not drift. The currency travels with the value, so an operation between two currencies is refused rather than producing a number with no meaning.

## Syntax

```sql
money_create(amount, currency)
```

## Returns

MONEY. NULL when either argument is NULL.

## Examples

```sql
SELECT money_create(1999, 'USD')
```

A MONEY value of 19.99 US dollars.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_add](money-add.md)
- [money_format](money-format.md)
- [money_currency_code](money-currency-code.md)
