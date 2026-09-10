# money_convert

Restates a money value in another currency by applying the rate given. The rate is supplied by the caller and is not looked up, so the conversion is reproducible and the statement records which rate was used. The result is rounded to the target currency's minor unit, which may have a different number of digits.

## Syntax

```sql
money_convert(money, currency, rate)
```

## Returns

MONEY in the target currency. NULL when any argument is NULL.

## Examples

```sql
SELECT money_convert(money_create(1000, 'USD'), 'EUR', 0.92)
```

A MONEY value of 9.20 euros.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_add](money-add.md)
- [currency_lookup](currency-lookup.md)
