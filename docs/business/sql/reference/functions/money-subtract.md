# money_subtract

Subtracts the second value from the first. Both must be in the same currency, and differing currencies raise an error. The result may be negative.

## Syntax

```sql
money_subtract(money, money)
```

## Returns

MONEY in the shared currency. NULL when either argument is NULL.

## Examples

```sql
SELECT money_subtract(money_create(2000, 'USD'), money_create(1, 'USD'))
```

A MONEY value of 19.99 US dollars.

## Refused

- The two values are in different currencies.

## On a cluster

This runs on the member it arrives at and reaches no other. Nothing about it enters the consensus log.

## See also

- [money_add](money-add.md)
- [money_convert](money-convert.md)
